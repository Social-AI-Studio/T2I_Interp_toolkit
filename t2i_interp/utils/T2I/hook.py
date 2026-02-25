from typing import Callable, Optional, Union, Tuple, List, Any, Dict, Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict
import torch as t
from t2i_interp.utils.generic import _extract_tensor_and_rebuild, StopForward, _extract_tensor

Tensor = t.Tensor
StepIndex = Optional[Union[int, str, Iterable[int], slice]]
# Policy = Union[
#     Callable[..., Tensor],
#     Mapping[Union[int, str], Callable[..., Tensor]],
# ]


@dataclass(kw_only=True)
class BaseHook:
    enabled: bool = True
    call_counter: Optional[int] = None
    step_index: StepIndex = None   # None/int/"all"/iterable/slice
    stop: bool = False

    # --- internal: normalize gating once ---
    def __post_init__(self):
        # call_counter: if None, we still want deterministic gating;
        # so initialize to 0 on first use.
        self._step_gate = self._make_step_gate(self.step_index)

    @staticmethod
    def _make_step_gate(step_index: StepIndex):
        """
        Returns either:
          - None  => take all
          - callable(int)->bool gate
        """
        if step_index is None or step_index == "all":
            return None  # take all

        if isinstance(step_index, int):
            target = step_index
            return lambda n: n == target

        if isinstance(step_index, slice):
            start = 0 if step_index.start is None else step_index.start
            stop = step_index.stop  # can be None
            step = 1 if step_index.step is None else step_index.step

            def gate(n: int) -> bool:
                if n < start:
                    return False
                if stop is not None and n >= stop:
                    return False
                return ((n - start) % step) == 0

            return gate

        # iterable of ints
        try:
            idx_set = set(int(x) for x in step_index)  # may raise TypeError
        except TypeError as e:
            raise TypeError(
                f"step_index must be None, 'all', int, slice, or an iterable of ints; got {type(step_index)}"
            ) from e

        return lambda n: n in idx_set

    def _take_it(self) -> bool:
        if not self.enabled:
            return False

        # initialize call_counter lazily
        if self.call_counter is None:
            self.call_counter = 0
        else:
            self.call_counter += 1

        # gate
        if self._step_gate is None:
            return True
        return bool(self._step_gate(self.call_counter))

    # override these in subclasses
    def on_forward(self, module, inputs, output):
        return None

    def hook(self, module, inputs, output=None):
        if not self._take_it():
            return None
        out = self.on_forward(module, inputs, output)
        if self.stop:
            raise StopForward(
                f"Stopped at call={self.call_counter} module={module.__class__.__name__} step_index={self.step_index}"
            )
        return out


@dataclass
class AlterHook(BaseHook):
    policy: Callable
    # missing: str = "identity"  # "identity" | "error" | "default"
    cache: Optional[Dict[int, Tensor]] = None

    def _resolve_cache(self) -> Optional[Callable[..., Tensor]]:
        """
        Returns the policy callable to use for the *current* call_counter.
        - If self.policy is callable -> returns it.
        - If mapping -> tries exact key self.call_counter, else "default" if present.
        """
        if self.cache is None:
            return None  # direct callable case
        # mapping case
        step = self.call_counter
        # if step is None:
        #     # this shouldn't happen if BaseHook initializes/increments call_counter
        #     step = 0

        if step in self.cache:
            return self.cache[step]  # type: ignore[index]

        # if "default" in self.policy:
        #     return self.policy["default"]  # type: ignore[index]

        # if self.missing == "default":
        #     # allow missing="default" even without "default" key -> identity
        #     return None

        # if self.missing == "error":
        #     raise KeyError(f"No policy found for step={step} and no 'default' policy.")

        # missing == "identity"
        return None

    def _apply(self, x: Tensor, module: t.nn.Module, **ctx) -> Tensor:
        value = self._resolve_cache()
        out = self.policy(x, **{"value":value})
        return out.to(x.device)

@dataclass
class TextEncoderAlterHook(AlterHook):
    """
    Adds token-index context. Useful for policies that need "last non-pad token" etc.
    """
    last_token_indices: Optional[Tensor] = None  # [B]

    def set_token_indices(self, idx: Tensor):
        # idx: [B] on any device; we move to output's device at use-time
        self.last_token_indices = idx

    def on_forward(self, module: t.nn.Module, inputs, output):
        x, rebuild = _extract_tensor_and_rebuild(output)
        if x is None:
            return None

        new_x = self._apply(x, module, token_indices=self.last_token_indices)
        return rebuild(new_x)
    
@dataclass
class UNetAlterHook(AlterHook):
    """
    Applies policy either to full batch or only CFG conditional half (second half).
    """
    cfg_cond_only: bool = False

    def on_forward(self, module: t.nn.Module, inputs, output):
        # if not self._take_it():
        #     return None

        x, rebuild = _extract_tensor_and_rebuild(output)
        if x is None:
            return None

        if self.cfg_cond_only:
            # x is typically [2B, ...] under CFG
            b2 = x.shape[0]
            if b2 % 2 != 0:
                # Can't reliably split; apply to all.
                new_x = self._apply(x, module)
                return rebuild(new_x)

            b = b2 // 2
            x_uncond, x_cond = x[:b], x[b:]
            x_cond_new = self._apply(x_cond, module)
            new_x = t.cat([x_uncond, x_cond_new], dim=0)
            return rebuild(new_x)

        new_x = self._apply(x, module)
        return rebuild(new_x)

@dataclass
class CaptureHook(BaseHook):
    """
    Generic capture hook.

    - capture="output": captures module output tensor (or `.sample` for diffusers outputs)
    - capture="input": captures first tensor input argument
    """

    capture: str = "output"  # "input" | "output"
    tensor_index: int = 0
    reduce_fn: Optional[Callable[[Tensor], Tensor]] = None
    last: Optional[Tensor] = None
    cache: Optional[Dict[int, Tensor]] = None

    def _post(self, x: Tensor) -> None:
        if self.reduce_fn is not None:
            x = self.reduce_fn(x)
        self.last = x
        if self.cache is None:
            self.cache = {}
        step = 0 if self.call_counter is None else int(self.call_counter)
        self.cache[step] = x

    def on_forward(self, module: t.nn.Module, inputs, output):
        if self.capture == "input":
            tensor = _extract_tensor(inputs, self.tensor_index)
        elif self.capture == "output":
            tensor = _extract_tensor(output, self.tensor_index)
        else:
            raise ValueError("capture must be 'input' or 'output'")

        if tensor is not None:
            self._post(tensor)
        return None






