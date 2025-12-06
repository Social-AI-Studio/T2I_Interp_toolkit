import copy
import fnmatch
from collections.abc import Callable
from typing import Union

import torch.nn as nn

ModuleOrFactory = Union[nn.Module, Callable[[nn.Module, str], nn.Module]]


def _parent_and_attr(model: nn.Module, dotted: str):
    """
    Return (parent_module, attr_name) for a dotted path, e.g. 'blocks.0.attn'.
    Works for ModuleList indices too (e.g. 'blocks.0').
    """
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _all_named_modules(model: nn.Module):
    """Stable snapshot of named_modules so we can mutate safely later."""
    return list(model.named_modules())  # (name, module), includes root as ("", model)


def _device_dtype_of(mod: nn.Module):
    for t in list(mod.parameters()) + list(mod.buffers()):
        return t.device, t.dtype
    return None, None


def replace_modules(
    model: nn.Module,
    module_to_replace: type | tuple[type, ...] | str | list[str],
    new_module: ModuleOrFactory,
    *,
    copy_state: bool = True,
    strict_state: bool = False,
    name_scope: str = "",  # optional: only replace under this prefix (glob supported)
) -> int:
    """
    Replace modules in `model`.

    Args
    ----
    module_to_replace:
        - type or tuple of types  -> replace ALL instances of that type
        - str                     -> exact dotted path or a glob (supports * and **)
        - list[str]               -> multiple paths/globs
    new_module:
        - nn.Module instance      -> used as-is for single replacement (copied for many)
        - callable(old, name)->nn.Module  -> factory that receives the old module & its dotted name
    copy_state:     try to load old.state_dict() into new (strict=False by default)
    strict_state:   pass strict=True to load_state_dict if shapes match exactly
    name_scope:     optional glob prefix to limit replacements (e.g., "unet.**")

    Returns
    -------
    count of modules replaced.
    """
    # Gather targets
    named = _all_named_modules(model)
    # Build the set of (name, module) to replace
    targets: list[tuple[str, nn.Module]] = []

    def _match_name(n: str) -> bool:
        if not name_scope:
            return True
        return fnmatch.fnmatch(n, name_scope)

    if isinstance(module_to_replace, (type, tuple)):
        # Type-based
        for n, m in named:
            if n == "":  # skip root
                continue
            if isinstance(m, module_to_replace) and _match_name(n):
                targets.append((n, m))
    else:
        # Name/glob-based
        patterns = module_to_replace if isinstance(module_to_replace, list) else [module_to_replace]
        matched_names = set()
        for pat in patterns:
            is_glob = any(ch in pat for ch in "*?[]")
            for n, m in named:
                if n == "":
                    continue
                ok = fnmatch.fnmatch(n, pat) if is_glob else (n == pat)
                if ok and _match_name(n):
                    matched_names.add(n)
        # preserve the traversal order (by appearance in named)
        for n, m in named:
            if n in matched_names:
                targets.append((n, m))

    if not targets:
        return 0

    # Perform replacements
    replaced = 0
    for name, old in targets:
        parent, attr = _parent_and_attr(model, name)

        # Build a fresh new module
        if callable(new_module):
            new = new_module(old, name)  # factory decides how to wrap/copy
        else:
            # If single replacement: use as-is; for many, deepcopy so params aren’t shared
            new = new_module if len(targets) == 1 else copy.deepcopy(new_module)

        # Keep mode/device/dtype aligned
        dev, dt = _device_dtype_of(old)
        if dev is not None:
            new.to(device=dev, dtype=dt)
        new.train(old.training)

        # Optionally load weights from old into new (best-effort)
        if copy_state:
            try:
                sd = old.state_dict()
                new.load_state_dict(sd, strict=strict_state)
            except Exception:
                # Best-effort: ignore if shapes/keys don't match
                pass

        # Swap
        setattr(parent, attr, new)
        replaced += 1

    return replaced


# usage example:
# replaced = replace_modules(
#     unet,
#     module_to_replace="**attn2",      # all cross-attn modules in diffusers UNet
#     new_module=HookedCrossAttention.wrap_factory,
#     copy_state=True                  # state copied by default
# )
# print("replaced:", replaced)

# Swap one block
# count = replace_modules(
#     model,
#     module_to_replace="blocks.5.attn",    # exact dotted path
#     new_module=lambda old, name: Dummy(old,name=name),
# )
