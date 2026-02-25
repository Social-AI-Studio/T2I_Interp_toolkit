"""  
Script to implement the hooking function for nethook
"""

# Libraries 
from typing import Optional, Union, Tuple, List, Callable, Dict, Any, Iterable
import torch as t
# from diffusers import StableDiffusionPipeline
# import torch.nn.functional as nnf
import numpy as np
import abc
import argparse 
import tqdm 
# from PIL import Image  
import contextlib
import copy
# import inspect
from collections import OrderedDict
from t2i_interp.utils.generic import StopForward

# Trace Class -- which performs the operation over only one layer
class Trace(contextlib.AbstractContextManager):
    """
    Attach ONE hook object to ONE module instance.
    No wrapper: we register hook_obj.hook directly.

    stop=True is supported only if hook_obj has attribute `.stop` and
    hook_obj.hook raises StopForward when self.stop is True.
    """

    def __init__(self, module: t.nn.Module, hook_obj: Any, stop: bool = False):
        if not isinstance(module, t.nn.Module):
            raise TypeError(f"module must be nn.Module, got {type(module)}")
        if not hasattr(hook_obj, "hook") or not callable(getattr(hook_obj, "hook")):
            raise TypeError("hook_obj must have a callable .hook(module, inputs, output)")

        self.module = module
        self.hook_obj = hook_obj
        self.stop = stop
        self._handle = None

        # for restoring hook_obj.stop (if present)
        self._has_stop_attr = hasattr(hook_obj, "stop")
        self._prev_stop = getattr(hook_obj, "stop", None)

        if self.stop:
            if not self._has_stop_attr:
                raise TypeError(
                    "stop=True requires hook_obj.stop to exist, and hook_obj.hook must raise StopForward when stop is enabled."
                )
            setattr(self.hook_obj, "stop", True)

        # IMPORTANT: no wrapper; direct registration
        if hasattr(self.hook_obj, "capture") and self.hook_obj.capture == "input":
             self._handle = self.module.register_forward_pre_hook(self.hook_obj.hook)
        else:
             self._handle = self.module.register_forward_hook(self.hook_obj.hook)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True  # suppress StopForward
        return False

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self.stop and self._has_stop_attr:
            setattr(self.hook_obj, "stop", self._prev_stop)


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    Attach hooks to multiple module instances.

    layers: iterable of nn.Module
    hook_objs: either a single hook object shared across layers,
              or a dict mapping module -> hook object

    stop=True enables stop only on the LAST module in the iterable, by toggling hook_obj.stop=True.
    """

    def __init__(
        self,
        layers: Iterable[t.nn.Module],
        hook_objs: Union[Any, Dict[t.nn.Module, Any]],
        stop: bool = False,
    ):
        super().__init__()
        self.stop = stop

        layers = list(layers)
        if len(layers) == 0:
            return

        last_layer = layers[-1]

        for layer in layers:
            if not isinstance(layer, t.nn.Module):
                raise TypeError(f"All layers must be nn.Module, got {type(layer)}")

            if isinstance(hook_objs, dict):
                if layer not in hook_objs:
                    raise KeyError("hook_objs dict missing a hook for a provided module instance")
                hook_obj = hook_objs[layer]
            else:
                hook_obj = hook_objs

            self[layer] = Trace(
                module=layer,
                hook_obj=hook_obj,
                stop=(stop and layer is last_layer),
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True
        return False

    def close(self):
        for _, tr in reversed(list(self.items())):
            tr.close()



