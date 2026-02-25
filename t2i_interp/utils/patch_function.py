# from contextlib import contextmanager
# from functools import wraps
# import inspect, importlib, types

# def _resolve_target(target):
#     """
#     target can be:
#       - dotted path: "torch.nn.Module.load_state_dict"
#       - (obj, "attrname"): (nn.Module, "load_state_dict")
#       - a function object (will need (obj, name) if it's a method)
#     Returns (owner_obj, attr_name, original_callable)
#     """
#     if isinstance(target, str):
#         mod_path, _, attr_path = target.rpartition(".")
#         obj = importlib.import_module(mod_path) if mod_path else globals()
#         owner = obj
#         for part in attr_path.split(".")[:-1]:
#             owner = getattr(owner, part)
#         name = attr_path.split(".")[-1]
#         return owner, name, getattr(owner, name)
#     elif isinstance(target, tuple) and len(target) == 2:
#         owner, name = target
#         return owner, name, getattr(owner, name)
#     elif isinstance(target, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
#         # Best-effort: find where it lives (may fail for some builtins)
#         owner = getattr(target, "__self__", None)
#         name = getattr(target, "__name__", None)
#         if owner is None or name is None:
#             raise ValueError("Pass a dotted path or (owner, 'name') for methods/builtins")
#         return owner, name, getattr(owner, name)
#     else:
#         raise TypeError("Unsupported target type")

# def _force_kwargs_wrapper(func, force_kwargs=None, arg_map=None, predicate=None):
#     """
#     force_kwargs: dict of kw -> value to force (overrides positional too)
#     arg_map: optional callable (*args, **kwargs) -> (args, kwargs) to mutate arbitrarily
#     predicate: optional callable (*args, **kwargs) -> bool to decide whether to intercept
#     """
#     force_kwargs = force_kwargs or {}
#     sig = None
#     try:
#         sig = inspect.signature(func)
#     except (TypeError, ValueError):
#         pass  # builtins may not have signatures

#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         if predicate and not predicate(*args, **kwargs):
#             return func(*args, **kwargs)

#         if arg_map:
#             args, kwargs = arg_map(*args, **kwargs)

#         if sig is not None:
#             # Robustly override even if library passed positionally
#             bound = sig.bind_partial(*args, **kwargs)
#             bound.apply_defaults()
#             for k, v in force_kwargs.items():
#                 if k in sig.parameters:
#                     bound.arguments[k] = v
#                 else:
#                     # If the kwarg isn't in signature, just add it (some funcs take **kwargs)
#                     bound.arguments.setdefault(k, v)
#             return func(*bound.args, **bound.kwargs)
#         else:
#             # Fallback: best effort for builtins
#             kwargs.update(force_kwargs)
#             return func(*args, **kwargs)

#     return wrapped

# @contextmanager
# def patch_function(target, *, force_kwargs=None, arg_map=None, predicate=None):
#     """
#     Temporarily patch any function/method to force kwargs and/or remap args.

#     target: dotted path str, (owner, 'attr'), or function (see _resolve_target)
#     force_kwargs: dict of kwarg -> value to force (overrides positionals)
#     arg_map: callable to rewrite (args, kwargs)
#     predicate: callable to decide which calls to intercept
#     """
#     owner, name, original = _resolve_target(target)
#     patched = _force_kwargs_wrapper(original, force_kwargs, arg_map, predicate)
#     try:
#         setattr(owner, name, patched)
#         yield
#     finally:
#         setattr(owner, name, original)

# @contextmanager
# def patch_many(patches):
#     """
#     patches: list of dicts, each with keys for patch_function()
#       e.g., [{"target": "torch.load", "force_kwargs": {"map_location": "cpu"}}]
#     """
#     originals = []
#     try:
#         for spec in patches:
#             owner, name, original = _resolve_target(spec["target"])
#             patched = _force_kwargs_wrapper(
#                 original,
#                 spec.get("force_kwargs"),
#                 spec.get("arg_map"),
#                 spec.get("predicate"),
#             )
#             originals.append((owner, name, original))
#             setattr(owner, name, patched)
#         yield
#     finally:
#         for owner, name, original in reversed(originals):
#             setattr(owner, name, original)
