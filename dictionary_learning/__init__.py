# repo_root/dictionary_learning/__init__.py
import importlib as _imp

_inner = _imp.import_module(".dictionary_learning", __name__)  # <-- relative

# Re-export public API
__all__ = getattr(_inner, "__all__", [n for n in dir(_inner) if not n.startswith("_")])
globals().update({k: getattr(_inner, k) for k in __all__})

# Let subpackages resolve (e.g., dictionary_learning.utils)
try:
    __path__ = list(getattr(_inner, "__path__", [])) + list(__path__)  # type: ignore[name-defined]
except NameError:
    __path__ = list(getattr(_inner, "__path__", []))


def __getattr__(name):
    if hasattr(_inner, name):
        return getattr(_inner, name)
    return _imp.import_module(f".{name}", __name__)
