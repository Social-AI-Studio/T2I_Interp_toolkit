# --- registry.py ---
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional
import fnmatch
import torch
import torch.nn as nn

@dataclass
class CanonicalEntry:
    path: str                 # original dotted path in the model
    module: nn.Module         # the module object
    hook_kind: str = "post"   # 'post', 'pre', or 'module_out'

class CanonicalRegistry:
    """
    Build a mapping: canonical_name -> (original path, module, hook_kind).
    Use this to attach hooks by canonical name, without renaming the model.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.map: Dict[str, CanonicalEntry] = {}
        # Fast lookups
        self._named_modules = dict(model.named_modules())

    def _iter_glob(self, pattern: str):
        for name, mod in self._named_modules.items():
            if fnmatch.fnmatch(name, pattern):
                yield name, mod

    def add_by_glob(self, canon_key: str, glob_pattern: str, hook_kind: str = "post"):
        """Map a canonical key to the FIRST match of a glob pattern."""
        for name, mod in self._iter_glob(glob_pattern):
            self.map[canon_key] = CanonicalEntry(name, mod, hook_kind)
            return name
        raise KeyError(f"No module matched glob: {glob_pattern}")

    def add_by_path(self, canon_key: str, path: str, hook_kind: str = "post"):
        mod = self._named_modules.get(path)
        if mod is None:
            raise KeyError(f"Path not found: {path}")
        self.map[canon_key] = CanonicalEntry(path, mod, hook_kind)

    # ---- Hooks by canonical name ----
    def hook(self, canon_key: str, fn: Callable, when: Optional[str] = None):
        ent = self.map[canon_key]
        kind = when or ent.hook_kind
        if kind == "pre":
            return ent.module.register_forward_pre_hook(lambda m, inp: fn(m, inp, None))
        elif kind == "post" or kind == "module_out":
            return ent.module.register_forward_hook(lambda m, inp, out: fn(m, inp, out))
        else:
            raise ValueError(f"Unknown hook kind: {kind}")

    def get(self, canon_key: str) -> nn.Module:
        return self.map[canon_key].module

    def info(self) -> Dict[str, Tuple[str, str]]:
        return {k: (v.path, v.hook_kind) for k, v in self.map.items()}
