# config_mapper.py
from __future__ import annotations

import copy
import fnmatch
import os
import pathlib
import re
from collections.abc import Iterable
from typing import Any

import yaml

# ---------------------------
# YAML loader: extends + $include (+ lists) + ${CONST} in KEYS & VALUES
# ---------------------------
PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = {**a}
        for k, v in b.items():
            out[k] = _deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    if isinstance(a, list) and isinstance(b, list):
        # merge lists of dicts with 'id' by id; else append
        if all(isinstance(x, dict) and "id" in x for x in a + b):
            by_id = {x["id"]: x for x in a}
            for x in b:
                i = x["id"]
                by_id[i] = _deep_merge(by_id[i], x) if i in by_id else copy.deepcopy(x)
            return list(by_id.values())
        return a + copy.deepcopy(b)
    return copy.deepcopy(b)


# --- file cache (avoid re-reading included files) ---
_YAML_CACHE: dict[pathlib.Path, Any] = {}


def _load_yaml_file(path: pathlib.Path) -> Any:
    p = path.resolve()
    if p not in _YAML_CACHE:
        _YAML_CACHE[p] = yaml.safe_load(p.read_text())
    return _YAML_CACHE[p]


def _select_subtree(obj: Any, spec: str) -> Any:
    # spec like "#a.b.c" or "#/a/b/c" or bare "a.b.c"
    if not spec:
        return obj
    if spec.startswith("#/"):
        parts = [p for p in spec[2:].split("/") if p]
    else:
        if spec.startswith("#"):
            spec = spec[1:]
        parts = [p for p in spec.split(".") if p]
    cur = obj
    for p in parts:
        cur = cur[int(p)] if isinstance(cur, list) and p.isdigit() else cur[p]
    return cur


def _resolve_includes(
    node: Any, base_dir: pathlib.Path, seen: set[tuple[pathlib.Path, str]] | None = None
) -> Any:
    """Resolve {"$include": "file.yaml#path"} (or list), prevent circular includes."""
    if seen is None:
        seen = set()

    if isinstance(node, dict):
        if "$include" in node:
            inc_spec = node["$include"]
            merged = {}
            specs = inc_spec if isinstance(inc_spec, list) else [inc_spec]
            for spec in specs:
                file, sep, sel = spec.partition("#")
                inc_path = (base_dir / file).resolve()
                key = (inc_path, sel)
                if key in seen:
                    raise RuntimeError(f"Circular $include detected: {inc_path}#{sel}")
                seen.add(key)
                inc_obj = _load_yaml_file(inc_path)
                inc_sub = _select_subtree(inc_obj, f"#{sel}" if sep else "")
                # Recurse within the included subtree, preserving base_dir of the include file
                merged = _deep_merge(merged, _resolve_includes(inc_sub, inc_path.parent, seen))
                seen.remove(key)
            # merge siblings (override included)
            siblings = {k: v for k, v in node.items() if k != "$include"}
            return _resolve_includes(_deep_merge(merged, siblings), base_dir, seen)
        return {k: _resolve_includes(v, base_dir, seen) for k, v in node.items()}

    if isinstance(node, list):
        return [_resolve_includes(x, base_dir, seen) for x in node]

    return node


def _subst_placeholders(obj: Any, consts: dict[str, Any]) -> Any:
    """Substitute ${CONST} in both keys and values. Supports ${NAME.path} and ${NAME:-default} and ${ENV:VAR}."""

    def resolve_token(tok: str):
        if tok.startswith("ENV:"):
            return os.getenv(tok.split(":", 1)[1], "")
        if ":-" in tok:
            name, default = tok.split(":-", 1)
        else:
            name, default = tok, None
        cur: Any = consts
        for seg in name.split("."):
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Missing const: {name}")
        return copy.deepcopy(cur)

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                # substitute in KEY
                if isinstance(k, str):
                    m = re.fullmatch(PLACEHOLDER_RE, k)
                    if m:
                        k = resolve_token(m.group(1))
                    else:
                        k = PLACEHOLDER_RE.sub(lambda mo: str(resolve_token(mo.group(1))), k)
                out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(i) for i in x]
        if isinstance(x, str):
            m = re.fullmatch(PLACEHOLDER_RE, x)
            if m:
                return resolve_token(m.group(1))
            return PLACEHOLDER_RE.sub(lambda mo: str(resolve_token(mo.group(1))), x)
        return x

    return walk(obj)


def load_config(yaml_path: str) -> dict[str, Any]:
    """Load top YAML with extends → includes → ${CONST} substitution (keys + values)."""
    path = pathlib.Path(yaml_path).resolve()
    raw = _load_yaml_file(path)

    # 1) extends (depth-first so nearer child wins on conflicts)
    base = {}
    for parent in raw.pop("extends", []) or []:
        base = _deep_merge(base, load_config(str((path.parent / parent).resolve())))

    merged = _deep_merge(base, raw)

    # 2) resolve $include (supports list, #subtree, circular-guard)
    merged = _resolve_includes(merged, base_dir=path.parent)

    # 3) collect constants from all layers (base/raw/merged.const)
    consts: dict[str, Any] = {}
    for src in (base.get("const"), raw.get("const"), merged.get("const")):
        if isinstance(src, dict):
            consts = _deep_merge(consts, src)

    # 4) substitute ${...} in keys and values
    merged = _subst_placeholders(merged, consts)

    # 5) optional cleanup
    merged.pop("const", None)
    return merged


# ---------------------------
# Mapper builder
# ---------------------------
def _all_module_names(root) -> dict[str, Any]:
    return {n: m for n, m in root.named_modules() if n != ""}


def _match(names: Iterable[str], pattern: str) -> Iterable[str]:
    # Support ** across dotted names
    pat = pattern.replace("**", "*")
    return [n for n in names if fnmatch.fnmatch(n, pat)]


def _join(base: str, sub: str | None) -> str:
    return f"{base}.{sub}" if sub else base


def build_module_mapper(
    hf_model: Any, top_yaml_path: str, *, strict: bool = False
) -> dict[str, str]:
    """
    Returns: dict { original_module_path_in_model : renamed_full_dotted_path }

    Behavior:
      - String slot  ("label": "sub.path"):
          base.sub.path             -> base.label
      - Dict slot with target ("label": {target: "ff", slots: {...}}):
          Parent:
            base.ff                 -> base.label (or .canonical if provided)
          Descendants:
            base.ff.<suffix>        -> base.label.<suffix>  (parent replaced)
            and if <suffix> starts with any child key (e.g., 'net.2'),
            replace that prefix with the child label (e.g., 'down'):
            base.ff.net.2(.rest)    -> base.label.down(.rest)
    """
    cfg = load_config(top_yaml_path)
    names = {n: m for n, m in hf_model.named_modules() if n != ""}
    out: dict[str, str] = {}

    def add(src: str, dst: str, ctx: str):
        if src in names:
            out[src] = dst
        elif strict:
            raise KeyError(f"[{ctx}] missing module: {src}")

    # glob match across dotted names (supports **)
    def _match(pattern: str) -> Iterable[str]:
        pat = pattern.replace("**", "*")
        return [n for n in names if fnmatch.fnmatch(n, pat)]

    def _join(a: str, b: str | None) -> str:
        return f"{a}.{b}" if b else a

    # apply child renames on a suffix using longest-prefix (segment) match
    def _apply_child_map(suffix: str, child_map: dict[str, str]) -> str:
        if not suffix:
            return suffix
        s = suffix[1:] if suffix.startswith(".") else suffix  # drop leading dot
        if not child_map:
            return suffix
        # longest key first so 'net.0.proj' beats 'net.0'
        for key in sorted(child_map.keys(), key=len, reverse=True):
            if s == key or s.startswith(key + "."):
                replaced = child_map[key] + s[len(key) :]
                return "." + replaced if replaced else ""
        return suffix

    # rename an entire subtree by replacing the parent segment and optionally remapping child prefixes
    def _rename_subtree(base_parent: str, new_parent: str, child_map: dict[str, str], ctx: str):
        # ensure parent itself maps
        add(base_parent, new_parent, ctx)
        prefix = base_parent + "."
        for path in names.keys():
            if path.startswith(prefix):
                suffix = path[len(base_parent) :]  # includes leading dot
                new_suffix = _apply_child_map(suffix, child_map)
                add(path, new_parent + new_suffix, ctx)

    # Uniform handler for any section that has 'discover' + 'slots'
    def handle_section(section: dict[str, Any], ctx: str):
        discover = section.get("discover")
        slots = section.get("slots", {})
        if not discover or not isinstance(slots, dict):
            return
        # print(discover, _match(discover))
        for base in _match(discover):
            for label, spec in slots.items():
                # Case 1: leaf mapping
                if isinstance(spec, str):
                    src = _join(base, spec)
                    dst = _join(base, label)
                    add(src, dst, ctx)
                    continue
                # print(245,spec)
                # Case 2: object mapping with target (+ optional nested slots)
                if isinstance(spec, dict):
                    target = spec.get("target")
                    canonical = spec.get("canonical")
                    inner = spec.get("slots", {}) or {}
                    if not target:
                        if strict:
                            raise ValueError(f"[{ctx}] slot '{label}' missing 'target'")
                        continue
                    # print(base,target)
                    src_parent = _join(base, target)
                    dst_parent = _join(base, (canonical or label))
                    # print(src_parent,dst_parent)
                    # build child prefix map: {"net.2": "down", "net.0.proj": "up", ...}
                    child_map: dict[str, str] = {}
                    for sublabel, subpath in inner.items():
                        if not isinstance(subpath, str):
                            if strict:
                                raise TypeError(
                                    f"[{ctx}] child slot '{sublabel}' must be str, got {type(subpath)}"
                                )
                            continue
                        child_map[subpath] = sublabel

                    # rename parent and all descendants under it
                    _rename_subtree(src_parent, dst_parent, child_map, ctx)
                    continue

                # Bad spec
                if strict:
                    raise TypeError(f"[{ctx}] slot '{label}' must be str or dict, got {type(spec)}")

    # Recursively find every section with {discover, slots} anywhere in the config
    def recurse(node: Any, prefix: str = "cfg"):
        if isinstance(node, dict):
            if "discover" in node and "slots" in node:
                handle_section(node, prefix)
            for k, v in node.items():
                recurse(v, f"{prefix}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                recurse(v, f"{prefix}[{i}]")

    # print(cfg)
    recurse(cfg)
    return out
