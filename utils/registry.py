import fnmatch
from typing import Any, Dict
from torch import nn

def infer_arch_by_name(model: nn.Module, arch_map: Dict[str, Any]) -> str:
    cls = model.__class__.__name__
    for arch, spec in arch_map.items():
        for pat in spec.get("class_names", []):
            if fnmatch.fnmatch(cls, pat):
                return arch
    return "unknown"

def _discover_and_add(mapping, names, base_key, root_glob, slots):
    roots = [n for n in names if fnmatch.fnmatch(n, root_glob)]
    roots.sort()
    # If there is no {i} in base_key, we don't enumerate (e.g., embeds/vae root)
    enumerate_roots = "{i}" in base_key
    for idx, root in enumerate(roots if enumerate_roots else [root_glob]):
        rpath = root if enumerate_roots else roots[0] if roots else None
        if rpath is None:
            continue
        for slot, spec in slots.items():
            if isinstance(spec, str):
                path = f"{rpath}.{spec}"
            else:
                # derived (pattern/z): map to attention/parent root
                path = rpath
            key = (base_key.format(i=idx) if enumerate_roots else base_key) + f".{slot}"
            mapping[key] = path

def build_canonical_map_with_encoders(model: nn.Module, schema: Dict[str, Any]) -> Dict[str, str]:
    names = dict(model.named_modules())
    arch = infer_arch_by_name(model, schema.get("arch_map", {}))
    out: Dict[str, str] = {}

    # ---- encoders: text ----
    for enc in schema.get("encoders", {}).get("text", []):
        base = f"encoders.text.{enc['id']}"
        # embed (optional)
        if "embed" in enc:
            _discover_and_add(out, names, base_key=f"{base}.embed",
                              root_glob=enc["embed"]["discover"],
                              slots=enc["embed"]["slots"])
        # blocks.self_attn
        ta = enc["blocks"]["self_attn"]
        _discover_and_add(out, names,
                          base_key=f"{base}.blocks.self_attn.{{i}}",
                          root_glob=ta["discover"], slots=ta["slots"])
        # blocks.mlp
        tm = enc["blocks"]["mlp"]
        _discover_and_add(out, names,
                          base_key=f"{base}.blocks.mlp.{{i}}",
                          root_glob=tm["discover"], slots=tm["slots"])
        # final norm (optional)
        if "final_norm" in enc:
            _discover_and_add(out, names, base_key=f"{base}.final_norm",
                              root_glob=enc["final_norm"]["discover"],
                              slots=enc["final_norm"]["slots"])

    # ---- encoders: image (VAE roots) ----
    for img in schema.get("encoders", {}).get("image", []):
        base = f"encoders.image.{img['id']}"
        if img.get("type") == "vae" and "vae" in img:
            vae = img["vae"]
            _discover_and_add(out, names, f"{base}.vae.encoder", vae["encoder"]["discover"], vae["encoder"]["slots"])
            _discover_and_add(out, names, f"{base}.vae.decoder", vae["decoder"]["discover"], vae["decoder"]["slots"])
            if "misc" in vae:
                _discover_and_add(out, names, f"{base}.vae.misc", vae["misc"]["discover"], vae["misc"]["slots"])

    # ---- denoisers (same as before; omitted here for brevity) ----
    # use your existing UNet/DiT logic to fill:
    # - denoisers.0.steps.{k}.down|mid|up.self_attn.{i}.(q|k|v|out_proj|pattern|z)
    # - denoisers.0.steps.{k}.blocks.self_attn.{i}.(...), cross_attn likewise

    return out

# usage
# import yaml
# schema = yaml.safe_load(open("t2i_schema.yaml"))
# canon = build_canonical_map_with_encoders(my_model, schema)

# # Example lookups:
# canon["encoders.text.0.blocks.self_attn.3.q"]         # -> "...text_model.encoder.layers.3.self_attn.q_proj"
# canon["encoders.image.0.vae.encoder.quant_conv"]      # (if present)
# canon["denoisers.0.steps.{k}.down.cross_attn.2.k"]    # -> "...attn2.to_k"

