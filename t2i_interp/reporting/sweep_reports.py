import os
import json
from glob import glob
import wandb
from typing import List, Dict, Any
from omegaconf import DictConfig

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_constant_and_flat_cfgs(job_results: List[Dict[str, Any]]):
    all_cfgs = [res.get("cfg", {}) for res in job_results]
    flat_cfgs = [flatten_dict(c) for c in all_cfgs]
    all_keys = set()
    for fc in flat_cfgs:
        all_keys.update(fc.keys())
        
    constant_keys = set()
    for k in all_keys:
        first_val = None
        for fc in flat_cfgs:
            if k in fc:
                first_val = fc[k]
                break
        is_constant = True
        for fc in flat_cfgs:
            if k not in fc or fc[k] != first_val:
                is_constant = False
                break
        if is_constant:
            constant_keys.add(k)
            
    return flat_cfgs, constant_keys

def generate_default_table(config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """Standard table with one row per job."""
    columns = ["Job ID", "Configuration", "Metrics", "Generated Artifacts"]
    table = wandb.Table(columns=columns)
    
    flat_cfgs, constant_keys = get_constant_and_flat_cfgs(job_results)

    for i, result in enumerate(job_results):
        job_id = result.get("job_number", "Unknown")
        metrics = result.get("metrics", {})
        out_dir = result.get("output_dir")
        
        images = []
        if out_dir and os.path.isdir(out_dir):
            for img_path in sorted(glob(os.path.join(out_dir, "*.png"))):
                images.append(wandb.Image(img_path))
        
        filtered_cfg = {k: v for k, v in flat_cfgs[i].items() if k not in constant_keys}
        cfg_str = json.dumps(filtered_cfg, indent=2, default=str)
        metrics_str = json.dumps(metrics, indent=2, default=str)

        table.add_data(job_id, cfg_str, metrics_str, images)
    return table


def generate_steer_table(config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """Pivoted table specific to steering runs: prompts as rows, individual metric columns."""
    flat_cfgs, constant_keys = get_constant_and_flat_cfgs(job_results)
    
    # 1. Discover all metric names
    base_metric_names = set()
    for res in job_results:
        for k in res.get("metrics", {}).keys():
            if k.startswith("steered/"):
                base_metric_names.add(k.replace("steered/", "", 1))
            elif k.startswith("baseline/"):
                base_metric_names.add(k.replace("baseline/", "", 1))
            else:
                base_metric_names.add(k)
    ordered_metrics = sorted(list(base_metric_names))

    # 2. Build Columns
    # To avoid WandB UI crashes from mixed types or too many columns, we make each Run a separate Row.
    # Columns will be strictly typed: Prompt (str), Run (str), Image (wandb.Image), Metric1 (float), Metric2 (float)...
    metric_cols = [m.split('/')[-1] for m in ordered_metrics]
    columns = ["Prompt", "Run", "Image"] + metric_cols
            
    table = wandb.Table(columns=columns)

    prompts = config.get("prompts", [])
    if not prompts and len(job_results) > 0:
        prompts = job_results[0].get("cfg", {}).get("prompts", [])

    # Helper to extract and format scalar or list value strictly to native python to prevent tensor crashes
    def get_metric_val(metric_dict, key, idx):
        import math
        val = metric_dict.get(key)
        if isinstance(val, list):
            val = val[idx] if idx < len(val) else val[-1]
            
        if val is None:
            return None
            
        try:
            if hasattr(val, "item"):
                val = val.item() # convert numpy/torch to native
            if (isinstance(val, float) and math.isnan(val)) or str(val).lower().strip() == "nan":
                return None
            if isinstance(val, (int, float)):
                return val
            return float(val)
        except Exception:
            return None

    # 3. Populate Rows
    for i, prompt in enumerate(prompts):
        
        # --- Baseline Row ---
        baseline_img = None
        baseline_metrics = {}
        for res in job_results:
            out_dir = res.get("output_dir")
            if out_dir:
                bpath = os.path.join(out_dir, f"baseline_{i}.png")
                if os.path.exists(bpath):
                    baseline_img = wandb.Image(bpath)
            
            b_mdict = res.get("metrics", {})
            for m in ordered_metrics:
                if f"baseline/{m}" in b_mdict and m not in baseline_metrics:
                    baseline_metrics[m] = get_metric_val(b_mdict, f"baseline/{m}", i)
                    
        b_row = [str(prompt), "Baseline", baseline_img]
        for m in ordered_metrics:
            b_row.append(baseline_metrics.get(m, None))
        table.add_data(*b_row)
        
        # --- Steered Run Rows ---
        for j, res in enumerate(job_results):
            job_id = res.get("job_number", f"Job {j}")
            cfg = flat_cfgs[j]
            
            if "layer_names" in cfg and cfg["layer_names"]:
                ln = cfg["layer_names"]
                first = ln[0] if isinstance(ln, list) else ln
                if isinstance(ln, list) and len(ln) > 1:
                    # Extract block name: unet.down_blocks.0... → down_blocks
                    parts = str(first).split(".")
                    block = parts[1] if len(parts) > 1 else str(first)
                    label = f"unet.{block} ({len(ln)} layers)"
                else:
                    label = str(first)
            elif "layer_name" in cfg:
                label = str(cfg["layer_name"])
            elif "steer.layer_name" in cfg:
                label = str(cfg["steer.layer_name"])
            else:
                filtered_cfg = {k: v for k, v in cfg.items() if k not in constant_keys}
                cfg_str = ", ".join(f"{k}={v}" for k, v in filtered_cfg.items())
                label = f"Run {job_id}"
                if cfg_str: label += f" [{cfg_str}]"
            
            out_dir = res.get("output_dir")
            steered_img = None
            if out_dir:
                spath = os.path.join(out_dir, f"steered_{i}.png")
                if os.path.exists(spath):
                    steered_img = wandb.Image(spath)
                    
            s_row = [str(prompt), label, steered_img]
            
            s_mdict = res.get("metrics", {})
            for m in ordered_metrics:
                v = get_metric_val(s_mdict, f"steered/{m}", i)
                if v is None: # fallback
                    v = get_metric_val(s_mdict, m, i)
                s_row.append(v)
                
            table.add_data(*s_row)
            
    return table

def generate_localise_table(config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """Pivoted table specific to localisation sweeps: prompts as rows, layer/heads as columns, individual metric columns."""
    flat_cfgs, constant_keys = get_constant_and_flat_cfgs(job_results)
    
    # 1. Discover all metric names strictly omitting baseline and dynamic run layer names
    base_metric_names = set()
    for res in job_results:
        for k in res.get("metrics", {}).keys():
            m_split = k.split('/')
            if len(m_split) > 1:
                base_metric_names.add(m_split[-1])
            else:
                base_metric_names.add(k)
    ordered_metrics = sorted(list(base_metric_names))

    # 2. Build Columns
    # Columns: Prompt, Layer, Head, Image, Metric1, Metric2...
    columns = ["Prompt", "Layer", "Head", "Image"] + ordered_metrics
    table = wandb.Table(columns=columns)

    # 3. Populate Rows
    # In localisation sweeps, the prompt is usually scalar from the config, but we'll try to extract it from job_results cfg
    prompt = config.get("prompt", None)
    if not prompt and len(job_results) > 0:
        prompt = job_results[0].get("cfg", {}).get("prompt", "Unknown Prompt")
        
    def get_metric_val(metric_dict, key):
        import math
        val = metric_dict.get(key)
        if isinstance(val, list):
            val = val[-1]
        if val is None:
            return None
        try:
            if hasattr(val, "item"):
                val = val.item()
            if (isinstance(val, float) and math.isnan(val)) or str(val).lower().strip() == "nan":
                return None
            if isinstance(val, (int, float)):
                return val
            return float(val)
        except Exception:
            return None

    # --- Baseline Row ---
    baseline_img = None
    baseline_metrics = {}
    
    # Find the output_dir from the first valid job result to grab the baseline
    for res in job_results:
        out_dir = res.get("output_dir")
        if out_dir:
            bpath = os.path.join(out_dir, "baseline.png")
            if os.path.exists(bpath):
                baseline_img = wandb.Image(bpath)
        
        b_mdict = res.get("metrics", {})
        for m in ordered_metrics:
            if f"baseline/{m}" in b_mdict and m not in baseline_metrics:
                baseline_metrics[m] = get_metric_val(b_mdict, f"baseline/{m}")

    b_row = [str(prompt), "Baseline", None, baseline_img]
    for m in ordered_metrics:
        b_row.append(baseline_metrics.get(m, None))
    table.add_data(*b_row)
    
    # --- Localisation Run Rows ---
    for j, res in enumerate(job_results):
        out_dir = res.get("output_dir")
        if not out_dir: continue
        
        metrics = res.get("metrics", {})
        
        # In run_localisation.py, metrics are saved as `<layer>__h<head>/<metric>`
        # We need to find all unique layer__h prefixes saved in this job's metrics
        run_prefixes = set()
        for k in metrics.keys():
            if k.startswith("baseline/"): continue
            prefix = k.split('/')[0]
            if "__h" in prefix:
                run_prefixes.add(prefix)
                
        for prefix in run_prefixes:
            # prefix format: layer_name__h0
            layer_name, head_str = prefix.split("__h")
            head_idx = int(head_str)
            
            # Find the corresponding image
            img_path = os.path.join(out_dir, f"{prefix}.png")
            loc_img = wandb.Image(img_path) if os.path.exists(img_path) else None
            
            # Extract metrics for this specific head
            r_row = [str(prompt), layer_name, head_idx, loc_img]
            for m in ordered_metrics:
                v = get_metric_val(metrics, f"{prefix}/{m}")
                if v is None: v = get_metric_val(metrics, m) # fallback
                r_row.append(v)
                
            table.add_data(*r_row)

    return table


def generate_caa_alpha_table(config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """CAA-specific table with alpha as an explicit column.

    Rows:    one per (prompt, layer_block, alpha); baseline rows use alpha=0.
    Columns: Prompt | Layer | alpha | Image | <metric cols>
    """
    from collections import defaultdict

    flat_cfgs, _ = get_constant_and_flat_cfgs(job_results)

    # Discover all metric names (mirrors generate_steer_table)
    base_metric_names = set()
    for res in job_results:
        for k in res.get("metrics", {}).keys():
            if k.startswith("steered/"):
                base_metric_names.add(k.replace("steered/", "", 1))
            elif k.startswith("baseline/"):
                base_metric_names.add(k.replace("baseline/", "", 1))
            else:
                base_metric_names.add(k)
    ordered_metrics = sorted(base_metric_names)
    metric_cols = [m.split("/")[-1] for m in ordered_metrics]

    columns = ["Prompt", "Layer", "alpha", "Image"] + metric_cols
    table = wandb.Table(columns=columns)

    prompts = list(config.get("prompts", []))
    if not prompts and job_results:
        prompts = list(job_results[0].get("cfg", {}).get("prompts", []))

    def layer_label(cfg):
        ln = cfg.get("layer_names")
        if ln and isinstance(ln, list) and ln:
            first = str(ln[0])
            if len(ln) > 1:
                block = first.split(".")[1] if "." in first else first
                return f"unet.{block} ({len(ln)} layers)"
            return first
        return str(cfg.get("layer_name", "unknown"))

    def get_metric_val(metric_dict, key, idx):
        import math
        val = metric_dict.get(key)
        if isinstance(val, list):
            val = val[idx] if idx < len(val) else val[-1]
        if val is None:
            return None
        try:
            if hasattr(val, "item"):
                val = val.item()
            if (isinstance(val, float) and math.isnan(val)) or str(val).lower().strip() == "nan":
                return None
            if isinstance(val, (int, float)):
                return val
            return float(val)
        except Exception:
            return None

    # layer_label → alpha → (flat_cfg_index, job_result)
    layer_alpha: dict = defaultdict(dict)
    for j, res in enumerate(job_results):
        lbl = layer_label(flat_cfgs[j])
        raw_alpha = flat_cfgs[j].get("alpha") or res.get("cfg", {}).get("alpha", 0)
        a = float(raw_alpha) if raw_alpha is not None else 0.0
        print(f"[generate_caa_alpha_table] job={j} layer={lbl!r} alpha={a}")
        layer_alpha[lbl][a] = (j, res)

    for i, prompt in enumerate(prompts):
        for lbl, alpha_map in layer_alpha.items():

            # --- Baseline row (alpha = 0) ---
            baseline_img = None
            baseline_metrics: dict = {}
            for _a, (_j, res) in alpha_map.items():
                out_dir = res.get("output_dir")
                if out_dir:
                    p = os.path.join(out_dir, f"baseline_{i}.png")
                    if os.path.exists(p) and baseline_img is None:
                        baseline_img = wandb.Image(p)
                b_mdict = res.get("metrics", {})
                for m in ordered_metrics:
                    if f"baseline/{m}" in b_mdict and m not in baseline_metrics:
                        baseline_metrics[m] = get_metric_val(b_mdict, f"baseline/{m}", i)

            b_row = [str(prompt), lbl, 0, baseline_img]
            for m in ordered_metrics:
                b_row.append(baseline_metrics.get(m, None))
            table.add_data(*b_row)

            # --- One steered row per alpha value ---
            for a in sorted(alpha_map.keys()):
                _j, res = alpha_map[a]
                out_dir = res.get("output_dir")
                steered_img = None
                if out_dir:
                    p = os.path.join(out_dir, f"steered_{i}.png")
                    if os.path.exists(p):
                        steered_img = wandb.Image(p)

                s_row = [str(prompt), lbl, a, steered_img]
                s_mdict = res.get("metrics", {})
                for m in ordered_metrics:
                    v = get_metric_val(s_mdict, f"steered/{m}", i)
                    if v is None:
                        v = get_metric_val(s_mdict, m, i)
                    s_row.append(v)
                table.add_data(*s_row)

    return table


def generate_stitch_steer_table(config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """Table for stitch steer_contrast -m multirun sweeps.

    Each Hydra job runs steer_contrast for one (pos, neg, apply) triplet and
    writes a ``steer_meta.json`` file.  This function reads those files and
    produces one row per job.

    Rows:    one per sweep job
    Columns: Job | pos | neg | alpha | apply_prompt | steered | baseline
    """
    columns = ["Job", "pos → neg", "alpha", "apply", "steered", "baseline"]
    table = wandb.Table(columns=columns)

    for j, res in enumerate(job_results):
        job_id  = res.get("job_number", j)
        out_dir = res.get("output_dir")

        # Read per-job metadata written by run_stitch.py
        meta = {}
        if out_dir:
            meta_path = os.path.join(out_dir, "steer_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[generate_stitch_steer_table] Failed to read {meta_path}: {e}")

        pos   = meta.get("pos",   res.get("cfg", {}).get("steer_pos_prompts", ["?"])[0] if isinstance(res.get("cfg", {}).get("steer_pos_prompts"), list) else "?")
        neg   = meta.get("neg",   res.get("cfg", {}).get("steer_neg_prompts", ["?"])[0] if isinstance(res.get("cfg", {}).get("steer_neg_prompts"), list) else "?")
        alpha = float(meta.get("alpha", res.get("cfg", {}).get("steer_alpha", 1.0)))
        apply_prompts    = meta.get("apply", [])
        steered_paths    = meta.get("steered_paths",  [os.path.join(out_dir, f"steered_{i}.png")  for i in range(len(apply_prompts))] if out_dir else [])
        baseline_paths   = meta.get("baseline_paths", [os.path.join(out_dir, f"baseline_{i}.png") for i in range(len(apply_prompts))] if out_dir else [])

        # One row per apply prompt (or one row if no apply prompts)
        n_rows = max(len(apply_prompts), 1)
        for ai in range(n_rows):
            ap           = apply_prompts[ai] if ai < len(apply_prompts) else ""
            steered_img  = wandb.Image(steered_paths[ai])  if ai < len(steered_paths)  and os.path.exists(steered_paths[ai])  else None
            baseline_img = wandb.Image(baseline_paths[ai]) if ai < len(baseline_paths) and os.path.exists(baseline_paths[ai]) else None
            table.add_data(str(job_id), f"{pos} → {neg}", alpha, ap, steered_img, baseline_img)

    return table


def build_steer_grid_from_jobs(job_results: List[Dict[str, Any]], output_path: str) -> str | None:
    """Read each job's ``steer_meta.json``, assemble a combined grid image, and
    save it to ``output_path``.  Returns the saved path, or None if no metadata
    was found (e.g. all jobs failed or used the multi-pair path).

    This is called by ``WandbMultirunCallback.on_multirun_end`` when
    ``sweep_report_type == "stitch_steer"`` so that all per-job steered/baseline
    images are stitched together into one overview grid.

    Layout (one column-group per sweep job):
        header:   "pos → neg  (alpha=X)"
        rows:     steered image then baseline image, one column per apply-prompt
    """
    from PIL import Image as _Image
    from t2i_interp.utils.plot import make_steer_grid

    pairs_results = []
    for res in job_results:
        out_dir = res.get("output_dir")
        if not out_dir:
            continue
        meta_path = os.path.join(out_dir, "steer_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[build_steer_grid] Failed to read {meta_path}: {e}")
            continue

        steered_imgs  = [_Image.open(p) for p in meta.get("steered_paths",  []) if os.path.exists(p)]
        baseline_imgs = [_Image.open(p) for p in meta.get("baseline_paths", []) if os.path.exists(p)]

        if not steered_imgs:
            continue

        pairs_results.append({
            "pos":      meta.get("pos",   "pos"),
            "neg":      meta.get("neg",   "neg"),
            "apply":    meta.get("apply", [""]),
            "steered":  steered_imgs,
            "baseline": baseline_imgs,
        })

    if not pairs_results:
        print("[build_steer_grid] No steer_meta.json files found — skipping grid.")
        return None

    grid = make_steer_grid(pairs_results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid.save(output_path)
    print(f"[build_steer_grid] Combined grid saved → {output_path}")
    return output_path


def generate_sweep_table(report_type: str, config: DictConfig, job_results: List[Dict[str, Any]]) -> wandb.Table:
    """Factory method to get the correct W&B table based on report_type."""
    if report_type == "steer":
        return generate_steer_table(config, job_results)
    elif report_type == "localise":
        return generate_localise_table(config, job_results)
    elif report_type == "caa_alpha":
        return generate_caa_alpha_table(config, job_results)
    elif report_type == "stitch_steer":
        return generate_stitch_steer_table(config, job_results)
    else:
        return generate_default_table(config, job_results)
