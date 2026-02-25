import os
import torch
import webdataset as wds
import numpy as np
from tqdm import tqdm
import datetime
from datasets import load_dataset, Dataset, IterableDataset
from torch.utils.data import DataLoader
from t2i_interp.t2i import T2IModel
from t2i_interp.accessors.accessor import ModuleAccessor, IOType
from functools import reduce

def collect_latents(
    accessors: list[str],
    dataset: str | Dataset | IterableDataset,
    model: T2IModel,
    save_path: str,
    columns: list[str] = ["caption"],
    split: str = "train",
    data_files: str | list | dict | None = None,
    streaming: bool = True,
    batch_size: int = 1,
    start_at: int = 0,
    finish_at: int = 30000,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    capture_step_index: int = 0,
    filters: dict[str, list | set | callable] | None = None,
    conditional_only: bool = False,
):
    """Collect latent activations from a T2I model for a dataset.

    Args:
        accessors: Layer names to capture activations from.
        dataset: HF dataset name, or a Dataset/IterableDataset object.
        model: A T2IModel instance (with the correct pipeline already loaded).
        save_path: Directory to save WebDataset tars.
        columns: Dataset columns containing text prompts. Each column produces
                 a separate tar file and a separate model forward pass.
                 All other batch data is saved alongside activations automatically.
        split: Dataset split to load (ignored if dataset is already an object).
        data_files: Optional data files for load_dataset.
        streaming: Whether to stream the dataset.
        batch_size: Batch size for the DataLoader.
        start_at: Skip batches before this index.
        finish_at: Stop after this batch index.
        num_inference_steps: Diffusion inference steps.
        guidance_scale: Classifier-free guidance scale.
        capture_step_index: Which diffusion step's activations to capture.
        conditional_only: If True, only capture the conditional (second half) latents (requires CFG).

    Returns:
        save_path: The directory where tars were written.
    """
    # ---- decide output directory ----
    ct = datetime.datetime.now()
    save_path = os.path.join(save_path, str(ct))
    os.makedirs(save_path, exist_ok=True)

    # Writers: one per (accessor, prompt_column)
    writers = {}
    for acc in accessors:
        for col in columns:
            filename = f"{acc}_{col}.tar"
            writers[(acc, col)] = wds.TarWriter(os.path.join(save_path, filename))

    print(f"[collect_latents] Writing tars under: {save_path} (Format: {{accessor}}_{{column}}.tar)")
    
    # Disable diffusion progress bar
    model.pipeline.set_progress_bar_config(disable=True)

    # ---- init dataset ----
    if isinstance(dataset, (Dataset, IterableDataset)):
        print(f"Using provided dataset object.")
        ds = dataset
    else:
        print(f"Loading dataset {dataset} (split={split}, streaming={streaming})")
        load_args = {
            "path": dataset,
            "split": split,
            "streaming": streaming,
        }
        if data_files is not None:
            load_args["data_files"] = data_files
        try:
            ds = load_dataset(**load_args)
        except Exception as e:
            print(f"Error loading dataset {dataset}: {e}")
            raise e

    dataloader = DataLoader(ds, batch_size=batch_size)

    # ---- resolve accessors to modules using T2IModel helper ----
    real_accessors = []
    for accessor_name in accessors:
        try:
            acc = model.resolve_accessor(accessor_name, io_type=IOType.OUTPUT)
            real_accessors.append(acc)
        except Exception as e:
            raise ValueError(f"Could not resolve accessor {accessor_name} in model: {e}")

    # Define reduce_fn if conditional_only
    reduce_fn = None
    if conditional_only:
        def slice_cond(x):
            # Assume CFG: [uncond, cond]. Take second half.
            # B_total = 2 * B_prompts
            if x.shape[0] % 2 != 0:
                # Should not happen in standard CFG but safety check
                return x
            half = x.shape[0] // 2
            return x[half:]
        reduce_fn = slice_cond

    
# ---- optional per-sample filtering ----
def _to_py_scalar(v):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    return v

def _select_batch(batch_dict, idxs):
    out = {}
    for k, v in batch_dict.items():
        if isinstance(v, (str, bytes)):
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v[idxs]
        elif isinstance(v, (list, tuple)):
            out[k] = [v[j] for j in idxs]
        else:
            # fallback: try __getitem__
            try:
                out[k] = v[idxs]
            except Exception:
                out[k] = v
    return out

def _compute_filter_idxs(batch_dict, filters_spec):
    if not filters_spec:
        return None
    # Start with all indices
    # Find batch length from any sequence-like value
    B = None
    for v in batch_dict.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            B = v.shape[0]; break
        if isinstance(v, (list, tuple)):
            B = len(v); break
    if B is None:
        return None
    keep = [True] * B
    for key, rule in filters_spec.items():
        if key not in batch_dict:
            continue
        col = batch_dict[key]
        # normalize to python list
        if torch.is_tensor(col):
            vals = col.detach().cpu().tolist()
        elif isinstance(col, (list, tuple)):
            vals = list(col)
        else:
            vals = [col] * B
        for i, val in enumerate(vals):
            if not keep[i]:
                continue
            pyv = _to_py_scalar(val)
            ok = True
            if callable(rule):
                ok = bool(rule(pyv))
            else:
                allowed = set(rule) if not isinstance(rule, set) else rule
                ok = pyv in allowed
            if not ok:
                keep[i] = False
    idxs = [i for i, k in enumerate(keep) if k]
    return idxs

# ---- iterate dataset ----
    for num_document, batch in tqdm(enumerate(dataloader)):
        if num_document < start_at:
            continue
        if num_document >= finish_at:
            break

        batch_start = num_document * batch_size

        # Iterate over prompt columns — each gets its own model run and tar
        for input_col in columns:
            # Apply per-sample filters (e.g., race/gender) if provided
            idxs = _compute_filter_idxs(batch, filters)
            filtered_batch = _select_batch(batch, idxs) if idxs is not None else batch
            prompts = filtered_batch.get(input_col)
            if prompts is None:
                print(f"Skipping column '{input_col}' for batch {num_document}: Not found.")
                continue

            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            B = len(prompts)
            if B == 0:
                continue
            # track original dataset indices within this batch (useful when filtering)
            orig_idxs = idxs if idxs is not None else list(range(B))

            # Run model
            try:
                cache = model.run_with_cache(
                    prompts,
                    accessors=real_accessors,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    return_output=False,
                    reduce_fn=reduce_fn
                )
            except Exception as e:
                print(f"Error running model on column {input_col}: {e}")
                continue

            # Write samples
            for i in range(B):
                key = f"sample_{batch_start + orig_idxs[i]}"

                # Save all batch data as extras
                sample_extras = {}
                for k, val in filtered_batch.items():
                    if hasattr(val, "__getitem__") and not isinstance(val, (str, bytes)):
                        v = val[i]
                        if torch.is_tensor(v):
                            v = v.detach().cpu()
                        sample_extras[f"{k}.pth"] = v
                    else:
                        sample_extras[f"{k}.pth"] = val

                for accessor_name in accessors:
                    act_data = cache[accessor_name]
                    if isinstance(act_data, dict):
                        if capture_step_index in act_data:
                            batch_tensor = act_data[capture_step_index]
                        else:
                            steps = sorted(list(act_data.keys()))
                            batch_tensor = act_data[steps[-1]]
                    else:
                        batch_tensor = act_data
                    
                    act = batch_tensor[i].detach().cpu()

                    block_sample = {
                        "__key__": key,
                        "output.pth": act,
                        "__orig_index__.pth": torch.tensor(batch_start + orig_idxs[i]),
                        **sample_extras
                    }
                    writers[(accessor_name, input_col)].write(block_sample)

    for w in writers.values():
        w.close()

    return save_path