import os
import torch
import webdataset as wds
import numpy as np
from tqdm import tqdm
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
    max_samples: int = 30000,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    capture_step_index: int = 0,
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
        max_samples: Stop after this batch index.
        num_inference_steps: Diffusion inference steps.
        guidance_scale: Classifier-free guidance scale.
        capture_step_index: Which diffusion step's activations to capture.
        conditional_only: If True, only capture the conditional (second half) latents (requires CFG).

    Returns:
        save_path: The directory where tars were written.
    """
    # ---- decide output directory ----
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


    # ---- iterate dataset ----
    for num_document, batch in tqdm(enumerate(dataloader)):
        if num_document < start_at:
            continue
        if max_samples and num_document >= max_samples:
            break

        batch_start = num_document * batch_size

        # Iterate over prompt columns — each gets its own model run and tar
        for input_col in columns:
            prompts = batch.get(input_col)
            if prompts is None:
                print(f"Skipping column '{input_col}' for batch {num_document}: Not found.")
                continue

            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            B = len(prompts)

            # Run model (no reduce_fn — we handle conditional slicing after retrieval
            # so we always know the expected batch size B)
            try:
                cache = model.run_with_cache(
                    prompts,
                    accessors=real_accessors,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    return_output=False,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error running model on column {input_col}: {e}")
                continue

            # Write samples
            for i in range(B):
                key = f"sample_{batch_start + i}"

                # Save all batch data as extras
                sample_extras = {}
                for k, val in batch.items():
                    if f"{k}.pth" in sample_extras: continue
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
                        if capture_step_index == "all":
                            steps = sorted(list(act_data.keys()))
                            batch_tensor = torch.stack([act_data[s] for s in steps], dim=1)
                        elif capture_step_index in act_data:
                            batch_tensor = act_data[capture_step_index]
                        else:
                            steps = sorted(list(act_data.keys()))
                            batch_tensor = act_data[steps[-1]]
                    else:
                        batch_tensor = act_data

                    # Resolve CFG doubling: if conditional_only=True and the hook
                    # captured the full [uncond, cond] batch (2*B), slice the cond half.
                    # If the tensor already has exactly B samples (e.g. text encoder was
                    # called separately for uncond and cond), use it as-is — the last
                    # call captured in cache corresponds to the cond pass.
                    n_captured = batch_tensor.shape[0]
                    if conditional_only and n_captured == 2 * B:
                        # full doubled batch: take the cond half
                        batch_tensor = batch_tensor[B:]
                    elif conditional_only and isinstance(act_data, dict):
                        # encoder was called per-half
                        # If we stacked all steps (batch_tensor is 4D), taking the cond half
                        # depends on how the last_key aligns, but typically if n_captured == B,
                        # it's already exactly just the cond half (because UNet runs twice or handles doubled internally).
                        # Let's ensure we only overwrite if it's NOT a stacked step-conditional array
                        if capture_step_index != "all":
                            last_key = sorted(act_data.keys())[-1]
                            batch_tensor = act_data[last_key]

                    act = batch_tensor[i].detach().cpu()

                    block_sample = {
                        "__key__": key,
                        "output.pth": act,
                        **sample_extras
                    }
                    writers[(accessor_name, input_col)].write(block_sample)

    for w in writers.values():
        w.close()

    return save_path


def collect_latents_inmemory(
    accessors: list[str],
    dataset,
    model: "T2IModel",
    columns: list[str] = ["caption"],
    batch_size: int = 1,
    max_samples: int = 5000,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    capture_step_index: int = 0,
    conditional_only: bool = False,
    device: str = "cuda",
    dtype=None,
) -> dict[str, torch.Tensor]:
    """Collect activations from a model directly into GPU tensors — no disk I/O.

    Returns a dict mapping each accessor name to a stacked tensor of shape
    ``(N, *activation_shape)``, already on ``device`` with ``dtype``.

    Suitable for small datasets (≤10k samples) where the full activation set
    fits in GPU memory.  For large datasets use :func:`collect_latents` instead.
    """
    from torch.utils.data import DataLoader

    model.pipeline.set_progress_bar_config(disable=True)

    real_accessors = []
    for accessor_name in accessors:
        acc = model.resolve_accessor(accessor_name, io_type=IOType.OUTPUT)
        real_accessors.append(acc)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    # accumulate per (accessor, column) → list of per-sample tensors
    buffers: dict[tuple[str, str], list[torch.Tensor]] = {
        (acc, col): [] for acc in accessors for col in columns
    }

    n_collected = 0
    for num_document, batch in tqdm(enumerate(dataloader), desc="collect_inmemory"):
        if n_collected >= max_samples:
            break

        for input_col in columns:
            prompts = batch.get(input_col)
            if prompts is None:
                continue
            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            B = len(prompts)

            try:
                cache = model.run_with_cache(
                    prompts,
                    accessors=real_accessors,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    return_output=False,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[collect_inmemory] Error on batch {num_document}: {e}")
                continue

            for accessor_name in accessors:
                act_data = cache[accessor_name]
                if isinstance(act_data, dict):
                    if capture_step_index in act_data:
                        batch_tensor = act_data[capture_step_index]
                    else:
                        steps = sorted(act_data.keys())
                        batch_tensor = act_data[steps[-1]]
                else:
                    batch_tensor = act_data

                n_captured = batch_tensor.shape[0]
                if conditional_only and n_captured == 2 * B:
                    batch_tensor = batch_tensor[B:]

                for i in range(min(B, max_samples - n_collected)):
                    t = batch_tensor[i].detach()
                    if dtype is not None:
                        t = t.to(dtype=dtype)
                    buffers[(accessor_name, input_col)].append(t.to(device))

        # count by first column of first accessor
        first_col = columns[0]
        prompts_batch = batch.get(first_col)
        if prompts_batch is not None:
            B = len(prompts_batch) if not isinstance(prompts_batch, str) else 1
            n_collected += B

    result = {}
    for (acc, col), tensors in buffers.items():
        if tensors:
            key = acc if len(columns) == 1 else f"{acc}__{col}"
            result[key] = torch.stack(tensors, dim=0)
    return result