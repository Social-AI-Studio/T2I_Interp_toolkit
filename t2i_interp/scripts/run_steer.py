"""run_steer — entry point: ``t2i-steer``

t2i-steer
t2i-steer model_key=CompVis/stable-diffusion-v1-4 device=cuda:1
t2i-steer alpha=20 steer_steps=10
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from t2i_interp.config._hydra_config import config_dir
from t2i_interp.utils.utils import save_json


@hydra.main(config_path=config_dir(), config_name="steer/run", version_base=None)
def main(cfg: DictConfig) -> None:
    import sys
    import types

    import torch

    # The dev diffusers (LCM_LoRA) unconditionally imports xformers.ops in
    # attention_processor.py. That version of xformers was built for a
    # different torch version, so its native .so files fail to load.
    # Pre-populating sys.modules with stubs prevents those imports from
    # triggering native library loads; diffusers falls back to standard attention.
    if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
        torch.backends.cuda.is_flash_attention_available = lambda: False
    for _xf in [
        "xformers.ops",
        "xformers.ops.fmha",
        "xformers.ops.fmha.flash",
        "xformers.ops.fmha.common",
        "xformers.ops.fmha.triton_splitk",
        "xformers.flash_attn_3",
        "xformers.flash_attn_3._C",
    ]:
        sys.modules.setdefault(_xf, types.ModuleType(_xf))

    import transformers
    import wandb
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    from datasets import Dataset, DatasetDict, load_dataset

    from t2i_interp.linear_steering import KSteer
    from t2i_interp.mapper import MLPMapper
    from t2i_interp.reporting.fingerprint import (
        RunFingerprint,
        mark_run_completed as _mark_completed_impl,
        record_wandb_run,
        seed_everything,
    )
    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.inference import Inference, InferenceSpec
    from t2i_interp.utils.T2I.buffer import ActivationsDataloader, PairedLoader
    from t2i_interp.utils.T2I.collect_latents import collect_latents
    from t2i_interp.utils.training import Training, TrainingSpec

    print("=== t2i-steer config ===")
    print(OmegaConf.to_yaml(cfg))

    # Reproducibility: seed all RNGs before model load / data ops.
    seed_everything(getattr(cfg, "seed", None))

    # `prompts_file` (JSON list of strings) is the comma-safe alternative to
    # passing `prompts=[a, b]` on the Hydra command line — prompts containing
    # commas/spaces break the list parser. The Streamlit page writes this file.
    prompts_file = getattr(cfg, "prompts_file", None)
    if prompts_file and os.path.exists(prompts_file):
        import json

        with open(prompts_file) as _pf:
            _loaded_prompts = json.load(_pf)
        if not isinstance(_loaded_prompts, list) or not all(
            isinstance(p, str) for p in _loaded_prompts
        ):
            raise ValueError(f"prompts_file={prompts_file!r} must contain a JSON list of strings")
        OmegaConf.set_struct(cfg, False)
        cfg.prompts = _loaded_prompts
        OmegaConf.set_struct(cfg, True)

    # Resolve layer_names + finalize output_dir BEFORE wandb.init / fingerprint
    # / model load so all three downstream snapshots see the SAME `cfg`
    # (W&B's run.config, the fingerprint JSON, and the actual output path
    # should all agree).
    _layer_names_cfg = OmegaConf.to_container(getattr(cfg, "layer_names", None), resolve=True)
    layer_names = list(_layer_names_cfg) if _layer_names_cfg is not None else [cfg.layer_name]

    OmegaConf.set_struct(cfg, False)
    cfg.save_dir = os.path.abspath(cfg.save_dir)
    cfg.output_dir = os.path.abspath(cfg.output_dir)
    if layer_names:
        parts = layer_names[0].split(".")
        block = parts[1] if len(parts) > 1 else parts[0]
        cfg.output_dir = f"{cfg.output_dir}_{block}"
    alpha_suffix = float(getattr(cfg, "alpha", 0))
    cfg.output_dir = f"{cfg.output_dir}_alpha={alpha_suffix:g}"
    OmegaConf.set_struct(cfg, True)

    # Optional wandb initialization — must come AFTER cfg.output_dir is
    # finalized so the W&B run snapshot matches the fingerprint's snapshot.
    run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.get("project"):
        base_name = cfg.wandb.get("name", None)
        alpha_val = float(getattr(cfg, "alpha", 0))
        run_name = f"{base_name}_alpha={alpha_val:g}" if base_name else f"alpha={alpha_val:g}"
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            tags=cfg.wandb.get("tags", []),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Run fingerprint: written BEFORE model load (matches README §"Reproducibility").
    fingerprint = RunFingerprint.from_cfg(
        cfg,
        workflow="steer",
        intervention={
            "steer_type": getattr(cfg, "steer_type", None),
            "layer_names": list(layer_names),
            "alpha": getattr(cfg, "alpha", None),
            "steer_steps": getattr(cfg, "steer_steps", None),
            "guidance_scale": getattr(cfg, "guidance_scale", None),
            "num_inference_steps": getattr(cfg, "num_inference_steps", None),
        },
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    fingerprint.write(os.path.join(cfg.output_dir, "fingerprint.json"))
    print(f"[fingerprint] {fingerprint.hash()} → {cfg.output_dir}/fingerprint.json")
    if run is not None:
        fingerprint.log_to_wandb(run)
        record_wandb_run(cfg.output_dir, run)

    # 1. Model
    from diffusers import AutoPipelineForText2Image

    model = T2IModel(
        cfg.model_key, automodal=AutoPipelineForText2Image, device=cfg.device, dtype=cfg.dtype
    )
    model.pipeline.set_progress_bar_config(disable=True)

    def _load_inline_pairs_dataset(cfg):
        """Build an in-memory `DatasetDict` from inline {pos, neg} prompt pairs.

        Used by the Streamlit playground / quick demos. Patches `cfg` with
        the matching column names so the rest of the script is unaware of
        the source. Returns None if no inline pairs are configured — the
        caller falls back to `load_dataset(cfg.dataset_name)`.

        See `t2i_interp.utils.inline_pairs` for the load/split helpers shared
        with run_stitch.py.
        """
        from t2i_interp.utils.inline_pairs import load_inline_pairs, make_disjoint_split

        pairs = load_inline_pairs(cfg)
        if not pairs:
            return None

        # Validate the {pos, neg} shape early — easier to debug than a
        # downstream KeyError.
        for i, p in enumerate(pairs):
            if not isinstance(p, dict) or "pos" not in p or "neg" not in p:
                raise ValueError(
                    f"inline_pairs[{i}] must be a dict with 'pos' and 'neg' keys; "
                    f"got {type(p).__name__}={p!r}"
                )
            if not p["pos"] or not p["neg"]:
                raise ValueError(f"inline_pairs[{i}] has empty 'pos' or 'neg': {p!r}")

        steer_type_early = getattr(cfg, "steer_type", "caa")
        if steer_type_early == "loreft":
            # LoReFT wants paired columns on the SAME row: one base, one teacher.
            rows = [{"base_prompt": p["neg"], "teacher_prompt": p["pos"]} for p in pairs]
            patch = {"prompt_cols": ["base_prompt", "teacher_prompt"]}
        else:
            # CAA / KSteer want a single `prompt_col` + binary label per row.
            rows = []
            for p in pairs:
                rows.append({"caption": p["pos"], "label": 1})
                rows.append({"caption": p["neg"], "label": 0})
            patch = {
                "prompt_col": "caption",
                "label_col": "label",
                "pos_labels": [1],
                "neg_labels": [0],
            }
        OmegaConf.set_struct(cfg, False)
        for k, v in patch.items():
            setattr(cfg, k, v)
        OmegaConf.set_struct(cfg, True)

        train_rows, val_rows = make_disjoint_split(rows, seed=getattr(cfg, "seed", None))
        return DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "validation": Dataset.from_list(val_rows),
            }
        )

    # 2. Dataset — either inline prompt pairs (playground) or a HuggingFace dataset.
    ds_full = _load_inline_pairs_dataset(cfg) or load_dataset(cfg.dataset_name)
    ds_train = ds_full["train"]
    ds_val = (
        ds_full.get("validation")
        or ds_full.get("test")
        or ds_train.train_test_split(test_size=0.2, seed=42)["test"]
    )

    if getattr(cfg, "max_samples", None):
        ds_train = ds_train.select(range(min(len(ds_train), cfg.max_samples)))
        ds_val = ds_val.select(range(min(len(ds_val), max(1, cfg.max_samples // 5))))

    # Map string labels to integer indices if necessary BEFORE latents collate
    data_key = getattr(cfg, "label_col", None)
    if data_key and isinstance(ds_train[0][data_key], str):
        print(f"Converting string column '{data_key}' to integer indices.")
        unique_labels = sorted(set(ds_train[data_key]))
        label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        ds_train = ds_train.map(
            lambda x: {f"{data_key}_idx": label2idx[x[data_key]]}, desc=f"Mapping {data_key}"
        )
        ds_val = ds_val.map(
            lambda x: {f"{data_key}_idx": label2idx[x[data_key]]}, desc=f"Mapping {data_key}"
        )

        # Override the label_col so the rest of the script uses the new int index column
        OmegaConf.set_struct(cfg, False)
        cfg.label_col = f"{data_key}_idx"

        # Also convert pos_labels/neg_labels if they are strings
        pos_labels = OmegaConf.to_container(getattr(cfg, "pos_labels", [1, True]), resolve=True)
        neg_labels = OmegaConf.to_container(getattr(cfg, "neg_labels", [0, False]), resolve=True)

        cfg.pos_labels = [label2idx[l] if l in label2idx else l for l in pos_labels]
        cfg.neg_labels = [label2idx[l] if l in label2idx else l for l in neg_labels]
        OmegaConf.set_struct(cfg, True)
        print(f"Mapped config labels: pos={cfg.pos_labels}, neg={cfg.neg_labels}")

    # 4. Resolve steer-type specific variables before collecting latents
    steer_type = getattr(cfg, "steer_type", "ksteer")

    if steer_type == "loreft":
        prompt_cols = list(
            getattr(
                cfg,
                "prompt_cols",
                [getattr(cfg, "prompt_col", None), getattr(cfg, "prompt_col", None)],
            )
        )
        if len(prompt_cols) < 2 or not all(prompt_cols):
            raise ValueError(
                f"LoReFT requires two valid text columns (e.g. prompt_cols: ['student', 'teacher']). Got: {prompt_cols}"
            )
    else:
        prompt_col = getattr(cfg, "prompt_col", None)
        if not prompt_col:
            raise ValueError(f"Method {steer_type} requires 'prompt_col' in config.")
        prompt_cols = [prompt_col]

    def _find_tars(split, layer):
        base = os.path.join(cfg.save_dir, split)
        return [os.path.join(base, f"{layer}_{col}.tar") for col in prompt_cols]

    def _delete_layer_cache(layer):
        """Remove tar files for *layer* once its steering vector has been fitted."""
        for split in ("train", "val"):
            for tar in _find_tars(split, layer):
                print(f"[delete_cache] checking {tar} … exists={os.path.exists(tar)}")
                if os.path.exists(tar):
                    os.remove(tar)
                    print(f"[delete_cache] deleted {tar}")

    # 3. Collect latents
    def _collect(dataset, split):
        save_path = os.path.join(cfg.save_dir, split)
        all_exist = all(
            os.path.exists(t) for layer in layer_names for t in _find_tars(split, layer)
        )
        if all_exist:
            print(f"Skipping latent collection for {split}, all tars found")
            return save_path

        return collect_latents(
            accessors=layer_names,
            dataset=dataset,
            model=model,
            save_path=save_path,
            columns=prompt_cols,
            batch_size=cfg.batch_size,
            guidance_scale=cfg.guidance_scale,
            conditional_only=cfg.conditional_only,
            capture_step_index=getattr(cfg, "capture_step_index", 0),
            num_inference_steps=getattr(cfg, "num_inference_steps", 1),
            max_samples=cfg.max_samples,
        )

    print("Collecting latents...")
    _collect(ds_train, "train")
    _collect(ds_val, "val")

    def _loader(split, shuffle, layer):
        tars = _find_tars(split, layer)
        for tar in tars:
            if not os.path.exists(tar):
                raise FileNotFoundError(f"{tar} not found")

        loaders = []
        for tar in tars:
            loaders.append(
                ActivationsDataloader(
                    paths_to_datasets=[tar],
                    block_name=layer,
                    batch_size=16,
                    flatten=True,
                    shuffle=shuffle,
                    device=cfg.device,
                )
            )

        data_key = getattr(cfg, "label_col", None)
        if data_key:
            loaders.append(
                ActivationsDataloader(
                    paths_to_datasets=[tars[0]],
                    block_name=layer,
                    batch_size=16,
                    flatten=False,
                    shuffle=shuffle,
                    device=cfg.device,
                    data_key=f"{data_key}.pth",
                )
            )

        if len(loaders) > 1:
            # Multiple loaders must not shuffle independently — that breaks pairing.
            # Disable per-loader shuffle and let PairedLoader apply one joint shuffle.
            for l in loaders:
                l.shuffle = False
            return PairedLoader(loaders, shuffle=shuffle)
        return loaders[0]

    train_shuffle = getattr(cfg, "shuffle", True)

    # For KSteer/LoReFT we use the first layer; for CAA we may use multiple
    train_loader = _loader("train", shuffle=train_shuffle, layer=layer_names[0])
    val_loader = _loader("val", shuffle=False, layer=layer_names[0])

    # 5. Train
    sample = next(train_loader.iterate())
    is_tuple = isinstance(sample, (list, tuple))

    act = sample[0] if is_tuple else sample
    input_dim = act.reshape(act.shape[0], -1).shape[-1]

    # Auto-detect label dimension length
    label = sample[1] if is_tuple and len(sample) > 1 else None

    if steer_type == "caa":
        from t2i_interp.linear_steering import CAA

        steer = CAA(model=model)
        mapper = None
    elif steer_type == "loreft":
        from t2i_interp.linear_steering import LoReFT

        steer = LoReFT(model=model)
        mapper = None
    else:
        from t2i_interp.linear_steering import KSteer

        steer = KSteer(model=model)
        if (
            label is not None
            and torch.is_tensor(label)
            and len(label.shape) > 1
            and label.shape[1] > 1
        ):
            # Multi-dimensional label (like race+gender)
            output_dims = [label.shape[1]] if len(label.shape) == 2 else list(label.shape[1:])
            # Since MLPMapperTwoHeads expects exactly 2 dims, if not 2, just use MLPMapper with flat output_dim
            if len(output_dims) == 2:
                from t2i_interp.mapper import MLPMapperTwoHeads

                mapper = MLPMapperTwoHeads(input_dim=input_dim, output_dims=output_dims)
            else:
                mapper = MLPMapper(input_dim=input_dim, output_dim=label.shape[1])
        else:
            mapper = MLPMapper(input_dim=input_dim, output_dim=2)

    def collect_from_loader(loader, target_labels=None, max_samples=1000):
        collected = []
        count = 0
        if target_labels is not None and not isinstance(target_labels, list):
            target_labels = [target_labels]
        iterator = loader.iterate()
        try:
            while count < max_samples:
                batch_data = next(iterator)
                if isinstance(batch_data, (tuple, list)):
                    act = batch_data[0]
                    label = batch_data[1] if len(batch_data) > 1 else None
                else:
                    act = batch_data
                    label = None
                batch_size = act.shape[0]
                if isinstance(label, list):
                    label = label[0]
                if torch.is_tensor(label):
                    label = label.cpu().tolist()
                if not isinstance(label, list) and label is not None:
                    label = [label] * batch_size
                # Defensive cap. The act loader (flatten=True) treats each tar
                # entry's (seq, dim) activation as `seq` rows, while the label
                # loader (flatten=False) treats one entry as one row. PairedLoader
                # zips them per-batch, so an act batch of size 16 can pair with a
                # label batch of size <16 when the underlying entry count is
                # small (typical for inline-pair runs). Iterating beyond the
                # shorter side raises IndexError. We honour the labels we have
                # and drop the rest of the act batch — same semantics as the
                # pre-existing zip-truncation behaviour for full-batch runs.
                if isinstance(label, list) and len(label) < batch_size:
                    batch_size = len(label)
                for i in range(batch_size):
                    l = label[i] if label else None
                    if torch.is_tensor(l):
                        l = l.item()
                    if isinstance(l, list):
                        l = l[0]
                    if target_labels is None or l in target_labels:
                        collected.append(act[i].detach().cpu())
                        count += 1
                    if count >= max_samples:
                        break
        except StopIteration:
            pass
        if not collected:
            return torch.tensor([])
        return torch.stack(collected)

    if steer_type == "loreft":
        if not hasattr(steer, "lorefts"):
            steer.lorefts = {}

        for layer_name in layer_names:
            print(f"[{steer_type}] Fitting adapter for {layer_name}...")
            # `_loader` automatically builds ActivationsDataloaders and wraps them in a PairedLoader if prompt_cols > 1
            layer_train_loader = _loader("train", shuffle=train_shuffle, layer=layer_name)

            # LoReFT automatically extracts validations under the same prompt columns
            layer_val_loader = _loader("val", shuffle=False, layer=layer_name)

            Training(
                TrainingSpec(
                    training_function=steer.fit,
                    kwargs={
                        "loader": layer_train_loader,
                        "val_loader": layer_val_loader,
                        "layer_name": layer_name,
                        "rank": getattr(cfg, "lora_rank", 16),
                        "num_steps": cfg.train_steps,
                        "lr": cfg.lr,
                        "device": cfg.device,
                        "log_steps": getattr(cfg, "log_steps", max(1, cfg.train_steps // 10)),
                    },
                )
            ).run_trainer()

            # steer.fit() stores the trained adapter in self.loreft (always the last
            # trained one). Copy it into lorefts keyed by layer so steer.steer() can
            # look up the correct adapter per layer during multi-layer inference.
            steer.lorefts[layer_name] = steer.loreft
            print(f"[loreft] Stored adapter for {layer_name} in steer.lorefts")

            if getattr(cfg, "delete_cache", False):
                _delete_layer_cache(layer_name)
    elif mapper is not None:
        # KSteer / mapper-based path. Reuses the train_loader / val_loader
        # built upstream from the latent tar files — the old code re-created
        # them here referencing undefined `train_latents_dir` / `val_latents_dir`
        # vars (NameError on every ksteer run), see PLAN.md item B.7.

        def _get_target_labels(loader, attr_name="default"):
            """Extract pos_acts and neg_acts from dataloader to train linear stealers"""
            pos_labels = OmegaConf.to_container(getattr(cfg, "pos_labels", [1, True]), resolve=True)
            neg_labels = OmegaConf.to_container(
                getattr(cfg, "neg_labels", [0, False]), resolve=True
            )

            if attr_name == "default":
                # For backward compatibility
                p_labels, n_labels = pos_labels, neg_labels
            else:
                p_labels = [
                    label2idx[l] if l in label2idx else l
                    for l in pos_labels.get(attr_name, [1, True])
                ]
                n_labels = [
                    label2idx[l] if l in label2idx else l
                    for l in neg_labels.get(attr_name, [0, False])
                ]

            max_samples = getattr(cfg, "max_samples", 100)

            pos_acts = _collect_samples(loader, p_labels, max_samples)
            neg_acts = _collect_samples(loader, n_labels, max_samples)
            return {"pos_acts": pos_acts, "neg_acts": neg_acts}

        def _collect_samples(loader, target_labels, max_samples):
            collected = []
            count = 0
            try:
                for batch in loader.iterate():
                    for accessor in layer_names:
                        act = batch[accessor]
                        batch_size = act.shape[0]
                        # Attempt to extract label
                        label = batch["label"] if "label" in batch else None

                        for i in range(batch_size):
                            l = label[i] if label else None
                            if torch.is_tensor(l):
                                l = l.item()
                            if isinstance(l, list):
                                l = l[0]
                            if target_labels is None or l in target_labels:
                                collected.append(act[i].detach().cpu())
                                count += 1
                            if count >= max_samples:
                                break
            except StopIteration:
                pass
            if not collected:
                return torch.tensor([])
            return torch.stack(collected)

        Training(
            TrainingSpec(
                training_function=steer.fit,
                kwargs={
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "mapper": mapper,
                    "loss_fn": torch.nn.CrossEntropyLoss(),
                    "train_steps": cfg.train_steps,
                    "lr": cfg.lr,
                },
            )
        ).run_trainer()
        if getattr(cfg, "delete_cache", False):
            _delete_layer_cache(layer_names[0])
    elif steer_type == "caa":
        pos_labels = list(getattr(cfg, "pos_labels", [1, True]))
        neg_labels = list(getattr(cfg, "neg_labels", [0, False]))
        for layer in layer_names:
            layer_loader = _loader("train", shuffle=train_shuffle, layer=layer)
            pos_acts = collect_from_loader(layer_loader, pos_labels, max_samples=cfg.max_samples)
            layer_loader.reset()
            neg_acts = collect_from_loader(layer_loader, neg_labels, max_samples=cfg.max_samples)
            steer.fit(pos_acts=pos_acts, neg_acts=neg_acts, attr_name=layer)
            if getattr(cfg, "delete_cache", False):
                _delete_layer_cache(layer)

    # 6. Steer & save
    os.makedirs(cfg.output_dir, exist_ok=True)

    def run_steer(**kwargs):
        if steer_type == "caa":
            steering_vecs = [steer.steering_vecs[layer] for layer in layer_names]
            alphas_cfg = getattr(cfg, "alphas", None)
            if alphas_cfg is not None:
                alphas = list(OmegaConf.to_container(alphas_cfg, resolve=True))
            else:
                alphas = [cfg.alpha] * len(layer_names)
            imgs = steer.steer(
                list(cfg.prompts),
                layer_names=layer_names,
                steering_vecs=steering_vecs,
                alphas=alphas,
            )
        elif steer_type == "loreft":
            imgs = steer.steer(
                prompts=list(cfg.prompts),
                layer_names=layer_names,
                num_inference_steps=cfg.steer_steps,
                alpha=float(getattr(cfg, "alpha", 1.0)),
            )
        else:
            imgs = steer.steer(
                list(cfg.prompts),
                target_idx=[[0], None],
                avoid_idx=None,
                alpha=cfg.alpha,
                layer_name=layer_names[0],
                steer_steps=cfg.steer_steps,
            )
        return imgs, list(cfg.prompts)

    scorers_dict = {}
    if getattr(cfg, "metrics", None):
        # Keep raw scorers around to evaluate manually afterwards.
        for metric_name, metric_cfg in cfg.metrics.items():
            try:
                scorer = hydra.utils.instantiate(metric_cfg)
                if hasattr(scorer, "score"):
                    scorers_dict[metric_name] = scorer
            except Exception as e:
                # The "Error locating target" wrapper from Hydra hides the real
                # cause (most often `ModuleNotFoundError: No module named
                # 'lpips' / 'cleanfid' / 'open_clip'`). Surface the install
                # hint so users don't grep the toolkit for nonexistent
                # `_target_` paths.
                print(
                    f"Failed to instantiate metric {metric_name}: {e}. "
                    "Install the metric backends with `uv sync --extra metrics` "
                    "or drop the entry from cfg.metrics."
                )

    specs = []
    if getattr(cfg, "use_baseline", False):

        def run_baseline(**kwargs):
            if steer_type == "caa":
                steering_vecs = [steer.steering_vecs[layer] for layer in layer_names]
                imgs = steer.steer(
                    list(cfg.prompts),
                    layer_names=layer_names,
                    steering_vecs=steering_vecs,
                    alphas=[0.0] * len(layer_names),
                )
            elif steer_type == "loreft":
                # LoReFT doesn't use alpha scaling, meaning baseline is just an unhooked diffusion pass
                imgs = steer.model.pipeline(
                    list(cfg.prompts),
                    num_inference_steps=cfg.steer_steps,
                ).images
            else:
                imgs = steer.steer(
                    list(cfg.prompts),
                    target_idx=None,
                    avoid_idx=None,
                    alpha=0.0,
                    layer_name=layer_names[0],
                    steer_steps=0,
                )
            return imgs, list(cfg.prompts)

        specs.append(InferenceSpec(name="baseline", inference_fn=run_baseline, kwargs={}))

    specs.append(InferenceSpec(name="steered", inference_fn=run_steer, kwargs={}))

    all_metric_results = {}
    # wandb image groups, keyed by spec prefix ("baseline" / "steered") so
    # the UI can compare baseline vs steered side-by-side rather than dumping
    # everything into one bucket.
    wandb_imgs_by_prefix: dict[str, list] = {}

    baseline_disk_paths = []

    for spec in specs:
        inference = Inference(spec)
        out = inference.run_inference()

        imgs = out.preds[0]
        prompts = out.preds[1]
        prefix = spec.name

        # Save images to disk
        local_paths = []
        for j, img in enumerate(imgs):
            path = os.path.join(cfg.output_dir, f"{prefix}_{j}.png")
            img.save(path)
            local_paths.append(path)
            if run:
                wandb_imgs_by_prefix.setdefault(prefix, []).append(
                    wandb.Image(path, caption=f"[{prefix}] {cfg.prompts[j % len(cfg.prompts)]}")
                )

        print(f"Saved {len(imgs)} {prefix} images → {cfg.output_dir}")

        # Track baselines to feed as references
        if prefix == "baseline":
            baseline_disk_paths = local_paths

        # Manually compute metrics
        metric_results = {}
        for m_name, scorer in scorers_dict.items():
            refs = baseline_disk_paths if prefix == "steered" and baseline_disk_paths else None
            # FID expects list of strings for ref_dir logic, CLIP expects PIL or strings.
            # Local paths handle LPIPS cleanly as well.
            try:
                res = scorer.score(images=local_paths, prompts=prompts, references=refs)
                if isinstance(res, dict):
                    metric_results.update({f"{m_name}/{k}": v for k, v in res.items()})
                else:
                    metric_results[m_name] = res
            except Exception as e:
                print(f"Metric {m_name} failed: {e}")

        if metric_results:
            print(f"[{prefix}] Metrics:", metric_results)
            all_metric_results.update({f"{prefix}/{k}": v for k, v in metric_results.items()})

        print(f"Saved {len(imgs)} {prefix} images → {cfg.output_dir}")

    if run:
        log_dict: dict = {f"{prefix}_images": imgs for prefix, imgs in wandb_imgs_by_prefix.items()}
        log_dict.update(all_metric_results)
        alpha_val = float(getattr(cfg, "alpha", 0))
        log_dict["alpha"] = alpha_val
        wandb.log(log_dict)
        run.summary["alpha"] = alpha_val
        run.finish()

    # Write metrics to disk so callbacks can read them without Hydra serialization bugs.
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    save_json(all_metric_results, metrics_path)

    # Completion marker: lets downstream consumers (Fingerprints page,
    # crash-detection scripts) distinguish a finished run from one that
    # left a fingerprint behind but crashed mid-train. Written last.
    _mark_completed_impl(cfg.output_dir, workflow="steer")

    return {"output_dir": cfg.output_dir, "metrics_file": metrics_path}


if __name__ == "__main__":
    main()
