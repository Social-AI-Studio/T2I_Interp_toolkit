"""run_stitch — entry point: ``t2i-stitch``

    t2i-stitch
    t2i-stitch dataset_name=my/ds num_steps=2000
    t2i-stitch inject_steps=[0,1,2]
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from t2i_interp.config._hydra_config import config_dir


@hydra.main(config_path=config_dir(), config_name="stitch/run", version_base=None)
def main(cfg: DictConfig) -> None:
    import sys, types, torch
    if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
        torch.backends.cuda.is_flash_attention_available = lambda: False
    for _xf in [
        "xformers.ops", "xformers.ops.fmha", "xformers.ops.fmha.flash",
        "xformers.ops.fmha.common", "xformers.ops.fmha.triton_splitk",
        "xformers.flash_attn_3", "xformers.flash_attn_3._C",
    ]:
        sys.modules.setdefault(_xf, types.ModuleType(_xf))
    
    import os
    import torch as th
    import wandb
    from diffusers import DiffusionPipeline
    from diffusers.utils import logging as diffusers_logging
    import transformers
    diffusers_logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    from datasets import load_dataset

    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.T2I.collect_latents import collect_latents, collect_latents_inmemory
    from t2i_interp.utils.T2I.buffer import ActivationsDataloader, PairedLoader, InMemoryPairedLoader
    from t2i_interp.mapper import MLPMapper
    from t2i_interp.stitch import Stitcher
    from t2i_interp.utils.training import TrainingSpec, Training
    from t2i_interp.utils.inference import InferenceSpec, Inference

    print("=== t2i-stitch config ===")
    print(OmegaConf.to_yaml(cfg))

    # Optional wandb initialization
    run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.get("project"):
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("name", None),
            tags=cfg.wandb.get("tags", []),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # 1. Models
    # model_a: owns layer_a (source activations)
    # model_b: owns layer_b (target activations); loaded separately only when
    #          model_key_b is set and differs from model_key — otherwise model_b
    #          is the same object as model_a (single-model mode).
    model_key_b = getattr(cfg, "model_key_b", None)
    
    lora_b_cfg  = OmegaConf.to_container(cfg.get("lora_b",  {}), resolve=True) if hasattr(cfg, "lora_b") else {}
    unet_b_cfg  = OmegaConf.to_container(cfg.get("unet_b",  {}), resolve=True) if hasattr(cfg, "unet_b") else {}
    lora_b_repo      = lora_b_cfg.get("repo")
    lora_b_scheduler = lora_b_cfg.get("scheduler")
    unet_b_repo      = unet_b_cfg.get("repo")
    unet_b_file      = unet_b_cfg.get("filename")
    unet_b_scheduler = unet_b_cfg.get("scheduler")
    
    cross_model  = bool((model_key_b and model_key_b != cfg.model_key) or lora_b_repo or unet_b_repo)

    model_a = T2IModel(cfg.model_key, automodal=DiffusionPipeline,
                       device=cfg.device, dtype=cfg.dtype)
    model_a.pipeline.set_progress_bar_config(disable=True)

    if cross_model:
        import importlib
        if lora_b_repo:
            # ── LCM-LoRA style: load base pipeline, apply LoRA weights, fuse ──────
            # e.g. model_key_b = "runwayml/stable-diffusion-v1-5"
            #      lora_b.repo  = "latent-consistency/lcm-lora-sdv1-5"
            #      lora_b.scheduler = "LCMScheduler"
            model_b = T2IModel(model_key_b or cfg.model_key, automodal=DiffusionPipeline,
                               device=cfg.device, dtype=cfg.dtype)
            if lora_b_scheduler:
                sched_cls = getattr(importlib.import_module("diffusers"), lora_b_scheduler)
                model_b.pipeline.scheduler = sched_cls.from_config(
                    model_b.pipeline.scheduler.config
                )
            model_b.pipeline.load_lora_weights(lora_b_repo)
            model_b.pipeline.fuse_lora()
            print(f"Cross-model (LCM-LoRA): model_a={cfg.model_key}  "
                  f"model_b={model_key_b or cfg.model_key} + LoRA={lora_b_repo}  "
                  f"scheduler={lora_b_scheduler or 'unchanged'}")

        elif unet_b_repo and unet_b_file:
            # ── Custom UNet checkpoint style (e.g. DMD2) ──────────────────────────
            from diffusers import UNet2DConditionModel
            from huggingface_hub import hf_hub_download
            th_dtype = getattr(th, cfg.dtype, th.float16)
            unet = UNet2DConditionModel.from_config(
                model_key_b or cfg.model_key, subfolder="unet"
            ).to(cfg.device, th_dtype)
            unet.load_state_dict(
                th.load(hf_hub_download(unet_b_repo, unet_b_file), map_location=cfg.device)
            )
            model_b = T2IModel(
                model_key_b or cfg.model_key, automodal=DiffusionPipeline,
                device=cfg.device, dtype=cfg.dtype, unet=unet
            )
            if unet_b_scheduler:
                sched_cls = getattr(importlib.import_module("diffusers"), unet_b_scheduler)
                model_b.pipeline.scheduler = sched_cls.from_config(
                    model_a.pipeline.scheduler.config
                )
            print(f"Cross-model (custom UNet): model_a={cfg.model_key}  "
                  f"model_b={model_key_b or cfg.model_key} + {unet_b_repo}/{unet_b_file}  "
                  f"scheduler={unet_b_scheduler or 'unchanged'}")

        else:
            # ── Plain different model ──────────────────────────────────────────────
            model_b = T2IModel(model_key_b, automodal=DiffusionPipeline,
                               device=cfg.device, dtype=cfg.dtype)
            print(f"Cross-model: model_a={cfg.model_key}  model_b={model_key_b}")

        model_b.pipeline.set_progress_bar_config(disable=True)
    else:
        model_b = None
        print(f"Single-model mode: {cfg.model_key}")

    # 2. Dataset
    collect_to_memory = getattr(cfg, "collect_to_memory", False)
    mode = getattr(cfg, "mode", "train")
    ds_full = ds_train = ds_val = None

    # steer_contrast / steer_transfer don't need a training dataset
    needs_dataset = (mode not in ("steer_contrast", "steer_transfer"))
    if needs_dataset:
        ds_full  = load_dataset(cfg.dataset_name)
        ds_train = ds_full["train"]
        ds_val   = (ds_full.get("validation") or ds_full.get("test")
                    or ds_train.train_test_split(test_size=0.2, seed=42)["test"])

        if getattr(cfg, "max_samples", None):
            ds_train = ds_train.select(range(min(len(ds_train), cfg.max_samples)))
            ds_val   = ds_val.select(range(min(len(ds_val), max(1, cfg.max_samples // 5))))

    # 3+4. Collect latents and build loaders
    capture_step = getattr(cfg, "capture_step_index", 0)
    n_steps_a    = getattr(cfg, "num_inference_steps_a", 1)
    n_steps_b    = getattr(cfg, "num_inference_steps_b", 1)
    train_loader = val_loader = None

    if needs_dataset:
        if collect_to_memory:
            # ── In-memory path: run models directly, skip all tar files ────────
            # Practical up to ~10k samples (e.g. 5k × 81920 × fp16 ≈ 800 MB per model).
            max_tr = getattr(cfg, "max_samples", 5000)
            max_va = max(1, max_tr // 5)
            cache_dtype = getattr(th, getattr(cfg, "gpu_cache_dtype", "float16"), th.float16)

            print("Collecting train activations (model_a) in memory...")
            acts_a_train = collect_latents_inmemory(
                accessors=[cfg.layer_a], dataset=ds_train,
                model=model_a, columns=[cfg.prompt_col_a],
                batch_size=cfg.batch_size, max_samples=max_tr,
                num_inference_steps=n_steps_a, guidance_scale=cfg.guidance_scale,
                capture_step_index=capture_step, conditional_only=cfg.conditional_only,
                device=cfg.device, dtype=cache_dtype,
            )[cfg.layer_a]  # (N, ...)

            print("Collecting train activations (model_b) in memory...")
            acts_b_train = collect_latents_inmemory(
                accessors=[cfg.layer_b], dataset=ds_train,
                model=model_b or model_a, columns=[cfg.prompt_col_b],
                batch_size=cfg.batch_size, max_samples=max_tr,
                num_inference_steps=n_steps_b, guidance_scale=cfg.guidance_scale,
                capture_step_index=capture_step, conditional_only=cfg.conditional_only,
                device=cfg.device, dtype=cache_dtype,
            )[cfg.layer_b]

            print("Collecting val activations in memory...")
            acts_a_val = collect_latents_inmemory(
                accessors=[cfg.layer_a], dataset=ds_val,
                model=model_a, columns=[cfg.prompt_col_a],
                batch_size=cfg.batch_size, max_samples=max_va,
                num_inference_steps=n_steps_a, guidance_scale=cfg.guidance_scale,
                capture_step_index=capture_step, conditional_only=cfg.conditional_only,
                device=cfg.device, dtype=cache_dtype,
            )[cfg.layer_a]

            acts_b_val = collect_latents_inmemory(
                accessors=[cfg.layer_b], dataset=ds_val,
                model=model_b or model_a, columns=[cfg.prompt_col_b],
                batch_size=cfg.batch_size, max_samples=max_va,
                num_inference_steps=n_steps_b, guidance_scale=cfg.guidance_scale,
                capture_step_index=capture_step, conditional_only=cfg.conditional_only,
                device=cfg.device, dtype=cache_dtype,
            )[cfg.layer_b]

            # Trim to same length and build InMemoryPairedLoader directly from tensors
            n_tr = min(len(acts_a_train), len(acts_b_train))
            n_va = min(len(acts_a_val),   len(acts_b_val))
            acts_a_train, acts_b_train = acts_a_train[:n_tr], acts_b_train[:n_tr]
            acts_a_val,   acts_b_val   = acts_a_val[:n_va],   acts_b_val[:n_va]
            print(f"In-memory: {n_tr} train pairs, {n_va} val pairs")

            train_loader = InMemoryPairedLoader.from_tensors(
                acts_a_train, acts_b_train,
                batch_size=cfg.loader_batch_size, shuffle=True, device=cfg.device,
            )
            val_loader = InMemoryPairedLoader.from_tensors(
                acts_a_val, acts_b_val,
                batch_size=cfg.loader_batch_size, shuffle=False, device=cfg.device,
            )

        else:
            # ── Tar-file path: collect → save to disk → stream back ─────────────
            save_dir_a = os.path.join(cfg.save_dir, "a") if cross_model else cfg.save_dir
            save_dir_b = os.path.join(cfg.save_dir, "b") if cross_model else cfg.save_dir

            def _collect(layer, col, dataset, split, model, save_root, n_steps):
                save_path = os.path.join(save_root, split)
                tar_path  = os.path.join(save_path, f"{layer}_{col}.tar")
                if os.path.isfile(tar_path) and os.path.getsize(tar_path) > 0:
                    print(f"[stitch] Skipping collection (already exists): {tar_path}")
                    return save_path
                if os.path.isfile(tar_path):
                    print(f"[stitch] Removing empty/corrupt tar: {tar_path}")
                    os.remove(tar_path)
                return collect_latents(
                    accessors=[layer], dataset=dataset, model=model,
                    save_path=save_path, columns=[col],
                    batch_size=cfg.batch_size, guidance_scale=cfg.guidance_scale,
                    conditional_only=cfg.conditional_only,
                    capture_step_index=capture_step,
                    num_inference_steps=n_steps,
                )

            print("Collecting latents...")
            for ds, split in [(ds_train, "train"), (ds_val, "val")]:
                _collect(cfg.layer_a, cfg.prompt_col_a, ds, split, model_a,               save_dir_a, n_steps_a)
                _collect(cfg.layer_b, cfg.prompt_col_b, ds, split, model_b or model_a,    save_dir_b, n_steps_b)

            def _find_tar(save_root, split, layer, col):
                base  = os.path.join(save_root, split)
                fname = f"{layer}_{col}.tar"
                cand_flat = os.path.join(base, fname)
                if os.path.isfile(cand_flat):
                    return cand_flat
                if os.path.isdir(base):
                    for sub in sorted(os.listdir(base), reverse=True):
                        cand = os.path.join(base, sub, fname)
                        if os.path.isfile(cand):
                            return cand
                raise FileNotFoundError(f"{fname} not found under {base}")

            def _loader(save_root, split, layer, col, shuffle=False):
                return ActivationsDataloader(
                    paths_to_datasets=[_find_tar(save_root, split, layer, col)],
                    block_name=layer, batch_size=cfg.loader_batch_size, flatten=False,
                    shuffle=shuffle, device=cfg.device,
                )

            print("Building loaders...")
            use_gpu_cache = getattr(cfg, "use_gpu_cache", False)
            if use_gpu_cache:
                cache_dtype = getattr(th, getattr(cfg, "gpu_cache_dtype", "float16"), th.float16)
                train_loader = InMemoryPairedLoader(
                    [
                        _loader(save_dir_a, "train", cfg.layer_a, cfg.prompt_col_a),
                        _loader(save_dir_b, "train", cfg.layer_b, cfg.prompt_col_b),
                    ],
                    batch_size=cfg.loader_batch_size, shuffle=True,
                    device=cfg.device, dtype=cache_dtype,
                )
                val_loader = InMemoryPairedLoader(
                    [
                        _loader(save_dir_a, "val", cfg.layer_a, cfg.prompt_col_a),
                        _loader(save_dir_b, "val", cfg.layer_b, cfg.prompt_col_b),
                    ],
                    batch_size=cfg.loader_batch_size, device=cfg.device, dtype=cache_dtype,
                )
            else:
                train_loader = PairedLoader([
                    _loader(save_dir_a, "train", cfg.layer_a, cfg.prompt_col_a, shuffle=False),
                    _loader(save_dir_b, "train", cfg.layer_b, cfg.prompt_col_b, shuffle=False),
                ], shuffle=True)
                val_loader = PairedLoader([
                    _loader(save_dir_a, "val", cfg.layer_a, cfg.prompt_col_a),
                    _loader(save_dir_b, "val", cfg.layer_b, cfg.prompt_col_b),
                ])

    stitcher = Stitcher()
    os.makedirs(cfg.output_dir, exist_ok=True)
    inject_steps = list(cfg.inject_steps) if cfg.inject_steps is not None else None

    # guidance_scale_b: used for model_b generation (steer/stitch inference).
    # Separate from guidance_scale which is used for latent collection on model_a.
    # e.g. LCM-LoRA needs 1.0, plain SD1.5 needs 7.0–7.5.
    _gs_b = getattr(cfg, "guidance_scale_b", None)
    guidance_scale_b = float(_gs_b) if _gs_b is not None else float(cfg.guidance_scale)

    # ── Helper: load a mapper from a .pt checkpoint ───────────────────────────
    def _load_mapper(path: str) -> th.nn.Module:

        ckpt = th.load(path, map_location="cpu", weights_only=False)
        m = MLPMapper(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            hidden_dim=ckpt["hidden_dim"],
        )
        m.load_state_dict(ckpt["state_dict"])
        return m

    # ── Shared helper for steer modes ────────────────────────────────────────
    def _run_steer_mode(imgs, steer_baseline, apply_prompts):
        wandb_imgs = []
        for j, img in enumerate(imgs):
            tag = "steered" if j % 2 == 0 else "baseline"
            prompt_idx = j // 2 if steer_baseline else j
            path = os.path.join(cfg.output_dir, f"{tag}_{prompt_idx}.png")
            img.save(path)
            if run:
                caption = f"{apply_prompts[prompt_idx % len(apply_prompts)]} [{tag}]"
                wandb_imgs.append(wandb.Image(path, caption=caption))
        print(f"Saved {len(imgs)} images → {cfg.output_dir}")
        return wandb_imgs

    # ── Mode: steer_contrast — compute direction from contrast prompts on model_a ─
    if mode == "steer_contrast":
        mapper_path = getattr(cfg, "mapper_path", None)
        if not mapper_path:
            raise ValueError("steer_contrast mode requires mapper_path.")

        print(f"[steer_contrast] Loading mapper from {mapper_path}")
        trained_mapper = _load_mapper(mapper_path).to(device=cfg.device, dtype=th.float32)

        steer_alpha    = float(getattr(cfg, "steer_alpha", 1.0))
        steer_baseline = bool(getattr(cfg, "steer_baseline", True))

        pos_prompts   = list(cfg.steer_pos_prompts)
        neg_prompts   = list(cfg.steer_neg_prompts)
        apply_prompts = list(cfg.steer_apply_prompts) or list(cfg.prompts)
        if not pos_prompts or not neg_prompts:
            raise ValueError("steer_contrast requires steer_pos_prompts and steer_neg_prompts.")
        imgs = stitcher.steer_contrast(
            pos_prompts=pos_prompts, neg_prompts=neg_prompts, apply_prompts=apply_prompts,
            module_a=cfg.layer_a, module_b=cfg.layer_b,
            mapper=trained_mapper, model_a=model_a, model_b=model_b or model_a,
            device=cfg.device, alpha=steer_alpha,
            num_inference_steps=cfg.num_inference_steps, inject_steps=inject_steps,
            also_generate_baseline=steer_baseline, guidance_scale=guidance_scale_b,
        )
        wandb_imgs = _run_steer_mode(imgs, steer_baseline, apply_prompts)

        # Save per-job metadata so WandbMultirunCallback can build a combined grid
        import json
        steered_paths  = [os.path.join(cfg.output_dir, f"steered_{i}.png")  for i in range(len(apply_prompts))]
        baseline_paths = [os.path.join(cfg.output_dir, f"baseline_{i}.png") for i in range(len(apply_prompts))] \
                         if steer_baseline else []
        meta = {
            "pos":      pos_prompts[0]  if pos_prompts  else "",
            "neg":      neg_prompts[0]  if neg_prompts  else "",
            "apply":    apply_prompts,
            "alpha":    steer_alpha,
            "steered_paths":  [p for p in steered_paths  if os.path.exists(p)],
            "baseline_paths": [p for p in baseline_paths if os.path.exists(p)],
        }
        with open(os.path.join(cfg.output_dir, "steer_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[steer_contrast] steer_meta.json saved → {cfg.output_dir}")

    # ── Mode: steer_transfer — skip training, load mapper + steer vector ──────
    elif mode == "steer_transfer":
        mapper_path    = getattr(cfg, "mapper_path", None)
        steer_vec_path = getattr(cfg, "steer_vec_path", None)
        if not mapper_path or not steer_vec_path:
            raise ValueError("steer_transfer mode requires mapper_path and steer_vec_path in config.")

        print(f"[steer_transfer] Loading mapper from {mapper_path}")
        trained_mapper = _load_mapper(mapper_path).to(device=cfg.device, dtype=th.float32)

        print(f"[steer_transfer] Loading steering vector from {steer_vec_path}")
        steer_vec = th.load(steer_vec_path, map_location="cpu", weights_only=False).float()

        ref_act_path = getattr(cfg, "ref_act_path", None)
        ref_act = th.load(ref_act_path, map_location="cpu", weights_only=False).float() \
                  if ref_act_path else None

        steer_alpha    = float(getattr(cfg, "steer_alpha", 1.0))
        steer_baseline = bool(getattr(cfg, "steer_baseline", True))

        imgs = stitcher.steer_transfer(
            steering_vector=steer_vec,
            module_b=cfg.layer_b,
            mapper=trained_mapper,
            model_b=model_b or model_a,
            prompts=list(cfg.prompts),
            device=cfg.device,
            alpha=steer_alpha,
            ref_activation=ref_act,
            num_inference_steps=cfg.num_inference_steps,
            inject_steps=inject_steps,
            also_generate_baseline=steer_baseline,
            guidance_scale=guidance_scale_b,
        )
        wandb_imgs = _run_steer_mode(imgs, steer_baseline, list(cfg.prompts))

    # ── Mode: train — train mapper + save + run stitch inference ─────────────
    else:
        # 5. Train
        mapper = MLPMapper(input_dim=cfg.input_dim, output_dim=cfg.output_dim,
                           hidden_dim=cfg.hidden_dim)
        print("Training mapper...")
        result = Training(TrainingSpec(
            training_function=stitcher.train_mapper,
            kwargs={
                "train_loader":    train_loader, "val_loader":      val_loader,
                "mapper":          mapper,
                "optimizers":      [th.optim.Adam(mapper.parameters(), lr=cfg.lr)],
                "num_steps":       cfg.num_steps, "loss_fn":         th.nn.MSELoss(),
                "training_device": cfg.device,   "autocast_dtype":  th.bfloat16,
                "log_steps":       cfg.log_steps,
            },
        )).run_trainer()
        trained_mapper = result.preds

        # 5b. Save mapper checkpoint
        mapper_save_path = os.path.join(cfg.output_dir, "mapper.pt")
        th.save({
            "state_dict": trained_mapper.state_dict(),
            "input_dim":  cfg.input_dim,
            "output_dim": cfg.output_dim,
            "hidden_dim": cfg.hidden_dim,
            "layer_a":    cfg.layer_a,
            "layer_b":    cfg.layer_b,
        }, mapper_save_path)
        print(f"Mapper saved → {mapper_save_path}")

        # 6. Stitch inference (standard map)
        result = Inference(InferenceSpec(
            name="mapper_stitch",
            inference_fn=stitcher.map,
            kwargs={
                "module_a": cfg.layer_a, "module_b": cfg.layer_b,
                "model_a":             model_a,
                "model_b":             model_b,
                "mapper":              trained_mapper.to(device=cfg.device, dtype=th.float16),
                "prompts":             list(cfg.prompts),
                "device":              cfg.device,
                "num_inference_steps": cfg.num_inference_steps,
                "inject_steps":        inject_steps,
                "guidance_scale":      guidance_scale_b,
            },
        )).run_inference()
        wandb_imgs = []
        for j, img in enumerate(result.preds):
            path = os.path.join(cfg.output_dir, f"stitched_{j}.png")
            img.save(path)
            if run:
                wandb_imgs.append(wandb.Image(path, caption=cfg.prompts[j % len(cfg.prompts)]))
        print(f"Saved {len(result.preds)} images → {cfg.output_dir}")

        # 7. Optional: steer_contrast run right after training if prompts are configured
        steer_alpha    = float(getattr(cfg, "steer_alpha",    1.0))
        steer_baseline = bool(getattr(cfg, "steer_baseline",  True))
        pos_prompts    = list(getattr(cfg, "steer_pos_prompts",   []) or [])
        neg_prompts    = list(getattr(cfg, "steer_neg_prompts",   []) or [])
        apply_prompts  = list(getattr(cfg, "steer_apply_prompts", []) or []) or list(cfg.prompts)

        if pos_prompts and neg_prompts:
            print("[train] steer_pos/neg_prompts found — running steer_contrast with trained mapper...")
            steer_mapper = trained_mapper.to(device=cfg.device, dtype=th.float32)
            steer_imgs = stitcher.steer_contrast(
                pos_prompts=pos_prompts, neg_prompts=neg_prompts, apply_prompts=apply_prompts,
                module_a=cfg.layer_a, module_b=cfg.layer_b,
                mapper=steer_mapper, model_a=model_a, model_b=model_b or model_a,
                device=cfg.device, alpha=steer_alpha,
                num_inference_steps=cfg.num_inference_steps, inject_steps=inject_steps,
                also_generate_baseline=steer_baseline, guidance_scale=guidance_scale_b,
            )
            steer_wandb = _run_steer_mode(steer_imgs, steer_baseline, apply_prompts)
            if run:
                wandb_imgs += steer_wandb

    if run:
        wandb.log({"stitched_images": wandb_imgs})
        run.finish()


if __name__ == "__main__":
    main()
