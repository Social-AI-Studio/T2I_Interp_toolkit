"""run_stitch — entry point: ``t2i-stitch``

    t2i-stitch
    t2i-stitch dataset_name=my/ds num_steps=2000
    t2i-stitch inject_steps=[0,1,2]
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from config._hydra_config import config_dir


@hydra.main(config_path=config_dir("stitch"), config_name="run", version_base=None)
def main(cfg: DictConfig) -> None:
    import os
    import torch as th
    import wandb
    from diffusers import AutoPipelineForText2Image
    from datasets import load_dataset

    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.T2I.collect_latents import collect_latents
    from t2i_interp.utils.T2I.buffer import ActivationsDataloader, PairedLoader
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

    # 1. Model
    model = T2IModel(cfg.model_key, automodal=AutoPipelineForText2Image,
                     device=cfg.device, dtype=cfg.dtype)
    model.pipeline.set_progress_bar_config(disable=True)

    # 2. Dataset
    ds_full  = load_dataset(cfg.dataset_name)
    ds_train = ds_full["train"]
    ds_val   = (ds_full.get("validation") or ds_full.get("test")
                or ds_train.train_test_split(test_size=0.2, seed=42)["test"])

    # 3. Collect latents for both layers
    def _collect(layer, col, dataset, split):
        return collect_latents(
            accessors=[layer], dataset=dataset, model=model,
            save_path=os.path.join(cfg.save_dir, split), columns=[col],
            batch_size=cfg.batch_size, guidance_scale=cfg.guidance_scale,
            conditional_only=cfg.conditional_only,
        )

    print("Collecting latents...")
    for ds, split in [(ds_train, "train"), (ds_val, "val")]:
        _collect(cfg.layer_a, cfg.prompt_col_a, ds, split)
        _collect(cfg.layer_b, cfg.prompt_col_b, ds, split)

    # 4. Loaders
    def _find_tar(split, layer, col):
        base  = os.path.join(cfg.save_dir, split)
        fname = f"{layer}_{col}.tar"
        for sub in sorted(os.listdir(base), reverse=True):
            cand = os.path.join(base, sub, fname)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"{fname} not found under {base}")

    def _loader(split, layer, col, shuffle=False):
        return ActivationsDataloader(
            paths_to_datasets=[_find_tar(split, layer, col)], block_name=layer,
            batch_size=cfg.loader_batch_size, flatten=False,
            shuffle=shuffle, device=cfg.device)

    print("Building loaders...")
    train_loader = PairedLoader([
        _loader("train", cfg.layer_a, cfg.prompt_col_a, shuffle=True),
        _loader("train", cfg.layer_b, cfg.prompt_col_b, shuffle=True),
    ])
    val_loader = PairedLoader([
        _loader("val", cfg.layer_a, cfg.prompt_col_a),
        _loader("val", cfg.layer_b, cfg.prompt_col_b),
    ])

    # 5. Train
    stitcher = Stitcher()
    mapper   = MLPMapper(input_dim=cfg.input_dim, output_dim=cfg.output_dim,
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

    # 6. Inference
    os.makedirs(cfg.output_dir, exist_ok=True)
    inject_steps = list(cfg.inject_steps) if cfg.inject_steps is not None else None
    result = Inference(InferenceSpec(
        name="mapper_stitch",
        inference_fn=stitcher.map,
        kwargs={
            "model": model, "module_a": cfg.layer_a, "module_b": cfg.layer_b,
            "mapper":              trained_mapper.to(device=cfg.device, dtype=th.float16),
            "prompts":             list(cfg.prompts),
            "device":              cfg.device,
            "num_inference_steps": cfg.num_inference_steps,
            "inject_steps":        inject_steps,
            "guidance_scale":      cfg.guidance_scale,
        },
    )).run_inference()
    wandb_imgs = []
    for j, img in enumerate(result.preds):
        path = os.path.join(cfg.output_dir, f"stitched_{j}.png")
        img.save(path)
        if run:
            wandb_imgs.append(wandb.Image(path, caption=cfg.prompts[j % len(cfg.prompts)]))
            
    print(f"Saved {len(result.preds)} images → {cfg.output_dir}")

    if run:
        wandb.log({"stitched_images": wandb_imgs})
        run.finish()


if __name__ == "__main__":
    main()
