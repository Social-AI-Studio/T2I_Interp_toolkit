"""run_steer — entry point: ``t2i-steer``

    t2i-steer
    t2i-steer model_key=CompVis/stable-diffusion-v1-4 device=cuda:1
    t2i-steer alpha=20 steer_steps=10
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from config._hydra_config import config_dir


@hydra.main(config_path=config_dir("steer"), config_name="run", version_base=None)
def main(cfg: DictConfig) -> None:
    import os
    import torch
    import wandb
    from diffusers import AutoPipelineForText2Image
    from datasets import load_dataset

    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.T2I.collect_latents import collect_latents
    from t2i_interp.utils.T2I.buffer import ActivationsDataloader
    from t2i_interp.utils.training import TrainingSpec, Training
    from t2i_interp.steer import KSteer
    from t2i_interp.mapper import MLPMapper

    print("=== t2i-steer config ===")
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

    # 3. Collect latents
    def _collect(dataset, split):
        return collect_latents(
            accessors=[cfg.layer_name], dataset=dataset, model=model,
            save_path=os.path.join(cfg.save_dir, split),
            columns=[cfg.prompt_col], batch_size=cfg.batch_size,
            guidance_scale=cfg.guidance_scale, conditional_only=cfg.conditional_only,
        )
    print("Collecting latents...")
    _collect(ds_train, "train")
    _collect(ds_val,   "val")

    # 4. Find tars & build loaders
    def _find_tar(split):
        base  = os.path.join(cfg.save_dir, split)
        fname = f"{cfg.layer_name}_{cfg.prompt_col}.tar"
        for sub in sorted(os.listdir(base), reverse=True):
            cand = os.path.join(base, sub, fname)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"{fname} not found under {base}")

    def _loader(split, shuffle):
        return ActivationsDataloader(
            paths_to_datasets=[_find_tar(split)], block_name=cfg.layer_name,
            batch_size=16, flatten=True, shuffle=shuffle, device=cfg.device)

    train_loader = _loader("train", shuffle=True)
    val_loader   = _loader("val",   shuffle=False)

    # 5. Train
    sample    = next(train_loader.iterate())
    input_dim = sample[0].reshape(sample[0].shape[0], -1).shape[-1]
    ksteer    = KSteer(model=model)
    mapper    = MLPMapper(input_dim=input_dim, output_dim=2)

    Training(TrainingSpec(
        training_function=ksteer.fit,
        kwargs={"train_loader": train_loader, "val_loader": val_loader,
                "mapper": mapper, "loss_fn": torch.nn.CrossEntropyLoss(),
                "train_steps": cfg.train_steps, "lr": cfg.lr},
    )).run_trainer()

    # 6. Steer & save
    os.makedirs(cfg.output_dir, exist_ok=True)
    imgs = ksteer.steer(list(cfg.prompts), target_idx=[[0], None], avoid_idx=None,
                        alpha=cfg.alpha, layer_name=cfg.layer_name,
                        steer_steps=cfg.steer_steps)
    wandb_imgs = []
    for j, img in enumerate(imgs):
        path = os.path.join(cfg.output_dir, f"steered_{j}.png")
        img.save(path)
        if run:
            wandb_imgs.append(wandb.Image(path, caption=cfg.prompts[j % len(cfg.prompts)]))
            
    print(f"Saved {len(imgs)} images → {cfg.output_dir}")

    if run:
        wandb.log({"steered_images": wandb_imgs})
        run.finish()


if __name__ == "__main__":
    main()
