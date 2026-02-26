"""run_localisation — entry point: ``t2i-localise``

    t2i-localise
    t2i-localise factor=0.5 prompt="a dragon"
    t2i-localise sweep_all_layers=true target_heads=[0,1,2]
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from config._hydra_config import config_dir


@hydra.main(config_path=config_dir("localisation"), config_name="run", version_base=None)
def main(cfg: DictConfig) -> None:
    import os
    import torch
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from diffusers import StableDiffusionPipeline

    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.T2I.hook import UNetAlterHook
    from t2i_interp.utils.inference import InferenceSpec, Inference

    print("=== t2i-localise config ===")
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
    model = T2IModel(cfg.model_key, automodal=StableDiffusionPipeline,
                     device=cfg.device, dtype=cfg.dtype)
    model.pipeline.set_progress_bar_config(disable=True)

    # 2. Cross-attn accessors
    all_cross_attn = {
        name: acc
        for name, acc in model.unet.accessors.items()
        if "attn2" in name and name.endswith("_out")
        and getattr(acc.module, "heads", None)
    }
    print(f"Found {len(all_cross_attn)} cross-attn accessors with heads.")

    # 3. Baseline
    g        = torch.Generator().manual_seed(cfg.seed)
    baseline = model.pipeline(
        prompt=[cfg.prompt], num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale, generator=g,
    ).images[0]
    os.makedirs(cfg.output_dir, exist_ok=True)
    baseline.save(os.path.join(cfg.output_dir, "baseline.png"))

    # 4. Hook factory
    def make_hook(n_heads, head_idx, factor, start, end):
        def _policy(x: torch.Tensor, **_) -> torch.Tensor:
            orig = x.shape
            hs   = x.view(x.shape[0], x.shape[-2], n_heads, -1).clone()
            hs[..., head_idx, :] *= factor
            return hs.view(orig)
        return UNetAlterHook(policy=_policy, step_index=slice(start, end))

    def run_head(model, acc, head_idx, factor, start_step, end_step,
                 prompt, n_steps, seed, guidance_scale):
        hook = make_hook(acc.module.heads, head_idx, factor, start_step, end_step)
        g    = torch.Generator().manual_seed(seed)
        return model.run_with_hooks(
            [prompt], hooks={acc.module: hook},
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale, generator=g,
        )

    # 5. Build specs
    target_heads = list(cfg.target_heads) if cfg.target_heads is not None else None
    sweep = (all_cross_attn if cfg.sweep_all_layers
             else {cfg.target_layer: all_cross_attn[cfg.target_layer]})

    specs = [
        InferenceSpec(
            name=f"{name}__h{h}",
            inference_fn=run_head,
            kwargs=dict(model=model, acc=acc, head_idx=h,
                        factor=cfg.factor, start_step=cfg.start_step,
                        end_step=cfg.end_step, prompt=cfg.prompt,
                        n_steps=cfg.num_inference_steps, seed=cfg.seed,
                        guidance_scale=cfg.guidance_scale),
        )
        for name, acc in sweep.items()
        for h in (target_heads if target_heads is not None else range(acc.module.heads))
    ]
    print(f"Running {len(specs)} spec(s)...")
    results = [Inference(s).run_inference() for s in tqdm(specs)]

    # 6. Save per-layer grids to local & W&B
    layer_names = sorted(set(s.name.split("__h")[0] for s in specs))
    wandb_grids = []

    for layer in layer_names:
        ls = [(s, r) for s, r in zip(specs, results) if s.name.startswith(layer)]
        fig, axes = plt.subplots(1, len(ls) + 1, figsize=(4 * (len(ls) + 1), 4))
        axes[0].imshow(baseline); axes[0].set_title("baseline"); axes[0].axis("off")
        for ax, (s, r) in zip(axes[1:], ls):
            ax.imshow(r.preds[0]); ax.set_title(s.name.split("__")[1]); ax.axis("off")
        plt.suptitle(layer, fontsize=9, wrap=True)
        plt.tight_layout()
        
        grid_path = os.path.join(cfg.output_dir, f"{layer[:80]}.png")
        plt.savefig(grid_path, dpi=80)
        plt.close()

        if run:
            wandb_grids.append(wandb.Image(grid_path, caption=f"Sweep for {layer}"))

        for s, r in ls:
            r.preds[0].save(os.path.join(cfg.output_dir, f"{s.name}.png"))

    print(f"Done. Results → {cfg.output_dir}")

    if run:
        wandb.log({"localisation_grids": wandb_grids})
        run.finish()


if __name__ == "__main__":
    main()
