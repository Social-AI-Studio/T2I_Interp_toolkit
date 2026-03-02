"""run_sae — entry point: ``t2i-sae``

    t2i-sae
    t2i-sae prompt="a red apple" n_top_features=6
    t2i-sae strengths="[-5,5]"
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from t2i_interp.config._hydra_config import config_dir


@hydra.main(config_path=config_dir(), config_name="sae/run", version_base=None)
def main(cfg: DictConfig) -> None:
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import torch
    import wandb
    from diffusers import AutoPipelineForText2Image

    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.T2I.policy import scale_indx_policy
    from t2i_interp.utils.inference import InferenceSpec, Inference

    print("=== t2i-sae config ===")
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

    # 2. SAE Manager construction
    from t2i_interp.build_sae import build_sae_manager
    sae_manager, sae_list = build_sae_manager(model, saes_config=cfg.saes, device=cfg.device, dtype=getattr(torch, cfg.dtype))

    # 3. Capture activations
    print("Capturing sparse activations...")
    output = Inference(InferenceSpec(
        name="sae_capture",
        inference_fn=sae_manager.run_with_steering,
        kwargs={"sae_list": sae_list, "prompt": cfg.prompt,
                "z_alter_fns": {}, "use_delta": False,
                "num_inference_steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale, "seed": cfg.seed},
    )).run_inference()

    sae_key     = cfg.target_sae
    if sae_key not in output.preds:
        raise ValueError(f"SAE key '{sae_key}' not found in captured activations! Available: {list(output.preds.keys())}")
        
    sparse_maps = output.preds[sae_key].view(cfg.spatial_h, cfg.spatial_w, -1)
    top_features = sparse_maps.mean(dim=(0, 1)).topk(cfg.n_top_features).indices.cpu().tolist()
    print(f"Top {cfg.n_top_features} features: {top_features}")

    # 4. Modulation grid
    def activate(feature_idx, strength):
        z_alter_fns = {sae_key: scale_indx_policy(strength, [feature_idx])}
        return sae_manager.run_with_steering(
            sae_list, cfg.prompt, z_alter_fns=z_alter_fns, use_delta=True,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale, seed=cfg.seed)[0]

    os.makedirs(cfg.output_dir, exist_ok=True)
    strengths = list(cfg.strengths)
    n_plot    = min(cfg.n_features_to_plot, len(top_features))

    fig, axes = plt.subplots(n_plot, len(strengths),
                             figsize=(3 * len(strengths), 3 * n_plot))
    for i, feat in enumerate(top_features[:n_plot]):
        for j, s in enumerate(strengths):
            ax = axes[i][j] if n_plot > 1 else axes[j]
            ax.imshow(activate(feat, s))
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Strength {s}")
            if j == 0:
                ax.set_ylabel(f"Feat {feat}", fontsize=10)
    plt.tight_layout()
    grid_path = os.path.join(cfg.output_dir, "feature_grid.png")
    plt.savefig(grid_path, dpi=100)
    print(f"Saved grid → {grid_path}")

    if run:
        wandb.log({"sae_feature_grid": wandb.Image(grid_path)})
        run.finish()


if __name__ == "__main__":
    main()
