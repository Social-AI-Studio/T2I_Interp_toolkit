"""run_sae — entry point: ``t2i-sae``

t2i-sae
t2i-sae prompt="a red apple" n_top_features=6
t2i-sae strengths="[-5,5]"
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from t2i_interp.config._hydra_config import config_dir


@hydra.main(config_path=config_dir(), config_name="sae/run", version_base=None)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import torch
    import wandb
    from diffusers import AutoPipelineForText2Image

    from t2i_interp.reporting.fingerprint import (
        RunFingerprint,
        mark_run_completed,
        record_wandb_run,
        seed_everything,
    )
    from t2i_interp.t2i import T2IModel
    from t2i_interp.utils.inference import Inference, InferenceSpec
    from t2i_interp.utils.T2I.policy import scale_indx_policy

    print("=== t2i-sae config ===")
    print(OmegaConf.to_yaml(cfg))

    # Reproducibility: seed all RNGs before model load / data ops.
    seed_everything(getattr(cfg, "seed", None))

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

    # Run fingerprint: canonical record of every reproducibility-relevant input.
    fingerprint = RunFingerprint.from_cfg(
        cfg,
        workflow="sae",
        intervention={
            "target_sae": getattr(cfg, "target_sae", None),
            "saes": OmegaConf.to_container(getattr(cfg, "saes", {}), resolve=True),
            "strengths": list(getattr(cfg, "strengths", []) or []),
            "n_top_features": getattr(cfg, "n_top_features", None),
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
    model = T2IModel(
        cfg.model_key, automodal=AutoPipelineForText2Image, device=cfg.device, dtype=cfg.dtype
    )
    model.pipeline.set_progress_bar_config(disable=True)

    # 2. SAE Manager construction
    from t2i_interp.build_sae import build_sae_manager

    # Resolve relative checkpoint paths against the repo root so `t2i-sae`
    # works from any CWD. The YAML default for `saes.<hook>.path` is the
    # readable `./sdxl-unbox/checkpoints/...`; invoking the CLI from the
    # repo or from a tmpdir should both find the bundled checkpoints.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OmegaConf.set_struct(cfg, False)
    for _hook, _sae_cfg in cfg.saes.items():
        _path = str(_sae_cfg.path)
        if not os.path.isabs(_path):
            _sae_cfg.path = os.path.normpath(os.path.join(_repo_root, _path))
    OmegaConf.set_struct(cfg, True)

    sae_manager, sae_list = build_sae_manager(
        model, saes_config=cfg.saes, device=cfg.device, dtype=getattr(torch, cfg.dtype)
    )

    # 3. Capture activations.
    # Use `capture_activations(return_images=False)` to get the {sae_name: z}
    # dict in output.preds — `run_with_steering` returns images, which is the
    # wrong shape for the downstream `output.preds[sae_key]` lookup.
    print("Capturing sparse activations...")
    output = Inference(
        InferenceSpec(
            name="sae_capture",
            inference_fn=sae_manager.capture_activations,
            kwargs={
                "sae_list": sae_list,
                "prompt": cfg.prompt,
                "use_delta": False,
                "return_images": False,  # we only want the latent dict
                "num_inference_steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
                "seed": cfg.seed,
            },
        )
    ).run_inference()

    sae_key = cfg.target_sae
    if sae_key not in output.preds:
        raise ValueError(
            f"SAE key '{sae_key}' not found in captured activations! Available: {list(output.preds.keys())}"
        )

    sparse_maps = output.preds[sae_key].view(cfg.spatial_h, cfg.spatial_w, -1)
    top_features = sparse_maps.mean(dim=(0, 1)).topk(cfg.n_top_features).indices.cpu().tolist()
    print(f"Top {cfg.n_top_features} features: {top_features}")

    # 4. Modulation grid
    def activate(feature_idx, strength):
        z_alter_fns = {sae_key: scale_indx_policy(strength, [feature_idx])}
        return sae_manager.run_with_steering(
            sae_list,
            cfg.prompt,
            z_alter_fns=z_alter_fns,
            use_delta=True,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
        )[0]

    os.makedirs(cfg.output_dir, exist_ok=True)
    strengths = list(cfg.strengths)
    n_plot = min(cfg.n_features_to_plot, len(top_features))

    fig, axes = plt.subplots(n_plot, len(strengths), figsize=(3 * len(strengths), 3 * n_plot))
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

    mark_run_completed(cfg.output_dir, workflow="sae")


if __name__ == "__main__":
    main()
