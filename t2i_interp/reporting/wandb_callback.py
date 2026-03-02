import os
import wandb
import json
from glob import glob
from typing import Any
from omegaconf import DictConfig, OmegaConf
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from t2i_interp.utils.utils import load_json

class WandbMultirunCallback(Callback):
    """
    Hydra callback that aggregates results from a sweep into a single master W&B table.
    - Activates only during `-m` (multirun) executions.
    - Runs once after all sweep iterations have finished.
    """
    def __init__(self):
        self.sweep_id = None
        self.job_results = []
        self.base_project = None

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Called upon the start of the sweep."""
        if not getattr(config, "wandb", None) or not config.wandb.get("project"):
            return

        self.base_project = config.wandb.project
        print(f"[WandbMultirunCallback] Initialized tracking for project: {self.base_project}")

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        """Called after each sweep configuration finishes execution."""
        if not self.base_project:
            return

        cfg_dict = OmegaConf.to_container(config, resolve=True)

        # Prefer the actual output_dir returned by main() — it includes the
        # auto-suffix (e.g. _down_blocks) added at runtime, which Hydra's
        # config object does not reflect.
        output_dir = None
        rv = getattr(job_return, "return_value", None)
        if isinstance(rv, dict):
            output_dir = rv.get("output_dir")
        if not output_dir:
            output_dir = config.get("output_dir", None)
        if output_dir:
            output_dir = os.path.abspath(output_dir)

        metrics = {}
        if output_dir:
            metrics_path = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    metrics = load_json(metrics_path)
                except Exception as e:
                    print(f"Failed to read metrics.json: {e}")
            else:
                print(f"[WandbMultirunCallback] metrics.json not found at {metrics_path}")

        print(f"[WandbMultirunCallback] job output_dir={output_dir}, metrics keys={list(metrics.keys())}")
        self.job_results.append({
            "job_number": getattr(job_return, "id", getattr(job_return, "job_name", len(self.job_results))),
            "cfg": cfg_dict,
            "output_dir": output_dir,
            "metrics": metrics
        })

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Called at the end of the entire sweep to aggregate tables."""
        if not self.base_project or not self.job_results:
            return

        print(f"[WandbMultirunCallback] Aggregating {len(self.job_results)} runs into master W&B table...")

        # Initialize the master sweep run
        run = wandb.init(
            project=self.base_project,
            name=f"Sweep-Summary-{config.wandb.get('name', 'Table')}",
            tags=["sweep_summary"] + list(config.wandb.get("tags", [])),
            config=OmegaConf.to_container(config, resolve=False)
        )

        from t2i_interp.reporting.sweep_reports import generate_sweep_table, build_steer_grid_from_jobs

        report_type = config.wandb.get("sweep_report_type", "default")
        table = generate_sweep_table(report_type, config, self.job_results)

        log_payload = {"Sweep Comparison Table": table}

        # For stitch_steer sweeps also build and log the combined grid image
        if report_type == "stitch_steer":
            import os
            # Save grid next to the first job's output dir, or fall back to cwd
            first_out = next(
                (r.get("output_dir") for r in self.job_results if r.get("output_dir")),
                "."
            )
            grid_path = os.path.join(os.path.dirname(first_out), "steer_sweep_grid.png")
            saved = build_steer_grid_from_jobs(self.job_results, grid_path)
            if saved:
                log_payload["Steer Sweep Grid"] = wandb.Image(
                    saved, caption=f"Combined grid — {len(self.job_results)} sweep jobs"
                )

        wandb.log(log_payload)
        run.finish()
        print(f"[WandbMultirunCallback] Master table ({report_type}) successfully uploaded to W&B.")
