"""Distributed inference using Ray for parallel processing.

This module provides utilities for running multiple inferences in parallel
using Ray's distributed computing framework.

Example:
    >>> from ray_runner import run_distributed_inference
    >>> from utils.inference import InferenceSpec
    >>>
    >>> specs = [InferenceSpec(...), InferenceSpec(...)]
    >>> results = run_distributed_inference(specs, num_workers=4)
"""

import os
from typing import Any

import ray

from utils.inference import Inference, InferenceSpec


@ray.remote
class RayInferenceWorker:
    """Ray actor for running inference tasks in parallel.

    This actor is initialized once per worker and can process multiple
    inference tasks without reloading the model.

    Args:
        model_config: Configuration dictionary for model initialization
        device: Device to run inference on (e.g., "cuda:0", "cpu")
    """

    def __init__(self, model_config: dict[str, Any] | None = None, device: str = "cpu"):
        self.model_config = model_config or {}
        self.device = device

    def run_inference(self, spec_dict: dict[str, Any], base_kwargs: dict[str, Any]) -> Any:
        """Run a single inference task.

        Args:
            spec_dict: Dictionary representation of InferenceSpec
            base_kwargs: Additional kwargs for inference

        Returns:
            Inference results
        """
        spec = InferenceSpec(**spec_dict)
        infer = Inference(inference_spec=spec, **base_kwargs)
        return infer.predict()


def run_distributed_inference(
    specs: list[InferenceSpec],
    base_kwargs: dict[str, Any] | None = None,
    num_workers: int = 4,
    model_config: dict[str, Any] | None = None,
    init_ray: bool = True,
) -> list[Any]:
    """Run multiple inferences in parallel using Ray.

    Args:
        specs: List of InferenceSpec objects to process
        base_kwargs: Base kwargs to pass to all inference calls
        num_workers: Number of Ray workers to spawn
        model_config: Configuration for model initialization
        init_ray: Whether to initialize Ray (set False if already initialized)

    Returns:
        List of inference results in same order as specs

    Example:
        >>> specs = [InferenceSpec(...) for _ in range(10)]
        >>> results = run_distributed_inference(
        ...     specs,
        ...     num_workers=4,
        ...     base_kwargs={"device": "cuda"}
        ... )
    """
    base_kwargs = base_kwargs or {}

    # Initialize Ray if requested
    if init_ray and not ray.is_initialized():
        ray.init(
            runtime_env={"working_dir": os.getcwd()},
            logging_level="INFO",
            ignore_reinit_error=True,
        )

    # Create worker pool
    workers = [RayInferenceWorker.remote(model_config=model_config) for _ in range(num_workers)]

    # Distribute tasks across workers
    futures = [
        workers[i % num_workers].run_inference.remote(spec.__dict__, base_kwargs)
        for i, spec in enumerate(specs)
    ]

    # Gather results
    results = ray.get(futures)

    return results


def run_single_inference_remote(
    spec: InferenceSpec,
    base_kwargs: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> Any:
    """Run a single inference task using Ray (useful for testing).

    Args:
        spec: InferenceSpec object
        base_kwargs: Additional kwargs for inference
        model_config: Model configuration

    Returns:
        Inference result
    """
    base_kwargs = base_kwargs or {}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    worker = RayInferenceWorker.remote(model_config=model_config)
    result = ray.get(worker.run_inference.remote(spec.__dict__, base_kwargs))

    return result


if __name__ == "__main__":
    # Example usage
    print("Ray Distributed Inference Example")
    print("=" * 50)
    print("To use this module:")
    print("1. Create InferenceSpec objects")
    print("2. Call run_distributed_inference(specs, num_workers=4)")
    print("3. Results will be returned in same order as input specs")
    print("=" * 50)
