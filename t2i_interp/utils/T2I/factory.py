from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image,
)

ModelName = Literal["sd14", "sd21", "sd21-turbo", "sdxl", "sdxl-turbo"]


@dataclass
class PipeFactory:
    cache_dir: str = "./cache"
    dtype: torch.dtype = torch.float16
    device: Optional[str] = "cuda"  # set None to not move
    use_safetensors: bool = True    # relevant for SDXL
    variant_fp16: Optional[str] = "fp16"  # relevant for turbo/sdxl fp16 repos

    # Map keys -> HF repo ids
    repo_id: Dict[str, str] = None

    def __post_init__(self):
        if self.repo_id is None:
            self.repo_id = {
                "sd14": "CompVis/stable-diffusion-v1-4",
                "sd21": "stabilityai/stable-diffusion-2-1",
                "sd21-turbo": "stabilityai/sd-turbo",
                "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
                "sdxl-turbo": "stabilityai/sdxl-turbo",
            }

    def create(self, model: ModelName, **overrides: Any):
        """
        Create and return a diffusers pipeline for the given model name.

        Pass any diffusers from_pretrained kwargs via **overrides (e.g. local_files_only=True).
        """
        repo = self.repo_id[model]

        common = dict(
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
            **overrides,
        )

        if model in ("sd14", "sd21"):
            pipe = StableDiffusionPipeline.from_pretrained(repo, **common)

        elif model in ("sd21-turbo", "sdxl-turbo"):
            # Turbo pipelines are commonly exposed via AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                repo,
                variant=self.variant_fp16,
                **common,
            )

        elif model == "sdxl":
            pipe = DiffusionPipeline.from_pretrained(
                repo,
                use_safetensors=self.use_safetensors,
                variant=self.variant_fp16,
                **common,
            )

        else:
            raise ValueError(f"Unknown model: {model}")

        if self.device is not None:
            pipe = pipe.to(self.device)

        return pipe

# usage example:
# factory = PipeFactory(cache_dir="./cache", device="cuda", dtype=torch.float16)

# pipe14 = factory.create("sd14")
# pipeXL = factory.create("sdxl")

# # Override any from_pretrained kwargs
# pipeTurbo = factory.create("sd21-turbo", local_files_only=True)

