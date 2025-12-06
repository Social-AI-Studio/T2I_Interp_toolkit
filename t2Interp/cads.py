# pip install diffusers==0.30.0 transformers accelerate safetensors
import math
import os

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


def _randn_like(x: torch.Tensor, generator=None):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@torch.no_grad()
def generate_with_cads_batch(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 6.5,
    height: int = 512,
    width: int = 512,
    num_images_per_prompt: int = 4,
    generator: torch.Generator | list[torch.Generator] | None = None,
    # CADS knobs
    cads_sigma0: float = 0.06,
    cads_early_frac: float = 0.33,
    cads_schedule: str = "linear",
):
    device = pipe.device
    dtype = pipe.unet.dtype
    B = num_images_per_prompt

    # Encode text once (Diffusers handles repetition when num_images_per_prompt>1)
    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=B,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    uncond_embeds, cond_embeds = prompt_embeds.chunk(2, dim=0)  # shapes: [B, seq, dim] each
    ref_std = float(cond_embeds.float().std().clamp(min=1e-6))

    # Timesteps & initial latents (B images)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = randn_tensor(
        (B, pipe.unet.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    def _scale_latents(latents, t):
        return (
            pipe.scheduler.scale_model_input(latents, t)
            if hasattr(pipe.scheduler, "scale_model_input")
            else latents
        )

    cutoff = int(math.ceil(cads_early_frac * num_inference_steps))

    def decay(i):
        if i >= cutoff:
            return 0.0
        x = 1.0 - (i / max(1, cutoff - 1))
        return 0.5 * (1 - math.cos(math.pi * x)) if cads_schedule == "cosine" else x

    for step_idx, t in enumerate(timesteps):
        d = decay(step_idx)
        if d > 0:
            eps = _randn_like(cond_embeds, generator=generator)  # per-sample, per-token noise
            cond_step = cond_embeds + (cads_sigma0 * d * ref_std) * eps
        else:
            cond_step = cond_embeds

        embeds_step = torch.cat([uncond_embeds, cond_step], dim=0)  # [2B, seq, dim]
        latent_in = torch.cat([latents, latents], dim=0)  # [2B, C, H, W]
        latent_in = _scale_latents(latent_in, t)

        noise = pipe.unet(latent_in, t, encoder_hidden_states=embeds_step).sample
        noise_u, noise_c = noise.chunk(2, dim=0)
        noise_guided = noise_u + guidance_scale * (noise_c - noise_u)

        latents = pipe.scheduler.step(noise_guided, t, latents).prev_sample

    # Decode all B images
    latents = latents / pipe.vae.config.scaling_factor
    imgs = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    pil_images = pipe.image_processor.postprocess(imgs, output_type="pil")
    return pil_images  # list of PIL.Image


# ---------------- Example usage ----------------
if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")

g = torch.Generator(device=pipe.device).manual_seed(12345)
images = generate_with_cads_batch(
    pipe,
    prompt="doctor",
    negative_prompt="low quality, blurry",
    num_inference_steps=30,
    guidance_scale=6.5,
    num_images_per_prompt=4,
    generator=g,  # or a list of Generators for per-image seeds
    cads_sigma0=0.06,
    cads_early_frac=0.33,
    cads_schedule="linear",
)
os.makedirs("./output", exist_ok=True)
for i, im in enumerate(images):
    im.save(f"./output/doctor_cads_{i}.png")
