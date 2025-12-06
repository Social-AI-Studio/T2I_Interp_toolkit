# pip install diffusers==0.30.0 transformers accelerate scikit-learn pillow torchvision
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from sklearn.cluster import KMeans
from torch import Tensor
from torchvision import transforms

# -------------------------- CONFIG --------------------------

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Roles (prompt buckets) you’ll audit; edit as needed
ROLES = ["doctor", "engineer", "teacher", "CEO"]

# Attribute vocabularies (you can expand these)
GENDER = ["female", "male"]
RACE = ["Black", "White", "East Asian", "South Asian", "Middle Eastern", "Latino"]
AGE = ["child", "young", "middle-aged", "elderly"]

# Number of clusters to keep per attribute value (per role)
K_CLUSTERS = 4

# SDEdit start (how deep into noise we push during mode test)
T_START_FRAC = 0.85  # 0.8–0.9 is typical

# Sampler settings (keep these fixed for atlas ↔ inference consistency)
NUM_STEPS = 30
GUIDANCE = 7.0

# --------------------- TEXT-IMAGE TAGGER (CLIP) ---------------------

from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def clip_argmax(img: Image.Image, texts: list[str]) -> int:
    inputs = clip_proc(text=texts, images=img, return_tensors="pt", padding=True).to(DEVICE)
    logits = clip_model(**inputs).logits_per_image[0]
    return int(logits.argmax().item())


def tag_gender(img: Image.Image) -> str:
    opts = ["female", "male"]
    i = clip_argmax(img, [f"a photo of a {o} person" for o in opts])
    return opts[i]


def tag_race(img: Image.Image) -> str:
    # very rough; replace with a calibrated detector if available
    opts = RACE
    i = clip_argmax(img, [f"a photo of a {o} person" for o in opts])
    return opts[i]


def tag_age(img: Image.Image) -> str:
    opts = AGE
    i = clip_argmax(img, [f"a photo of a {o} person" for o in opts])
    return opts[i]


# --------------------- SD PIPE + UTILITIES ---------------------

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)  # DDIM(η=0) pairs well with SDEdit
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

to_tensor = transforms.ToTensor()


@torch.no_grad()
def encode_vae(img: Image.Image) -> Tensor:
    x = to_tensor(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    x = (x - 0.5) * 2
    z = pipe.vae.encode(x.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
    return z


@torch.no_grad()
def decode_vae(z: Tensor) -> Image.Image:
    z = z / pipe.vae.config.scaling_factor
    x = pipe.vae.decode(z.half()).sample
    x = (x / 2 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(x[0].float().cpu())


def add_noise_at_index(latent: Tensor, t_index: int) -> Tensor:
    # Use scheduler betas/alphas to inject noise consistent with DDIM schedule
    betas = pipe.scheduler.betas.to(latent.device, latent.dtype)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    t = pipe.scheduler.timesteps[t_index].long().item()
    a = alpha_bar[t]
    eps = torch.randn_like(latent)
    return (a**0.5) * latent + ((1 - a) ** 0.5) * eps


@torch.no_grad()
def denoise_from_index(z_t: Tensor, neutral_prompt: str, start_index: int) -> Image.Image:
    # Manual loop from t=start_index down to 0 with CFG
    pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps[: start_index + 1]
    embeds = pipe._encode_prompt(
        neutral_prompt,
        DEVICE,
        1,
        True,
        negative_prompt="",
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    uncond, cond = embeds.chunk(2, dim=0)

    latents = z_t.clone()
    for t in timesteps:
        x = torch.cat([latents, latents], dim=0)
        if hasattr(pipe.scheduler, "scale_model_input"):
            x = pipe.scheduler.scale_model_input(x, t)
        ehs = torch.cat([uncond, cond], dim=0)
        noise = pipe.unet(x, t, encoder_hidden_states=ehs).sample
        n_u, n_c = noise.chunk(2, dim=0)
        n = n_u + GUIDANCE * (n_c - n_u)
        latents = pipe.scheduler.step(n, t, latents).prev_sample
    return decode_vae(latents)


@torch.no_grad()
def txt2img(prompt: str, seed: int) -> Image.Image:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE, generator=g).images[
        0
    ]


# --------------------- MODE TEST → LATENT COLLECTION ---------------------


def explicit_prompt(role: str, attribute: str, value: str) -> str:
    # How we write explicit prompts for exemplars (edit to taste)
    if attribute == "gender":
        return f"a photo of a {value} {role}"
    if attribute == "race":
        return f"a photo of a {value} {role}"
    if attribute == "age":
        return f"a photo of a {value} {role}"
    return f"a photo of a {role}"


def neutral_prompt(role: str) -> str:
    return f"a photo of a {role}"


def tag_attribute(img: Image.Image, attribute: str) -> str:
    return {"gender": tag_gender, "race": tag_race, "age": tag_age}[attribute](img)


def mode_test_latents_for_value(
    role: str,
    attribute: str,
    value: str,
    seeds: list[int],
    t_start_frac: float = T_START_FRAC,
) -> list[np.ndarray]:
    """
    For an attribute value (e.g., female), collect high-t latents z_t that,
    when denoised with the neutral prompt, often regenerate the same value.
    """
    latents = []
    exp_prompt = explicit_prompt(role, attribute, value)
    neut = neutral_prompt(role)

    pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    start_idx = int(math.floor(t_start_frac * (len(pipe.scheduler.timesteps) - 1)))

    for s in seeds:
        # 1) explicit exemplar
        img_exp = txt2img(exp_prompt, s)
        # 2) push toward noise
        z0 = encode_vae(img_exp)
        zt = add_noise_at_index(z0, start_idx)
        # 3) denoise with neutral prompt
        img_neu = denoise_from_index(zt, neut, start_idx)
        # 4) accept if attribute reappears (acts as a filter)
        pred = tag_attribute(img_neu, attribute)
        if pred == value:
            latents.append(zt.squeeze(0).detach().float().cpu().numpy().reshape(-1))
    return latents


# --------------------- CLUSTERING → (mu, Sigma) ---------------------


def cluster_latents(latents: list[np.ndarray], k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    KMeans centroids + empirical covariances (regularized).
    Returns ([mu_k], [Sigma_k]).
    """
    X = np.stack(latents, axis=0)  # [N, D]
    km = KMeans(n_clusters=min(k, len(X)), n_init=10, random_state=0)
    labels = km.fit_predict(X)
    mus, covs = [], []
    for c in range(km.n_clusters):
        pts = X[labels == c]
        mu = pts.mean(axis=0)
        # covariance with ridge regularization
        dif = pts - mu
        Sigma = (dif.T @ dif) / max(1, (len(pts) - 1))
        # add small ridge to avoid singular
        Sigma = Sigma + (1e-3 * np.eye(Sigma.shape[0], dtype=Sigma.dtype))
        mus.append(mu)
        covs.append(Sigma)
    return mus, covs


# --------------------- ATLAS BUILDER ---------------------


@dataclass
class AtlasEntry:
    mus: list[np.ndarray]
    covs: list[np.ndarray]
    count: int


Atlas = dict[str, dict[str, AtlasEntry]]  # atlas[attribute][value] -> entry


def build_atlas_for_role(
    role: str,
    seeds_per_value: int = 64,
    attributes: dict[str, list[str]] = None,
    k_clusters: int = K_CLUSTERS,
) -> Atlas:
    """
    For a role (e.g., 'doctor'), build per-attribute atlases via the mode test.
    """
    if attributes is None:
        attributes = {"gender": GENDER, "race": RACE, "age": AGE}

    atlas: Atlas = {}
    for attr, values in attributes.items():
        atlas[attr] = {}
        for v in values:
            seeds = list(range(1000, 1000 + seeds_per_value))
            zts = mode_test_latents_for_value(role, attr, v, seeds)
            if len(zts) == 0:
                print(f"[WARN] No accepted latents for {role}/{attr}={v}")
                atlas[attr][v] = AtlasEntry(mus=[], covs=[], count=0)
                continue
            mus, covs = cluster_latents(zts, k_clusters)
            atlas[attr][v] = AtlasEntry(mus=mus, covs=covs, count=len(zts))
            print(f"[OK] {role}/{attr}={v}: {len(zts)} latents -> {len(mus)} clusters")
    return atlas


# --------------------- WEIGHTED INFERENCE ---------------------


def sample_mog(
    mu: np.ndarray, Sigma: np.ndarray, tau: float, shape: tuple[int, int, int]
) -> Tensor:
    """
    Sample from N(mu, tau^2 Sigma) then reshape to latent tensor [C,H,W].
    """
    D = mu.shape[0]
    z = np.random.randn(D)
    # For efficiency we sample with diagonal of Sigma (or do a cheap eig)
    # Here: use full Sigma via cholesky if possible; fallback to diag.
    try:
        L = np.linalg.cholesky(Sigma)
        sample = mu + tau * (L @ z)
    except np.linalg.LinAlgError:
        diag = np.sqrt(np.maximum(np.diag(Sigma), 1e-6))
        sample = mu + tau * (diag * z)
    arr = sample.astype(np.float32).reshape(shape)
    return torch.from_numpy(arr).to(DEVICE, dtype=DTYPE)


def choose_value_by_weights(weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(keys, p=probs))


def pick_centroid(entry: AtlasEntry) -> tuple[np.ndarray, np.ndarray]:
    # Uniform over clusters; you can weight by cluster size if you keep counts
    idx = np.random.randint(len(entry.mus))
    return entry.mus[idx], entry.covs[idx]


@torch.no_grad()
def generate_with_atlas(
    pipe: StableDiffusionPipeline,
    prompt: str,
    role: str,
    atlas_by_role: dict[str, Atlas],
    weights: dict[
        str, dict[str, float]
    ],  # e.g., {"gender":{"female":0.7,"male":0.3}, "race":{...}, "age":{...}}
    mode: str = "seed",  # "seed" (mixture seeding) or "nudge" (first-step nudge)
    pi: float = 0.5,  # prob. to use atlas vs standard N(0,I) (seed mode)
    tau: float = 0.7,  # covariance shrink (seed mode)
    eps: float = 0.10,  # nudge magnitude (nudge mode)
    height: int = 512,
    width: int = 512,
    negative_prompt: str = "",
) -> Image.Image:
    """
    Weighted inference: pick an attribute and value by 'weights', then either
    (A) seed from that centroid’s Gaussian, or (B) nudge x_T toward that centroid for early steps.
    """
    if role not in atlas_by_role:
        raise ValueError(f"No atlas for role {role}")

    atlas = atlas_by_role[role]
    # choose which attribute to steer this time (you can randomize or prioritize)
    which_attr = random.choice([a for a in weights.keys() if a in atlas])
    val = choose_value_by_weights(weights[which_attr])
    entry = atlas[which_attr].get(val, AtlasEntry([], [], 0))
    use_atlas = len(entry.mus) > 0

    # Prepare timesteps
    pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps

    # Build initial latent x_T
    C = pipe.unet.in_channels
    H = height // pipe.vae_scale_factor
    W = width // pipe.vae_scale_factor

    if mode == "seed" and use_atlas and (np.random.rand() < pi):
        mu, Sigma = pick_centroid(entry)
        zT = sample_mog(mu, Sigma, tau=tau, shape=(C, H, W))
    else:
        zT = torch.randn((1, C, H, W), device=DEVICE, dtype=DTYPE)

    latents = zT if zT.ndim == 4 else zT.unsqueeze(0)

    # Text embeddings (CFG)
    embeds = pipe._encode_prompt(prompt, DEVICE, 1, True, negative_prompt, None, None)
    uncond, cond = embeds.chunk(2, dim=0)

    # Optional: early-step nudge (works even if you didn't seed from atlas)
    # Only if we have a centroid to aim at
    if mode == "nudge" and use_atlas:
        mu, Sigma = pick_centroid(entry)
        # reshape mu to latent tensor and compute a tiny move
        mu_t = torch.from_numpy(mu.astype(np.float32).reshape(C, H, W)).to(DEVICE, dtype=DTYPE)
        direction = mu_t - latents[0]
        step0 = eps * direction / (direction.norm() + 1e-6)
        latents[0] = latents[0] + step0  # first-step nudge; you can decay more steps if you like

    # Denoising
    for t in timesteps:
        x = torch.cat([latents, latents], dim=0)
        if hasattr(pipe.scheduler, "scale_model_input"):
            x = pipe.scheduler.scale_model_input(x, t)
        ehs = torch.cat([uncond, cond], dim=0)
        noise = pipe.unet(x, t, encoder_hidden_states=ehs).sample
        n_u, n_c = noise.chunk(2, dim=0)
        n = n_u + GUIDANCE * (n_c - n_u)
        latents = pipe.scheduler.step(n, t, latents).prev_sample

    # Decode
    latents = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


# --------------------- EXAMPLE WORKFLOW ---------------------
if __name__ == "__main__":
    role = "CEO"
    os.makedirs("./output", exist_ok=True)

    # 1) Build an atlas for a role (this can take time; do once and save)
    attributes = {"gender": GENDER, "race": RACE, "age": AGE}
    # check if atlas already exists

    if os.path.exists(f"./output/atlas_{role}.json"):
        print(f"Loading existing atlas for role {role}...")
        with open(f"./output/atlas_{role}.json") as f:
            serial = json.load(f)

        def list_to_np(a):
            return np.array(a) if isinstance(a, list) else a

        atlas_role = {
            attr: {
                val: AtlasEntry(
                    mus=[list_to_np(m) for m in ent["mus"]],
                    covs=[list_to_np(c) for c in ent["covs"]],
                    count=ent["count"],
                )
                for val, ent in d.items()
            }
            for attr, d in serial.items()
        }
    else:
        print(f"Building atlas for role {role}...")
        atlas_role = build_atlas_for_role(
            role, seeds_per_value=48, attributes=attributes, k_clusters=K_CLUSTERS
        )

    # Save to disk
    def np_to_list(a):
        return a.tolist() if isinstance(a, np.ndarray) else a

    serial = {
        attr: {
            val: {
                "mus": [np_to_list(m) for m in ent.mus],
                "covs": [np_to_list(c) for c in ent.covs],
                "count": ent.count,
            }
            for val, ent in d.items()
        }
        for attr, d in atlas_role.items()
    }
    with open(f"./output/atlas_{role}.json", "w") as f:
        json.dump(serial, f)

    # 2) Load atlas for multiple roles (here just one for demo)
    atlas_by_role = {role: atlas_role}

    # 3) Weighted inference examples
    #   (A) seed mixture: 70% female; otherwise sample standard Gaussian
    weights = {
        "gender": {"female": 0.5, "male": 0.5, "black": 0.5, "white": 0.5},
    }
    img_seed = generate_with_atlas(
        pipe,
        f"a photo of a {role}",
        role,
        atlas_by_role,
        weights=weights,
        mode="seed",
        pi=0.7,
        tau=0.7,
    )

    img_seed.save("./output/ceo_seed.png")

    #   (B) tiny early nudge: prefer "East Asian" race this time
    weights2 = {"race": {"East Asian": 1.0}}
    img_nudge = generate_with_atlas(
        pipe, f"a photo of a {role}", role, atlas_by_role, weights=weights2, mode="nudge", eps=0.12
    )
    img_nudge.save("./output/ceo_nudge.png")
