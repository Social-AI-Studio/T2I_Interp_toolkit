import torch as t
from typing import Any, Callable, Dict, List, Union
Tensor = t.Tensor


def resolve_module_from_layer_name(pipe, layer_name: str):
    """
    Resolve a submodule from a dotted/attr path safely.
    Accepts either
      - 'unet.down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0'
      - 'pipe.unet.down_blocks.2....' (with leading 'pipe.')
    """
    # allow both "pipe." prefix and bare
    path = layer_name
    if path.startswith("pipe."):
        path = path[len("pipe.") :]

    obj = pipe
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

# Total number of computable operations / modules -- 709
def high_level_layers(model):
    # Counter for the list
    c = 0
    # Stores the relevant layers to perform causal tracing on 
    relevant_modules = []
    # Total list of all modules
    named_module_list = []
    for n,m in model.unet.named_modules():
        c += 1
        named_module_list.append(n)

    # Ends with 'attn2', 'attn1'
    attn_list = []
    for item in named_module_list:
        if 'attn2' in item and ('to_k' in item or 'to_v' in item):
            attn_list.append(item)
    
    #print(attn_list)
    # Layernames
    return attn_list

def _prep_prompts_images(
        batch: Union[List[Any], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Accepts:
          - list of mixed samples (str prompts and/or images tensors/PIL) -> split into dict
          - dict with keys 'prompt' and/or 'image' -> used directly
        Ensures at least an empty prompt list if only images are provided.
        """
        # Case 1: batch is a dict already
        if isinstance(batch, dict):
            prompts = batch.get("prompt", None)
            images = batch.get("image", None)

            # Normalize prompts
            if prompts is None:
                if images is not None:
                    bs = len(images) if hasattr(images, "__len__") else 1
                    prompts = [""] * bs
            elif isinstance(prompts, str):
                prompts = [prompts]

            out = {}
            if prompts is not None:
                out["prompt"] = prompts
            if images is not None:
                out["image"] = images
            return out

        # Case 2: batch is a list of mixed entries
        prompts: List[str] = []
        images: List[Any] = []
        for x in batch:
            if isinstance(x, str):
                prompts.append(x)
            else:
                images.append(x)

        if not prompts and images:
            prompts = [""] * len(images)

        out: Dict[str, Any] = {}
        if prompts:
            out["prompt"] = prompts
        if images:
            out["image"] = images
        return out
    
def last_token_indices(tokenizer, prompts):
    enc = tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    last_by_mask = mask.sum(dim=1) - 1

    eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "sep_token_id", None)
    if eos_id is None:
        return last_by_mask

    eos_mask = (ids == eos_id)
    has_eos = eos_mask.any(dim=1)
    # first EOS position (works even if EOS is repeated to the end)
    eos_pos_first = eos_mask.int().argmax(dim=1)
    return t.where(has_eos, eos_pos_first, last_by_mask)    

# def run_with_hook(
#     pipe,
#     batch,
#     module: t.nn.Module,
#     hook_obj: Any, 
#     **pipe_kwargs,
#     ) -> Any:
#         """
#         Register hook_obj.hook on `module`, run `runner()`, then remove the hook.
#         Returns (hook_obj, result).
#         """
#         # handle = _register(module, hook_obj.hook)
#         hook_obj.register(module)
#         try:
#             io = _prep_prompts_images(batch)
#             if io.get("prompt", None) is not None:
#                 prompt_inputs = io["prompt"]
#                 # hook_obj.last_token_indices = last_token_indices(pipe.tokenizer, prompt_inputs)
                
#             # If all prompts empty, avoid CFG pulling toward text by mistake
#             if "prompt" in io and isinstance(io["prompt"], list) and all(p == "" for p in io["prompt"]):
#                 print("All prompts empty; setting guidance_scale=1.0 to avoid CFG with empty text.")
#                 pipe_kwargs.setdefault("guidance_scale", 1.0)
                
#             if io.get("image", None) is None:   
#                 result = pipe(**io, **pipe_kwargs)
#             else:
#                 if isinstance(io["image"][0], Image):
#                     batch_tensors = t.stack([
#                         preprocess_image_for_vae(img) for img in io["image"]
#                     ]).squeeze(1)
#                 else:
#                     batch_tensors = t.stack(io["image"]).squeeze(1)
                    
#                 batch_tensors = batch_tensors.to(device=pipe.device, dtype=pipe.dtype)    
#                 with t.no_grad():
#                     latents = pipe.vae.encode(batch_tensors).latent_dist.sample()
#                     latents = latents * pipe.vae.config.scaling_factor  
                
#                 result = pipe.unet(
#                     prompt=io.get("prompt", None),
#                     latents=latents,
#                     **pipe_kwargs
#                 )
#         finally:
#             hook_obj._handle.remove()
#         return result, getattr(hook_obj,"last",None)
    
@t.no_grad()
def compute_last_token_indices(pipe, prompts, device=None) -> Tensor:
    if isinstance(prompts, str):
        prompts = [prompts]
    enc = pipe.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    attn = enc["attention_mask"]  # [B, T], 1 for valid tokens, 0 for pad
    idx = (attn.to(t.int).sum(dim=-1) - 1).clamp(min=0)  # [B]
    if device is not None:
        idx = idx.to(device)
    return idx    