import webdataset as wds
import torch

tar_path = "/mnt/data2/nirmal/toolkit/interp_works/latents_cache/loreft_steering/train/unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2_base_prompt.tar"
ds = wds.WebDataset(tar_path).decode()
sample = next(iter(ds))
tensor = sample["unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.pth"]
print(f"Tensor shape: {tensor.shape}")
