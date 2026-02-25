from typing import Any

import numpy as np
from PIL import Image


def _as_pil(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().clamp(0, 1)
            if x.ndim == 3 and x.shape[0] in (1, 3):  # C,H,W
                x = x.permute(1, 2, 0).numpy()
            else:
                x = x.numpy()
    except Exception:
        pass
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)
