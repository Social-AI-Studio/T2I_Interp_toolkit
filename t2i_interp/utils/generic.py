from typing import Callable, Optional, Union, Tuple, List, Any, Dict
import inspect
import torch as t
import torch.nn as nn
from torchvision import transforms

Tensor = t.Tensor

def call_with_filtered_kwargs(fn, *args, **kwargs):
    """
    Calls fn(*args, **kwargs) but filters kwargs to only those accepted by fn's signature,
    unless fn accepts **kwargs (VAR_KEYWORD), in which case all are passed.
    """
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **allowed)

# Forward
class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass

def _to_dtype_device(x: Tensor, ref: Tensor) -> Tensor:
    # keep gradients, but match dtype/device for numerical safety
    if x.dtype != ref.dtype:
        x = x.to(ref.dtype)
    if x.device != ref.device:
        x = x.to(ref.device)
    return x

def reshape_like(vec, x):
    """
    vec: tensor with total elements == x.numel()
    x:   target tensor (e.g., (B, C, H, W))
    """
    v = vec.to(dtype=x.dtype, device=x.device)
    if v.numel() != x.numel():
        raise ValueError(f"vec.numel()={v.numel()} != x.numel()={x.numel()}")
    # .reshape handles non-contiguous; .view requires contiguity
    return v.reshape(x.shape)

def flatten_batch(acts: Tensor, device: Optional[Union[str, t.device]] = None) -> Tensor:
    """
    (B, ...) -> (B, -1)
    """
    if isinstance(acts, tuple):
        acts = acts[0]
    if not isinstance(acts, Tensor):
        raise TypeError(f"Captured value is not a Tensor: {type(acts)}")
    if acts.dim() == 0:
        acts = acts.unsqueeze(0)
    B = acts.shape[0] if acts.dim() >= 1 else 1
    if device is not None:
        acts = acts.to(device)
    return acts.view(B, -1)

def _extract_tensor_and_rebuild(output: Any, tensor_index: int = 0) -> Tuple[Optional[Tensor], Callable[[Tensor], Any]]:
    """
    Returns (tensor, rebuild_fn) where rebuild_fn(new_tensor) reconstructs the output.
    Supports:
      - Tensor
      - tuple/list whose first element is a Tensor
      - objects with .sample (diffusers outputs)
    """
    if isinstance(output, t.Tensor):
        return output, (lambda new: new)

    if hasattr(output, "sample") and isinstance(output.sample, t.Tensor):
        # UNet2DConditionOutput-like
        return output.sample, (lambda new: output.__class__(**{**output.__dict__, "sample": new}))

    if isinstance(output, (tuple, list)) and len(output) > 0:
        idx = int(tensor_index)
        # Check boundaries
        if idx >= len(output):
            target_val = output[0]
            write_idx = 0
        else:
            target_val = output[idx]
            write_idx = idx
            
        if isinstance(target_val, t.Tensor):
             def rebuild(new: Tensor):
                if isinstance(output, tuple):
                    # Reconstruct tuple
                    lst = list(output)
                    lst[write_idx] = new
                    return tuple(lst)
                # Reconstruct list
                out = list(output)
                out[write_idx] = new
                return out
             return target_val, rebuild

    return None, (lambda _: output)

def _extract_tensor(data: Any, tensor_index: int = 0) -> Optional[Tensor]:
    """
    Simplified extraction that only returns the tensor, ignoring reconstruction.
    Useful for 'input' hooks or read-only 'output' hooks.
    """
    tensor, _ = _extract_tensor_and_rebuild(data, tensor_index)
    return tensor

def preprocess_image_for_vae(image, target_size=512):
    """
    Preprocess PIL image for VAE encoder.
    
    Parameters:
    -----------
    image : PIL.Image
    target_size : int
    
    Returns:
    --------
    torch.Tensor : Preprocessed image tensor [1, 3, H, W] in range [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    return transform(image).unsqueeze(0)


def build_loss(loss: Union[str, Dict[str, Any], nn.Module] = "mse", **kwargs) -> nn.Module:
    """
    Small loss factory for training scripts.

    Accepts:
    - string name, e.g. "mse", "l1"/"mae", "huber"/"smooth_l1"
    - dict spec, e.g. {"name": "huber", "delta": 1.0, "reduction": "mean"}
    - an nn.Module (returned as-is)

    Extra kwargs override dict fields.
    """
    if isinstance(loss, nn.Module):
        return loss

    params: Dict[str, Any]
    if isinstance(loss, dict):
        name = str(loss.get("name", loss.get("type", "mse"))).lower()
        params = {k: v for k, v in loss.items() if k not in ("name", "type")}
        params.update(kwargs)
    else:
        name = str(loss).lower()
        params = dict(kwargs)

    reduction = str(params.pop("reduction", "mean"))

    if name in ("mse", "mse_loss"):
        return nn.MSELoss(reduction=reduction)
    if name in ("l1", "mae", "l1_loss"):
        return nn.L1Loss(reduction=reduction)
    if name in ("huber", "smooth_l1", "smoothl1"):
        delta = params.pop("delta", params.pop("huber_delta", 1.0))
        return nn.HuberLoss(delta=float(delta), reduction=reduction)

    raise ValueError(f"Unknown loss {name!r}. Supported: mse, l1/mae, huber/smooth_l1.")