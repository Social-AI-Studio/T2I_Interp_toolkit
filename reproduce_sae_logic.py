
import torch
import torch.nn as nn
from typing import Tuple, Callable

# Mocking the minimal logic from SAEManager for flattening
def _get_sae_dim(sae: nn.Module) -> int:
    if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
        return sae.encoder.weight.shape[1]
    if hasattr(sae, "config") and hasattr(sae.config, "d_in"):
        return sae.config.d_in
    if hasattr(sae, "d_in"):
        return sae.d_in
    raise ValueError("Cannot determine input dimension of SAE.")

def _flatten_for_sae(x: torch.Tensor, sae: nn.Module) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    d_in = _get_sae_dim(sae)
    
    # Check if d_in matches x.shape[-1]
    if x.shape[-1] == d_in:
        orig_shape = x.shape
        x_flat = x.reshape(-1, d_in)
        def restore(flat):
            return flat.reshape(orig_shape)
        return x_flat, restore
        
    dims = [i for i, d in enumerate(x.shape) if d == d_in]
    if not dims:
        raise RuntimeError(f"SAE expects input dim {d_in}, but input shape is {x.shape}. No dimension matches.")
    
    if len(dims) > 1:
         if (len(x.shape) - 1) in dims:
             target_dim = len(x.shape) - 1
         else:
             if 1 in dims:
                 target_dim = 1
             else:
                 target_dim = dims[0]
    else:
        target_dim = dims[0]
        
    ndim = x.ndim
    perm = list(range(ndim))
    perm.pop(target_dim)
    perm.append(target_dim)
    
    x_perm = x.permute(*perm)
    orig_perm_shape = x_perm.shape
    x_flat = x_perm.reshape(-1, d_in)
    
    def restore(flat):
        unflat = flat.reshape(orig_perm_shape)
        inv_perm = [0] * ndim
        for dst_idx, src_idx in enumerate(perm):
            inv_perm[src_idx] = dst_idx
        return unflat.permute(*inv_perm)
        
    return x_flat, restore

# Mock SAE
class MockSAE(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.d_in = d_in
        self.encoder = nn.Linear(d_in, d_in) # Simple identity-like
        self.decoder = nn.Linear(d_in, d_in)

def test_flattening_logic():
    B, C, H, W = 1, 320, 16, 16
    d_in = 320
    
    x = torch.randn(B, C, H, W)
    sae = MockSAE(d_in)
    
    print(f"Input shape: {x.shape}")
    print(f"SAE d_in: {d_in}")
    
    x_flat, restore_fn = _flatten_for_sae(x, sae)
    print(f"Flattened shape: {x_flat.shape}")
    
    # Expected: (B*H*W, C) = (256, 320)
    expected_flat_shape = (B * H * W, C)
    if x_flat.shape == expected_flat_shape:
        print("PASS: Flattened shape matches expected (B*H*W, C)")
    else:
        print(f"FAIL: Flattened shape {x_flat.shape} != expected {expected_flat_shape}")

    # Check mapping
    # x[0, :, 0, 0] should match x_flat[0, :]?
    # x_perm was (B, H, W, C) [0, 2, 3, 1]
    # x_flat is (B*H*W, C)
    # index 0 -> (0, 0, 0)
    # index 1 -> (0, 0, 1)
    
    val0 = x[0, :, 0, 0]
    flat0 = x_flat[0]
    if torch.allclose(val0, flat0):
        print("PASS: Value mapping (0,0) correct")
    else:
        print("FAIL: Value mapping (0,0) incorrect")
        
    val_last = x[0, :, 15, 15]
    flat_last = x_flat[-1]
    if torch.allclose(val_last, flat_last):
        print("PASS: Value mapping (15,15) correct")
    else:
        print("FAIL: Value mapping (15,15) incorrect")

    # Test restoration
    x_restored = restore_fn(x_flat)
    print(f"Restored shape: {x_restored.shape}")
    if torch.equal(x, x_restored):
        print("PASS: Restoration exact match")
    else:
        print("FAIL: Restoration mismatch")

    print("-" * 20)
    # Test example.ipynb logic comparison
    # example.ipynb: diff.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0)
    # If input diff was (1, 1, C, H, W)?
    # Let's assume input diff was 5D. 
    # Logic: if _flatten_for_sae works on (B, C, H, W) and user does .view(H, W, -1), does it match?
    
    # In sae_steer.ipynb, user does: output.preds["..."].view(16, 16, -1)
    # If output.preds is x_flat (256, 320)
    # view(16, 16, 320)
    # This corresponds to (H, W, C)
    
    viewed = x_flat.view(16, 16, -1)
    # viewed[0, 0, :] should be flat0 which is x[0, :, 0, 0] ?
    # Wait. x[0, :, 0, 0] is the channel vector at (0,0).
    # x_flat[0] is the channel vector at (0,0).
    # viewed[0, 0] is x_flat[0].
    # So viewed[0, 0] contains channels at spatial (0,0).
    
    # example.ipynb: heatmap = sparse_maps[:, :, feature]
    # This expects spatial (H, W).
    # If viewed is (16, 16, C), then viewed[:, :, feature] gives spatial map of feature.
    # This assumes x_flat was row-major H*W.
    # _flatten_for_sae used permute(0, 2, 3, 1) -> (B, H, W, C).
    # reshape(-1, C) collapses B, H, W in that order.
    # So first dims are B, then H, then W.
    # Yes, it is row-major.
    print("Logic check: _flatten_for_sae produces row-major spatial flattening (B, H, W, C).")
    print("User .view(16, 16, -1) assumes output is (H, W, C).")
    print("This matches.")

if __name__ == "__main__":
    test_flattening_logic()
