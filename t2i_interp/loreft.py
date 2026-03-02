import torch
import torch.nn as nn
import math

class LoReFTLayer(nn.Module):
    """
    LoReFT edit on hidden states h: (B, T, D).

    Φ(h) = h + R^T ( W h + b - R h )

    - R: (r, D)
    - W: (r, D)
    - b: (r,)
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # R: rows ~ orthonormal (we'll renormalize in forward)
        self.R = nn.Parameter(torch.randn(rank, d_model) / math.sqrt(d_model))
        # W, b start near identity / small edit
        self.W = nn.Parameter(torch.zeros(rank, d_model))
        self.b = nn.Parameter(torch.zeros(rank))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, D)
        returns h_edit: (B, T, D)
        """
        h_shape = h.shape
        D = h_shape[-1]
        
        # Flatten everything before the last dimension to handle both (B, D) and (B, T, D)
        h_flat = h.reshape(-1, D)

        # Normalize R rows to keep them roughly orthonormal
        R = torch.nn.functional.normalize(self.R, dim=-1)  # (r, D)

        # Rh, Wh: (BT, r)
        Rh = h_flat @ R.t()
        Wh = h_flat @ self.W.t() + self.b

        # Δh = R^T (Wh - Rh)
        delta = (Wh - Rh) @ R  # (BT, D)
        # alpha = getattr(self, "alpha", 1.0)
        h_edit = h_flat + delta

        # Restore original shape
        return h_edit.view(*h_shape)

class StepConditionalLoReFT(nn.Module):
    """
    A container that holds a separate LoReFTLayer for each diffusion step.
    During training, it processes a stacked tensor [B, num_steps, ...].
    During inference, it processes a single step tensor given `step=...`.
    """
    def __init__(self, d_model: int, rank: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.layers = nn.ModuleList([LoReFTLayer(d_model, rank) for _ in range(num_steps)])

    def forward(self, h: torch.Tensor, step: int | None = None) -> torch.Tensor:
        """
        - If `step` is provided (inference), h is [B, ...]. The specific step's layer is applied.
        - If `step` is None (training), h is [B, num_steps, ...]. Each step's layer is applied to h[:, t].
        """
        if step is not None:
            # Inference applies a specific step's adapter
            adapter_step = min(step, self.num_steps - 1)
            return self.layers[adapter_step](h)
        else:
            # Training applies all active steps simultaneously
            # h is expected to be [B, num_steps, sequence_length, d_model]
            assert h.dim() >= 3 and h.shape[1] == self.num_steps, "For StepConditional training, h must be [B, num_steps, ...]"
            out = []
            for t in range(self.num_steps):
                out.append(self.layers[t](h[:, t]))
            return torch.stack(out, dim=1)
