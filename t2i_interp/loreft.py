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
        B, T, D = h.shape
        h_flat = h.reshape(-1, D)  # (BT, D)

        # Normalize R rows to keep them roughly orthonormal
        R = torch.nn.functional.normalize(self.R, dim=-1)  # (r, D)

        # Rh, Wh: (BT, r)
        Rh = h_flat @ R.t()
        Wh = h_flat @ self.W.t() + self.b

        # Δh = R^T (Wh - Rh)
        delta = (Wh - Rh) @ R  # (BT, D)
        h_edit = h_flat + delta

        return h_edit.view(B, T, D)
