from dataclasses import dataclass

from t2Interp.accessors import ModuleAccessor


@dataclass
class TransformerBlock:
    """
    Holds accessors for one transformer layer.

    - For non-attention parts, store individual ModuleAccessor objects:
        in_, mlp_in, mlp_out, out_
    - For attention, use a *single* AttentionAccessor shared by
      attn_in / attn_q / attn_k / attn_v / attn_out.
    """

    # layer: int
    in_: ModuleAccessor | None = None
    mlp_in: ModuleAccessor | None = None
    mlp_out: ModuleAccessor | None = None
    out_: ModuleAccessor | None = None
    attn_in: ModuleAccessor | None = None
    attn_out: ModuleAccessor | None = None
    WO_in: ModuleAccessor | None = None
    WO_out: ModuleAccessor | None = None
    q_in: ModuleAccessor | None = None
    q_out: ModuleAccessor | None = None
    k_in: ModuleAccessor | None = None
    k_out: ModuleAccessor | None = None
    v_in: ModuleAccessor | None = None
    v_out: ModuleAccessor | None = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"TransformerBlock has no attribute '{k}'")
            setattr(self, k, v)

    # ---- helpers ----
    def summary(self) -> str:
        return (
            "in_ | attn_in | attn_out | WO_in | WO_out | mlp_in | mlp_out | out_ | "
            "q_in | k_in | v_in | q_out | k_out | v_out"
        )

    def __repr__(self):
        return self.summary()


@dataclass
class UnetTransformerBlock:
    """
    Holds accessors for one transformer layer.

    - For non-attention parts, store individual ModuleAccessor objects:
        in_, mlp_in, mlp_out, out_
    - For attention, use a *single* AttentionAccessor shared by
      attn_in / attn_q / attn_k / attn_v / attn_out.
    """

    in_: ModuleAccessor | None = None
    mlp_in: ModuleAccessor | None = None
    mlp_out: ModuleAccessor | None = None
    out_: ModuleAccessor | None = None
    cross_attn_in: ModuleAccessor | None = None
    cross_attn_out: ModuleAccessor | None = None
    cross_q_in: ModuleAccessor | None = None
    cross_q_out: ModuleAccessor | None = None
    cross_k_in: ModuleAccessor | None = None
    cross_k_out: ModuleAccessor | None = None
    cross_v_in: ModuleAccessor | None = None
    cross_v_out: ModuleAccessor | None = None
    self_attn_in: ModuleAccessor | None = None
    self_attn_out: ModuleAccessor | None = None
    self_attn_q_in: ModuleAccessor | None = None
    self_attn_q_out: ModuleAccessor | None = None
    self_attn_k_in: ModuleAccessor | None = None
    self_attn_k_out: ModuleAccessor | None = None
    self_attn_v_in: ModuleAccessor | None = None
    self_attn_v_out: ModuleAccessor | None = None
    self_attn_WO_in: ModuleAccessor | None = None
    self_attn_WO_out: ModuleAccessor | None = None
    cross_attn_WO_in: ModuleAccessor | None = None
    cross_attn_WO_out: ModuleAccessor | None = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"TransformerBlock has no attribute '{k}'")
            setattr(self, k, v)

    # ---- helpers ----
    def summary(self) -> str:
        return (
            "in | self_attn_in | self_attn_out | "
            "cross_attn_in | cross_attn_out | "
            "mlp_in | mlp_out | out | "
            "cross_q_in | cross_k_in | cross_v_in | "
            "self_q_in | self_k_in | self_v_in |"
            "cross_q_out | cross_k_out | cross_v_out | "
            "self_q_out | self_k_out | self_v_out"
        )

    def __repr__(self):
        return self.summary()


class SAEBlock:
    """
    Holds accessors for one SAE layer.
    """

    encoder_in: ModuleAccessor = None
    encoder_out: ModuleAccessor = None
    decoder_in: ModuleAccessor = None
    decoder_out: ModuleAccessor = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"TransformerBlock has no attribute '{k}'")
            setattr(self, k, v)

    # ---- helpers ----
    def summary(self) -> str:
        return "encoder_in | encoder_out | decoder_in | decoder_out"

    def __repr__(self):
        return self.summary()
