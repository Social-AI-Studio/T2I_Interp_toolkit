from dataclasses import dataclass

from t2i_interp.accessors.accessor import ModuleAccessor




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
