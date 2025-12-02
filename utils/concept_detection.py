from email.mime import text
from platform import processor
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch.nn.functional as F
from typing import Tuple
try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    
class ConceptLocalizer:
    def __init__(self, device: str = "cuda:0"):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
        self.model.eval()
        self.device = device

    def get_mask(self, image: Image.Image, text: str,
             threshold: float = 0.5,
             dominant_region: bool = False) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # e.g. [T, H', W']

        if logits.ndim == 3:
            logits_for_concept = logits[0]       # [H', W']
        elif logits.ndim == 4:
            logits_for_concept = logits[0, 0]
        else:
            raise RuntimeError(f"Unexpected logits shape {logits.shape}")

        probs = torch.sigmoid(logits_for_concept)   # [H', W']
        probs_4d = probs.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']

        probs_up = F.interpolate(
            probs_4d,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]                                     # [H_img, W_img]

        base_mask = probs_up > threshold            # bool [H_img, W_img]

        if not dominant_region:
            return base_mask

        # --- dominant_region=True ---

        # If no positive pixel, just return the (all-False) mask
        if base_mask.sum() == 0:                    # sum() is scalar → OK
            return base_mask

        if not _HAS_SCIPY:
            raise RuntimeError("dominant_region=True but SciPy is not installed.")

        # Connected components on CPU
        mask_np = base_mask.cpu().numpy().astype("int32")
        labels, n_labels = ndi.label(mask_np)       # 0 = background

        if n_labels <= 1:
            return base_mask  # only one blob anyway

        probs_np = probs_up.cpu().numpy()
        best_label = None
        best_score = -1.0
        for lab in range(1, n_labels + 1):
            comp_mask = (labels == lab)
            score = probs_np[comp_mask].sum()
            if score > best_score:
                best_score = score
                best_label = lab

        final_mask_np = (labels == best_label)
        final_mask = torch.from_numpy(final_mask_np).to(base_mask.device)
        return final_mask.bool()

    def mask_to_q_indices(
        self,
        mask: torch.Tensor,               # [H_img, W_img], bool or 0/1
        unet_hw: Tuple[int, int],
        ) -> torch.Tensor:
            """
            Downsample a pixel mask to UNet resolution and return flattened Q indices.
            """
            if mask.dtype != torch.float32:
                mask = mask.float()
            H_img, W_img = mask.shape
            H_u, W_u     = unet_hw

            # [1,1,H,W] -> bilinear -> [1,1,H_u,W_u] -> bool
            mask_small = torch.nn.functional.interpolate(
                mask[None, None, :, :],
                size=(H_u, W_u),
                mode="bilinear",
                align_corners=False,
            )[0, 0]  # [H_u, W_u]

            mask_small = mask_small > 0.5
            q_idx = mask_small.view(-1).nonzero(as_tuple=True)[0].to(self.device)
            return q_idx

    def segment_concept(self, image: Image.Image, text: str, threshold: float = 0.5, unet_hw: Tuple[int, int]=(64,64)) -> torch.Tensor:
        mask = self.get_mask(image, text, threshold=threshold)   # [H_img, W_img]
        if mask.sum() == 0:
            return
        q_idx = self.mask_to_q_indices(mask, unet_hw=unet_hw)    # [N_selected]
        return q_idx
    
    def get_topk_heads(self, concept_indices: torch.Tensor, attn_activation: torch.Tensor, n_heads: int, topk: int = 5) -> torch.Tensor:
        """
        Get top-k heads which attend most to the indices in concept.

        concept_indices: [N_selected]
        attn_activations: [B, S, d]  or [B, S, H, d]
        returns:        [topk] indices of heads
        """
        orig_shape = attn_activation.shape
        orig_ndim  = attn_activation.ndim
        S = attn_activation.shape[-2]
        # reshape to expose heads
        if orig_ndim == 3:       # (B, S, D) -> (B, S, H, d)
            hs = attn_activation.view(attn_activation.shape[0], S, n_heads, -1)
        elif orig_ndim == 2:     # (S, D)    -> (S, H, d)
            hs = attn_activation.view(S, n_heads, -1)
        else:
            raise ValueError(f"Unexpected hidden_states shape: {tuple(orig_shape)}")
        
        for index in concept_indices.tolist():
            assert 0 <= index < S, f"concept index {index} out of range for spatial length {S}"
            # filter other indices
        concept_hs = hs[:, concept_indices, ...] if orig_ndim == 3 else hs[concept_indices, ...]  # [B, N_selected, H, d] or [N_selected, H, d
        concept_hs = concept_hs.sum(dim=-1)   # [B, N_selected, H] or [N_selected, H]
        if orig_ndim == 3:
            concept_hs = concept_hs.sum(dim=0)    # [N_selected, H]
        # get top-k heads
        _, topk_heads = torch.topk(concept_hs, k=topk, dim=-1, largest=True, sorted=True)  # [B, N_selected, topk] or [N_selected, topk]
        return topk_heads
        