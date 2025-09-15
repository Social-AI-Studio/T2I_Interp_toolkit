import torch
from PIL import Image
import numpy as np
from utils.output import Output        
from PIL.Image import Image
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import numpy as np
from typing import List, Union, Sequence, Optional
from transformers import CLIPModel, CLIPProcessor

class MetricBase:
    def __init__(self):
        pass

    def compute(self, *args, **kwargs):
        pass

class CLIPImageDataset(torch.utils.data.Dataset):
    """If you still want a DataLoader path; not used in compute() below."""
    def __init__(self, data: Sequence[Union[str, Image.Image]]):
        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx]
        img = Image.open(x) if isinstance(x, str) else x
        return {"image": img.convert("RGB")}

    def __len__(self):
        return len(self.data)


        
class CLIPScore(MetricBase):
    """
    Cosine similarity between a GT image and each candidate image using CLIP image encoder.
    Uses Hugging Face Transformers checkpoint names, e.g. 'openai/clip-vit-large-patch14'.
    """
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

    def compute(self, images: Union[List[Image.Image], Output], **kwargs) -> Output:
        # Accept Output or raw list[Image]
        if isinstance(images, Output):
            out = images
            gt_image = kwargs.get("gt", None)
            assert gt_image is not None, "Need gt image when passing raw images"
            cands = images.preds
        else:
            assert len(out.preds) >= 2, "Need [gt, cand1, cand2, ...] in Output.preds"
            gt_image, cands = out.preds[0], out.preds[1:]
            out = Output()
            

        # Ensure PIL and RGB
        gt_image = gt_image.convert("RGB")
        cands = [im.convert("RGB") for im in cands]

        with torch.no_grad():
            # Encode candidates
            cand_inputs = self.processor(images=cands, return_tensors="pt").to(self.device)
            cand_feats = self.model.get_image_features(**cand_inputs)        # [N, D]
            cand_feats = self._l2_normalize(cand_feats)

            # Encode ground truth
            gt_inputs = self.processor(images=gt_image, return_tensors="pt").to(self.device)
            gt_feat = self.model.get_image_features(**gt_inputs)             # [1, D]
            gt_feat = self._l2_normalize(gt_feat)

            # Cosine similarity
            sim = (gt_feat @ cand_feats.T).squeeze(0).detach().cpu().numpy()  # [N]

        # Attach metrics and return Output
        out.metrics = (out.metrics or {})
        out.metrics["clip_score"] = sim.tolist()
        return out