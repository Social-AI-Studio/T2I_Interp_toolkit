import torch
from typing import Optional, Union
from PIL import Image
from dataclasses import dataclass
import open_clip

@dataclass
class CLIPScorer:
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    device: Optional[Union[str, torch.device]] = None
    cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        # create once
        self._cos = torch.nn.CosineSimilarity(dim=-1)

    @torch.no_grad()
    def score(
        self,
        image: Union[str, Image.Image],
        prompt: str,
    ) -> float:
        """
        Returns cosine similarity between image and text embeddings (both L2-normalized).
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"image must be a path or PIL.Image.Image, got {type(image)}")

        # Image -> features
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)  # [1, C, H, W]
        img_feat = self.model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Text -> features
        text_tokens = self.tokenizer([prompt]).to(self.device)
        txt_feat = self.model.encode_text(text_tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        return float(self._cos(img_feat, txt_feat).item())
    
# usage
# scorer = CLIPScorer(cache_dir="/cmlscratch/krezaei/cache")
# s = scorer.score("some.png", "style of van gogh")
# print(s) 