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
    def compute(self, out: 'Output', prompts: list[str] = None) -> 'Output':
        """
        Computes CLIP score for each image in out.preds against the corresponding prompt.
        If prompts is not provided, it tries to get it from out.labels or out.run_metadata.
        Returns the modified Output object.
        """
        from t2i_interp.utils.output import Output
        if not isinstance(out, Output):
            raise TypeError("out must be of type Output")

        images = out.preds
        if not images:
            return out

        if prompts is None:
            if hasattr(out, "labels") and out.labels is not None:
                prompts = out.labels
            elif hasattr(out, "run_metadata") and out.run_metadata is not None and "prompts" in out.run_metadata:
                prompts = out.run_metadata["prompts"]
            else:
                raise ValueError("prompts must be provided either as argument or in out.labels / out.run_metadata['prompts']")

        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        if len(prompts) != len(images):
            # If there's 1 prompt and multiple images
            if len(prompts) == 1:
                prompts = prompts * len(images)
            else:
                raise ValueError(f"Number of prompts ({len(prompts)}) does not match number of images ({len(images)})")

        scores = []
        for img, prompt in zip(images, prompts):
            scores.append(self.score(img, prompt))

        if out.metrics is None:
            out.metrics = []
        
        # Determine list format or dict format
        if not out.metrics:
            out.metrics = [{"clip_score": s} for s in scores]
        elif len(out.metrics) == len(scores):
            for i, m in enumerate(out.metrics):
                m["clip_score"] = scores[i]
        else:
            # Recreate or append
            out.metrics = [{"clip_score": s} for s in scores]

        return out

    @torch.no_grad()
    def score(
        self,
        images: Union[str, Image.Image, list],
        prompts: Union[str, list],
        **kwargs
    ) -> dict[str, list[float]]:
        """
        Returns cosine similarity between image and text embeddings (both L2-normalized).
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
            
        scores = []
        for image, prompt in zip(images, prompts):
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
            text_tokens = open_clip.tokenize([prompt]).to(self.device)
            txt_feat = self.model.encode_text(text_tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    
            scores.append(float(self._cos(img_feat, txt_feat).item()))
            
        return {"clip_score": scores}
    
# usage
# scorer = CLIPScorer(cache_dir="/cmlscratch/krezaei/cache")
# s = scorer.score("some.png", "style of van gogh")
# print(s) 