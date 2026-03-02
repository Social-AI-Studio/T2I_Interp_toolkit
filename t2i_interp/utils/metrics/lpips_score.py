# lpips_score.py
# pip install lpips torch torchvision pillow

import argparse
import os
from typing import List, Tuple

import torch
import lpips
from PIL import Image
import torchvision.transforms as T
def list_pairs(ref_dir: str, pred_dir: str) -> List[Tuple[str, str]]:
    # Pair by filename intersection
    ref_files = {f for f in os.listdir(ref_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}
    pred_files = {f for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}
    common = sorted(ref_files & pred_files)
    return [(os.path.join(ref_dir, f), os.path.join(pred_dir, f)) for f in common]

def load_img_pil(img: Image.Image, device: str) -> torch.Tensor:
    img = img.convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0
    return x

def load_img(path: str, device: str) -> torch.Tensor:
    return load_img_pil(Image.open(path), device)

class LPIPSScorer:
    def __init__(self, net: str = "alex", device: str = None):
        self.net = net
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net=self.net).to(self.device).eval()

    @torch.no_grad()
    def compute(self, out: 'Output', ref_images=None) -> 'Output':
        from t2i_interp.utils.output import Output
        if not isinstance(out, Output):
            raise TypeError("out must be of type Output")

        preds = out.preds
        
        if ref_images is None:
            if hasattr(out, "baselines") and out.baselines is not None:
                ref_images = out.baselines
            else:
                return out

        if not isinstance(ref_images, list) and not isinstance(ref_images, tuple):
            ref_images = [ref_images] * len(preds)
        
        if len(ref_images) != len(preds):
            if len(ref_images) == 1:
                ref_images = ref_images * len(preds)
            else:
                raise ValueError("Mismatch in number of preds and ref_images")

        scores = []
        for ref, pred in zip(ref_images, preds):
            if isinstance(ref, str):
                ref_t = load_img(ref, self.device)
            else:
                ref_t = load_img_pil(ref, self.device)
                
            if isinstance(pred, str):
                pred_t = load_img(pred, self.device)
            else:
                pred_t = load_img_pil(pred, self.device)

            v = self.loss_fn(ref_t, pred_t)
            scores.append(float(v.item()))

        if out.metrics is None:
            out.metrics = []
            
        metric_dict = {"lpips_score": sum(scores) / len(scores)}
        if not out.metrics:
            out.metrics = [metric_dict] * len(preds)
        else:
            for m in out.metrics:
                m["lpips_score"] = float(metric_dict["lpips_score"])

        return out

    @torch.no_grad()
    def score(self, images, references=None, **kwargs):
        """Standardized interface for run_steer.py"""
        if references is None:
            # LPIPS requires a reference image. If not provided, it fails gracefully.
            return {"lpips_score": float('nan')}
            
        preds = images
        ref_images = references

        if not isinstance(ref_images, list) and not isinstance(ref_images, tuple):
            ref_images = [ref_images] * len(preds)
        
        if len(ref_images) != len(preds):
            if len(ref_images) == 1:
                ref_images = ref_images * len(preds)
            else:
                return {"lpips_score": float('nan')}

        scores = []
        for ref, pred in zip(ref_images, preds):
            if isinstance(ref, str):
                ref_t = load_img(ref, self.device)
            elif isinstance(ref, Image.Image):
                ref_t = load_img_pil(ref, self.device)
            else:
                ref_t = ref.to(self.device).float()
                
            if isinstance(pred, str):
                pred_t = load_img(pred, self.device)
            elif isinstance(pred, Image.Image):
                pred_t = load_img_pil(pred, self.device)
            else:
                pred_t = pred.to(self.device).float()

            s = self.loss_fn(ref_t, pred_t).item()
            scores.append(s)
            
        return {"lpips_score": sum(scores) / len(scores)}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", type=str, required=True, help="Folder of reference images")
    ap.add_argument("--pred_dir", type=str, required=True, help="Folder of predicted images (same filenames)")
    ap.add_argument("--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pairs = list_pairs(args.ref_dir, args.pred_dir)
    if not pairs:
        raise ValueError("No matching filenames between ref_dir and pred_dir.")

    scorer = LPIPSScorer(net=args.net, device=args.device)
    vals = []
    
    for ref_path, pred_path in pairs:
        ref = load_img(ref_path, args.device)
        pred = load_img(pred_path, args.device)
        v = scorer.loss_fn(ref, pred)
        vals.append(float(v.item()))

    mean_lpips = sum(vals) / len(vals)
    print(f"LPIPS ({args.net}) over {len(vals)} pairs = {mean_lpips:.6f}")

if __name__ == "__main__":
    main()