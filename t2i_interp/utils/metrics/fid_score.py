# fid_score.py
# pip install clean-fid pillow

import argparse
from cleanfid import fid

import tempfile
import os
from PIL import Image

class FIDScorer:
    def __init__(self, mode: str = "clean"):
        self.mode = mode

    def compute(self, out: 'Output', ref_dir: str = None) -> 'Output':
        from t2i_interp.utils.output import Output
        if not isinstance(out, Output):
            raise TypeError("out must be of type Output")

        preds = out.preds
        if not preds:
            return out
        
        # If ref_dir is not provided, try to find in run_metadata
        if ref_dir is None:
            if hasattr(out, "run_metadata") and out.run_metadata and "ref_dir" in out.run_metadata:
                ref_dir = out.run_metadata["ref_dir"]
            else:
                return out # Can't compute FID without reference directory or baselines
        
        with tempfile.TemporaryDirectory() as tmp_gen:
            for i, img in enumerate(preds):
                if isinstance(img, Image.Image):
                    img.save(os.path.join(tmp_gen, f"{i}.png"))
                elif isinstance(img, str):
                    Image.open(img).save(os.path.join(tmp_gen, f"{i}.png"))
                    
            if isinstance(ref_dir, list) or isinstance(ref_dir, tuple):
                with tempfile.TemporaryDirectory() as tmp_ref:
                    for i, img in enumerate(ref_dir):
                        if isinstance(img, Image.Image):
                            img.save(os.path.join(tmp_ref, f"{i}.png"))
                        elif isinstance(img, str):
                            Image.open(img).save(os.path.join(tmp_ref, f"{i}.png"))
                    score = fid.compute_fid(tmp_ref, tmp_gen, mode=self.mode)
            else:
                score = fid.compute_fid(ref_dir, tmp_gen, mode=self.mode)

        if out.metrics is None:
            out.metrics = []
        
        metric_dict = {"fid_score": float(score)}
        if not out.metrics:
            out.metrics = [metric_dict] * len(preds)
        else:
            for m in out.metrics:
                m["fid_score"] = float(score)

        return out

    def score(self, images, references=None, **kwargs):
        """Standardized interface for run_steer.py"""
        if references is None:
            # FID needs a reference directory or reference images
            return {"fid_score": float('nan')}
            
        preds = images
        ref_dir = references

        from cleanfid import fid
        with tempfile.TemporaryDirectory() as tmp_gen:
            for i, img in enumerate(preds):
                if isinstance(img, Image.Image):
                    img.save(os.path.join(tmp_gen, f"{i}.png"))
                elif isinstance(img, str):
                    Image.open(img).save(os.path.join(tmp_gen, f"{i}.png"))
                    
            if isinstance(ref_dir, list) or isinstance(ref_dir, tuple):
                with tempfile.TemporaryDirectory() as tmp_ref:
                    for i, img in enumerate(ref_dir):
                        if isinstance(img, Image.Image):
                            img.save(os.path.join(tmp_ref, f"{i}.png"))
                        elif isinstance(img, str):
                            Image.open(img).save(os.path.join(tmp_ref, f"{i}.png"))
                    score = fid.compute_fid(tmp_ref, tmp_gen, mode=self.mode)
            else:
                score = fid.compute_fid(ref_dir, tmp_gen, mode=self.mode)

        return {"fid_score": float(score)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True, help="Folder of real images")
    ap.add_argument("--gen_dir", type=str, required=True, help="Folder of generated images")
    ap.add_argument("--mode", type=str, default="clean", choices=["clean", "legacy"], help="FID mode")
    args = ap.parse_args()

    # compute FID between two folders
    score = fid.compute_fid(args.real_dir, args.gen_dir, mode=args.mode)
    print(f"FID ({args.mode}) = {score:.6f}")

if __name__ == "__main__":
    main()