from typing import List, Optional, Union
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils.output import Output   

def visualize_outputs(
    outputs: Union[Output, List[Output]],
    metric_name: str = "clip_score",
    labels: Optional[List[str]] = None,
    cols: int = 4,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    dpi: int = 100,
):
    """
    Visualize one or more Output objects where:
      - out.preds[0] is a PIL image
      - out.metrics[metric_name] is a float or [float]

    Args:
        outputs: Output or list[Output].
        metric_name: key in out.metrics to show.
        labels: optional text label per Output (same length as outputs).
        cols: number of columns in the matplotlib grid.
        show: whether to show a grid in the current notebook.
        save_dir: if not None, saves individual images with metric text overlaid.
        dpi: DPI for the matplotlib figure.
    """

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    n = len(outputs)
    if labels is not None:
        assert len(labels) == n, "labels length must match number of outputs"

    def get_item(i):
        out = outputs[i]
        assert len(out.preds) >= 1, f"Output[{i}] has no preds."
        img = out.preds[0]
        assert isinstance(img, Image.Image), f"Output[{i}].preds[0] must be a PIL image."

        score = None
        if out.metrics is not None and metric_name in out.metrics:
            val = out.metrics[metric_name]
            if isinstance(val, (list, tuple)):
                assert len(val) >= 1, f"metrics['{metric_name}'] is empty for index {i}"
                score = float(val[0])
            else:
                score = float(val)

        label = labels[i] if labels is not None else None
        return img, score, label

    # ---- 1) Show grid in notebook ----
    if show:
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=dpi)

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            if idx < n:
                img, score, label = get_item(idx)
                ax.imshow(img)
                ax.axis("off")

                title_parts = []
                if label is not None:
                    title_parts.append(str(label))
                if score is not None:
                    title_parts.append(f"{metric_name}: {score:.3f}")
                if title_parts:
                    ax.set_title(" | ".join(title_parts), fontsize=7)  # smaller font
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ---- 2) Save individual images with metric text at the top ----
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Try to get a small-ish TrueType font if available; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", size=12)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=12)
            except Exception:
                font = ImageFont.load_default()

        for i in range(n):
            img, score, label = get_item(i)
            img = img.convert("RGB")

            txt_parts = []
            if label is not None:
                txt_parts.append(str(label))
            if score is not None:
                txt_parts.append(f"{metric_name}: {score:.3f}")
            title_text = " | ".join(txt_parts) if txt_parts else ""

            if title_text:
                pad = 32  # slightly smaller padding is enough for smaller text
                canvas = Image.new("RGB", (img.width, img.height + pad), "white")
                canvas.paste(img, (0, pad))
                draw = ImageDraw.Draw(canvas)
                draw.text((5, 8), title_text, fill="black", font=font)
            else:
                canvas = img

            safe_label = (label or f"img_{i}").replace(" ", "_")
            if score is not None:
                fname = f"{i:03d}_{safe_label}_{metric_name}_{score:.3f}.png"
            else:
                fname = f"{i:03d}_{safe_label}.png"

            canvas.save(save_dir / fname)
