import numpy as np
import abc
import tqdm 
from PIL import Image, ImageDraw, ImageFont
#import open_clip
import pickle 
import os

from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DataLike = Union[pd.DataFrame, Mapping[str, Any]]


def plot_key_wise(
    data: DataLike,
    *,
    key_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    preserve_order: bool = True,
    sort_keys: bool = False,
    scale: str = "global",  # "global" or "per_row"
    cmap: str = "Greens",
    figsize: Optional[Tuple[float, float]] = None,
    max_xticks: int = 12,
    show_xticks: bool = True,
    rotate_xticks: int = 90,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Draws stripe-like heatmap rows (one row per numeric column) across keys.

    - If `data` is a DataFrame: first column (or `key_col`) is the key, other numeric columns are plotted.
    - If `data` is a dict:
        * {key: scalar} -> one row called "value"
        * {key: {col: scalar, ...}} -> multiple rows for each inner key
    """
    df = _to_dataframe(data, key_col=key_col, preserve_order=preserve_order)

    # Determine key column
    if key_col is None:
        key_col = df.columns[0]
    if key_col not in df.columns:
        raise ValueError(f"key_col='{key_col}' not found in columns: {list(df.columns)}")

    # Determine value columns (numeric)
    if value_cols is None:
        numeric_cols = df.drop(columns=[key_col]).select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found to plot.")
        value_cols = numeric_cols
    else:
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"value_cols not found in df: {missing}")

    # Optional sorting
    if sort_keys:
        df = df.sort_values(by=key_col, kind="mergesort")  # stable sort

    keys = df[key_col].astype(str).tolist()
    values = df[list(value_cols)].to_numpy(dtype=float).T  # shape: (R, N)
    R, N = values.shape

    if figsize is None:
        figsize = (12, max(2.0, 0.55 * R + 1.0))

    fig, axes = plt.subplots(
        nrows=R, ncols=1, sharex=True, figsize=figsize,
        gridspec_kw={"hspace": 0.25}
    )
    if R == 1:
        axes = [axes]

    # Handle NaNs nicely
    cm = plt.get_cmap(cmap).copy()
    try:
        cm.set_bad(alpha=0.0)
    except Exception:
        pass

    # Scaling
    if scale not in {"global", "per_row"}:
        raise ValueError("scale must be 'global' or 'per_row'")

    if scale == "global":
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        vmins = [vmin] * R
        vmaxs = [vmax] * R
    else:
        vmins = [np.nanmin(values[i]) for i in range(R)]
        vmaxs = [np.nanmax(values[i]) for i in range(R)]

    im_last = None
    for i, (ax, col) in enumerate(zip(axes, value_cols)):
        row = values[i:i+1, :]  # (1, N)
        im = ax.imshow(
            row,
            aspect="auto",
            interpolation="nearest",
            cmap=cm,
            vmin=vmins[i],
            vmax=vmaxs[i],
        )
        im_last = im

        ax.set_yticks([])
        ax.set_ylabel(str(col), rotation=0, ha="right", va="center", labelpad=25)

        # Clean look like your screenshot
        for spine in ax.spines.values():
            spine.set_visible(False)

    # X ticks (only bottom axis)
    axes[-1].set_xlabel(str(key_col))
    if show_xticks and N > 0:
        step = max(1, N // max_xticks)
        tick_positions = list(range(0, N, step))
        axes[-1].set_xticks(tick_positions)
        axes[-1].set_xticklabels([keys[j] for j in tick_positions], rotation=rotate_xticks, ha="right")
    else:
        axes[-1].set_xticks([])

    if suptitle:
        fig.suptitle(suptitle, y=0.98)

    if colorbar and im_last is not None:
        cb = fig.colorbar(im_last, ax=axes, fraction=0.03, pad=0.02)
        cb.set_label(colorbar_label or "Higher score →")

    fig.tight_layout()
    return fig, axes


def _to_dataframe(
    data: DataLike,
    *,
    key_col: Optional[str],
    preserve_order: bool,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        return df

    if not isinstance(data, Mapping):
        raise TypeError("data must be a pandas DataFrame or a dict-like Mapping[str, ...].")

    # dict[str, scalar]  OR  dict[str, dict[str, scalar]]
    keys = list(data.keys()) if preserve_order else sorted(list(data.keys()))
    first_val = next(iter(data.values())) if data else None

    if data and isinstance(first_val, Mapping):
        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.loc[keys]  # preserve order
        df = df.reset_index().rename(columns={"index": key_col or "key"})
    else:
        df = pd.DataFrame({key_col or "key": keys, "value": [data[k] for k in keys]})

    return df

# usage (DataFrame (first col is layer string, rest numeric))
# fig, axes = plot_layer_stripes(df, colorbar_label="Higher CLIP-Score →", suptitle="UNet")
# plt.show()

# usage (Dict {layer: score})
# d = {"unet.up_blocks.0...": 0.38, "unet.down_blocks.2...": 0.41}
# fig, axes = plot_layer_stripes(d, colorbar_label="Higher score →")
# plt.show()

# usage (Dict {layer: {metric1:..., metric2:...}} (multiple stripe rows))
# d = {
#   "layerA": {"clip_to_clean": 0.38, "other_metric": 1.2},
#   "layerB": {"clip_to_clean": 0.41, "other_metric": 0.9},
# }
# fig, axes = plot_layer_stripes(d, scale="global")
# plt.show()

def view_images(
    images,
    num_rows=1,
    offset_ratio=0.02,
    bg=255,
    resize_to_first=False,
    labels=None,                 # list[str] same length as #images (before padding)
    label_color=(0, 0, 0),
    label_bg=(255, 255, 255),
    label_bg_alpha=180,          # 0..255
    label_pad=6,
    label_pos="top-left",        # "top-left", "top-right", "bottom-left", "bottom-right"
    font_size=16,
):
    """
    Returns a single PIL.Image grid with optional per-tile labels.

    Accepts:
      - list/tuple of PIL.Image | torch.Tensor | np.ndarray
      - np.ndarray batch: (B,H,W,C) or (B,C,H,W)
      - torch.Tensor batch: (B,H,W,C) or (B,C,H,W)
      - single image of any of the above
    """

    def to_uint8_hwc3(x):
        if hasattr(x, "detach"):  # torch
            x = x.detach().cpu().numpy()
        if isinstance(x, Image.Image):
            x = np.array(x)
        x = np.asarray(x)

        if x.ndim == 4:
            raise ValueError("Got a 4D tensor/array inside list; pass the batch directly, not nested.")

        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)

        # CHW -> HWC if needed
        if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))

        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if x.shape[-1] == 4:
            x = x[..., :3]

        if x.dtype != np.uint8:
            x = x.astype(np.float32)
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)

        return x

    # --- normalize input into list ---
    if isinstance(images, (list, tuple)):
        items = list(images)
    else:
        if hasattr(images, "detach"):  # torch
            arr = images.detach().cpu().numpy()
            items = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 4 else [arr]
        else:
            if isinstance(images, Image.Image):
                items = [images]
            else:
                arr = np.asarray(images)
                items = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 4 else [images]

    if len(items) == 0:
        raise ValueError("No images provided.")

    n_orig = len(items)
    if labels is not None and len(labels) != n_orig:
        raise ValueError(f"labels must have length {n_orig}, got {len(labels)}")

    num_rows = int(num_rows)
    if num_rows <= 0:
        raise ValueError("num_rows must be >= 1")

    imgs = [to_uint8_hwc3(x) for x in items]

    # size handling
    h0, w0 = imgs[0].shape[:2]
    if resize_to_first:
        imgs2 = []
        for im in imgs:
            if im.shape[:2] != (h0, w0):
                imgs2.append(np.array(Image.fromarray(im).resize((w0, h0), Image.BILINEAR)))
            else:
                imgs2.append(im)
        imgs = imgs2
    else:
        for im in imgs:
            if im.shape[:2] != (h0, w0):
                raise ValueError(f"All images must have same size. First is {(h0,w0)}, got {im.shape[:2]}. "
                                 f"Set resize_to_first=True to auto-resize.")

    # pad empties to fill grid
    rem = n_orig % num_rows
    num_empty = (num_rows - rem) % num_rows
    empty = np.ones((h0, w0, 3), dtype=np.uint8) * bg
    imgs = imgs + [empty] * num_empty
    labels_padded = (list(labels) if labels is not None else None)
    if labels_padded is not None:
        labels_padded += [""] * num_empty

    num_items = len(imgs)
    num_cols = num_items // num_rows

    offset = int(h0 * offset_ratio)
    grid_h = h0 * num_rows + offset * (num_rows - 1)
    grid_w = w0 * num_cols + offset * (num_cols - 1)
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * bg

    # paste tiles
    for idx in range(num_items):
        i = idx // num_cols
        j = idx % num_cols
        y0 = i * (h0 + offset)
        x0 = j * (w0 + offset)
        canvas[y0:y0 + h0, x0:x0 + w0] = imgs[idx]

    out = Image.fromarray(canvas).convert("RGBA")

    # draw labels
    if labels_padded is not None:
        draw = ImageDraw.Draw(out, "RGBA")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        def label_xy(x0, y0, text_w, text_h):
            if label_pos == "top-left":
                return x0 + label_pad, y0 + label_pad
            if label_pos == "top-right":
                return x0 + w0 - label_pad - text_w, y0 + label_pad
            if label_pos == "bottom-left":
                return x0 + label_pad, y0 + h0 - label_pad - text_h
            if label_pos == "bottom-right":
                return x0 + w0 - label_pad - text_w, y0 + h0 - label_pad - text_h
            raise ValueError("label_pos must be one of top-left/top-right/bottom-left/bottom-right")

        for idx, lab in enumerate(labels_padded):
            if not lab:
                continue
            i = idx // num_cols
            j = idx % num_cols
            y0 = i * (h0 + offset)
            x0 = j * (w0 + offset)

            # measure text
            bbox = draw.textbbox((0, 0), lab, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            x, y = label_xy(x0, y0, tw, th)

            # background box (semi-transparent)
            bg_rgba = (*label_bg, int(label_bg_alpha))
            pad = 3
            draw.rounded_rectangle(
                [x - pad, y - pad, x + tw + pad, y + th + pad],
                radius=6,
                fill=bg_rgba,
                outline=None,
            )
            draw.text((x, y), lab, font=font, fill=(*label_color, 255))

    return out.convert("RGB")

def plot_image_heatmap(output, sparse_maps, feature):
    """
    Visualizes a feature activation map overlaid on the generated image.
    
    Args:
        output: Pipeline output object containing .images[0]
        sparse_maps: Feature activations tensor (H, W, D) or similar (on CPU/RAM)
        feature: Index of the feature to visualize
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image
    
    # sparse_maps should be (H, W, D)
    heatmap = sparse_maps[:, :, feature]
    if hasattr(heatmap, 'cpu'):
        heatmap = heatmap.cpu().numpy()
    elif hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
        
    image = output.images[0]
    if image.mode != 'RGBA':
        image = image.convert("RGBA")
    
    jet = plt.cm.jet
    cmap = jet(np.arange(jet.N))
    cmap[:1, -1] = 0
    cmap[1:, -1] = 0.6
    cmap = ListedColormap(cmap)
    
    # Normalize heatmap
    h_min, h_max = np.min(heatmap), np.max(heatmap)
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)
    
    heatmap_rgba = cmap(heatmap)
    heatmap_image = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    
    # Resize heatmap to match image size (Nearest Neighbor to preserve grid structure)
    heatmap_image = heatmap_image.resize(image.size, resample=Image.NEAREST)
    
    heatmap_with_transparency = Image.alpha_composite(image, heatmap_image)

    return heatmap_with_transparency


def make_steer_grid(
    pairs_results: list[dict],
    cell_size: float = 4.0,
    dpi: int = 120,
) -> "Image.Image":
    """Build a steer grid with Baseline and Steered columns.

    Layout (N rows × 2 columns)::

                   Baseline              Steered
                   ─────────────────     ─────────────────
        row 0:     [  base img  ]        [  steer img  ]
                   "apply prompt"        pos → neg
                   ─────────────────     ─────────────────
        row 1:     [  base img  ]        [  steer img  ]
                   "apply prompt"        pos → neg

    - Column header above first row: ``"Baseline"`` / ``"Steered"``
    - Caption below each image via ``set_xlabel``:
      baseline column → apply prompt; steered column → ``pos → neg``

    Args:
        pairs_results: List of dicts, one per contrast pair::

            {
              "pos":      str,
              "neg":      str,
              "apply":    list[str],
              "steered":  list[PIL.Image],
              "baseline": list[PIL.Image],
            }

        cell_size: Inches per image cell.
        dpi: Resolution of the saved PNG.

    Returns:
        A PIL Image of the rendered grid.
    """
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    def _clean_ax(ax):
        """Hide spines and ticks but keep title / xlabel / ylabel."""
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Flatten to one row per (pair × apply-prompt)
    rows: list[dict] = []
    for pair in pairs_results:
        pos      = pair.get("pos", "")
        neg      = pair.get("neg", "")
        apply    = pair.get("apply", [""]) or [""]
        steered  = pair.get("steered",  []) or []
        baseline = pair.get("baseline", []) or []
        for ai, ap in enumerate(apply):
            rows.append({
                "pos":      pos,
                "neg":      neg,
                "apply":    ap,
                "steered":  steered[ai]  if ai < len(steered)  else None,
                "baseline": baseline[ai] if ai < len(baseline) else None,
            })

    has_baseline = any(r["baseline"] is not None for r in rows)
    n_cols = 2 if has_baseline else 1
    n_rows = len(rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size),
        squeeze=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.05},
    )

    for ri, row in enumerate(rows):
        ax_base  = axes[ri][0]
        ax_steer = axes[ri][1] if has_baseline else axes[ri][0]

        # ── column headers on the first row ─────────────────────────────────
        if ri == 0:
            if has_baseline:
                ax_base.set_title("Baseline", fontsize=11, fontweight="bold", pad=8)
            ax_steer.set_title("Steered", fontsize=11, fontweight="bold", pad=8)

        # ── baseline image + caption below ──────────────────────────────────
        if has_baseline:
            if row["baseline"] is not None:
                ax_base.imshow(np.asarray(row["baseline"]))
            else:
                ax_base.set_facecolor("#eeeeee")
            ax_base.set_xlabel(f'"{row["apply"]}"', fontsize=8, labelpad=6)
            _clean_ax(ax_base)

        # ── steered image + caption below ───────────────────────────────────
        if row["steered"] is not None:
            ax_steer.imshow(np.asarray(row["steered"]))
        else:
            ax_steer.set_facecolor("#eeeeee")
        ax_steer.set_xlabel(f'{row["pos"]} → {row["neg"]}', fontsize=8, labelpad=6)
        _clean_ax(ax_steer)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def show_grid(images, labels, cols=3):
    """
    Plots a list of images in a grid with corresponding labels.
    
    Args:
        images: List of PIL Images or numpy arrays.
        labels: List of label strings.
        cols: Number of columns in the grid.
    """
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('off')
        
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()
