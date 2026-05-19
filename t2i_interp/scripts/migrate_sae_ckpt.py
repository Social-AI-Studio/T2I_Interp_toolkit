"""migrate_sae_ckpt — convert legacy sdxl-unbox SAE checkpoints to the format
expected by ``dictionary_learning.trainers.top_k.AutoEncoderTopK``.

The legacy checkpoints (from https://github.com/surkovv/sdxl-unbox) store
state dicts as ``{"state_dict": {...}}`` with a ``pre_bias`` key. ``AutoEncoderTopK``
expects flat state dicts with ``b_dec`` instead of ``pre_bias`` and extra fields
``encoder.bias`` / ``k`` / ``threshold``.

Originally a 4× copy-paste block at the top of ``notebooks/sae.ipynb``;
extracted here so the notebook stays focused on analysis. Idempotent — safe
to re-run.

Usage:
    python -m t2i_interp.scripts.migrate_sae_ckpt \\
        --checkpoint-dir ./sdxl-unbox/checkpoints \\
        --hidden-dim 5120 --k 10

    # or via CLI script
    t2i-migrate-sae --checkpoint-dir ./sdxl-unbox/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

LEGACY_KEYS = {"encoder.weight", "pre_bias", "decoder.weight"}


def migrate_one(ckpt_path: Path, hidden_dim: int, k: int) -> bool:
    """Rewrite a single checkpoint file in-place. Returns True if modified.

    Note: ``torch.load`` unpickles arbitrary code from the checkpoint, so
    only run this on checkpoints from sources you trust (e.g. the sdxl-unbox
    repo). We use ``weights_only=True`` where possible (PyTorch ≥2.0) to
    restrict unpickling to tensors + a safelist of containers.
    """
    try:
        sd_full = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # Older torch (no `weights_only` kwarg) or legacy checkpoints that
        # need full pickle support. Fall back loudly — callers must trust
        # the checkpoint source.
        print(
            f"  [WARN] {ckpt_path}: weights_only=True load failed, falling back "
            "to full pickle. Only use this on trusted checkpoints."
        )
        sd_full = torch.load(ckpt_path, map_location="cpu")
    sd = sd_full.get("state_dict", sd_full)

    # Already migrated? (flat dict, has encoder.bias / k / threshold)
    if "encoder.bias" in sd and "k" in sd and "pre_bias" not in sd:
        return False

    sd = {k_.replace("pre_bias", "b_dec"): v for k_, v in sd.items() if k_ in LEGACY_KEYS}
    # Derive bias length from the encoder weight (dict_size = weight.shape[0])
    # rather than the --hidden-dim CLI arg. AutoEncoderTopK's load_state_dict
    # checks bias shape against the weight, so a stale --hidden-dim would
    # silently produce a checkpoint that fails to load.
    enc_weight = sd.get("encoder.weight")
    if enc_weight is None:
        raise RuntimeError(f"{ckpt_path}: no 'encoder.weight' found in state dict")
    dict_size = enc_weight.shape[0]
    if dict_size != hidden_dim:
        print(
            f"  [info] {ckpt_path}: hidden_dim arg ({hidden_dim}) differs from "
            f"encoder.weight.shape[0] ({dict_size}); using actual weight shape."
        )
    sd.update(
        {
            "encoder.bias": torch.zeros((dict_size,)),
            "k": torch.tensor(k),
            "threshold": torch.tensor(-1.0),
        }
    )
    torch.save(sd, ckpt_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./sdxl-unbox/checkpoints"),
        help="Directory containing legacy SAE checkpoint subfolders.",
    )
    parser.add_argument(
        "--pattern",
        default="**/state_dict.pth",
        help="Glob pattern for checkpoints to migrate (default: **/state_dict.pth).",
    )
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    ckpts = sorted(args.checkpoint_dir.glob(args.pattern))
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {args.checkpoint_dir}/{args.pattern}")

    print(f"Found {len(ckpts)} checkpoint(s) under {args.checkpoint_dir}:")
    n_migrated = 0
    for p in ckpts:
        changed = migrate_one(p, hidden_dim=args.hidden_dim, k=args.k)
        status = "migrated" if changed else "already-migrated (skipped)"
        print(f"  [{status}] {p}")
        if changed:
            n_migrated += 1
    print(f"\nDone. {n_migrated}/{len(ckpts)} checkpoints rewritten.")


if __name__ == "__main__":
    main()
