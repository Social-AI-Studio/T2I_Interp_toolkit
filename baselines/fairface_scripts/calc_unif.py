"""
- Detect the column that identifies the face crop or original image
- Parse race_scores_fair_4 (order: [White, Black, Asian, Indian])
- Map to four buckets: white, black, asian, neither
- Collapse multiple faces from the same original image using highest confidence
- Save image_race_labels.csv
- Print counts and proportions across all detected faces
"""

import sys
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import re

# -------- config --------
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "test_outputs.csv"
OUTPUT_PER_IMAGE = "image_race_labels.csv"

# Expected order from FairFace for race_4
RACE4_ORDER = ["White", "Black", "Asian", "Indian"]

# Candidate columns that may contain the image reference
# Added face_name_align since FairFace writes that by default
CANDIDATE_KEYS = [
    "img_path", "file", "image", "img", "path", "filename", "name", "src", "face_name_align"
]

def parse_scores_cell(cell):
    """Convert score cell to a numpy array of length 4."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    s = str(cell).strip()
    # Try a safe literal eval first
    try:
        v = ast.literal_eval(s)
        return np.asarray(v, dtype=float)
    except Exception:
        pass
    # Fallback: strip brackets and split on comma or whitespace
    s = s.replace("array(", "").replace(")", "")
    s = s.strip().lstrip("[").rstrip("]")
    parts = [p for p in s.replace(",", " ").split() if p]
    return np.asarray([float(p) for p in parts], dtype=float)

def to_bucket(label):
    if label == "White":
        return "white"
    if label == "Black":
        return "black"
    if label == "Asian":
        return "asian"
    return "neither"

def original_from_facecrop(path_str: str) -> str:
    """
    Convert a face crop path like:
      detected_faces/some_image_face0.jpg
    back to an original image stub:
      some_image
    This helps collapse multiple faces per original image.
    """
    base = Path(str(path_str)).name  # some_image_face0.jpg
    stem = Path(base).stem           # some_image_face0
    # Remove the trailing _faceN part if present
    return re.sub(r"_face\d+$", "", stem)

def main():
    # Load FairFace output
    if not Path(INPUT_CSV).exists():
        sys.exit(f"Cannot find {INPUT_CSV}. Run predict.py first.")

    df = pd.read_csv(INPUT_CSV)

    # Pick key column that points to the crop or image
    key_col = next((c for c in CANDIDATE_KEYS if c in df.columns), None)
    if key_col is None:
        sys.exit(f"Could not find an image path column. Columns were: {list(df.columns)}")

    # Find the column with 4-race scores
    score_col_candidates = ["race_scores_fair_4", "race4_scores", "scores_fair_4"]
    score_col = next((c for c in score_col_candidates if c in df.columns), None)
    if score_col is None:
        sys.exit(
            "Could not find a 4-race score column. "
            f"Looked for {score_col_candidates}. Columns were: {list(df.columns)}"
        )

    # Parse scores, derive predictions and confidence
    scores = df[score_col].apply(parse_scores_cell)
    bad_len = scores.map(lambda a: len(a) != 4)
    if bad_len.any():
        bad_count = int(bad_len.sum())
        sys.exit(
            f"Found {bad_count} rows where {score_col} does not have 4 values. "
            "Make sure you ran the 4-class head."
        )

    df["race4_idx"] = scores.apply(np.argmax)
    df["race4_conf"] = scores.apply(lambda a: float(a.max()))
    df["race4_pred"] = df["race4_idx"].apply(lambda i: RACE4_ORDER[i])
    df["race_bucket"] = df["race4_pred"].apply(to_bucket)

    # Create a column that groups rows by original image, not by face crop
    df["orig_image"] = df[key_col].apply(original_from_facecrop)

    # One label per original image using the highest confidence face
    best = (
        df.sort_values("race4_conf", ascending=False)
          .drop_duplicates(subset=["orig_image"])
          .copy()
    )

    # Save per-image labels
    cols_to_save = ["orig_image", "race_bucket", "race4_pred", "race4_conf"]
    best[cols_to_save].to_csv(OUTPUT_PER_IMAGE, index=False)

    # Counts and proportions across all detected faces
    counts = (
        df["race_bucket"]
        .value_counts()
        .reindex(["white", "black", "asian", "neither"], fill_value=0)
    )
    total = counts.sum()
    props = (counts / total).round(6) if total > 0 else counts.astype(float)

    print(f"Detected faces: {total}")
    print("\nCounts per bucket:")
    print(counts.to_string())
    print("\nProportions per bucket:")
    print(props.to_string())
    print(f"\nWrote per-image labels to {OUTPUT_PER_IMAGE}")
    print(f"Image key column used: {key_col}")
    print(f"Score column used: {score_col}")

if __name__ == "__main__":
    main()
