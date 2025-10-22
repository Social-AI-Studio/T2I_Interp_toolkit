import csv
import pathlib

folder = pathlib.Path("debias_race_14_v2_50imgs")
csv_path = "debias_race_14_v2_50imgs.csv"

# Extensions to include
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder names to skip explicitly
SKIP_DIRS = {".ipynb_checkpoints", ".ipynb", "checkpts", "checkpoints"}

def should_skip(path: pathlib.Path, root: pathlib.Path) -> bool:
    # Skip if any part is a hidden component (starts with ".")
    try:
        parts = path.relative_to(root).parts
    except Exception:
        parts = path.parts
    if any(part.startswith(".") for part in parts):
        return True
    # Skip if any part matches known checkpoint dirs
    if any(part in SKIP_DIRS for part in parts):
        return True
    return False

paths = []
for p in folder.rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS and not should_skip(p, folder):
        paths.append(str(p.resolve()))

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["img_path"])
    w.writerows([[p] for p in sorted(paths)])

print(f"Wrote {len(paths)} rows to {csv_path}")
