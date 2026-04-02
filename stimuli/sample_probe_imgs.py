"""
sample_probe_imgs.py
-------------------
Sample one image per latent state (16 states, 4 binary dimensions).

Latent state encoding (from dlbt/constants.py):
  bit 3 (DIM_LEFT_RIGHT):  1 = right  (pos_xy[0] >= 0.0)
  bit 2 (DIM_TRANSP):      1 = transp (transparency >= 0.5)
  bit 1 (DIM_GLOSS):       1 = glossy (glossiness   >= 0.5)
  bit 0 (DIM_SMALL_LARGE): 1 = large  (scale        >= 0.63)

Outputs (in stimuli/probe_imgs/):
  - one symlink / copy of each selected image
  - probe_imgs_metadata.jsonl   (full metadata for the 16 images)
  - probe_imgs.csv              (filename, verbal_description)

Usage:
    cd <repo root>
    python stimuli/sample_probe_imgs.py
"""

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED          = 14
METADATA      = Path("stimuli/imgs/metadata.jsonl")
IMG_ROOT      = Path("stimuli/imgs")          # images live at IMG_ROOT / rec["image_file"]
OUT_DIR       = Path("stimuli/imgs/probe_imgs")

# Thresholds (must match dlbt/constants.py)
X_THRESHOLD   = 0.0
TRANSP_THRESH = 0.5
GLOSS_THRESH  = 0.5
SCALE_THRESH  = 0.63

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def latent_state(z: dict) -> int:
    lr    = int(z["pos_xy"][0]    >= X_THRESHOLD)
    tr    = int(z["transparency"] >= TRANSP_THRESH)
    gl    = int(z["glossiness"]   >= GLOSS_THRESH)
    sl    = int(z["scale"]        >= SCALE_THRESH)
    return (lr << 3) | (tr << 2) | (gl << 1) | sl


def verbal_description(state: int) -> str:
    lr = "right"       if (state >> 3) & 1 else "left"
    tr = "transparent" if (state >> 2) & 1 else "opaque"
    gl = "glossy"      if (state >> 1) & 1 else "matte"
    sl = "large"       if (state >> 0) & 1 else "small"
    return f"{lr}_{tr}_{gl}_{sl}"


# ---------------------------------------------------------------------------
# Load metadata and bucket by latent state
# ---------------------------------------------------------------------------
buckets: dict[int, list] = defaultdict(list)

with open(METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec   = json.loads(line)
        if rec.get("tag") != "random":      # skip grid images
            continue
        state = latent_state(rec["z"])
        buckets[state].append(rec)

print(f"Loaded images per state:")
for s in range(16):
    print(f"  state {s:02d} ({verbal_description(s):30s}): {len(buckets[s])} images")

# ---------------------------------------------------------------------------
# Sample one image per state
# ---------------------------------------------------------------------------
rng = random.Random(SEED)

selected: list[dict] = []
for state in range(16):
    pool = buckets[state]
    if not pool:
        raise RuntimeError(f"No images found for state {state} ({verbal_description(state)})")
    rec = rng.choice(pool)
    rec["_latent_state"]       = state
    rec["_verbal_description"] = verbal_description(state)
    selected.append(rec)

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_rows = []
with open(OUT_DIR / "probe_imgs_metadata.jsonl", "w") as meta_f:
    for rec in selected:
        src = IMG_ROOT / rec["image_file"]
        dst = OUT_DIR  / src.name
        shutil.copy2(src, dst)
        meta_f.write(json.dumps(rec) + "\n")
        csv_rows.append({
            "filename":           src.name,
            "verbal_description": rec["_verbal_description"],
            "latent_state":       rec["_latent_state"],
        })
        print(f"  [{rec['_latent_state']:02d}] {rec['_verbal_description']:30s}  →  {src.name}")

with open(OUT_DIR / "probe_imgs.csv", "w", newline="") as csv_f:
    writer = csv.DictWriter(csv_f, fieldnames=["filename", "verbal_description", "latent_state"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nSaved {len(selected)} probe_imgs to {OUT_DIR}/")
print(f"  probe_imgs_metadata.jsonl")
print(f"  probe_imgs.csv")
