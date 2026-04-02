"""
select_prototype_imgs.py
------------------------
Select one *prototypical* image per latent state (16 states, 4 binary dims).

"Prototypical" = furthest from ALL four decision boundaries simultaneously.

For each image the signed margin on dimension d is:
    margin_d = (continuous_value - threshold_d) * sign_d

where sign_d = +1 if the state requires the positive side, -1 otherwise.
The margins are normalised by the full observed range of each dimension so
they are comparable.  The prototypicality score is the *minimum* normalised
margin across all four dimensions (bottleneck metric).  The image with the
highest score is the most unambiguously representative of its state.

Latent state encoding (matches dlbt/constants.py):
  bit 3 (DIM_LEFT_RIGHT):  1 = right  (pos_xy[0] >= 0.0)
  bit 2 (DIM_TRANSP):      1 = transp (transparency >= 0.5)
  bit 1 (DIM_GLOSS):       1 = glossy (glossiness   >= 0.5)
  bit 0 (DIM_SMALL_LARGE): 1 = large  (scale        >= 0.63)

Outputs (in stimuli/imgs/prototype_imgs/):
  - one copy of each selected image
  - prototype_imgs_metadata.jsonl
  - prototype_imgs.csv  (filename, verbal_description, latent_state,
                         prototypicality_score)

Usage:
    cd <repo root>
    python stimuli/select_prototype_imgs.py
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA  = Path("stimuli/imgs/metadata.jsonl")
IMG_ROOT  = Path("stimuli/imgs")
OUT_DIR   = Path("stimuli/imgs/prototype_imgs")

# Thresholds (must match dlbt/constants.py)
X_THRESHOLD   = 0.0
TRANSP_THRESH = 0.5
GLOSS_THRESH  = 0.5
SCALE_THRESH  = 0.63

# Full observed ranges for normalisation (from config.json / README)
X_RANGE     = (-2.5, 2.5)
TRANSP_RANGE = (0.0,  1.0)
GLOSS_RANGE  = (0.0,  1.0)
SCALE_RANGE  = (0.38, 0.88)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_margin(val: float, threshold: float, val_range: tuple, positive: bool) -> float:
    """Signed, range-normalised distance from threshold (positive = correct side)."""
    span   = val_range[1] - val_range[0]
    margin = (val - threshold) / span        # normalised signed distance
    return margin if positive else -margin


def latent_state(z: dict) -> int:
    lr = int(z["pos_xy"][0]    >= X_THRESHOLD)
    tr = int(z["transparency"] >= TRANSP_THRESH)
    gl = int(z["glossiness"]   >= GLOSS_THRESH)
    sl = int(z["scale"]        >= SCALE_THRESH)
    return (lr << 3) | (tr << 2) | (gl << 1) | sl


def prototypicality(z: dict, state: int) -> float:
    """Bottleneck normalised margin: min over the four dimensions."""
    lr_bit = (state >> 3) & 1
    tr_bit = (state >> 2) & 1
    gl_bit = (state >> 1) & 1
    sl_bit = (state >> 0) & 1

    margins = [
        _norm_margin(z["pos_xy"][0],    X_THRESHOLD,   X_RANGE,     bool(lr_bit)),
        _norm_margin(z["transparency"], TRANSP_THRESH, TRANSP_RANGE, bool(tr_bit)),
        _norm_margin(z["glossiness"],   GLOSS_THRESH,  GLOSS_RANGE,  bool(gl_bit)),
        _norm_margin(z["scale"],        SCALE_THRESH,  SCALE_RANGE,  bool(sl_bit)),
    ]
    return min(margins)


def verbal_description(state: int) -> str:
    lr = "right"       if (state >> 3) & 1 else "left"
    tr = "transparent" if (state >> 2) & 1 else "opaque"
    gl = "glossy"      if (state >> 1) & 1 else "matte"
    sl = "large"       if (state >> 0) & 1 else "small"
    return f"{lr}_{tr}_{gl}_{sl}"


# ---------------------------------------------------------------------------
# Load metadata, bucket by latent state, attach prototypicality score
# ---------------------------------------------------------------------------
buckets: dict[int, list] = defaultdict(list)

with open(METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("tag") != "random":      # skip grid images
            continue
        state = latent_state(rec["z"])
        score = prototypicality(rec["z"], state)
        rec["_state"] = state
        rec["_score"] = score
        buckets[state].append(rec)

print("Images per state:")
for s in range(16):
    print(f"  state {s:02d} ({verbal_description(s):30s}): {len(buckets[s])} candidates")

# ---------------------------------------------------------------------------
# Select the most prototypical image per state
# ---------------------------------------------------------------------------
selected: list[dict] = []
for state in range(16):
    pool = buckets[state]
    if not pool:
        raise RuntimeError(f"No images for state {state} ({verbal_description(state)})")
    best = max(pool, key=lambda r: r["_score"])
    best["_verbal_description"] = verbal_description(state)
    selected.append(best)

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_rows = []
with open(OUT_DIR / "prototype_imgs_metadata.jsonl", "w") as meta_f:
    for rec in selected:
        src = IMG_ROOT / rec["image_file"]
        dst = OUT_DIR  / src.name
        shutil.copy2(src, dst)
        meta_f.write(json.dumps(rec) + "\n")
        csv_rows.append({
            "filename":             src.name,
            "verbal_description":   rec["_verbal_description"],
            "latent_state":         rec["_state"],
            "prototypicality_score": round(rec["_score"], 4),
        })
        print(
            f"  [{rec['_state']:02d}] {rec['_verbal_description']:30s}"
            f"  score={rec['_score']:.3f}  →  {src.name}"
        )

with open(OUT_DIR / "prototype_imgs.csv", "w", newline="") as csv_f:
    writer = csv.DictWriter(
        csv_f,
        fieldnames=["filename", "verbal_description", "latent_state", "prototypicality_score"],
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nSaved {len(selected)} prototype images to {OUT_DIR}/")
print("  prototype_imgs_metadata.jsonl")
print("  prototype_imgs.csv")
