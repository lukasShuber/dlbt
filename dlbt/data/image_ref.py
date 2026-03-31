"""
ImageRef: a lightweight, immutable pointer to a rendered stimulus image.

Each ImageRef carries the image's UID, its absolute path, and its
pre-computed latent state index (an integer in [0, K-1]).

The latent state is determined deterministically from the image's
generative parameters via the five binary splits defined in constants.py:
  - front/back    (y-position threshold)
  - left/right    (x-position threshold)
  - transparency  (threshold)
  - glossiness    (threshold)
  - small/large   (scale threshold)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dlbt.constants import (
    K,
    DIM_FRONT_BACK, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    Y_THRESHOLD, X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)


@dataclass(frozen=True)
class ImageRef:
    """Immutable pointer to a single rendered stimulus."""
    uid: str          # 6-digit string ID, e.g. "000048"
    path: Path        # absolute path to the PNG file
    latent_state: int # index in [0, K-1]

    def __post_init__(self):
        if not (0 <= self.latent_state < K):
            raise ValueError(f"latent_state {self.latent_state} out of range [0, {K})")


# ---------------------------------------------------------------------------
# Latent-state computation
# ---------------------------------------------------------------------------

def _latent_state_from_z(z: dict) -> int:
    """
    Map a latent parameter dict (as stored in metadata.jsonl) to a state index.

    Bit layout:
      bit 4 (front_back):  1 if y >= Y_THRESHOLD, else 0
      bit 3 (left_right):  1 if x >= X_THRESHOLD, else 0
      bit 2 (transp):      1 if transparency >= TRANSP_THRESH, else 0
      bit 1 (gloss):       1 if glossiness >= GLOSS_THRESH, else 0
      bit 0 (small_large): 1 if scale >= SCALE_THRESH, else 0
    """
    front_back  = int(z["pos_xy"][1] >= Y_THRESHOLD)
    left_right  = int(z["pos_xy"][0] >= X_THRESHOLD)
    transp      = int(z["transparency"] >= TRANSP_THRESH)
    gloss       = int(z["glossiness"]   >= GLOSS_THRESH)
    small_large = int(z["scale"]        >= SCALE_THRESH)

    return (front_back  << DIM_FRONT_BACK
            | left_right  << DIM_LEFT_RIGHT
            | transp      << DIM_TRANSP
            | gloss       << DIM_GLOSS
            | small_large << DIM_SMALL_LARGE)


# ---------------------------------------------------------------------------
# Loading from metadata.jsonl
# ---------------------------------------------------------------------------

def load_image_refs(
    metadata_path: str | Path,
    images_dir: str | Path | None = None,
) -> Dict[str, ImageRef]:
    """
    Load all ImageRefs from a metadata.jsonl file produced by the Blender renderer.

    Args:
        metadata_path: path to metadata.jsonl
        images_dir: directory containing the PNG files. If None, inferred as
                    <metadata_path parent>/images/.

    Returns:
        dict mapping uid -> ImageRef, in file order.
    """
    metadata_path = Path(metadata_path)
    if images_dir is None:
        images_dir = metadata_path.parent / "images"
    images_dir = Path(images_dir)

    refs: Dict[str, ImageRef] = {}
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = record["id"]
            image_file = record["image_file"]  # e.g. "images/000048_shcub_...png"
            fname = Path(image_file).name
            path = images_dir / fname
            latent_state = _latent_state_from_z(record["z"])
            refs[uid] = ImageRef(uid=uid, path=path, latent_state=latent_state)

    return refs


def image_refs_as_list(refs: Dict[str, ImageRef]) -> List[ImageRef]:
    """Return image refs sorted by UID."""
    return [refs[uid] for uid in sorted(refs)]


def balanced_refs(
    task,
    image_refs: List[ImageRef],
    rng=None,
) -> List[ImageRef]:
    """
    Return a balanced subset of image_refs for a given task.

    Splits images into two response classes (action 0 vs action 1) according
    to task.delta_u, then subsamples the larger class to match the smaller.
    This implements the stimulus balancing described in the paper (Section X):
    P_t(X) is chosen so each response class is represented equally.

    Args:
        task:        Task whose delta_u defines the two response classes.
        image_refs:  full list of ImageRef objects.
        rng:         optional numpy.random.Generator for reproducible sampling.
                     If None, uses a fixed seed (0).

    Returns:
        Balanced list of ImageRefs (shuffled).
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(0)

    class0 = [r for r in image_refs if task.optimal_action(r.latent_state) == 0]
    class1 = [r for r in image_refs if task.optimal_action(r.latent_state) == 1]

    n = min(len(class0), len(class1))
    if n == 0:
        raise ValueError(f"Task '{task.name}' has no images in one response class.")

    idx0 = rng.choice(len(class0), size=n, replace=False)
    idx1 = rng.choice(len(class1), size=n, replace=False)

    balanced = [class0[i] for i in idx0] + [class1[i] for i in idx1]
    rng.shuffle(balanced)
    return balanced
