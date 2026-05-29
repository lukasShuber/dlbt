"""
render_canonical_states.py
--------------------------
Render one maximally-prototypical image per latent state (K=16 = 2^4).

Shape and color are held fixed across all 16 images; only the four
binary latent dimensions vary, each set to a clearly extreme value
well away from the decision boundary:

  Dimension       bit  threshold   low value  high value
  ----------      ---  ---------   ---------  ----------
  left / right     3    x=0.0      x=-1.8     x=+1.8
  opaque / transp  2    t=0.5      t=0.10     t=0.90
  matte / glossy   1    g=0.5      g=0.10     g=0.90
  small / large    0    s=0.63     s=0.45     s=0.82

Run from repo root:
    /Applications/Blender.app/Contents/MacOS/Blender -b \
        -P stimuli/render_canonical_states.py -- \
        --config stimuli/config.json \
        [--out-dir stimuli/imgs/canonical_states]
"""

import importlib.util
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Import rendering helpers from the main render script
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rdr", os.path.join(_HERE, "render_dataset_polyhedra.py")
)
rdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdr)


# ---------------------------------------------------------------------------
# Canonical settings — change these to explore different fixed conditions
# ---------------------------------------------------------------------------
FIXED_SHAPE      = "icosahedron"      # shape held constant across all 16 images
FIXED_FACE_INDEX = 3                  # which face rests on the floor
FIXED_YAW_DEG    = 20.0              # yaw (degrees) around vertical axis
FIXED_DEPTH      = 0.5               # depth in camera-relative coords (fixed position)
FIXED_LAB        = [65.0, 38.0, 42.0]  # Lab color — warm orange, vivid, in-gamut

# Canonical extreme values for each binary dimension
LATERAL_NEG  = -1.8   # clearly LEFT   (x threshold = 0.0, range [-2.5, 2.5])
LATERAL_POS  = +1.8   # clearly RIGHT
TRANSP_LOW   =  0.10  # clearly OPAQUE (threshold = 0.5)
TRANSP_HIGH  =  0.90  # clearly TRANSPARENT
GLOSS_LOW    =  0.10  # clearly MATTE  (threshold = 0.5)
GLOSS_HIGH   =  0.90  # clearly GLOSSY
SCALE_SMALL  =  0.45  # clearly SMALL  (threshold = 0.63, range [0.38, 0.88])
SCALE_LARGE  =  0.82  # clearly LARGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args(argv):
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1:]
    out  = {"config": None, "out_dir": None}
    it   = iter(args)
    for a in it:
        if a == "--config":
            out["config"] = next(it)
        elif a == "--out-dir":
            out["out_dir"] = next(it)
    return out


def verbal_description(state: int) -> str:
    lr = "right"       if (state >> 3) & 1 else "left"
    tr = "transparent" if (state >> 2) & 1 else "opaque"
    gl = "glossy"      if (state >> 1) & 1 else "matte"
    sl = "large"       if (state >> 0) & 1 else "small"
    return f"{lr}_{tr}_{gl}_{sl}"


def canonical_latents(state: int, cfg: dict) -> dict:
    """Build a latent dict at prototypically extreme values for *state*."""
    lr_bit = (state >> 3) & 1
    tr_bit = (state >> 2) & 1
    gl_bit = (state >> 1) & 1
    sl_bit = (state >> 0) & 1

    lateral      = LATERAL_POS  if lr_bit else LATERAL_NEG
    transparency = TRANSP_HIGH  if tr_bit else TRANSP_LOW
    glossiness   = GLOSS_HIGH   if gl_bit else GLOSS_LOW
    scale        = SCALE_LARGE  if sl_bit else SCALE_SMALL

    rgb, in_gamut = rdr.lab_to_rgb(*FIXED_LAB)
    rgb = [min(1.0, max(0.0, c)) for c in rgb]

    return {
        "shape_name":   FIXED_SHAPE,
        "face_index":   FIXED_FACE_INDEX,
        "yaw_deg":      FIXED_YAW_DEG,
        "scale":        scale,
        "transparency": transparency,
        "glossiness":   glossiness,
        "lab":          FIXED_LAB,
        "rgb":          rgb,
        # camera-relative coords: [lateral, depth]
        "pos_xy":       [lateral, FIXED_DEPTH],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args        = parse_args(sys.argv)
    config_path = args.get("config")
    if not config_path:
        raise SystemExit("Missing --config path.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Output directory (override via --out-dir or default to imgs/canonical_states)
    out_dir = args.get("out_dir") or os.path.join(
        rdr.resolve_path(config_path, cfg["out_dir"]), "canonical_states"
    )
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "canonical_states_metadata.jsonl")

    # Ensure "obj_lateral_range" is present so render_one uses camera-relative coords
    if "obj_lateral_range" not in cfg:
        cfg["obj_lateral_range"] = [-2.5, 2.5]

    engine = cfg.get("engine", "CYCLES").upper()

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("")

    print(f"Rendering 16 canonical states → {out_dir}")
    print(f"  shape={FIXED_SHAPE}  face={FIXED_FACE_INDEX}  yaw={FIXED_YAW_DEG}°"
          f"  Lab={FIXED_LAB}")
    print()

    for state in range(16):
        desc = verbal_description(state)
        uid  = f"state{state:02d}"
        z    = canonical_latents(state, cfg)

        print(f"  [{state:02d}] {desc}  "
              f"lat={z['pos_xy'][0]:+.1f}  t={z['transparency']:.2f}  "
              f"gl={z['glossiness']:.2f}  s={z['scale']:.2f}")

        rdr.render_one(
            cfg      = cfg,
            engine   = engine,
            img_dir  = out_dir,
            meta_path= meta_path,
            uid      = uid,
            z        = z,
            tag      = desc,
        )

    print(f"\nDone. 16 images in {out_dir}/")
    print("  canonical_states_metadata.jsonl")


if __name__ == "__main__":
    main()
