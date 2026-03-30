# make_grid_composite.py
# Usage:
#   python polyhedra/make_grid_composite.py \
#       --img_dir polyhedra/imgs/images \
#       --out polyhedra/imgs/grid_5x5.png \
#       --grid_n 5 \
#       --labels \
#       --row_prefix gloss \
#       --col_prefix trans
#
# Expects filenames ending with: _grid_rXX_tYY.png

import argparse
import os
import re
from PIL import Image, ImageDraw, ImageFont


def parse_rgb(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("RGB color must be like '255,255,255'")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid_n", type=int, default=5)

    # Updated pattern: matches filenames ending in _grid_rXX_tYY.png
    ap.add_argument("--pattern", default=r"_grid_r(\d+)_t(\d+)\.png$")

    ap.add_argument("--padding", type=int, default=8, help="Pixels between cells")
    ap.add_argument("--margin", type=int, default=40, help="Outer margin")
    ap.add_argument("--bg", default="255,255,255", help="Background RGB, e.g. 255,255,255")
    ap.add_argument("--border", type=int, default=0, help="Cell border width in pixels")
    ap.add_argument("--border_color", default="180,180,180", help="Cell border RGB")

    ap.add_argument("--labels", action="store_true", help="Draw row/column labels")
    ap.add_argument("--row_prefix", default="r", help="Row label prefix")
    ap.add_argument("--col_prefix", default="t", help="Column label prefix")
    ap.add_argument("--label_space_left", type=int, default=70, help="Extra left space for row labels")
    ap.add_argument("--label_space_top", type=int, default=45, help="Extra top space for column labels")
    ap.add_argument("--font_size", type=int, default=18)

    args = ap.parse_args()

    bg = parse_rgb(args.bg)
    border_color = parse_rgb(args.border_color)
    rx = re.compile(args.pattern)

    cells = {}
    for fn in os.listdir(args.img_dir):
        if not fn.lower().endswith(".png"):
            continue
        m = rx.search(fn)
        if not m:
            continue
        r = int(m.group(1))
        t = int(m.group(2))
        cells[(r, t)] = os.path.join(args.img_dir, fn)

    n = args.grid_n
    missing = [(r, t) for r in range(n) for t in range(n) if (r, t) not in cells]
    if missing:
        raise SystemExit(f"Missing {len(missing)} grid cells, e.g. {missing[:10]}")

    im0 = Image.open(cells[(0, 0)]).convert("RGB")
    w, h = im0.size

    left_extra = args.label_space_left if args.labels else 0
    top_extra = args.label_space_top if args.labels else 0

    total_w = left_extra + 2 * args.margin + n * w + (n - 1) * args.padding
    total_h = top_extra + 2 * args.margin + n * h + (n - 1) * args.padding

    canvas = Image.new("RGB", (total_w, total_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", args.font_size)
    except Exception:
        font = ImageFont.load_default()

    # rows = gloss/roughness-like dimension, cols = transparency
    for r in range(n):
        for t in range(n):
            im = Image.open(cells[(r, t)]).convert("RGB")
            if im.size != (w, h):
                im = im.resize((w, h))

            x = left_extra + args.margin + t * (w + args.padding)
            y = top_extra + args.margin + r * (h + args.padding)

            canvas.paste(im, (x, y))

            if args.border > 0:
                for k in range(args.border):
                    draw.rectangle(
                        [x - k, y - k, x + w - 1 + k, y + h - 1 + k],
                        outline=border_color,
                    )

    if args.labels:
        # column labels
        for t in range(n):
            label = f"{args.col_prefix}{t}"
            x = left_extra + args.margin + t * (w + args.padding) + w // 2
            y = args.margin + top_extra // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), label, fill=(0, 0, 0), font=font)

        # row labels
        for r in range(n):
            label = f"{args.row_prefix}{r}"
            x = args.margin + left_extra // 2
            y = top_extra + args.margin + r * (h + args.padding) + h // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), label, fill=(0, 0, 0), font=font)

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved composite to {out_path}")


if __name__ == "__main__":
    main()