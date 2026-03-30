import os
import random
from PIL import Image


def make_random_matrix(
    img_dir,
    out_path,
    rows=5,
    cols=10,
    seed=0,
    padding=8,
    bg_color=(255, 255, 255),
):
    rng = random.Random(seed)
    print(f"Looking for images in {img_dir}...")
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".png") and "_random" in f
    ]

    if len(files) < rows * cols:
        raise ValueError(f"Need at least {rows*cols} images, found {len(files)}")

    chosen = rng.sample(files, rows * cols)

    im0 = Image.open(os.path.join(img_dir, chosen[0])).convert("RGB")
    w, h = im0.size

    canvas_w = cols * w + (cols - 1) * padding
    canvas_h = rows * h + (rows - 1) * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    k = 0
    for r in range(rows):
        for c in range(cols):
            im = Image.open(os.path.join(img_dir, chosen[k])).convert("RGB")

            x = c * (w + padding)
            y = r * (h + padding)

            canvas.paste(im, (x, y))
            k += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

    print(f"Saved random matrix → {out_path}")


if __name__ == "__main__":
    make_random_matrix(
        img_dir="polyhedra/imgs/images",
        out_path="polyhedra/imgs/random_5x10.png",
        rows=5,
        cols=10,
        seed=0,
    )