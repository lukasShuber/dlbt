[![CI](https://github.com/lukasShuber/complete_prcept/actions/workflows/ci.yml/badge.svg)](https://github.com/lukasShuber/complete_prcept/actions/workflows/ci.yml)

## Installation instructions

* `cd` to this directory.
* Run `pip install -e .`

---

## Stimulus filename format

Each rendered image is named according to all latent parameters used to generate it:

```
000999_shoct_f05_yaw140_s046_lab042--027-+034_t031_gl035_xy-0164-+0058_random.png
```

| Field | Example | Description | Range | Task threshold |
|---|---|---|---|---|
| `000999` | `000999` | Zero-padded image UID | 0 … N | — |
| `sh` | `oct` | Shape — `tet` tetrahedron, `cub` cube, `oct` octahedron, `dod` dodecahedron, `ico` icosahedron | 5 shapes | — |
| `f` | `05` | Face index — which face rests on the floor (0-indexed) | tet 0–3, cub 0–5, oct 0–7, dod 0–11, ico 0–19 | — |
| `yaw` | `140` | Yaw rotation around the vertical axis (degrees, integer-rounded) | 0–360 | — |
| `s` | `046` | Scale × 100 (integer-rounded) | 38–88 (config.json) | `s` ≥ 63 → **large** |
| `lab` | `042--027-+034` | CIELab colour: L, a, b (integer-rounded, signed). Negative `a` produces a double-dash (e.g. `a=−27` → `--027`) | sampled or fixed per config | — |
| `t` | `031` | Transparency × 100 (integer-rounded) | 0–100 | `t` ≥ 50 → **transparent** |
| `gl` | `035` | Glossiness × 100 (integer-rounded) | 0–100 | `gl` ≥ 50 → **glossy** |
| `xy` | `-0164-+0058` | Camera-relative lateral × 100 then depth × 100 (both signed). `xy[0]` is the left/right axis. | lateral −250–+250, depth −250–+250 | `xy[0]` ≥ 0 → **right** |
| suffix | `random` | Sampling mode: `random` (randomly sampled) or `grid` (grid sweep) | — | — |