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

| Field | Example | Description | Range |
|---|---|---|---|
| `000999` | `000999` | Zero-padded image UID | 0 … N |
| `sh` | `oct` | Shape — `tet` tetrahedron, `cub` cube, `oct` octahedron, `dod` dodecahedron, `ico` icosahedron | 5 shapes |
| `f` | `05` | Face index — which face rests on the floor (0-indexed) | tet 0–3, cub 0–5, oct 0–7, dod 0–11, ico 0–19 |
| `yaw` | `140` | Yaw rotation around the vertical axis (degrees, integer-rounded) | 0–360 |
| `s` | `046` | Scale × 100 (integer-rounded) | 38–88 (config.json) / 20–80 (config_pink.json) |
| `lab` | `042--027-+034` | CIELab colour: L, a, b (integer-rounded, signed). Note: negative `a` produces a double-dash (e.g. `a=−27` → `--027`) | sampled or fixed per config |
| `t` | `031` | Transparency × 100 (integer-rounded) | 0–100 |
| `gl` | `035` | Glossiness × 100 (integer-rounded) | 0–100 |
| `xy` | `-0164-+0058` | Camera-relative position × 100 — lateral (signed) then depth (signed). `xy[0]` is the left/right axis used for the position task split. | lateral −250 to +250, depth +30 to +70 |
| suffix | `random` | Sampling mode: `random` (randomly sampled) or `grid` (grid sweep) | — |