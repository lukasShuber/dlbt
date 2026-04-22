"""
oracle_check.py
---------------
Three-stage sanity check: how well can a Dirichlet model predict human choices?

Stage 1 — Isotropic oracle sweep
    Alpha factored from ground-truth metadata (pos, transparency, glossiness, scale)
    via sigmoid with a single shared softness. Sweep over concentration x softness.

Stage 2 — Anisotropic oracle (coordinate scan)
    Same factored form but with independent softness per dimension, fitted by
    coordinate scan using only the 1-way tasks that isolate each dimension.

Stage 3 — Per-image unconstrained Dirichlet MLE  ("DLBT on steroids")
    For each probe image, directly optimise K=16 free Dirichlet parameters to
    maximise the (soft) multinomial log-likelihood of the observed choice counts
    across all tasks.  No metadata structure assumed — this is the tightest
    possible Dirichlet upper bound.

If Stage 3 also fails on some tasks (e.g. glossy), the Dirichlet family itself
cannot explain those choices, independent of how alpha is constructed.

Usage:
    cd <repo root>
    python experiments/behavior/run0/oracle_check.py
"""

import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from preprocess import load_and_preprocess

from dlbt.constants import K, X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH
from dlbt.data.task import TASKS

try:
    import torch
    from torch.distributions import Dirichlet as TorchDirichlet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available — Stage 3 will be skipped.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MC        = 2000    # MC samples for final evaluation
N_MC_OPT    = 500     # MC samples during coordinate scan (Stage 2)
RNG         = np.random.default_rng(42)
RNG_OPT     = np.random.default_rng(0)    # fixed seed → deterministic scan
PLOTS_DIR   = cfg.RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 grids
CONCENTRATIONS = [1, 5, 10, 50, 100, 500]
SOFTNESSES     = [0.05, 0.10, 0.20, 0.40]

# Stage 2 grids
SOFTNESS_FINE = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00]
CONC_FINE     = [0.5, 1, 2, 5, 10, 20, 50, 100, 500]

# Stage 2: which 1-way tasks isolate each dimension
DIM_TASKS = {
    "lr": ["right", "left"],
    "tr": ["transparent", "opaque"],
    "gl": ["glossy", "matte"],
    "sl": ["large", "small"],
}

# Stage 3 (torch)
N_MC_FIT    = 2000    # MC samples per gradient step
N_STEPS_FIT = 1000    # gradient steps per image
LR_FIT      = 0.05
TEMPERATURE = 50.0    # sigmoid sharpness for soft indicator (≈ hard at ±0.1)

# ---------------------------------------------------------------------------
# Load metadata  uid -> continuous z
# ---------------------------------------------------------------------------
print("Loading metadata...")
cont_meta: dict = {}
with open(cfg.METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cont_meta[rec["id"]] = rec["z"]


def _gt_state(uid: str) -> int:
    """Ground-truth latent state k for a probe image (from metadata thresholds)."""
    z  = cont_meta[uid]
    lr = int(z["pos_xy"][0]      > X_THRESHOLD)
    tr = int(z["transparency"]   > TRANSP_THRESH)
    gl = int(z["glossiness"]     > GLOSS_THRESH)
    sl = int(z["scale"]          > SCALE_THRESH)
    return lr * 8 + tr * 4 + gl * 2 + sl


# ---------------------------------------------------------------------------
# Load + preprocess behavioural data
# ---------------------------------------------------------------------------
print("Loading behavioural data...")
ds_full, probe_uids, main_uids, diag = load_and_preprocess(
    cfg.BEHAVIOR_CSV,
    beh_id_to_task     = cfg.BEH_ID_TO_TASK,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds    = cfg.USE_TRIAL_KINDS,
    seed               = cfg.SEED,
)
print(f"  {len(ds_full)} (uid, task) cells  |  "
      f"{ds_full.df['uid'].nunique()} images  |  "
      f"{ds_full.df['task_name'].nunique()} tasks")

df = ds_full.df[ds_full.df["uid"].isin(probe_uids)].copy()
df["emp_p"]  = df["count_1"] / (df["count_0"] + df["count_1"])
df["totals"] = df["count_0"] + df["count_1"]
print(f"  Probe-only: {len(df)} cells  |  {df['uid'].nunique()} images  |  "
      f"{df['task_name'].nunique()} tasks")

# Flat arrays (hot-loop targets)
_uids       = df["uid"].values
_task_names = df["task_name"].values
_emp_p      = df["emp_p"].values
_valid      = np.isfinite(_emp_p) & (df["totals"].values > 0)
_delta_us   = np.stack([TASKS[t].delta_u for t in _task_names])   # [N_cells, K]

# Precompute 1-way task masks per dimension (Stage 2)
_dim_masks: dict = {}
for _dim, _tasks in DIM_TASKS.items():
    _m = np.zeros(len(df), dtype=bool)
    for _t in _tasks:
        _m |= (_task_names == _t)
    _dim_masks[_dim] = _m & _valid

# Ground-truth state per probe uid
probe_uid_list  = sorted(probe_uids)   # stable order
gt_states       = {uid: _gt_state(uid) for uid in probe_uid_list}


# ---------------------------------------------------------------------------
# Oracle alpha construction (factored sigmoid, per-dimension softness)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def oracle_alpha(uid: str,
                 concentration: float,
                 softness_lr: float,
                 softness_tr: float,
                 softness_gl: float,
                 softness_sl: float) -> np.ndarray:
    z      = cont_meta[uid]
    p_lr   = _sigmoid((z["pos_xy"][0]    - X_THRESHOLD)  / softness_lr)
    p_tr   = _sigmoid((z["transparency"] - TRANSP_THRESH) / softness_tr)
    p_gl   = _sigmoid((z["glossiness"]   - GLOSS_THRESH)  / softness_gl)
    p_sl   = _sigmoid((z["scale"]        - SCALE_THRESH)  / softness_sl)
    probs  = np.array([
        (1-p_lr)*(1-p_tr)*(1-p_gl)*(1-p_sl),  # 0  L Op Mt Sm
        (1-p_lr)*(1-p_tr)*(1-p_gl)*   p_sl ,  # 1  L Op Mt Lg
        (1-p_lr)*(1-p_tr)*   p_gl *(1-p_sl),  # 2  L Op Gl Sm
        (1-p_lr)*(1-p_tr)*   p_gl *   p_sl ,  # 3  L Op Gl Lg
        (1-p_lr)*   p_tr *(1-p_gl)*(1-p_sl),  # 4  L Tr Mt Sm
        (1-p_lr)*   p_tr *(1-p_gl)*   p_sl ,  # 5  L Tr Mt Lg
        (1-p_lr)*   p_tr *   p_gl *(1-p_sl),  # 6  L Tr Gl Sm
        (1-p_lr)*   p_tr *   p_gl *   p_sl ,  # 7  L Tr Gl Lg
           p_lr *(1-p_tr)*(1-p_gl)*(1-p_sl),  # 8  R Op Mt Sm
           p_lr *(1-p_tr)*(1-p_gl)*   p_sl ,  # 9  R Op Mt Lg
           p_lr *(1-p_tr)*   p_gl *(1-p_sl),  # 10 R Op Gl Sm
           p_lr *(1-p_tr)*   p_gl *   p_sl ,  # 11 R Op Gl Lg
           p_lr *   p_tr *(1-p_gl)*(1-p_sl),  # 12 R Tr Mt Sm
           p_lr *   p_tr *(1-p_gl)*   p_sl ,  # 13 R Tr Mt Lg
           p_lr *   p_tr *   p_gl *(1-p_sl),  # 14 R Tr Gl Sm
           p_lr *   p_tr *   p_gl *   p_sl ,  # 15 R Tr Gl Lg
    ], dtype=np.float64)
    return np.clip(concentration * probs, 1e-6, None)


def build_all_alphas(concentration, softness_lr, softness_tr,
                     softness_gl, softness_sl) -> np.ndarray:
    uid_set = list(dict.fromkeys(_uids))
    by_uid  = {u: oracle_alpha(u, concentration, softness_lr,
                               softness_tr, softness_gl, softness_sl)
               for u in uid_set}
    return np.stack([by_uid[u] for u in _uids])


# ---------------------------------------------------------------------------
# MC P(yes) estimation
# ---------------------------------------------------------------------------

def mc_p_yes_batch(alphas: np.ndarray,
                   delta_us: np.ndarray,
                   rng: np.random.Generator,
                   n: int = N_MC) -> np.ndarray:
    """Estimate P(yes_i) for each row i via MC. Returns [N]."""
    preds = np.zeros(len(alphas))
    for i in range(len(alphas)):
        b = rng.dirichlet(alphas[i], size=n)
        preds[i] = float((b @ delta_us[i] > 0).mean())
    return preds


def evaluate_model(alphas_by_uid: dict,
                   rng: np.random.Generator = None,
                   n_mc: int = N_MC,
                   label: str = "") -> dict:
    """
    Evaluate any model given a dict uid -> alpha [K].
    Returns a result dict compatible with plotting functions.
    """
    if rng is None:
        rng = RNG
    alphas = np.stack([alphas_by_uid[u] for u in _uids])
    preds  = mc_p_yes_batch(alphas, _delta_us, rng, n=n_mc)
    mse    = float(np.mean((preds[_valid] - _emp_p[_valid]) ** 2))
    rho, _ = spearmanr(preds[_valid], _emp_p[_valid])
    return dict(label=label, mse=mse, rho=rho,
                preds=preds, emp_p=_emp_p, valid=_valid, task_names=_task_names)


# ---------------------------------------------------------------------------
# Stage 1 — Isotropic sweep
# ---------------------------------------------------------------------------
print(f"\n[Stage 1] Isotropic sweep: {len(CONCENTRATIONS)} conc x "
      f"{len(SOFTNESSES)} soft ({N_MC} MC samples)...")

iso_results = []
best_iso    = None
for conc, soft in product(CONCENTRATIONS, SOFTNESSES):
    print(f"  conc={conc:>5}  soft={soft:.2f} ...", end="  ", flush=True)
    adict = {u: oracle_alpha(u, conc, soft, soft, soft, soft)
             for u in probe_uid_list}
    r = evaluate_model(adict, label=f"iso c={conc} s={soft}")
    r.update(concentration=conc, softness=soft)
    iso_results.append(r)
    print(f"MSE={r['mse']:.4f}  rho={r['rho']:.3f}")
    if best_iso is None or r["mse"] < best_iso["mse"]:
        best_iso = r

print(f"\nBest isotropic: conc={best_iso['concentration']}  "
      f"soft={best_iso['softness']:.2f}  "
      f"MSE={best_iso['mse']:.4f}  rho={best_iso['rho']:.3f}")

# Stage 1 heatmap
print("\nPlotting Stage 1 heatmap...")
mse_grid = np.array([r["mse"] for r in iso_results]).reshape(
    len(CONCENTRATIONS), len(SOFTNESSES))
rho_grid = np.array([r["rho"] for r in iso_results]).reshape(
    len(CONCENTRATIONS), len(SOFTNESSES))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, grid, title, fmt in [
    (axes[0], mse_grid, "MSE",   ".4f"),
    (axes[1], rho_grid, "rho",   ".3f"),
]:
    sns.heatmap(grid, ax=ax,
                xticklabels=[f"{s}" for s in SOFTNESSES],
                yticklabels=[f"{c}" for c in CONCENTRATIONS],
                annot=True, fmt=fmt,
                cmap="viridis_r" if title == "MSE" else "viridis")
    ax.set_xlabel("softness"); ax.set_ylabel("concentration"); ax.set_title(title)
fig.suptitle("Stage 1: Isotropic oracle sweep", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "oracle_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR / 'oracle_sweep.png'}")


# ---------------------------------------------------------------------------
# Stage 2 — Anisotropic coordinate scan
# ---------------------------------------------------------------------------
print("\n[Stage 2] Anisotropic coordinate scan...")


def _scan_dim(dim, conc, s_lr, s_tr, s_gl, s_sl):
    mask = _dim_masks[dim]
    du   = _delta_us[mask]
    yp   = _emp_p[mask]
    best_s, best_mse = None, np.inf
    for s in SOFTNESS_FINE:
        kw = {"s_lr": s_lr, "s_tr": s_tr, "s_gl": s_gl, "s_sl": s_sl}
        kw[f"s_{dim}"] = s
        alphas = build_all_alphas(conc, kw["s_lr"], kw["s_tr"], kw["s_gl"], kw["s_sl"])
        preds  = mc_p_yes_batch(alphas[mask], du, RNG_OPT, n=N_MC_OPT)
        mse_v  = float(np.mean((preds - yp) ** 2))
        if mse_v < best_mse:
            best_mse, best_s = mse_v, s
    return best_s, best_mse


def _scan_conc(conc_grid, s_lr, s_tr, s_gl, s_sl):
    best_c, best_mse = None, np.inf
    for c in conc_grid:
        alphas = build_all_alphas(c, s_lr, s_tr, s_gl, s_sl)
        preds  = mc_p_yes_batch(alphas, _delta_us, RNG_OPT, n=N_MC_OPT)
        mse_v  = float(np.mean((preds[_valid] - _emp_p[_valid]) ** 2))
        if mse_v < best_mse:
            best_mse, best_c = mse_v, c
    return best_c, best_mse


s_lr = s_tr = s_gl = s_sl = best_iso["softness"]
conc = best_iso["concentration"]
print(f"  Init: conc={conc}  soft={s_lr:.2f}\n")

for rnd in range(2):
    print(f"  -- Round {rnd+1} --")
    for dim in ["lr", "tr", "gl", "sl"]:
        bs, bm = _scan_dim(dim, conc, s_lr, s_tr, s_gl, s_sl)
        if   dim == "lr": s_lr = bs
        elif dim == "tr": s_tr = bs
        elif dim == "gl": s_gl = bs
        else:             s_sl = bs
        print(f"    soft_{dim} -> {bs:.3f}  (dim-MSE={bm:.4f})")
    conc, bm = _scan_conc(CONC_FINE, s_lr, s_tr, s_gl, s_sl)
    print(f"    conc   -> {conc:.3f}  (full-MSE={bm:.4f})\n")

aniso_alphas_by_uid = {u: oracle_alpha(u, conc, s_lr, s_tr, s_gl, s_sl)
                       for u in probe_uid_list}
best_aniso = evaluate_model(aniso_alphas_by_uid, label="anisotropic")
best_aniso.update(concentration=conc,
                  softness_lr=s_lr, softness_tr=s_tr,
                  softness_gl=s_gl, softness_sl=s_sl)

print(f"Anisotropic: conc={conc:.3f}  "
      f"s_lr={s_lr:.3f}  s_tr={s_tr:.3f}  s_gl={s_gl:.3f}  s_sl={s_sl:.3f}")
print(f"  MSE={best_aniso['mse']:.4f}  rho={best_aniso['rho']:.3f}")
print(f"  (isotropic best: MSE={best_iso['mse']:.4f}  rho={best_iso['rho']:.3f})")


# ---------------------------------------------------------------------------
# Stage 3 — Per-image unconstrained Dirichlet MLE (torch Adam)
# ---------------------------------------------------------------------------
print("\n[Stage 3] Per-image unconstrained Dirichlet MLE...")

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    best_fitted = None
    fitted_alphas_by_uid = None
else:
    # Build per-image task data  uid -> [(delta_u [K], count_0, count_1), ...]
    uid_task_data: dict = {uid: [] for uid in probe_uid_list}
    for idx, row in df.iterrows():
        uid  = row["uid"]
        task = row["task_name"]
        c0   = int(row["count_0"])
        c1   = int(row["count_1"])
        if c0 + c1 > 0:
            uid_task_data[uid].append((TASKS[task].delta_u, c0, c1))

    def fit_alpha_for_image(uid: str,
                             init_alpha: np.ndarray = None,
                             n_steps: int = N_STEPS_FIT,
                             task_data: list = None) -> np.ndarray:
        """
        Optimise K=16 Dirichlet parameters for one image via Adam, maximising
        the soft multinomial NLL of observed choice counts across a set of tasks.

        task_data: list of (delta_u [K], count_0, count_1).
                   If None, uses all tasks for this image (uid_task_data[uid]).
        Uses sigmoid(T * b @ delta_u) as a smooth surrogate for the hard
        indicator, with reparameterised Dirichlet sampling for gradients.
        """
        if task_data is None:
            task_data = uid_task_data[uid]
        if not task_data:
            return np.ones(K, dtype=np.float64)

        delta_us_t = torch.tensor(
            np.stack([du for du, c0, c1 in task_data]),
            dtype=torch.float32)                         # [T, K]
        counts_t   = torch.tensor(
            [[c0, c1] for du, c0, c1 in task_data],
            dtype=torch.float32)                         # [T, 2]

        # Initialise from anisotropic oracle (warm start) if available
        if init_alpha is not None:
            log_alpha = torch.log(torch.tensor(init_alpha, dtype=torch.float32))
        else:
            log_alpha = torch.zeros(K, dtype=torch.float32)
        log_alpha = log_alpha.requires_grad_(True)

        optimizer  = torch.optim.Adam([log_alpha], lr=LR_FIT)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=1e-4)

        for _ in range(n_steps):
            optimizer.zero_grad()
            alpha    = torch.exp(log_alpha).clamp(min=1e-3)
            b        = TorchDirichlet(alpha).rsample((N_MC_FIT,))  # [n_mc, K]
            # logits[i, t] = b[i] . delta_u[t]
            logits   = b @ delta_us_t.T                             # [n_mc, T]
            p_yes    = torch.sigmoid(TEMPERATURE * logits).mean(0)  # [T]
            p_no     = 1.0 - p_yes
            probs    = torch.stack([p_no, p_yes], dim=1).clamp(1e-7, 1-1e-7)  # [T, 2]
            nll      = -(counts_t * probs.log()).sum()
            nll.backward()
            optimizer.step()
            scheduler.step()

        return torch.exp(log_alpha).detach().clamp(min=1e-6).numpy().astype(np.float64)

    fitted_alphas_by_uid: dict = {}
    for uid in probe_uid_list:
        print(f"  Fitting {uid}  (gt_state={gt_states[uid]:2d}) ...", end="  ", flush=True)
        init = aniso_alphas_by_uid.get(uid)
        alpha_fit = fit_alpha_for_image(uid, init_alpha=init)
        fitted_alphas_by_uid[uid] = alpha_fit
        # Quick NLL check
        task_data = uid_task_data[uid]
        if task_data:
            du_arr = np.stack([du for du, c0, c1 in task_data])
            preds  = mc_p_yes_batch(
                np.tile(alpha_fit, (len(du_arr), 1)), du_arr, RNG, n=500)
            mse_q  = float(np.mean([
                (p - c1/(c0+c1))**2
                for p, (du, c0, c1) in zip(preds, task_data)
                if c0+c1 > 0
            ]))
            print(f"img-MSE={mse_q:.4f}")
        else:
            print("no data")

    best_fitted = evaluate_model(fitted_alphas_by_uid, label="unconstrained MLE")
    print(f"\nUnconstrained MLE: MSE={best_fitted['mse']:.4f}  "
          f"rho={best_fitted['rho']:.3f}")
    print(f"  (anisotropic:    MSE={best_aniso['mse']:.4f}  "
          f"rho={best_aniso['rho']:.3f})")
    print(f"  (isotropic best: MSE={best_iso['mse']:.4f}  "
          f"rho={best_iso['rho']:.3f})")

    # -----------------------------------------------------------------------
    # Stage 4 — Held-out task generalisation: fit on TRAIN_TASKS, eval on ALL
    # -----------------------------------------------------------------------
    print("\n[Stage 4] Held-out task generalisation (fit on TRAIN_TASKS only)...")

    train_task_set = set(cfg.TRAIN_TASKS)
    val_task_set   = set(cfg.VAL_TASKS)

    # Build per-image task data restricted to TRAIN_TASKS
    uid_train_task_data: dict = {uid: [] for uid in probe_uid_list}
    for _, row in df.iterrows():
        if row["task_name"] not in train_task_set:
            continue
        uid = row["uid"]
        c0, c1 = int(row["count_0"]), int(row["count_1"])
        if c0 + c1 > 0:
            uid_train_task_data[uid].append(
                (TASKS[row["task_name"]].delta_u, c0, c1))

    fitted_train_alphas: dict = {}
    for uid in probe_uid_list:
        n_train = len(uid_train_task_data[uid])
        print(f"  Fitting {uid}  (gt={gt_states[uid]:2d}, "
              f"{n_train} train tasks) ...", end="  ", flush=True)
        init = aniso_alphas_by_uid.get(uid)
        alpha_fit = fit_alpha_for_image(
            uid, init_alpha=init,
            task_data=uid_train_task_data[uid])
        fitted_train_alphas[uid] = alpha_fit
        # Quick per-image MSE on VAL tasks only
        val_rows = [(TASKS[row["task_name"]].delta_u,
                     int(row["count_0"]), int(row["count_1"]))
                    for _, row in df[df["uid"] == uid].iterrows()
                    if row["task_name"] in val_task_set
                    and int(row["count_0"]) + int(row["count_1"]) > 0]
        if val_rows:
            du_v = np.stack([du for du, c0, c1 in val_rows])
            pv   = mc_p_yes_batch(
                np.tile(alpha_fit, (len(du_v), 1)), du_v, RNG, n=500)
            mse_v = float(np.mean([(p - c1/(c0+c1))**2
                                   for p, (du, c0, c1) in zip(pv, val_rows)]))
            print(f"val-MSE={mse_v:.4f}")
        else:
            print("no val data")

    best_fitted_train = evaluate_model(
        fitted_train_alphas, label="MLE (train tasks only)")

    # Compute held-out val-only MSE and rho
    val_mask  = np.array([t in val_task_set for t in _task_names]) & _valid
    train_mask = np.array([t in train_task_set for t in _task_names]) & _valid
    preds_tr  = best_fitted_train["preds"]
    if val_mask.sum() > 1:
        val_mse  = float(np.mean((preds_tr[val_mask]  - _emp_p[val_mask])  ** 2))
        val_rho, _ = spearmanr(preds_tr[val_mask],  _emp_p[val_mask])
        trn_mse  = float(np.mean((preds_tr[train_mask] - _emp_p[train_mask]) ** 2))
        trn_rho, _ = spearmanr(preds_tr[train_mask], _emp_p[train_mask])
    else:
        val_mse = val_rho = trn_mse = trn_rho = float("nan")

    print(f"\nMLE (train only) — all tasks:   MSE={best_fitted_train['mse']:.4f}  "
          f"rho={best_fitted_train['rho']:.3f}")
    print(f"               — train tasks:  MSE={trn_mse:.4f}  rho={trn_rho:.3f}")
    print(f"               — VAL tasks:    MSE={val_mse:.4f}  rho={val_rho:.3f}  "
          f"(generalization)")
    print(f"  (unconstrained full MLE:      MSE={best_fitted['mse']:.4f}  "
          f"rho={best_fitted['rho']:.3f})")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _task_color(name):
    n = name.count("_and_") + 1
    return {1: "#457B9D", 2: "#E76F51", 3: "#9B5DE5"}.get(n, "gray")


def _plot_scatter(result: dict, out_path: Path,
                  val_tasks: set = None) -> None:
    """
    Per-task scatter: oracle P(yes) vs human P(yes).

    val_tasks: optional set of task names.  When provided, panels for val tasks
               receive a red spine and a "(held-out)" subtitle so it is clear
               the model never saw those tasks during fitting.
    """
    all_tasks = sorted(df["task_name"].unique())
    n_cols    = 8
    n_rows    = int(np.ceil(len(all_tasks) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, task_name in enumerate(all_tasks):
        row, col  = divmod(idx, n_cols)
        ax        = axes[row, col]
        is_val    = val_tasks is not None and task_name in val_tasks
        mask      = (result["task_names"] == task_name) & result["valid"]
        x, y      = result["preds"][mask], result["emp_p"][mask]
        col_c     = _task_color(task_name)

        ax.scatter(x, y, s=10, alpha=0.8, color=col_c, linewidths=0)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, zorder=0)
        rho_t, _  = spearmanr(x, y) if len(x) > 2 else (float("nan"), None)
        mse_t     = float(np.mean((x - y)**2)) if len(x) > 0 else float("nan")
        ax.text(0.05, 0.95, f"r={rho_t:.2f}", transform=ax.transAxes,
                fontsize=6, va="top", color=col_c)
        ax.text(0.05, 0.82, f"e={mse_t:.3f}", transform=ax.transAxes,
                fontsize=6, va="top", color=col_c)

        title_str = task_name.replace("_and_", " & ").replace("_", "/")
        if is_val:
            title_str += "\n(held-out)"
        ax.set_title(title_str, fontsize=6, pad=2,
                     color="#CC2222" if is_val else "black")

        # Red frame for held-out val tasks
        if is_val:
            for spine in ax.spines.values():
                spine.set_edgecolor("#CC2222")
                spine.set_linewidth(1.4)
            ax.set_facecolor("#FFF5F5")

        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    for idx in range(len(all_tasks), n_rows * n_cols):
        r2, c2 = divmod(idx, n_cols)
        axes[r2, c2].set_visible(False)

    label   = result.get("label", "")
    mse_all = result["mse"]
    rho_all = result["rho"]
    extra   = "  |  red frame = held-out val task" if val_tasks else ""
    fig.suptitle(
        f"Oracle ({label})  |  overall MSE={mse_all:.4f}  rho={rho_all:.3f}{extra}\n"
        "colour: 1-way (blue)  2-way (orange)  3-way (purple)",
        fontsize=8, y=1.01)
    fig.text(0.5, -0.01, "Oracle P(yes)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5, "Human P(yes)", va="center", rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-task scatter plots
# ---------------------------------------------------------------------------
print("\nPlotting per-task scatters...")
_plot_scatter(best_iso,   PLOTS_DIR / "oracle_scatter_iso.png")
_plot_scatter(best_aniso, PLOTS_DIR / "oracle_scatter_aniso.png")
if best_fitted is not None:
    _plot_scatter(best_fitted, PLOTS_DIR / "oracle_scatter_fitted.png")
if HAS_TORCH and best_fitted is not None:
    _plot_scatter(best_fitted_train, PLOTS_DIR / "oracle_scatter_fitted_train.png",
                  val_tasks=set(cfg.VAL_TASKS))


# ---------------------------------------------------------------------------
# Comparison bar plots: rho and MSE per task, all models
# ---------------------------------------------------------------------------
print("Plotting comparison bar charts...")

all_tasks = sorted(df["task_name"].unique())

# Bar plots show only Stage 3 and Stage 4 (the meaningful Dirichlet upper bounds)
models       = []
model_labels = []
model_colors = []
if best_fitted is not None:
    models.append(best_fitted)
    model_labels.append("unconstrained MLE (all tasks)")
    model_colors.append("#43AA8B")
if HAS_TORCH and best_fitted is not None:
    models.append(best_fitted_train)
    model_labels.append("MLE (train tasks only)")
    model_colors.append("#9B5DE5")

# Build val_task_set for plot annotations (may be undefined if torch absent)
_val_task_set = set(cfg.VAL_TASKS) if HAS_TORCH else set()

if not models:
    print("Skipping bar plots (Stage 3/4 not available).")
n_models = len(models)
rho_data = [[] for _ in range(n_models)]
mse_data = [[] for _ in range(n_models)]
labels   = []
bar_cols = []

for task_name in all_tasks:
    mask = (_task_names == task_name) & _valid
    if mask.sum() < 3:
        continue
    y = _emp_p[mask]
    for mi, m in enumerate(models):
        x = m["preds"][mask]
        rho_data[mi].append(spearmanr(x, y)[0])
        mse_data[mi].append(float(np.mean((x - y)**2)))
    labels.append(task_name.replace("_and_", " & ").replace("_", "/"))
    bar_cols.append(_task_color(task_name))

x_pos  = np.arange(len(labels))
width  = 0.8 / n_models
figw   = max(12, len(labels) * 0.65)
offset = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width


def _barplot(data_list, ylabel, title, out_path, ref_line=None, val_tasks=None):
    """val_tasks: set of task names to mark with ★ (held-out tasks)."""
    fig, ax = plt.subplots(figsize=(figw, 4.5))
    for mi, (dat, lbl, col) in enumerate(zip(data_list, model_labels, model_colors)):
        # hatch the train-only model's bars for val tasks to signal held-out
        is_train_pred = (lbl == "MLE (train tasks only)")
        for xi, (d, task_lbl) in enumerate(zip(dat, labels)):
            raw_name = task_lbl.replace(" & ", "_and_").replace("/", "_")
            is_val   = val_tasks is not None and any(
                vt.replace("_and_", " & ") == task_lbl for vt in val_tasks)
            hatch = "//" if (is_train_pred and is_val) else None
            ax.bar(xi + offset[mi], d, width,
                   color=col, alpha=0.82, zorder=3,
                   hatch=hatch, edgecolor="white" if not hatch else "black",
                   linewidth=0.4,
                   label=lbl if xi == 0 else None)
    ax.set_xticks(x_pos)
    # Add ★ to val task labels
    tick_labels = []
    for task_lbl in labels:
        is_val = val_tasks is not None and any(
            vt.replace("_and_", " & ") == task_lbl for vt in val_tasks)
        tick_labels.append(("★ " if is_val else "") + task_lbl)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    for tick, col in zip(ax.get_xticklabels(), bar_cols):
        tick.set_color(col)
    if ref_line is not None:
        ax.axhline(ref_line, color="gray", lw=0.8, ls="--", zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    sns.despine(ax=ax, trim=True)
    ax.grid(axis="y", lw=0.3, color="lightgray", zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if models:
    _barplot(rho_data, "Spearman rho",
             "Dirichlet MLE: unconstrained (all tasks) vs train-tasks-only  "
             "|  ★ = held-out val task  |  hatching = held-out prediction",
             PLOTS_DIR / "oracle_rho_comparison.png",
             ref_line=0, val_tasks=_val_task_set)

    _barplot(mse_data, "MSE",
             "Dirichlet MLE: unconstrained (all tasks) vs train-tasks-only  "
             "|  ★ = held-out val task  |  hatching = held-out prediction",
             PLOTS_DIR / "oracle_mse_comparison.png",
             val_tasks=_val_task_set)


# ---------------------------------------------------------------------------
# Stage 3: Alpha heatmap — fitted alpha per probe image
# ---------------------------------------------------------------------------
if fitted_alphas_by_uid is not None:
    print("Plotting Stage 3 alpha heatmap...")

    # Sort probe images by ground-truth state
    sorted_uids = sorted(probe_uid_list, key=lambda u: gt_states[u])

    # Build normalised alpha matrix (= E[b] = alpha / sum(alpha))
    alpha_mat = np.stack([
        fitted_alphas_by_uid[u] / fitted_alphas_by_uid[u].sum()
        for u in sorted_uids
    ])  # [16, 16]

    gt_sorted = [gt_states[u] for u in sorted_uids]

    # State labels
    def _state_label(k):
        lr = "R" if k & 8 else "L"
        tr = "Tr" if k & 4 else "Op"
        gl = "Gl" if k & 2 else "Mt"
        sl = "Lg" if k & 1 else "Sm"
        return f"{lr}·{tr}·{gl}·{sl}"

    state_labels = [_state_label(k) for k in range(K)]
    row_labels   = [f"k={gt_states[u]:2d}  {_state_label(gt_states[u])}"
                    for u in sorted_uids]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(alpha_mat, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=alpha_mat.max())
    plt.colorbar(im, ax=ax, label="E[b_k] = alpha_k / sum(alpha)")

    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sorted_uids)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Mark ground-truth diagonal
    for row_i, k in enumerate(gt_sorted):
        ax.add_patch(plt.Rectangle((k - 0.5, row_i - 0.5), 1, 1,
                                   fill=False, edgecolor="blue",
                                   linewidth=2, zorder=5))

    ax.set_xlabel("Latent state k (fitted alpha mean)")
    ax.set_ylabel("Probe image (sorted by ground-truth state)")
    ax.set_title(
        "Stage 3: Fitted Dirichlet E[b] per probe image\n"
        "Blue border = ground-truth state  |  "
        f"overall MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f}",
        fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "oracle_alpha_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'oracle_alpha_heatmap.png'}")

    # Dual heatmap: full MLE vs train-only MLE side by side
    print("Plotting dual alpha heatmap (full vs train-only)...")
    alpha_mat_train = np.stack([
        fitted_train_alphas[u] / fitted_train_alphas[u].sum()
        for u in sorted_uids
    ])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    vmax = max(alpha_mat.max(), alpha_mat_train.max())
    for ax, mat, subtitle in [
        (axes[0], alpha_mat,       "Fit on ALL tasks (upper bound)"),
        (axes[1], alpha_mat_train, "Fit on TRAIN tasks only (generalisation)"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="E[b_k]", shrink=0.7)
        ax.set_xticks(range(K))
        ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(sorted_uids)))
        ax.set_yticklabels(row_labels, fontsize=7)
        for row_i, k in enumerate(gt_sorted):
            ax.add_patch(plt.Rectangle((k - 0.5, row_i - 0.5), 1, 1,
                                       fill=False, edgecolor="blue",
                                       linewidth=2, zorder=5))
        ax.set_xlabel("Latent state k")
        ax.set_title(subtitle, fontsize=9)

    fig.suptitle(
        "Fitted Dirichlet E[b] per probe image\n"
        "Blue border = ground-truth state  |  "
        f"Full MLE: MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f}    "
        f"Train-only: MSE={best_fitted_train['mse']:.4f}  "
        f"rho={best_fitted_train['rho']:.3f}",
        fontsize=9)
    axes[0].set_ylabel("Probe image (sorted by ground-truth state)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "oracle_alpha_heatmap_dual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'oracle_alpha_heatmap_dual.png'}")


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
all_tasks = sorted(df["task_name"].unique())

header  = f"{'task':<39}  {'N':>5}"
for lbl in model_labels:
    header += f"  {'MSE':>7}  {'rho':>5}  ({lbl[:8]})"
print(f"\n{'='*len(header)}")
print(header)
print("=" * len(header))

for task_name in all_tasks:
    mask = (_task_names == task_name) & _valid
    if mask.sum() < 2:
        continue
    n_way = task_name.count("_and_") + 1
    row   = f"[{n_way}-way] {task_name:<33}  {mask.sum():>5}"
    y     = _emp_p[mask]
    for m in models:
        x       = m["preds"][mask]
        mse_v   = float(np.mean((x - y)**2))
        rho_v,_ = spearmanr(x, y)
        row    += f"  {mse_v:>7.4f}  {rho_v:>5.3f}"
    print(row)

print(f"\n{'='*60}")
print("Overall (all tasks):")
for m in models:
    print(f"  {m['label']:<28}  MSE={m['mse']:.4f}  rho={m['rho']:.3f}")

if HAS_TORCH and best_fitted is not None:
    print(f"\nGeneralisation (MLE train→val):")
    print(f"  train tasks:  MSE={trn_mse:.4f}  rho={trn_rho:.3f}")
    print(f"  VAL tasks:    MSE={val_mse:.4f}  rho={val_rho:.3f}  ← generalisation")

if HAS_TORCH:
    print(f"\nStage 2 optimised softness:")
    print(f"  concentration : {best_aniso['concentration']:.4f}")
    print(f"  softness_lr   : {best_aniso['softness_lr']:.4f}")
    print(f"  softness_tr   : {best_aniso['softness_tr']:.4f}")
    print(f"  softness_gl   : {best_aniso['softness_gl']:.4f}")
    print(f"  softness_sl   : {best_aniso['softness_sl']:.4f}")
