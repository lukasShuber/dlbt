"""
Analysis for simulation 00 — identifiability.

Produces:
  plot_01_alpha_recovery.png  — scatter true alpha* vs predicted alpha (all images × K states)
  plot_02_learning_curve.png  — training NLL / MSE over epochs
  plot_03_behavior.png        — scatter true vs predicted P(right), all images × all tasks

Run from repo root:
    python experiments/simulations/00_identifiability/analysis.py
"""

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

import config as cfg

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
if not path.exists():
    raise FileNotFoundError(f"No results found at {path}. Run run.py first.")

with open(path, "rb") as f:
    res = pickle.load(f)

n_seeds     = res["n_seeds"]
n_trials    = res["n_trials"]
noise_floor = res["noise_floor"]
curves      = res["curves"]
dlbt        = res["dlbt"]       # {task: {pred:[n_seeds,n_pts], true, uids}}
true_alphas = res["true_alphas"]
alpha_preds = res["alpha_preds"]

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

uids = sorted(true_alphas.keys())
true_mat = np.array([true_alphas[u]              for u in uids])   # [N, K]
pred_mat = np.array([alpha_preds[u].mean(axis=0) for u in uids])   # [N, K]
pred_sem = np.array([alpha_preds[u].std(axis=0) / np.sqrt(n_seeds)
                     for u in uids])                                 # [N, K]

# ---------------------------------------------------------------------------
# Plot 1 — Alpha recovery  (K panels, one per latent state)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(10, 10),
                         gridspec_kw={"hspace": 0.5, "wspace": 0.35})

rhos = []
for k, ax in enumerate(axes.flat):
    t = true_mat[:, k]
    p = pred_mat[:, k]

    lo = min(t.min(), p.min()) * 0.95
    hi = max(t.max(), p.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=0.8, zorder=0)
    ax.scatter(t, p, s=4, alpha=0.25, color="#457B9D", linewidths=0)

    rho, _ = spearmanr(t, p)
    rhos.append(rho)
    ax.set_title(f"state {k}   ρ={rho:.2f}", fontsize=7, pad=3)
    ax.tick_params(labelsize=6)

fig.supxlabel("True α*", fontsize=11)
fig.supylabel("Predicted α (mean over seeds)", fontsize=11, x=0.02)
fig.suptitle(f"Parameter recovery   (mean ρ = {np.mean(rhos):.2f})", fontsize=13)

plt.tight_layout()
out = plots_dir / "plot_01_alpha_recovery.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2 — Learning curve
# ---------------------------------------------------------------------------
epochs = np.arange(len(curves["train_nlls"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

for ax, key_tr, key_va, ylabel in zip(
    axes,
    ["train_nlls", "train_mses"],
    ["val_nlls",   "val_mses"],
    ["NLL",        "MSE"],
):
    ax.plot(epochs, curves[key_tr], color="#E76F51", lw=1.5, label="train")
    if ylabel == "MSE":
        ax.axhline(noise_floor, ls="--", color="#E76F51", alpha=0.4, lw=1,
                   label="noise floor")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

fig.suptitle("Learning curve — DLBT (frozen encoder)", fontsize=12)
plt.tight_layout()
out = plots_dir / "plot_02_learning_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3 — Behavioral prediction  (all tasks pooled)
# ---------------------------------------------------------------------------
all_true = np.concatenate([dlbt[t]["true"] for t in dlbt])
all_pred_mean = np.concatenate(
    [dlbt[t]["pred"].mean(axis=0) for t in dlbt]
)
all_pred_sem = np.concatenate(
    [dlbt[t]["pred"].std(axis=0) / np.sqrt(n_seeds) for t in dlbt]
)
all_true_sem = np.sqrt(np.clip(all_true * (1 - all_true), 0, None) / n_trials)

rho, _  = spearmanr(all_pred_mean, all_true)
r,   _  = pearsonr(all_pred_mean,  all_true)
mse     = float(np.mean((all_pred_mean - all_true) ** 2))

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
ax.errorbar(all_pred_mean, all_true,
            xerr=all_pred_sem, yerr=all_true_sem,
            fmt="o", ms=3, alpha=0.15, color="#457B9D",
            elinewidth=0.4, capsize=0, linewidth=0)
ax.set_xlabel("Predicted P(right)", fontsize=12)
ax.set_ylabel("True P(right)", fontsize=12)
ax.set_title(f"Behavioral prediction (all tasks)\nρ={rho:.3f}   r={r:.3f}   MSE={mse:.4f}",
             fontsize=11)
ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])

import seaborn as sns
sns.despine(fig=fig)
plt.tight_layout()
out = plots_dir / "plot_03_behavior.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print(f"\nSummary:")
print(f"  Alpha recovery  mean ρ = {np.mean(rhos):.3f}  (per-state range: {min(rhos):.2f}–{max(rhos):.2f})")
print(f"  Behavioral pred      ρ = {rho:.3f}   MSE = {mse:.5f}   noise floor = {noise_floor:.5f}")
