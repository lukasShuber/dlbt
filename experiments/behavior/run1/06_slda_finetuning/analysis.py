"""
run1/06_slda_finetuning/analysis.py — SLDA Phase 2 diagnostic plots.

Reads results/slda_sandbox.pkl (or any pkl passed via --pkl) and produces:

  plot_nll.png        — Training + validation NLL across epochs (one subplot per LR).
  plot_cmse_curve.png — Probe cMSE-NF across epochs per LR variant (from epoch hook),
                        with Phase 1 reference line.
  plot_final_bar.png  — Final cMSE-NF and ρ bar chart for all conditions.

Run from repo root:
    python experiments/behavior/run1/06_slda_finetuning/analysis.py
    python experiments/behavior/run1/06_slda_finetuning/analysis.py --pkl PATH
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to a specific pkl.  Default: results/slda_sandbox.pkl.")
args = parser.parse_args()

if args.pkl:
    pkl_path = Path(args.pkl)
else:
    pkl_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No pkl found at {pkl_path}.  Run run.py first.")

print(f"Loading: {pkl_path.name}")
with open(pkl_path, "rb") as f:
    d = pickle.load(f)

variants          = d["variants"]
phase1_cmse       = d["phase1_cmse"]
phase1_rho        = d["phase1_rho"]
random_cmse_net   = d["random_cmse_net"]
rho_noise_ceiling = d.get("rho_noise_ceiling", float("nan"))

plots_dir = cfg.RESULTS_DIR / "plots" / pkl_path.stem
plots_dir.mkdir(parents=True, exist_ok=True)

n_variants = len(variants)
lrs        = [v["lr"] for v in variants]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    exp = int(round(np.log10(lr)))
    return rf"$lr=10^{{{exp}}}$"


def _despine(ax):
    sns.despine(ax=ax, top=True, right=True)


# ---------------------------------------------------------------------------
# Plot 1: NLL curves  (train + val per LR)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, n_variants, figsize=(4.5 * n_variants, 4.0),
                         sharey=False)
if n_variants == 1:
    axes = [axes]

for ax, v, c in zip(axes, variants, cfg.C_PHASE2):
    epochs     = list(range(len(v["train_nll"])))
    train_nll  = v["train_nll"]
    val_nll    = v["val_nll"]
    best_ep    = v["best_epoch"]

    ax.plot(epochs, train_nll, color=c,       lw=1.5, label="train NLL", zorder=3)
    ax.plot(epochs, val_nll,   color=c, ls="--", lw=1.5, label="val NLL",   zorder=3)
    ax.axvline(best_ep, color="#999999", lw=1.0, ls=":", zorder=2)
    ax.text(best_ep + max(1, len(epochs) * 0.01), ax.get_ylim()[1] * 0.98,
            f"best={best_ep}", color="#999999", fontsize=7, va="top")

    ax.set_title(_lr_label(v["lr"]), fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("NLL (total)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _despine(ax)

plt.suptitle("Phase 2 — NLL training curves", fontsize=12, y=1.01)
plt.tight_layout()
out = plots_dir / "plot_nll.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out.relative_to(cfg.RESULTS_DIR)}")

# ---------------------------------------------------------------------------
# Plot 2: probe cMSE-NF curves across epochs
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.5))

# Phase 1 reference
ax.axhline(phase1_cmse, color=cfg.C_PHASE1, lw=1.8, ls="--", zorder=2,
           label="Phase 1 (frozen CLIP)")

for v, c in zip(variants, cfg.C_PHASE2):
    hook_ep   = v["hook_epochs"]
    hook_cmse = v["hook_cmse"]
    if not hook_ep:
        continue
    ax.plot(hook_ep, hook_cmse, color=c, lw=1.8, zorder=3,
            label=_lr_label(v["lr"]))
    # Mark final value
    ax.plot(hook_ep[-1], hook_cmse[-1], "o", color=c, ms=6,
            mfc=c, mew=1.4, zorder=4)

# Chance reference
ax.axhline(random_cmse_net, color=cfg.C_RNDINI, lw=1.2,
           ls=(0, (4, 3)), zorder=1)
ax.annotate("chance (P=0.5)",
            xy=(1.0, random_cmse_net),
            xycoords=("axes fraction", "data"),
            xytext=(-4, -5), textcoords="offset points",
            color=cfg.C_RNDINI, fontsize=8, style="italic",
            va="top", ha="right", zorder=6)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("cMSE − noise floor", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, frameon=False)
_despine(ax)
plt.tight_layout()

out = plots_dir / "plot_cmse_curve.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out.relative_to(cfg.RESULTS_DIR)}")

# ---------------------------------------------------------------------------
# Plot 3: final bar chart (cMSE-NF and ρ)
# ---------------------------------------------------------------------------
# Build condition labels and values
cond_labels  = ["Phase 1\n(frozen CLIP)"]
cmse_vals    = [phase1_cmse]
rho_vals     = [phase1_rho]
bar_colors   = [cfg.C_PHASE1]
bar_hatches  = [""]

for v, c_solid, c_light in zip(variants, cfg.C_PHASE2, cfg.C_REFIT):
    lr_str = _lr_label(v["lr"])
    # Phase 2, frozen scaler
    cond_labels.append(f"Phase 2\n{lr_str}\n(frozen sc.)")
    cmse_vals.append(v["phase2_cmse"])
    rho_vals.append(v["phase2_rho"])
    bar_colors.append(c_solid)
    bar_hatches.append("")
    # Phase 2, refit scaler
    cond_labels.append(f"Phase 2\n{lr_str}\n(refit sc.)")
    cmse_vals.append(v["refit_scaler_cmse"])
    rho_vals.append(v["refit_scaler_rho"])
    bar_colors.append(c_light)
    bar_hatches.append("//")

n_bars = len(cond_labels)
xs     = np.arange(n_bars)

fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(max(7, n_bars * 1.1), 4.5))

# ── cMSE-NF bars ──
for x, h, c, hatch in zip(xs, cmse_vals, bar_colors, bar_hatches):
    ax_c.bar(x, h, color=c, hatch=hatch, edgecolor="white",
             linewidth=0.8, width=0.7, zorder=3)
ax_c.axhline(random_cmse_net, color=cfg.C_RNDINI, lw=1.2,
             ls=(0, (4, 3)), zorder=2)
ax_c.annotate("chance",
              xy=(1.0, random_cmse_net),
              xycoords=("axes fraction", "data"),
              xytext=(-4, -5), textcoords="offset points",
              color=cfg.C_RNDINI, fontsize=7, style="italic",
              va="top", ha="right", zorder=6)
ax_c.set_xticks(xs)
ax_c.set_xticklabels(cond_labels, fontsize=7)
ax_c.set_ylabel("cMSE − noise floor", fontsize=10, fontweight="bold")
ax_c.set_ylim(0, max(cmse_vals + [random_cmse_net]) * 1.15)
_despine(ax_c)

# ── ρ bars ──
for x, h, c, hatch in zip(xs, rho_vals, bar_colors, bar_hatches):
    ax_r.bar(x, h, color=c, hatch=hatch, edgecolor="white",
             linewidth=0.8, width=0.7, zorder=3)
if not np.isnan(rho_noise_ceiling):
    ax_r.axhline(rho_noise_ceiling, color="#555555", lw=1.2,
                 ls=(0, (2, 2)), zorder=2)
    ax_r.annotate("noise ceiling",
                  xy=(0.0, rho_noise_ceiling),
                  xycoords=("axes fraction", "data"),
                  xytext=(4, 5), textcoords="offset points",
                  color="#555555", fontsize=7, style="italic",
                  va="bottom", ha="left", zorder=6)
ax_r.set_xticks(xs)
ax_r.set_xticklabels(cond_labels, fontsize=7)
ax_r.set_ylabel(r"Spearman $\rho$", fontsize=10, fontweight="bold")
ax_r.set_ylim(0, 1.05)
_despine(ax_r)

plt.suptitle("Phase 2 — final performance comparison", fontsize=12)
plt.tight_layout()
out = plots_dir / "plot_final_bar.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out.relative_to(cfg.RESULTS_DIR)}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  {'Condition':<35}  {'cMSE-NF':>10}  {'ρ':>8}")
print("  " + "-" * 57)
print(f"  {'Phase 1 (frozen CLIP)':<35}  {phase1_cmse:+10.5f}  {phase1_rho:8.4f}")
for v in variants:
    lr_str = f"lr={v['lr']:.0e}"
    print(f"  {'Phase 2 ('+lr_str+', frozen sc.)':<35}  "
          f"{v['phase2_cmse']:+10.5f}  {v['phase2_rho']:8.4f}"
          f"  [best_ep={v['best_epoch']}]")
    print(f"  {'Phase 2 ('+lr_str+', refit sc.)':<35}  "
          f"{v['refit_scaler_cmse']:+10.5f}  {v['refit_scaler_rho']:8.4f}")
print("=" * 60)
