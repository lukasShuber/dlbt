"""
01_fit/run_flex.py — FlexAgent (logistic-normal SEU) fit on real human data.

Trains two variants in one script:
  • FlexAgent(cov_type="diag") — 2K mapper outputs, diagonal Gaussian Σ.
  • FlexAgent(cov_type="full") — K + K(K+1)/2 mapper outputs, full Cholesky Σ.

Uses the identical 10% cell-level eval split as run.py (same RNG seed) so
DLBT and Flex results are directly comparable.

Run from repo root:
    python experiments/behavior/run0/01_fit/run_flex.py
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset
from dlbt.agents.flex import FlexAgent
from dlbt.training.train_flex import train_flex

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ---------------------------------------------------------------------------
# Load stimuli & behavioural data  (identical to run.py)
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

print("\nLoading behavioural data...")
full_ds, probe_uids, main_uids, diag = load_and_preprocess(
    cfg.BEHAVIOR_CSV,
    beh_id_to_task    = cfg.BEH_ID_TO_TASK,
    min_catch_perf    = cfg.MIN_CATCH_PERF,
    main_perf_quantile= cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds   = cfg.USE_TRIAL_KINDS,
    seed              = cfg.SEED,
)

missing = [t for t in (cfg.TRAIN_TASKS + cfg.VAL_TASKS)
           if t not in full_ds.df["task_name"].unique()]
if missing:
    raise ValueError(f"Tasks missing from data: {missing}")

train_uids = set(main_uids)
test_uids  = set(probe_uids)

# ---------------------------------------------------------------------------
# Cell-level eval split — identical to run.py (same seed)
# ---------------------------------------------------------------------------
main_train_mask = (
    full_ds.df["uid"].isin(main_uids) &
    full_ds.df["task_name"].isin(cfg.TRAIN_TASKS)
)
main_train_df = full_ds.df[main_train_mask].copy().reset_index(drop=True)

rng_split = np.random.default_rng(cfg.SEED)   # same seed → same split
n_eval    = max(1, int(len(main_train_df) * cfg.EVAL_CELL_FRAC))
eval_idx  = rng_split.choice(len(main_train_df), size=n_eval, replace=False)
eval_mask = np.zeros(len(main_train_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df  = main_train_df[eval_mask].reset_index(drop=True)
train_df = main_train_df[~eval_mask].reset_index(drop=True)

train_ds     = BehavioralDataset(train_df)
eval_ds      = BehavioralDataset(eval_df)

def _slice(ds, task_names, uids):
    sub = ds.df[ds.df["task_name"].isin(task_names) & ds.df["uid"].isin(uids)].copy()
    return BehavioralDataset(sub)

stim_gen_ds  = _slice(full_ds, cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = _slice(full_ds, cfg.VAL_TASKS,   train_uids)
joint_gen_ds = _slice(full_ds, cfg.VAL_TASKS,   test_uids)

print(f"\nCell split: train={len(train_df)} / eval={len(eval_df)}")

# ---------------------------------------------------------------------------
# CLIP feature cache (shared)
# ---------------------------------------------------------------------------
from dlbt.agents.dlbt import DlbtAgent as _DlbtAgent
_tmp = _DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                  device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    _tmp.load_cache(str(cache_path))
else:
    _tmp.precompute_features(list(refs_dict.values()))
    _tmp.save_cache(str(cache_path))
frozen_clip = {uid: feat.clone() for uid, feat in _tmp._cache.items()}
del _tmp

# ---------------------------------------------------------------------------
# Empirical truth
# ---------------------------------------------------------------------------
_emp_lookup: dict = {}
for row in full_ds.df.itertuples(index=False):
    total = row.count_0 + row.count_1
    p     = row.count_1 / total if total > 0 else np.nan
    _emp_lookup[(row.uid, row.task_name)] = (p, total)

def emp_p(uid, tn): v = _emp_lookup.get((uid, tn)); return v[0] if v else np.nan
def emp_n(uid, tn): v = _emp_lookup.get((uid, tn)); return v[1] if v else 0

# ---------------------------------------------------------------------------
# Noise floors
# ---------------------------------------------------------------------------
noise_floors = {
    "train":     train_ds.noise_floor(),
    "eval":      eval_ds.noise_floor(),
    "stim_gen":  stim_gen_ds.noise_floor(),
    "task_gen":  task_gen_ds.noise_floor(),
    "joint_gen": joint_gen_ds.noise_floor(),
}
print(f"Noise floors: {noise_floors}")

# ---------------------------------------------------------------------------
# Helper: collect predictions after training
# ---------------------------------------------------------------------------
cond_ds = [
    ("train", train_ds,     cfg.TRAIN_TASKS),
    ("eval",  eval_ds,      cfg.TRAIN_TASKS),
    ("stim",  stim_gen_ds,  cfg.TRAIN_TASKS),
    ("task",  task_gen_ds,  cfg.VAL_TASKS),
    ("joint", joint_gen_ds, cfg.VAL_TASKS),
]

def collect_preds(agent: FlexAgent):
    preds = {cond: {} for cond, _, _ in cond_ds}
    agent.eval()
    for cond, ds, _ in cond_ds:
        for task_name, group in ds.iter_tasks():
            task       = TASKS[task_name]
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
            totals     = np.array([emp_n(r.uid, task_name) for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
            preds[cond][task_name] = {
                "pred": pred, "true": true_p, "totals": totals,
                "uids": [r.uid for r in batch_refs],
            }
    return preds


def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


# ---------------------------------------------------------------------------
# Training loop over variants
# ---------------------------------------------------------------------------
VARIANTS = [
    dict(cov_type="diag", run_tag="flex_diag_frozen"),
    dict(cov_type="full", run_tag="flex_full_frozen"),
]

for var in VARIANTS:
    cov_type = var["cov_type"]
    run_tag  = var["run_tag"]
    print(f"\n{'='*60}")
    print(f"FlexAgent  cov_type={cov_type!r}  run_tag={run_tag}")

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    agent = FlexAgent(
        freeze_encoder = True,
        n_mc_samples   = cfg.N_MC,
        device         = device,
        mapper_hidden  = cfg.MAPPER_HIDDEN,
        cov_type       = cov_type,
    )
    print(f"  {agent.param_summary()}")

    # Share the pre-computed CLIP features
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}

    result = train_flex(
        agent, train_ds, eval_ds, refs_dict,
        n_epochs  = cfg.N_EPOCHS_PHASE1,
        lr        = cfg.LR,
        patience  = cfg.PATIENCE_PHASE1,
        extra_val_datasets = {
            "stim_gen":  stim_gen_ds,
            "task_gen":  task_gen_ds,
            "joint_gen": joint_gen_ds,
        },
        kl_weight = 0.0,   # pure NLL; add regularisation if beliefs collapse
        prior_std = 1.0,
    )
    print(f"  best epoch: {result.best_epoch}  eval_mse: {result.best_val_mse:.4f}")

    preds = collect_preds(agent)

    # Save agent weights
    torch.save(agent.state_dict(), cfg.RESULTS_DIR / f"agent_{run_tag}.pt")
    end_state = result.end_state if result.end_state else agent.state_dict()
    torch.save(end_state, cfg.RESULTS_DIR / f"agent_{run_tag}_end.pt")

    curves = dict(
        train_nlls  = result.train_nlls,
        eval_nlls   = result.val_nlls,
        train_mses  = result.train_mses,
        eval_mses   = result.val_mses,
        stim_nlls   = result.extra_val_nlls.get("stim_gen",  []),
        task_nlls   = result.extra_val_nlls.get("task_gen",  []),
        joint_nlls  = result.extra_val_nlls.get("joint_gen", []),
        stim_mses   = result.extra_val_mses.get("stim_gen",  []),
        task_mses   = result.extra_val_mses.get("task_gen",  []),
        joint_mses  = result.extra_val_mses.get("joint_gen", []),
    )

    results = dict(
        model_label    = f"FlexAgent ({cov_type} Σ, frozen CLIP)",
        run_tag        = run_tag,
        cov_type       = cov_type,
        n_seeds        = 1,
        seeds          = cfg.SEEDS,
        phase_boundary = 0,
        best_epoch     = result.best_epoch,
        noise_floors   = noise_floors,
        curves         = curves,
        dlbt           = preds,   # same key as DLBT results for analysis.py compatibility
        slda           = {},
        train_uids     = train_uids,
        test_uids      = test_uids,
        main_uids      = main_uids,
        probe_uids     = probe_uids,
        eval_uids      = set(eval_df["uid"].unique()),
        diag           = diag,
        eval_cell_frac = cfg.EVAL_CELL_FRAC,
    )

    out = cfg.RESULTS_DIR / f"results_{run_tag}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved -> {out}")

print("\nDone.")
