"""
03_ebm/run.py — Energy-Based SEU agent fit on real human data.

Architecture:
    CLIP(x) [1024] → compress [128] ─┐
                                      concat [144] → hidden [256] → f(p̃,x) ∈ ℝ
                  p̃ [16] ────────────┘

Choice probability:
    P(yes | x, T) ≈ Σ_i  softmax(f(p̃_1,x),...,f(p̃_N,x))_i · I[⟨p̃_i,Δu⟩>0]

Uses the identical 10% cell-level eval split as 01_fit (same RNG seed).

Run from repo root:
    python experiments/behavior/run0/03_ebm/run.py
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
from dlbt.agents.ebm import EBMAgent
from dlbt.training.train_ebm import train_ebm

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ---------------------------------------------------------------------------
# Load stimuli & behavioural data
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
# Cell-level eval split — identical to 01_fit (same RNG seed)
# ---------------------------------------------------------------------------
main_train_mask = (
    full_ds.df["uid"].isin(main_uids) &
    full_ds.df["task_name"].isin(cfg.TRAIN_TASKS)
)
main_train_df = full_ds.df[main_train_mask].copy().reset_index(drop=True)

rng_split = np.random.default_rng(cfg.SEED)
n_eval    = max(1, int(len(main_train_df) * cfg.EVAL_CELL_FRAC))
eval_idx  = rng_split.choice(len(main_train_df), size=n_eval, replace=False)
eval_mask = np.zeros(len(main_train_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df  = main_train_df[eval_mask].reset_index(drop=True)
train_df = main_train_df[~eval_mask].reset_index(drop=True)

train_ds = BehavioralDataset(train_df)
eval_ds  = BehavioralDataset(eval_df)

def _slice(ds, task_names, uids):
    sub = ds.df[ds.df["task_name"].isin(task_names) & ds.df["uid"].isin(uids)].copy()
    return BehavioralDataset(sub)

stim_gen_ds  = _slice(full_ds, cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = _slice(full_ds, cfg.VAL_TASKS,   train_uids)
joint_gen_ds = _slice(full_ds, cfg.VAL_TASKS,   test_uids)

print(f"\nCell split: train={len(train_df)} / eval={len(eval_df)}")
for name, ds in [("stim_gen", stim_gen_ds), ("task_gen", task_gen_ds),
                 ("joint_gen", joint_gen_ds)]:
    print(f"  {name:12s}: {ds}")

# ---------------------------------------------------------------------------
# EBM agent
# ---------------------------------------------------------------------------
agent = EBMAgent(
    freeze_encoder = True,
    n_mc_samples   = cfg.N_MC_SAMPLES,
    device         = device,
    compress_dim   = cfg.COMPRESS_DIM,
    hidden_dim     = cfg.HIDDEN_DIM,
    mc_seed        = cfg.MC_SEED,
)
print(f"\n{agent.param_summary()}")

# Optionally bootstrap CLIP cache from the 01_fit cache (saves re-encoding)
clip_cache_path = Path(cfg.CACHE_PATH)
if clip_cache_path.exists():
    print(f"Loading CLIP feature cache from {clip_cache_path}")
    agent.load_cache(str(clip_cache_path))

# ---------------------------------------------------------------------------
# Empirical truth lookup
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
# Train
# ---------------------------------------------------------------------------
print(f"\nTraining EBM  (N_MC={cfg.N_MC_SAMPLES}, "
      f"compress={cfg.COMPRESS_DIM}, hidden={cfg.HIDDEN_DIM})")
print(f"  inner_batch_size={cfg.INNER_BATCH_SIZE}, lr={cfg.LR}, "
      f"patience={cfg.PATIENCE}, ent_weight={cfg.ENT_WEIGHT}")

result = train_ebm(
    agent, train_ds, eval_ds, refs_dict,
    n_epochs         = cfg.N_EPOCHS,
    lr               = cfg.LR,
    patience         = cfg.PATIENCE,
    inner_batch_size = cfg.INNER_BATCH_SIZE,
    grad_clip        = cfg.GRAD_CLIP,
    ent_weight       = cfg.ENT_WEIGHT,
    extra_val_datasets = {
        "stim_gen":  stim_gen_ds,
        "task_gen":  task_gen_ds,
        "joint_gen": joint_gen_ds,
    },
)
print(f"\nBest epoch: {result.best_epoch}  eval_mse: {result.best_val_mse:.4f}")
print(f"Final ESS/N:  {result.train_ess[-1]:.3f}  (1.0 = uniform, 1/N = collapsed)")
print(f"Final H(w):   {result.train_entropies[-1]:.3f}  "
      f"(max = {float(__import__('math').log(cfg.N_MC_SAMPLES)):.2f})")

# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------
cond_ds = [
    ("train", train_ds,     cfg.TRAIN_TASKS),
    ("eval",  eval_ds,      cfg.TRAIN_TASKS),
    ("stim",  stim_gen_ds,  cfg.TRAIN_TASKS),
    ("task",  task_gen_ds,  cfg.VAL_TASKS),
    ("joint", joint_gen_ds, cfg.VAL_TASKS),
]

preds = {cond: {} for cond, _, _ in cond_ds}
agent.eval()
for cond, ds, _ in cond_ds:
    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
        totals     = np.array([emp_n(r.uid, task_name) for r in batch_refs])
        # inner-batch for memory safety during collection
        preds_list = []
        for b0 in range(0, len(batch_refs), 64):
            refs_b = batch_refs[b0:b0+64]
            with torch.no_grad():
                p_b = agent.choice_probs(refs_b, task)[:, 1].cpu().numpy()
            preds_list.append(p_b)
        pred = np.concatenate(preds_list)
        preds[cond][task_name] = {
            "pred": pred, "true": true_p, "totals": totals,
            "uids": [r.uid for r in batch_refs],
        }

# ---------------------------------------------------------------------------
# Save agent weights
# ---------------------------------------------------------------------------
agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
torch.save(agent.state_dict(), agent_path)
end_state = result.end_state if result.end_state else agent.state_dict()
torch.save(end_state, cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_end.pt")
print(f"\nSaved agent -> {agent_path}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
curves = dict(
    train_nlls       = result.train_nlls,
    eval_nlls        = result.val_nlls,
    train_mses       = result.train_mses,
    eval_mses        = result.val_mses,
    train_ess        = result.train_ess,
    train_entropies  = result.train_entropies,
    stim_nlls        = result.extra_val_nlls.get("stim_gen",  []),
    task_nlls        = result.extra_val_nlls.get("task_gen",  []),
    joint_nlls       = result.extra_val_nlls.get("joint_gen", []),
    stim_mses        = result.extra_val_mses.get("stim_gen",  []),
    task_mses        = result.extra_val_mses.get("task_gen",  []),
    joint_mses       = result.extra_val_mses.get("joint_gen", []),
)

results = dict(
    model_label  = (f"EBM (N={cfg.N_MC_SAMPLES}, "
                    f"C={cfg.COMPRESS_DIM}, H={cfg.HIDDEN_DIM})"),
    run_tag      = cfg.RUN_TAG,
    best_epoch   = result.best_epoch,
    noise_floors = noise_floors,
    curves       = curves,
    dlbt         = preds,    # same key as 01_fit for analysis.py compatibility
    slda         = {},
    train_uids   = train_uids,
    test_uids    = test_uids,
    main_uids    = main_uids,
    probe_uids   = probe_uids,
    diag         = diag,
    ebm_config   = dict(
        n_mc_samples = cfg.N_MC_SAMPLES,
        compress_dim = cfg.COMPRESS_DIM,
        hidden_dim   = cfg.HIDDEN_DIM,
        mc_seed      = cfg.MC_SEED,
        ent_weight   = cfg.ENT_WEIGHT,
    ),
)

out = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(out, "wb") as f:
    pickle.dump(results, f)
print(f"Saved results -> {out}")
