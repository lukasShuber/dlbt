"""
Stage-2 fine-tuning of CLIP attention pooling for the sklearn-based SLDA baseline.

Context
-------
In experiments with FREEZE_ENCODER = False, DLBT runs a two-phase procedure:
  Phase 1 — train mapper with frozen CLIP (fast, CPU-friendly).
  Phase 2 — fine-tune attnpool jointly with mapper through the DLBT NLL.

To keep the comparison fair, SLDA should mirror this:
  Phase 1 — fit per-task ridge decoders on frozen CLIP features (standard fit_slda()).
  Phase 2 — fine-tune attnpool through the SLDA binomial NLL with decoders FIXED.

This module provides `finetune_slda_attnpool()` for Phase 2.

Loss
----
For each training cell (image x, task t, counts c0/c1):

    z_xt  = (f_φ(x) − μ_t) / σ_t  ·  w_t  +  b_t
    p_xt  = σ(z_xt / τ_t)
    ℓ_xt  = − [c1 · log p_xt  +  c0 · log(1 − p_xt)]

where (μ_t, σ_t, w_t, b_t, τ_t) are the FIXED Stage-1 sklearn artifacts and
φ (attnpool parameters) is the only trainable parameter.

The backbone (everything before attnpool) is kept frozen and its output is
pre-cached, so each training step is cheap: one attnpool forward per unique
image in the batch.
"""

from __future__ import annotations

import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SldaAttnpoolResult:
    best_epoch:    int
    best_val_nll:  float
    train_nll:     list   # per-epoch training NLL (unnormalised sum)
    val_nll:       list   # per-epoch validation NLL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_decoder_tensors(
    scalers: dict,
    models:  dict,
    temps:   dict,
    tasks:   list[str],
    device:  torch.device,
    feature_dim: int = 1024,
) -> dict[str, dict]:
    """
    Convert sklearn Stage-1 artifacts to torch tensors for each task.

    Returns a dict  task_name → {mean, std, w, b, tau}  (all on `device`).
    Tasks missing from models are omitted.
    """
    dec = {}
    for t in tasks:
        if t not in models:
            continue
        scaler = scalers[t]
        model  = models[t]
        tau    = float(temps[t])

        # StandardScaler may have with_mean/with_std = False for small tasks;
        # fall back to identity transform in that case.
        if scaler.with_mean and scaler.mean_ is not None:
            mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        else:
            mean = torch.zeros(feature_dim, device=device)

        if scaler.with_std and scaler.scale_ is not None:
            std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
            std = std.clamp(min=1e-8)
        else:
            std = torch.ones(feature_dim, device=device)

        w = torch.tensor(model.coef_,          dtype=torch.float32, device=device)  # [D]
        b = torch.tensor(float(model.intercept_), dtype=torch.float32, device=device)

        dec[t] = {"mean": mean, "std": std, "w": w, "b": b, "tau": tau}
    return dec


def _nll_on_dataset(
    agent:    DlbtAgent,
    ds:       BehavioralDataset,
    dec:      dict,
    refs_map: dict[str, ImageRef],
    device:   torch.device,
) -> float:
    """Compute total binomial NLL over a dataset (no gradient)."""
    agent.eval()
    total_nll = 0.0
    with torch.no_grad():
        uid_groups = ds.df.groupby("uid")
        for uid, group in uid_groups:
            if uid not in refs_map:
                continue
            ref   = refs_map[uid]
            feats = agent._encode([ref])          # [1, D]
            f     = feats[0]                      # [D]
            for row in group.itertuples(index=False):
                t_dec = dec.get(row.task_name)
                if t_dec is None:
                    continue
                f_sc = (f - t_dec["mean"]) / t_dec["std"]
                z    = f_sc @ t_dec["w"] + t_dec["b"]
                p    = torch.sigmoid(z / t_dec["tau"]).clamp(1e-7, 1 - 1e-7)
                total_nll += -(row.count_1 * torch.log(p)
                               + row.count_0 * torch.log(1 - p)).item()
    return total_nll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finetune_slda_attnpool(
    agent:    DlbtAgent,
    scalers:  dict,
    models:   dict,
    temps:    dict,
    train_ds: BehavioralDataset,
    eval_ds:  BehavioralDataset,
    refs_dict: Dict[str, ImageRef],
    n_epochs:  int   = 3000,
    patience:  int   = 50,
    lr:        float = 1e-5,
    batch_size: int  = 128,
) -> SldaAttnpoolResult:
    """
    Fine-tune only the CLIP attention-pooling layer through the SLDA NLL.

    The per-task ridge decoders (scalers, models, temps) are FIXED throughout.
    Only `agent.encoder.attnpool` parameters are updated.

    The backbone (everything before attnpool) must already be cached in
    `agent._backbone_cache`; call `agent.precompute_backbone_features()`
    before this function if needed.

    The agent is modified in-place.  Best weights (lowest val NLL) are
    restored before returning.

    Args
    ----
    agent      : DlbtAgent with freeze_encoder=False and a populated backbone
                 cache.
    scalers, models, temps : Stage-1 sklearn artifacts (from _fit_slda()).
    train_ds   : BehavioralDataset for gradient updates (main trials, 90 %).
    eval_ds    : BehavioralDataset for early stopping (held-out 10 %).
    refs_dict  : uid → ImageRef for all images.
    n_epochs   : maximum training epochs.
    patience   : early-stopping patience (epochs without val improvement).
    lr         : Adam learning rate for attnpool parameters.
    batch_size : number of (uid, task) cells per gradient step.
    """
    device = agent.device

    # ---- Freeze everything except attnpool --------------------------------
    for p in agent.parameters():
        p.requires_grad_(False)
    for p in agent.encoder.attnpool.parameters():
        p.requires_grad_(True)

    optimizer = optim.Adam(agent.encoder.attnpool.parameters(), lr=lr)

    # ---- Convert sklearn decoders to torch tensors ------------------------
    tasks     = train_ds.df["task_name"].unique().tolist()
    dec       = _build_decoder_tensors(scalers, models, temps, tasks,
                                        device, agent.feature_dim)

    # refs_map: uid -> ImageRef (needed by _encode which uses backbone cache)
    refs_map  = {uid: refs_dict[uid] for uid in refs_dict}

    # ---- Pre-shuffle cell records for batching ----------------------------
    train_cells = train_ds.df.to_dict("records")   # list of dicts

    # ---- Training loop ----------------------------------------------------
    best_val_nll  = float("inf")
    best_state    = {k: v.cpu().clone()
                     for k, v in agent.encoder.attnpool.state_dict().items()}
    no_improve    = 0
    train_nll_log = []
    val_nll_log   = []

    for epoch in range(n_epochs):
        agent.train()
        rng  = np.random.default_rng(epoch)
        idx  = rng.permutation(len(train_cells))

        epoch_nll = 0.0
        n_batches = 0

        for start in range(0, len(train_cells), batch_size):
            batch_idx   = idx[start : start + batch_size]
            batch_cells = [train_cells[int(i)] for i in batch_idx]

            # Collect unique UIDs in this batch
            batch_uids  = list({c["uid"] for c in batch_cells
                                 if c["uid"] in refs_map})
            if not batch_uids:
                continue

            batch_refs  = [refs_map[u] for u in batch_uids]
            uid_to_idx  = {u: i for i, u in enumerate(batch_uids)}

            optimizer.zero_grad()

            # Forward: attnpool on backbone cache → CLIP features
            feats = agent._encode(batch_refs)   # [n_uids, D]

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_terms = 0

            for cell in batch_cells:
                uid   = cell["uid"]
                tname = cell["task_name"]
                if uid not in uid_to_idx or tname not in dec:
                    continue
                f     = feats[uid_to_idx[uid]]          # [D]
                t_dec = dec[tname]
                f_sc  = (f - t_dec["mean"]) / t_dec["std"]
                z     = f_sc @ t_dec["w"] + t_dec["b"]
                p     = torch.sigmoid(z / t_dec["tau"]).clamp(1e-7, 1 - 1e-7)
                c1    = float(cell["count_1"])
                c0    = float(cell["count_0"])
                loss  = loss + (-(c1 * torch.log(p) + c0 * torch.log(1 - p)))
                n_terms += 1

            if n_terms == 0:
                continue

            loss.backward()
            optimizer.step()
            epoch_nll += loss.item()
            n_batches += 1

        train_nll_log.append(epoch_nll)

        # Validation
        val_nll = _nll_on_dataset(agent, eval_ds, dec, refs_map, device)
        val_nll_log.append(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone()
                            for k, v in agent.encoder.attnpool.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best weights
    agent.encoder.attnpool.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()}
    )
    agent.eval()

    return SldaAttnpoolResult(
        best_epoch   = best_epoch,
        best_val_nll = best_val_nll,
        train_nll    = train_nll_log,
        val_nll      = val_nll_log,
    )
