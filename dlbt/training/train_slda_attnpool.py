"""
Phase 2 fine-tuning of CLIP attention pooling for the SLDA baseline.

Pipeline
--------
  Phase 1 — fit per-task LogReg decoders on frozen CLIP features
             (fit_slda_logreg in train_slda.py).
  Phase 2 — fine-tune attnpool through those FIXED decoders (this module).
             Phase-1 scalers/models/use_base are kept unchanged after Phase 2;
             fine-tuned features are used directly for probe prediction.

Loss (Phase 2)
--------------
For each training cell (image x, task t, counts c0/c1):

    f_φ(x) = attnpool(backbone(x))              [trainable attnpool only]
    z_xt   = (f_φ(x) − μ_t) / σ_t  ·  w_t  +  b_t
    p_xt   = σ(z_xt)
    ℓ_xt   = − [c1 · log p_xt  +  c0 · log(1 − p_xt)]

where (μ_t, σ_t, w_t, b_t) are the FIXED Phase-1 logistic decoder artifacts and
φ (attnpool parameters) is the only thing being updated.

The backbone (everything before attnpool) is frozen and pre-cached in
agent._backbone_cache, so each training step is cheap: one attnpool forward
per unique image in the batch.

Epoch hook
----------
Pass `epoch_hook(epoch, agent) -> any` to evaluate the model (e.g. probe cMSE)
periodically during training.  Results accumulate in SldaAttnpoolResult.
"""

from __future__ import annotations

import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SldaAttnpoolResult:
    best_epoch:   int
    best_val_nll: float
    train_nll:    list   # per-epoch training NLL (unnormalised sum)
    val_nll:      list   # per-epoch validation NLL
    hook_epochs:  list = dataclasses.field(default_factory=list)
    hook_results: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_decoder_tensors(
    scalers: dict,
    models:  dict,
    tasks:   list[str],
    device:  torch.device,
    feature_dim: int = 1024,
) -> dict[str, dict]:
    """
    Convert Phase-1 sklearn artifacts to torch tensors.

    Returns task_name → {mean, std, w, b} — no temperature, which is fit
    separately in Phase 3.
    """
    dec = {}
    for t in tasks:
        if t not in models or t not in scalers:
            continue
        scaler = scalers[t]
        model  = models[t]

        if scaler.with_mean and scaler.mean_ is not None:
            mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        else:
            mean = torch.zeros(feature_dim, device=device)

        if scaler.with_std and scaler.scale_ is not None:
            std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
            std = std.clamp(min=1e-8)
        else:
            std = torch.ones(feature_dim, device=device)

        # LogisticRegressionCV: coef_ is [1, D], intercept_ is [1]
        # Ridge:                coef_ is [D],    intercept_ is scalar
        coef = model.coef_
        if coef.ndim == 2:
            coef = coef[0]
        intercept = model.intercept_
        if hasattr(intercept, "__len__"):
            intercept = intercept[0]

        w = torch.tensor(coef,            dtype=torch.float32, device=device)
        b = torch.tensor(float(intercept), dtype=torch.float32, device=device)

        dec[t] = {"mean": mean, "std": std, "w": w, "b": b}
    return dec


def _nll_on_dataset(
    agent:    SldaAgent,
    ds:       BehavioralDataset,
    dec:      dict,
    refs_map: dict[str, ImageRef],
    device:   torch.device,
) -> float:
    """Compute total binomial NLL over a dataset (no gradient, τ=1)."""
    agent.eval()
    total_nll = 0.0
    with torch.no_grad():
        uid_groups = ds.df.groupby("uid")
        for uid, group in uid_groups:
            if uid not in refs_map:
                continue
            ref   = refs_map[uid]
            feats = agent._encode([ref])   # [1, D] — uses backbone cache + attnpool
            f     = feats[0]               # [D]
            for row in group.itertuples(index=False):
                t_dec = dec.get(row.task_name)
                if t_dec is None:
                    continue
                f_sc = (f - t_dec["mean"]) / t_dec["std"]
                z    = f_sc @ t_dec["w"] + t_dec["b"]
                p    = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)
                total_nll += -(row.count_1 * torch.log(p)
                               + row.count_0 * torch.log(1 - p)).item()
    return total_nll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finetune_slda_attnpool(
    agent:       SldaAgent,
    scalers:     dict,
    models:      dict,
    train_ds:    BehavioralDataset,
    eval_ds:     BehavioralDataset,
    refs_dict:   Dict[str, ImageRef],
    n_epochs:    int   = 3000,
    patience:    int   = 50,
    lr:          float = 1e-5,
    batch_size:  int   = 128,
    epoch_hook   = None,
    eval_every:  int   = 10,
) -> SldaAttnpoolResult:
    """
    Phase 2: fine-tune only the CLIP attention-pooling layer through fixed
    Phase-1 logistic decoders.

    The agent must have freeze_encoder=False and a populated _backbone_cache
    (call agent.precompute_backbone_features() before this function).

    The agent is modified in-place.  Best weights (lowest val NLL) are
    restored before returning.

    Args
    ----
    agent      : SldaAgent with freeze_encoder=False and backbone cache ready.
    scalers    : Phase-1 StandardScaler per task.
    models     : Phase-1 LogisticRegressionCV per task.
    train_ds   : BehavioralDataset for gradient updates.
    eval_ds    : BehavioralDataset for early stopping.
    refs_dict  : uid → ImageRef for all images.
    n_epochs   : maximum training epochs.
    patience   : early-stopping patience (epochs without val improvement).
    lr         : Adam learning rate for attnpool parameters.
    batch_size : number of (uid, task) cells per gradient step.
    epoch_hook : optional callable(epoch: int, agent: SldaAgent) → any.
                 Called every eval_every epochs (after val NLL, agent in eval mode).
                 Results stored in SldaAttnpoolResult.hook_results.
    eval_every : interval (in epochs) between epoch_hook calls.
    """
    device = agent.device

    # ---- Freeze everything, then re-enable only attnpool ------------------
    for p in agent.parameters():
        p.requires_grad_(False)
    for p in agent.encoder.attnpool.parameters():
        p.requires_grad_(True)

    optimizer = optim.Adam(agent.encoder.attnpool.parameters(), lr=lr)

    # ---- Convert sklearn decoders to torch tensors (no temperature) -------
    tasks = train_ds.df["task_name"].unique().tolist()
    dec   = _build_decoder_tensors(scalers, models, tasks,
                                   device, agent.feature_dim)

    refs_map    = dict(refs_dict)
    train_cells = train_ds.df.to_dict("records")

    # ---- Training loop ----------------------------------------------------
    best_val_nll  = float("inf")
    best_epoch    = 0
    best_state    = {k: v.cpu().clone()
                     for k, v in agent.encoder.attnpool.state_dict().items()}
    no_improve    = 0
    train_nll_log  = []
    val_nll_log    = []
    hook_epochs_log  = []
    hook_results_log = []

    pbar = tqdm(range(n_epochs), desc="slda-attnpool", unit="epoch")
    for epoch in pbar:
        agent.train()
        rng = np.random.default_rng(epoch)
        idx = rng.permutation(len(train_cells))

        epoch_nll = 0.0

        for start in range(0, len(train_cells), batch_size):
            batch_idx   = idx[start : start + batch_size]
            batch_cells = [train_cells[int(i)] for i in batch_idx]

            # Unique UIDs in this batch
            batch_uids = list({c["uid"] for c in batch_cells
                               if c["uid"] in refs_map})
            if not batch_uids:
                continue

            batch_refs  = [refs_map[u] for u in batch_uids]
            uid_to_idx  = {u: i for i, u in enumerate(batch_uids)}

            optimizer.zero_grad()

            # Forward: backbone cache → attnpool → CLIP features
            feats = agent._encode(batch_refs)   # [n_uids, D]

            # Accumulate NLL over cells in this batch
            batch_loss: torch.Tensor | None = None
            for cell in batch_cells:
                uid   = cell["uid"]
                tname = cell["task_name"]
                if uid not in uid_to_idx or tname not in dec:
                    continue
                f     = feats[uid_to_idx[uid]]
                t_dec = dec[tname]
                f_sc  = (f - t_dec["mean"]) / t_dec["std"]
                z     = f_sc @ t_dec["w"] + t_dec["b"]
                p     = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)   # τ = 1
                c1    = float(cell["count_1"])
                c0    = float(cell["count_0"])
                term  = -(c1 * torch.log(p) + c0 * torch.log(1 - p))
                batch_loss = term if batch_loss is None else batch_loss + term

            if batch_loss is None:
                continue

            batch_loss.backward()
            optimizer.step()
            epoch_nll += batch_loss.item()

        train_nll_log.append(epoch_nll)

        # Validation
        val_nll = _nll_on_dataset(agent, eval_ds, dec, refs_map, device)
        val_nll_log.append(val_nll)

        pbar.set_postfix(
            train=f"{epoch_nll:.1f}",
            val  =f"{val_nll:.1f}",
        )

        # Epoch hook  (agent is in eval mode after _nll_on_dataset)
        if epoch_hook is not None and epoch % eval_every == 0:
            hook_result = epoch_hook(epoch, agent)
            hook_epochs_log.append(epoch)
            hook_results_log.append(hook_result)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone()
                            for k, v in agent.encoder.attnpool.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop epoch {epoch}  (best: {best_epoch})")
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
        hook_epochs  = hook_epochs_log,
        hook_results = hook_results_log,
    )
