"""
Fitting procedure for the logistic-regression SLDA baseline.

Pipeline (called from experiment run scripts)
---------------------------------------------
Phase 1  fit_slda_logreg()
    Per task: fit L2 LogisticRegressionCV on frozen (or fine-tuned) CLIP
    features.  Returns per-task scalers, models, and a use_base flag derived
    from held-out val-cell model selection (fitted model vs. P=0.5 base).

Phase 2  finetune_slda_attnpool()   [train_slda_attnpool.py]
    Fine-tune the CLIP attention-pooling layer through the FIXED Phase-1
    logistic decoders.  Only runs when FREEZE_ENCODER_SLDA = False.
    Phase-1 scalers/models/use_base are kept unchanged; fine-tuned features
    are used for probe prediction.

Probe prediction  slda_probe_matrix()
    Build the [n_probe × n_tasks] prediction matrix from fitted models.
    Tasks not fitted or where model selection chose base → predict 0.5.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from dlbt.data.dataset import BehavioralDataset


# ---------------------------------------------------------------------------
# Phase 1 / Phase 3 — per-task logistic regression
# ---------------------------------------------------------------------------

def fit_slda_logreg(
    tasks:        List[str],
    train_ds:     BehavioralDataset,
    val_ds:       BehavioralDataset,
    clip_features: Dict[str, "torch.Tensor"],
    Cs:           List[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    max_iter:     int         = 1000,
    cv:           int         = 3,
) -> Tuple[dict, dict, dict]:
    """
    Fit per-task L2 logistic regression.

    Args
    ----
    tasks         : ordered list of task names.
    train_ds      : 90 % training cells.
    val_ds        : 10 % validation cells (cell-level split).
    clip_features : uid → [D] tensor (frozen or fine-tuned CLIP features).
    Cs            : regularisation grid for LogisticRegressionCV.
    max_iter      : solver iteration limit.
    cv            : number of internal CV folds for C selection.

    Returns
    -------
    scalers  : task_name → fitted StandardScaler
    models   : task_name → fitted LogisticRegressionCV
    use_base : task_name → bool  (True ↔ base model wins on val)

    Tasks with < 2 training images, all-same labels, or fitting errors
    are omitted from scalers/models; probe predictions fall back to 0.5
    (see slda_probe_matrix).
    """
    scalers:  dict = {}
    models:   dict = {}
    use_base: dict = {}

    for task_name in tasks:
        train_grp = train_ds.df[train_ds.df["task_name"] == task_name]
        uids_tr   = [uid for uid in train_grp["uid"].tolist()
                     if uid in clip_features]

        if len(uids_tr) < 2:
            continue

        # ---- Build (X, y, w) — one row per non-zero outcome per cell --------
        X_list, y_list, w_list = [], [], []
        for row in train_grp[train_grp["uid"].isin(uids_tr)].itertuples(index=False):
            feat = clip_features[row.uid].cpu().numpy()
            c0, c1 = int(row.count_0), int(row.count_1)
            if c1 > 0:
                X_list.append(feat); y_list.append(1); w_list.append(c1)
            if c0 > 0:
                X_list.append(feat); y_list.append(0); w_list.append(c0)

        if not X_list or len(set(y_list)) < 2:
            continue

        X = np.array(X_list)
        y = np.array(y_list)
        w = np.array(w_list, dtype=float)

        # ---- Scaler fit on unique feature vectors ---------------------------
        X_unique = np.array([clip_features[uid].cpu().numpy() for uid in uids_tr])
        scaler   = StandardScaler()
        scaler.fit(X_unique)
        X_sc = scaler.transform(X)

        # ---- Fit logistic regression -----------------------------------------
        try:
            model = LogisticRegressionCV(
                Cs=list(Cs), max_iter=max_iter,
                solver="lbfgs", cv=cv,
            )
            model.fit(X_sc, y, sample_weight=w)
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"  WARNING: LogReg fitting FAILED for task '{task_name}'")
            print(f"  Exception: {type(e).__name__}: {e}")
            print(f"  Task will fall back to P=0.5 for all probe images.")
            print(f"{'!'*60}\n")
            continue

        # ---- Model selection on val cells ------------------------------------
        val_grp  = val_ds.df[val_ds.df["task_name"] == task_name]
        uids_val = [uid for uid in val_grp["uid"].tolist()
                    if uid in clip_features]
        if len(uids_val) >= 1:
            val_sub    = val_grp[val_grp["uid"].isin(uids_val)]
            tot_val    = (val_sub["count_0"] + val_sub["count_1"]).values.astype(float)
            p_obs_val  = val_sub["count_1"].values / np.clip(tot_val, 1, None)
            X_val_sc   = scaler.transform(
                np.array([clip_features[uid].cpu().numpy() for uid in uids_val]))
            p_pred_val = model.predict_proba(X_val_sc)[:, 1]
            fitted_mse = float(np.mean((p_pred_val - p_obs_val) ** 2))
            base_mse   = float(np.mean((0.5 - p_obs_val) ** 2))
            use_base[task_name] = base_mse < fitted_mse
        else:
            use_base[task_name] = False   # no val data → use fitted model

        scalers[task_name] = scaler
        models[task_name]  = model

    return scalers, models, use_base


# ---------------------------------------------------------------------------
# Probe matrix
# ---------------------------------------------------------------------------

def slda_probe_matrix(
    scalers:        dict,
    models:         dict,
    use_base:       dict,
    probe_features: Dict[str, "np.ndarray"],
    tasks_ordered:  List[str],
    uid_to_row:     Dict[str, int],
    n_probe:        int,
) -> np.ndarray:
    """
    Build [n_probe × n_tasks] prediction matrix from fitted SLDA models.

    Args
    ----
    scalers        : task_name → StandardScaler  (from fit_slda_logreg).
    models         : task_name → LogisticRegressionCV  (from fit_slda_logreg).
    use_base       : task_name → bool  (True ↔ base wins on val).
    probe_features : uid → np.ndarray [D]  (CLIP features for probe images).
    tasks_ordered  : column ordering for the output matrix.
    uid_to_row     : uid → row index in the output matrix.
    n_probe        : number of probe images (rows).

    Fallback (base model, P=0.5) is used for:
      - Tasks where fitting was skipped (< 2 images, all-same labels, exception).
      - Tasks where model selection chose the base.
    """
    n_tasks = len(tasks_ordered)
    pred    = np.full((n_probe, n_tasks), 0.5)   # default: base model

    probe_uids = [uid for uid in uid_to_row if uid in probe_features]
    if not probe_uids:
        return pred

    probe_X_np = np.array([probe_features[uid] for uid in probe_uids])

    for j, task_name in enumerate(tasks_ordered):
        if task_name not in models or use_base.get(task_name, False):
            continue   # stays 0.5
        X_sc   = scalers[task_name].transform(probe_X_np)
        p_pred = models[task_name].predict_proba(X_sc)[:, 1]
        for i_p, uid in enumerate(probe_uids):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_pred[i_p])

    return pred
