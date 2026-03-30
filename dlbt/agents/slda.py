"""
SldaAgent: Supervised Linear Decoder Agent (SLDA) baseline.

For each task, fits a separate linear decoder (with bias) from frozen CLIP
RN50 features to optimal-action labels via ordinary least squares.

At test time, decoder logits are passed through a softmax with a learnable
temperature (fitted on the validation set via a simple line search).

The CLIP encoder is always frozen for SLDA — there is no end-to-end
training signal.  This is the key contrast with DlbtAgent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dlbt.agents.base import Agent
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class SldaAgent(Agent):
    """
    SLDA agent.

    For each task t, a linear decoder W_t (shape [1024+1, 2] with bias)
    is fitted by solving the least-squares problem:
        min_W  ||X W - Y||^2
    where:
        X: [N, 1024] CLIP features (with a prepended bias column)
        Y: [N, 2]    one-hot optimal-action labels
                     label[i, a] = 1 if task.delta_u[image_refs[i].latent_state] > 0 else 0

    Choice probabilities use softmax over the decoder logits, with a
    per-task temperature (default 1.0, tunable on val set via fit_temperature).
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device

        # ---- CLIP RN50 encoder (always frozen) ----------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device).eval()
        self.preprocess = preprocess
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- Per-task state -----------------------------------------------
        # Fitted decoders: task_name -> weight matrix [1025, 2] (includes bias row)
        self._decoders: Dict[str, torch.Tensor] = {}
        # Per-task softmax temperatures (fitted on val set)
        self._temperatures: Dict[str, float] = {}

        # ---- Feature cache (uid -> [1024] tensor) -------------------------
        self._cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def precompute_features(
        self,
        image_refs: List[ImageRef],
        batch_size: int = 16,
    ) -> None:
        """Cache CLIP features in mini-batches. Skips already-cached UIDs."""
        uncached = [r for r in image_refs if r.uid not in self._cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing CLIP features", unit="batch"):
            batch = uncached[i : i + batch_size]
            imgs  = [Image.open(r.path).convert("RGB") for r in batch]
            tensor = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            feats  = self.encoder(tensor).float()
            for ref, feat in zip(batch, feats):
                self._cache[ref.uid] = feat

    def _get_features(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """Return [B, 1024] features, using cache where possible."""
        if all(r.uid in self._cache for r in image_refs):
            return torch.stack([self._cache[r.uid] for r in image_refs])
        # Fall back to fresh encoding for any uncached refs
        self.precompute_features(image_refs)
        return torch.stack([self._cache[r.uid] for r in image_refs])

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        task: Task,
        image_refs: List[ImageRef],
    ) -> None:
        """
        Fit the linear decoder for one task.

        Uses all provided image_refs; labels come from each image's
        latent_state and the task's delta_u.

        Args:
            task:        the Task to fit for.
            image_refs:  all training images (features are cached / fetched).
        """
        # Features
        X = self._get_features(image_refs).cpu()  # [N, 1024]

        # Optimal-action labels (deterministic from latent state)
        optimal = np.array(
            [task.optimal_action(r.latent_state) for r in image_refs],
            dtype=np.int64,
        )
        Y = torch.zeros(len(image_refs), 2)
        Y[torch.arange(len(image_refs)), torch.from_numpy(optimal)] = 1.0  # [N, 2]

        # Prepend bias column: [N, 1025]
        X_bias = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)

        # Least-squares solve on CPU (CUDA lstsq can produce NaNs)
        result = torch.linalg.lstsq(X_bias, Y)
        W = result.solution.to(self.device)  # [1025, 2]

        self._decoders[task.name]     = W
        self._temperatures[task.name] = 1.0  # default; tune with fit_temperature()

    def fit_temperature(
        self,
        task: Task,
        image_refs: List[ImageRef],
        counts: torch.Tensor,
        temperatures: Optional[List[float]] = None,
    ) -> float:
        """
        Line-search for the softmax temperature that minimises NLL on held-out data.

        Args:
            task:         the Task.
            image_refs:   validation images.
            counts:       [N, 2] observed choice counts on val set.
            temperatures: candidate values (default: log-spaced 0.1 to 10).

        Returns:
            best temperature (also stored internally).
        """
        from dlbt.training.metrics import multinomial_nll

        if temperatures is None:
            temperatures = list(np.logspace(-1, 1, 50))

        counts = counts.to(self.device)
        logits = self._logits(image_refs, task)  # [N, 2]

        best_temp, best_nll = 1.0, float("inf")
        for temp in temperatures:
            probs = F.softmax(logits / temp, dim=-1).clamp(min=1e-8)
            nll   = multinomial_nll(probs, counts).item()
            if nll < best_nll:
                best_nll, best_temp = nll, temp

        self._temperatures[task.name] = best_temp
        return best_temp

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def _logits(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Raw decoder logits, shape [B, 2]."""
        if task.name not in self._decoders:
            raise RuntimeError(
                f"No decoder fitted for task '{task.name}'. Call fit() first."
            )
        X = self._get_features(image_refs)                   # [B, 1024]
        X_bias = torch.cat([X, torch.ones(X.shape[0], 1, device=self.device)], dim=1)
        W = self._decoders[task.name]                        # [1025, 2]
        return X_bias @ W                                    # [B, 2]

    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Return [B, 2] softmax probabilities with the fitted temperature."""
        temp   = self._temperatures.get(task.name, 1.0)
        logits = self._logits(image_refs, task)
        return F.softmax(logits / temp, dim=-1)
