"""
SldaAgent: Softmax Linear Discriminant Agent.

Architecture:
    image → CLIP RN50 (frozen) → [1024]
          → Linear(1024, K) → u(x) [K]          (utility map)
          → logit = u(x) · ΔU_t / τ             (temperature-scaled inner product)
          → P(right | image, task) = σ(logit)   (sigmoid choice probability)

Deterministic baseline for DLBT. Uses the same frozen CLIP features but
replaces the Dirichlet belief distribution with a direct linear utility map.

Key differences from DlbtAgent:
  - No Dirichlet: u(x) is a point estimate, not a belief distribution.
  - No Monte Carlo: choice_probs() is deterministic (no sampling overhead).
  - Temperature τ = exp(log_τ) is learned jointly via NLL, calibrating the
    confidence of the sigmoid output from behavioral data.
  - Can generalise to val tasks because u(x) is task-agnostic (same as DLBT).

Compatible with train_dlbt() out of the box:
    freeze_encoder = True  → precompute_features() is called automatically
    n_mc_samples   = None  → corrected_mse() skips MC variance correction
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from dlbt.agents.base import Agent
from dlbt.constants import K
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class SldaAgent(nn.Module, Agent):
    """
    SLDA agent backed by a frozen CLIP RN50 visual encoder.

    Args:
        device: torch device.
    """

    # Signals for train_dlbt compatibility
    freeze_encoder = True   # always frozen — precompute_features() will be called
    n_mc_samples   = None   # deterministic — no MC correction in corrected_mse

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        # ---- CLIP RN50 encoder (always frozen) ----------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device)
        self.preprocess = preprocess
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- Utility mapper: 1024 → K ------------------------------------
        # Unconstrained linear map — utilities are real-valued, no Softplus.
        self.mapper = nn.Linear(1024, K).to(device)
        nn.init.xavier_uniform_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

        # ---- Learnable log-temperature -----------------------------------
        # τ = exp(log_τ) > 0. Initialised at 0 → τ = 1 (no scaling).
        # Trained jointly with the mapper via NLL: automatically calibrates
        # the confidence of sigmoid outputs from behavioral data.
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        # ---- Feature cache (uid → [1024]) --------------------------------
        self._cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        """Save the feature cache to disk."""
        torch.save(self._cache, path)

    def load_cache(self, path: str) -> None:
        """Load a previously saved feature cache from disk."""
        loaded = torch.load(path, map_location=self.device)
        self._cache.update(loaded)

    @torch.no_grad()
    def precompute_features(
        self,
        image_refs: List[ImageRef],
        batch_size: int = 16,
    ) -> None:
        """Encode all images and store in cache. Already-cached UIDs skipped."""
        uncached = [r for r in image_refs if r.uid not in self._cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing CLIP features (SLDA)", unit="batch"):
            batch    = uncached[i : i + batch_size]
            imgs     = [Image.open(r.path).convert("RGB") for r in batch]
            tensor   = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            features = self.encoder(tensor).float()
            for ref, feat in zip(batch, features):
                self._cache[ref.uid] = feat

    def _encode(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """Return CLIP features [B, 1024], using cache when available."""
        if all(r.uid in self._cache for r in image_refs):
            return torch.stack([self._cache[r.uid] for r in image_refs])
        imgs   = [Image.open(r.path).convert("RGB") for r in image_refs]
        batch  = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            return self.encoder(batch).float()

    # -----------------------------------------------------------------------
    # Core computation
    # -----------------------------------------------------------------------

    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        P(action | image, task) via temperature-scaled sigmoid.

            logit   = u(x) · ΔU_t / τ
            P(right) = σ(logit)

        Deterministic — no MC sampling.

        Returns:
            Tensor [B, 2], columns [P(left), P(right)], sums to 1 along dim=1.
        """
        features = self._encode(image_refs)                         # [B, 1024]
        u        = self.mapper(features)                            # [B, K]
        delta_u  = torch.tensor(
            task.delta_u, dtype=torch.float32, device=self.device
        )
        tau      = self.log_temperature.exp()                       # scalar > 0
        logit    = (u @ delta_u) / tau                              # [B]
        p_right  = torch.sigmoid(logit)                             # [B]
        return torch.stack([1.0 - p_right, p_right], dim=-1)       # [B, 2]

    def trainable_parameters(self):
        """Mapper weights/bias + log_temperature. Encoder is always frozen."""
        return list(self.mapper.parameters()) + [self.log_temperature]
