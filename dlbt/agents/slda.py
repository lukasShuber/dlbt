"""
SldaAgent: Softmax Linear Discriminant Agent.

Architecture:
    image → CLIP RN50 (backbone frozen; attnpool optionally trainable) → [1024]
          → Linear(1024, K) → u(x) [K]          (utility map)
          → logit = u(x) · ΔU_t / τ             (temperature-scaled inner product)
          → P(right | image, task) = σ(logit)   (sigmoid choice probability)

Two variants:
    SldaAgent(freeze_encoder=True)   — frozen CLIP: only mapper + log_temperature
                                       are trained.  Compatible with train_dlbt().
    SldaAgent(freeze_encoder=False)  — backbone frozen, attnpool trainable.
                                       Used for Phase 2 fine-tuning: call
                                       precompute_backbone_features() once, then
                                       pass to finetune_slda_attnpool().

Key differences from DlbtAgent:
  - No Dirichlet: u(x) is a point estimate, not a belief distribution.
  - No Monte Carlo: choice_probs() is deterministic (no sampling overhead).
  - Temperature τ = exp(log_τ) is learned jointly via NLL.
  - Task-agnostic utility map generalises to validation tasks (same as DLBT).

Compatible with train_dlbt() when freeze_encoder=True:
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
    SLDA agent backed by a CLIP RN50 visual encoder.

    Args:
        freeze_encoder: if True (default), the full encoder is frozen and only
                        the mapper + log_temperature are trained.
                        If False, the backbone is frozen but the attention-
                        pooling layer (attnpool) has requires_grad=True — for
                        use with finetune_slda_attnpool() in Phase 2.
        device: torch device.
    """

    # Signal for train_dlbt compatibility
    n_mc_samples = None   # deterministic — no MC correction in corrected_mse

    def __init__(
        self,
        freeze_encoder: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device        = device
        self.freeze_encoder = freeze_encoder   # instance attribute
        self.feature_dim   = 1024

        # ---- CLIP RN50 encoder -----------------------------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device)
        self.preprocess = preprocess

        # Freeze full encoder unconditionally, then selectively re-enable attnpool.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if not freeze_encoder:
            for p in self.encoder.attnpool.parameters():
                p.requires_grad_(True)

        # ---- Utility mapper: 1024 → K ----------------------------------------
        # Unconstrained linear map — utilities are real-valued, no Softplus.
        self.mapper = nn.Linear(1024, K).to(device)
        nn.init.xavier_uniform_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

        # ---- Learnable log-temperature ----------------------------------------
        # τ = exp(log_τ) > 0. Initialised at 0 → τ = 1 (no scaling).
        # Trained jointly with the mapper via NLL.
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        # ---- Feature caches ---------------------------------------------------
        # _cache:          uid → [1024]    full CLIP features (freeze_encoder=True)
        # _backbone_cache: uid → [C, H, W] pre-attnpool spatial maps
        #                                   (freeze_encoder=False)
        self._cache:          Dict[str, torch.Tensor] = {}
        self._backbone_cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        """Save the full-feature cache to disk."""
        torch.save(self._cache, path)

    def load_cache(self, path: str) -> None:
        """Load a previously saved full-feature cache from disk."""
        loaded = torch.load(path, map_location=self.device)
        self._cache.update(loaded)

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def precompute_features(
        self,
        image_refs: List[ImageRef],
        batch_size: int = 16,
    ) -> None:
        """
        Encode all images (full CLIP) and store in _cache.
        Call once before training when freeze_encoder=True.
        Already-cached UIDs are skipped.
        """
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

    @torch.no_grad()
    def precompute_backbone_features(
        self,
        image_refs: List[ImageRef],
        batch_size: int = 16,
    ) -> None:
        """
        Cache pre-attnpool spatial feature maps for all images.
        Call once before Phase 2 fine-tuning when freeze_encoder=False.

        The backbone (everything before attnpool) is frozen — caching it means
        each training step runs only the small attnpool layer, not the full
        ResNet.  Already-cached UIDs are skipped.
        """
        uncached = [r for r in image_refs if r.uid not in self._backbone_cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing backbone features (SLDA)", unit="batch"):
            batch_refs = uncached[i : i + batch_size]
            imgs  = [Image.open(r.path).convert("RGB") for r in batch_refs]
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            feats = self._run_backbone(batch)   # [B, C, H, W]
            for ref, feat in zip(batch_refs, feats):
                self._backbone_cache[ref.uid] = feat.cpu()  # kept off-GPU

    def _run_backbone(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run images through the frozen backbone up to (not including) attnpool.
        Returns spatial feature maps [B, C, H, W].
        """
        enc = self.encoder
        x   = batch.type(enc.conv1.weight.dtype)
        x   = enc.act1(enc.bn1(enc.conv1(x)))
        x   = enc.act2(enc.bn2(enc.conv2(x)))
        x   = enc.act3(enc.bn3(enc.conv3(x)))
        x   = enc.avgpool(x)
        x   = enc.layer1(x)
        x   = enc.layer2(x)
        x   = enc.layer3(x)
        x   = enc.layer4(x)
        return x

    def _encode(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """
        Return CLIP features [B, 1024].

        freeze_encoder=True  → full-feature cache (or fresh full encode).
        freeze_encoder=False → backbone cache + runs attnpool (grad-enabled).
                               Falls back to full fresh encode if backbone
                               cache is not populated (should not happen in
                               practice after precompute_backbone_features()).
        """
        if self.freeze_encoder:
            if all(r.uid in self._cache for r in image_refs):
                return torch.stack([self._cache[r.uid] for r in image_refs])
            imgs  = [Image.open(r.path).convert("RGB") for r in image_refs]
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            with torch.no_grad():
                return self.encoder(batch).float()
        else:
            if all(r.uid in self._backbone_cache for r in image_refs):
                spatial = torch.stack(
                    [self._backbone_cache[r.uid] for r in image_refs]
                ).to(self.device)
                return self.encoder.attnpool(spatial).float()
            # Fallback: full fresh encode
            imgs  = [Image.open(r.path).convert("RGB") for r in image_refs]
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
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

            logit    = u(x) · ΔU_t / τ
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
        """Return all parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def extract_features(
        self,
        image_refs: List[ImageRef],
        batch_size: int = 16,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract CLIP features for all image_refs using the current encoder
        (frozen or fine-tuned attnpool).

        When freeze_encoder=False this runs backbone cache + fine-tuned
        attnpool, giving updated features for Phase 3 LogReg refit.
        When freeze_encoder=True this is equivalent to _cache lookups.

        Returns uid → [D] tensor (on CPU).
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        uncached = [r for r in image_refs if r.uid not in features]
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            feats = self._encode(batch)   # [B, D]
            for ref, feat in zip(batch, feats):
                features[ref.uid] = feat.cpu()
        return features
