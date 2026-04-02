"""
DlbtAgent: Deep Latent Belief Tomography agent.

Architecture:
    image → CLIP RN50 encoder → [1024]
          → Linear(1024, K) + Softplus → α [K]      (distribution mapper)
          → b̃ ~ Dirichlet(α)                         (reparameterised sample)
          → logit = b̃ · ΔU_t                         (SEU difference, scalar)
          → straight-through argmax → P(right | image, task)

Two variants:
    DlbtAgent(freeze_encoder=True)   — DLBT-frozen: only mapper is trained.
    DlbtAgent(freeze_encoder=False)  — DLBT-attnpool: backbone frozen, only the
                                       CLIP attention-pooling layer + mapper are
                                       trained (~1M + 16K params). Parsimonious
                                       alternative to full encoder finetuning.

For the frozen variant, call precompute_features() once before training to
cache CLIP representations; subsequent forward passes become cheap lookups.
The attnpool variant cannot use the cache (attnpool weights change during
training), so every forward pass runs through the full encoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Dirichlet

from tqdm import tqdm

from dlbt.agents.base import Agent
from dlbt.constants import K
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class DlbtAgent(nn.Module, Agent):
    """
    DLBT agent backed by a CLIP RN50 visual encoder.

    Args:
        freeze_encoder: if True, the full encoder is frozen and only the
                        mapper is trained (DLBT-frozen). If False, the
                        backbone is frozen but the attention-pooling layer
                        (attnpool) is unfrozen and trained jointly with the
                        mapper (DLBT-attnpool).
        n_mc_samples:   number of Monte Carlo samples for the Dirichlet
                        expectation during choice_probs().
        device:         torch device.
        mapper_hidden:  if None (default), use a single Linear(1024, K).
                        If an int, insert a hidden layer of that width with
                        GELU activation: 1024 → mapper_hidden → K.
    """

    def __init__(
        self,
        freeze_encoder: bool = True,
        n_mc_samples: int = 1000,
        device: torch.device = torch.device("cpu"),
        mapper_hidden: Optional[int] = None,
        feature_dim: int = 1024,
    ):
        super().__init__()

        self.freeze_encoder = freeze_encoder
        self.n_mc_samples   = n_mc_samples
        self.device         = device
        self.feature_dim    = feature_dim

        # ---- CLIP RN50 encoder --------------------------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device)
        self.preprocess = preprocess   # torchvision transform, kept on CPU

        # Freeze the full backbone unconditionally.
        # When freeze_encoder=False, selectively re-enable attnpool only.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if not freeze_encoder:
            for p in self.encoder.attnpool.parameters():
                p.requires_grad_(True)

        # ---- Distribution mapper: feature_dim -> K ------------------------
        # Linear + Softplus guarantees strictly positive Dirichlet parameters.
        # Bias initialised so Softplus output starts at ~2.0 (moderate Dirichlet).
        # feature_dim defaults to 1024 (CLIP RN50 output); set to a smaller
        # value (e.g. 16 or 4) to use oracle features instead of CLIP.
        # mapper_hidden=None → single linear layer (default, fast, interpretable).
        # mapper_hidden=256  → MLP with one hidden layer (more expressive).
        if mapper_hidden is None:
            linear = nn.Linear(feature_dim, K)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 1.1)
            self.mapper = nn.Sequential(linear, nn.Softplus()).to(device)
        else:
            h1 = nn.Linear(feature_dim, mapper_hidden)
            h2 = nn.Linear(mapper_hidden, K)
            nn.init.xavier_uniform_(h1.weight);  nn.init.zeros_(h1.bias)
            nn.init.xavier_uniform_(h2.weight);  nn.init.constant_(h2.bias, 1.1)
            self.mapper = nn.Sequential(h1, nn.GELU(), h2, nn.Softplus()).to(device)

        # ---- Feature caches -----------------------------------------------
        # _cache:          uid -> [feature_dim]  CLIP or oracle features (freeze_encoder=True)
        # _backbone_cache: uid -> [C, H, W]      pre-attnpool spatial maps (freeze_encoder=False)
        # In the attnpool variant the backbone is frozen, so its output is
        # constant — caching it reduces each epoch to attnpool + mapper only.
        self._cache:          Dict[str, torch.Tensor] = {}
        self._backbone_cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        """Save the feature cache to disk."""
        torch.save(self._cache, path)

    def load_cache(self, path: str) -> None:
        """Load a previously saved feature cache from disk."""
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
        Encode all images in mini-batches and store results in the cache.
        Call once before training when freeze_encoder=True.
        Already-cached UIDs are skipped.
        """
        uncached = [r for r in image_refs if r.uid not in self._cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing CLIP features", unit="batch"):
            batch = uncached[i : i + batch_size]
            features = self._encode_fresh(batch)  # [B, 1024]
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
        Call once before training when freeze_encoder=False.
        The backbone (everything before attnpool) is frozen, so its output
        is constant — caching it means each training epoch only runs the
        small attnpool layer + mapper instead of the full ResNet.
        Already-cached UIDs are skipped.
        """
        uncached = [r for r in image_refs if r.uid not in self._backbone_cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing backbone features", unit="batch"):
            batch_refs = uncached[i : i + batch_size]
            imgs  = [Image.open(r.path).convert("RGB") for r in batch_refs]
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            feats = self._run_backbone(batch)          # [B, C, H, W]
            for ref, feat in zip(batch_refs, feats):
                self._backbone_cache[ref.uid] = feat.cpu()  # keep off-GPU; moved to device in _encode

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

    def _encode_fresh(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """Run images through the full CLIP encoder. Returns [B, 1024]."""
        imgs  = [Image.open(r.path).convert("RGB") for r in image_refs]
        batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        ctx   = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            return self.encoder(batch).float()

    def _encode(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """
        Return CLIP features [B, 1024].
        - freeze_encoder=True:  uses full-feature cache.
        - freeze_encoder=False: uses backbone cache + runs attnpool (trainable).
          Falls back to full fresh encode if backbone cache is not populated.
        """
        if self.freeze_encoder and all(r.uid in self._cache for r in image_refs):
            return torch.stack([self._cache[r.uid] for r in image_refs])
        if not self.freeze_encoder and all(r.uid in self._backbone_cache for r in image_refs):
            spatial = torch.stack([self._backbone_cache[r.uid] for r in image_refs]).to(self.device)
            return self.encoder.attnpool(spatial).float()
        return self._encode_fresh(image_refs)

    # -----------------------------------------------------------------------
    # Core computation
    # -----------------------------------------------------------------------

    def get_alpha(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """
        Return Dirichlet concentration parameters α for each image.
        Shape: [B, K], all entries strictly positive.
        """
        features = self._encode(image_refs)          # [B, 1024]
        return self.mapper(features).clamp(min=1e-6) # [B, K]

    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        Estimate P(action | image, task) via Monte Carlo Dirichlet sampling.

        Dispatches to the training or eval path depending on self.training:

        Training path — straight-through estimator:
          Hard argmax in the forward pass (discrete decisions), soft softmax
          gradient in the backward pass. Necessary for gradient flow through
          the discrete argmax.

        Eval path — clean hard MC average:
          Simply averages hard argmax decisions over MC samples.
          No ST entanglement, no gradient overhead.
          Uses n_mc_samples from __init__ (increase for lower-variance eval).

        Returns:
            Tensor of shape [B, 2], summing to 1 along dim=1.
            Differentiable w.r.t. mapper (and encoder if not frozen) in
            training mode only.
        """
        if self.training:
            return self._choice_probs_train(image_refs, task)
        else:
            return self._choice_probs_eval(image_refs, task)

    def _choice_probs_train(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        Training forward pass: straight-through argmax over MC Dirichlet samples.

          1. α = mapper(encoder(x))                      [B, K]
          2. b̃ ~ Dirichlet(α)  (rsample, differentiable) [N, B, K]
          3. logit = b̃ · ΔU_t                            [N, B]
          4. Straight-through argmax → choice indicator  [N, B, 2]
          5. Average over N samples                      [B, 2]
        """
        N     = self.n_mc_samples
        alpha = self.get_alpha(image_refs)                          # [B, K]
        delta_u = torch.tensor(
            task.delta_u, dtype=torch.float32, device=self.device
        )

        # Clamp to a numerically safe minimum before rsample.
        # PyTorch's Dirichlet uses Gamma sampling internally; very small
        # concentrations (< ~0.01) cause NaN on GPU even though the values
        # are technically positive.  1e-6 in get_alpha is not enough.
        alpha  = alpha.clamp(min=0.1)
        b      = Dirichlet(alpha).rsample((N,))                     # [N, B, K]
        logit  = torch.einsum("nbk,k->nb", b, delta_u)             # [N, B]

        logits_2d  = torch.stack([-logit, logit], dim=-1)          # [N, B, 2]
        probs_soft = F.softmax(logits_2d, dim=-1)
        hard       = F.one_hot(logits_2d.argmax(-1), 2).float()

        # Straight-through: hard forward, soft gradient
        st = (hard - probs_soft).detach() + probs_soft
        return st.mean(dim=0)                                       # [B, 2]

    @torch.no_grad()
    def _choice_probs_eval(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        Eval forward pass: clean hard MC average, no ST, no gradients.

          1. α = mapper(encoder(x))                      [B, K]
          2. b̃ ~ Dirichlet(α)  (sample, not rsample)    [N, B, K]
          3. logit = b̃ · ΔU_t                            [N, B]
          4. Hard argmax → {0, 1}                        [N, B]
          5. Average over N samples                      [B, 2]
        """
        N     = self.n_mc_samples
        alpha = self.get_alpha(image_refs).clamp(min=0.1)          # [B, K]
        delta_u = torch.tensor(
            task.delta_u, dtype=torch.float32, device=self.device
        )

        b     = Dirichlet(alpha).sample((N,))                      # [N, B, K]
        logit = torch.einsum("nbk,k->nb", b, delta_u)             # [N, B]
        hard  = (logit > 0).float()                                # [N, B]

        p_right = hard.mean(dim=0)                                 # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)        # [B, 2]

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        """Return the parameters that should be passed to the optimiser."""
        return [p for p in self.parameters() if p.requires_grad]
