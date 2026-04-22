"""
EBMAgent: Energy-Based SEU agent with a learned density over the K-simplex.

Instead of parameterising a specific distributional family (Dirichlet,
logistic-normal), this agent learns an unnormalized log-density over the
full K-simplex via a small neural network:

    q(p̃ | x)  ∝  exp( f(p̃, x) )

Choice probability is computed by importance-weighted Monte Carlo:

    P(yes | x, T)  ≈  Σ_i  w_i · I[⟨p̃_i, Δu⟩ > 0]

where the importance weights are

    w_i  =  softmax( f(p̃_1, x), …, f(p̃_N, x) )_i

and {p̃_i} are N points sampled uniformly from Δ^{K-1} and fixed for the
lifetime of the agent.

Because the normalisation constant V(x) = ∫ exp(f) dp̃ appears in both
numerator and denominator it cancels exactly, so the partition function
is never needed.

Scoring network architecture
─────────────────────────────
    CLIP(x) [1024]  ──► compress  Linear(1024→C) + GELU  ──► [C]   ─┐
                                                                       concat [C+K]
                 p̃ [K]  ─────────────────────────────────────────────┘
                         ──► hidden   Linear(C+K → H) + GELU  ──► [H]
                         ──► out      Linear(H → 1)            ──► scalar f(p̃, x)

All three linear layers are learnable and trained end-to-end from the
behavioural NLL.  The CLIP encoder is frozen (only the scoring MLP trains).

The gradient through P(yes) is the contrastive form:
    ∇θ log P(yes) = E_{w+}[∇θ f] − E_w[∇θ f]
where E_{w+} is the expectation over MC samples on the "yes" side and E_w
is the expectation over all MC samples.  No straight-through trick needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dlbt.agents.base import Agent
from dlbt.constants import K as K_DEFAULT
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class EBMAgent(nn.Module, Agent):
    """
    Energy-based SEU agent.

    Args:
        freeze_encoder: if True (default), only the scoring MLP is trained.
        n_mc_samples:   number of fixed MC samples on Δ^{K-1}.
        device:         torch device.
        compress_dim:   width of the image compression layer (1024 → C).
        hidden_dim:     width of the one hidden layer (C+K → H).
        K:              number of latent states (16 for standard DLBT).
        mc_seed:        RNG seed for the fixed simplex sample set.
    """

    def __init__(
        self,
        freeze_encoder: bool  = True,
        n_mc_samples:   int   = 1000,
        device: torch.device  = torch.device("cpu"),
        compress_dim:   int   = 128,
        hidden_dim:     int   = 256,
        K:              int   = K_DEFAULT,
        mc_seed:        int   = 0,
    ):
        super().__init__()

        self.freeze_encoder = freeze_encoder
        self.n_mc_samples   = n_mc_samples
        self.device         = device
        self.compress_dim   = compress_dim
        self.hidden_dim     = hidden_dim
        self.K              = K

        # ---- CLIP RN50 encoder (frozen) ------------------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device)
        self.preprocess = preprocess

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- Scoring MLP ---------------------------------------------------
        # compress: 1024 → compress_dim   (makes CLIP and p̃ roughly equal-scale)
        # hidden:   compress_dim + K → hidden_dim
        # out:      hidden_dim → 1
        self.compress = nn.Linear(1024, compress_dim)
        self.hidden   = nn.Linear(compress_dim + K, hidden_dim)
        self.out      = nn.Linear(hidden_dim, 1)
        self.act      = nn.GELU()

        # Initialisation: start from ~zero scores → uniform weights → ESS ≈ N
        nn.init.xavier_uniform_(self.compress.weight);  nn.init.zeros_(self.compress.bias)
        nn.init.xavier_uniform_(self.hidden.weight);    nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.out.weight);                nn.init.zeros_(self.out.bias)

        for m in (self.compress, self.hidden, self.out):
            m.to(device)

        # ---- Fixed uniform simplex samples ---------------------------------
        # Draw N points uniformly from Δ^{K-1} using the Dirichlet(1,…,1) trick:
        #   x_k ~ Exp(1)  →  p̃ = x / Σx  is uniform on the simplex.
        gen = torch.Generator()          # CPU generator — works regardless of device
        gen.manual_seed(mc_seed)
        x   = torch.zeros(n_mc_samples, K).exponential_(generator=gen)  # CPU
        mc  = (x / x.sum(dim=1, keepdim=True)).to(device)               # [N, K]
        self.register_buffer("mc_samples", mc)         # non-trainable, moves with .to()

        # ---- Task indicator cache (lazy, keyed by task.name) ---------------
        # indicator[i] = 1  iff  ⟨p̃_i, Δu⟩ > 0  (stays constant during training)
        self._indicator_cache: Dict[str, torch.Tensor] = {}

        # ---- CLIP feature cache -------------------------------------------
        self._cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        torch.save(self._cache, path)

    def load_cache(self, path: str) -> None:
        self._cache.update(torch.load(path, map_location=self.device))

    @torch.no_grad()
    def precompute_features(self, image_refs: List[ImageRef], batch_size: int = 32) -> None:
        uncached = [r for r in image_refs if r.uid not in self._cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing CLIP features (EBM)", unit="batch"):
            batch = uncached[i : i + batch_size]
            feats = self._encode_fresh(batch)
            for ref, feat in zip(batch, feats):
                self._cache[ref.uid] = feat.detach().cpu()

    def _encode_fresh(self, refs: List[ImageRef]) -> torch.Tensor:
        imgs  = [Image.open(r.path).convert("RGB") for r in refs]
        batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            return self.encoder(batch).float()

    def _encode(self, refs: List[ImageRef]) -> torch.Tensor:
        """Return [B, 1024] CLIP features, using cache when available."""
        if all(r.uid in self._cache for r in refs):
            return torch.stack([self._cache[r.uid] for r in refs]).to(self.device)
        return self._encode_fresh(refs)

    # -----------------------------------------------------------------------
    # Task indicator (cached)
    # -----------------------------------------------------------------------

    def _indicator(self, task: Task) -> torch.Tensor:
        """Return [N] binary indicator: 1 where ⟨p̃_i, Δu⟩ > 0."""
        if task.name not in self._indicator_cache:
            delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=self.device)
            self._indicator_cache[task.name] = (self.mc_samples @ delta_u > 0).float()
        return self._indicator_cache[task.name]

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def _scores(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Compute energy scores for all N MC samples for a batch of images.

        Args:
            feats: [B, 1024] CLIP features (already on device).
        Returns:
            scores: [B, N]  — unnormalized log-density at each MC sample.
        """
        B = feats.shape[0]
        N = self.n_mc_samples

        # Compressed image embedding: [B, C]
        g = self.act(self.compress(feats))

        # Broadcast to [B, N, C+K] and run the hidden layer
        g_exp = g.unsqueeze(1).expand(B, N, self.compress_dim)          # [B, N, C]
        p_exp = self.mc_samples.unsqueeze(0).expand(B, N, self.K)        # [B, N, K]
        inp   = torch.cat([g_exp, p_exp], dim=-1)                        # [B, N, C+K]

        scores = self.out(self.act(self.hidden(inp))).squeeze(-1)         # [B, N]
        return scores

    # -----------------------------------------------------------------------
    # Choice probabilities
    # -----------------------------------------------------------------------

    def choice_probs(self, refs: List[ImageRef], task: Task) -> torch.Tensor:
        """
        Estimate P(action | image, task) for a batch of images.

        Returns [B, 2] tensor (columns: P(no), P(yes)).
        Differentiable in training mode; no-grad in eval mode.
        """
        indicator = self._indicator(task)              # [N]
        feats     = self._encode(refs)                 # [B, 1024]

        if self.training:
            return self._forward(feats, indicator)
        else:
            with torch.no_grad():
                return self._forward(feats, indicator)

    def _forward(self, feats: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor:
        """Core computation: [B, 1024] × [N] → [B, 2]."""
        scores  = self._scores(feats)                              # [B, N]
        weights = F.softmax(scores, dim=1)                         # [B, N]  (sum to 1)
        p_yes   = (weights * indicator.unsqueeze(0)).sum(dim=1)    # [B]
        return torch.stack([1.0 - p_yes, p_yes], dim=-1)          # [B, 2]

    # -----------------------------------------------------------------------
    # Entropy regularisation helper  (differentiable)
    # -----------------------------------------------------------------------

    def mean_entropy(self, refs: List[ImageRef]) -> torch.Tensor:
        """
        Mean Shannon entropy of the importance-weight distribution.

            H_i = −Σ_n w_{in} log w_{in}   ∈ [0, log N]

        Returns a scalar (mean over images in batch).
        Fully differentiable — use as a regularisation term:

            loss += −ent_weight * agent.mean_entropy(refs)   # maximise H

        This prevents the softmax weights from collapsing onto a tiny
        fraction of MC samples (ESS collapse), keeping the Monte Carlo
        estimate of the choice probability well-supported across the simplex.
        """
        feats   = self._encode(refs)
        scores  = self._scores(feats)                   # [B, N]
        weights = F.softmax(scores, dim=1)              # [B, N]
        H = -(weights * (weights + 1e-10).log()).sum(dim=1)   # [B]
        return H.mean()                                 # scalar

    # -----------------------------------------------------------------------
    # Diagnostic: effective sample size  (no-grad)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ess(self, refs: List[ImageRef]) -> float:
        """
        Mean effective sample size (ESS) over a batch of images.

        ESS = 1 / Σ_i w_i²  (= N for uniform weights, → 1 for collapsed).
        Reported as a fraction of N so it's in [1/N, 1].
        """
        feats   = self._encode(refs)
        scores  = self._scores(feats)
        weights = F.softmax(scores, dim=1)              # [B, N]
        ess_per_image = 1.0 / (weights ** 2).sum(dim=1) # [B]
        return float(ess_per_image.mean().item()) / self.n_mc_samples

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> str:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"EBMAgent(K={self.K}, N={self.n_mc_samples}, "
            f"compress={self.compress_dim}, hidden={self.hidden_dim}, "
            f"trainable={n_train:,})"
        )
