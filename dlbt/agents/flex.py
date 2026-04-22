"""
FlexAgent: SEU agent with a logistic-normal distribution over the K-simplex.

Replaces the Dirichlet with a softmax-normal (logistic-normal):

    z ~ N(μ(x), Σ(x))          [pre-softmax latent]
    b  = softmax(z)              [K-simplex belief]
    P(yes | x, task) ≈ MC mean of [b · δu > 0]

This is more expressive than the Dirichlet in two key ways:
  1. Bimodality: the logistic-normal CAN represent bimodal distributions on
     the simplex (e.g. "high mass on opposite corners").  Dirichlet cannot.
  2. Flexible correlations: the full-covariance variant models arbitrary
     inter-dimension covariances.  Dirichlet only allows negative correlation.

Two covariance modes (controlled by cov_type):
  "diag"   — Σ = diag(σ²),  mapper outputs (μ, log σ) ∈ R^K × R^K.
             2K mapper outputs total.  Fast; most useful first step up.
  "full"   — Σ = L Lᵀ (Cholesky),  mapper outputs (μ, L_flat).
             L_flat is the lower triangle flattened (K(K+1)/2 values).
             Diagonal of L stored in log-space → exp() ensures positivity.
             K + K(K+1)/2 = 152 mapper outputs for K=16.  Maximum flexibility.

The choice_probs interface is identical to DlbtAgent, so FlexAgent is a
drop-in replacement in the training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dlbt.agents.base import Agent
from dlbt.constants import K as K_DEFAULT
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class FlexAgent(nn.Module, Agent):
    """
    Flexible SEU agent backed by a CLIP RN50 encoder and a logistic-normal
    (softmax-normal) belief distribution.

    Args:
        freeze_encoder: if True, only the mapper heads are trained (frozen CLIP).
        n_mc_samples:   MC samples for the choice-probability estimate.
        device:         torch device.
        mapper_hidden:  None → single linear layer for each head.
                        int  → hidden layer of that width (GELU), shared trunk.
        feature_dim:    CLIP feature dimensionality (1024 for RN50).
        K:              number of latent states (default 16 = 2^4).
        cov_type:       "diag" (diagonal Σ) or "full" (Cholesky Σ = LLᵀ).
    """

    def __init__(
        self,
        freeze_encoder: bool = True,
        n_mc_samples:   int  = 1000,
        device: torch.device = torch.device("cpu"),
        mapper_hidden:  Optional[int] = None,
        feature_dim:    int  = 1024,
        K:              int  = K_DEFAULT,
        cov_type:       str  = "diag",   # "diag" or "full"
    ):
        super().__init__()

        assert cov_type in ("diag", "full"), f"cov_type must be 'diag' or 'full', got {cov_type!r}"

        self.freeze_encoder = freeze_encoder
        self.n_mc_samples   = n_mc_samples
        self.device         = device
        self.feature_dim    = feature_dim
        self.K              = K
        self.cov_type       = cov_type

        # n_cov_params: K for diagonal, K(K+1)/2 for full Cholesky
        self.n_cov_params = K if cov_type == "diag" else K * (K + 1) // 2

        # ---- CLIP RN50 encoder (same as DlbtAgent) -------------------------
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self.encoder    = clip_model.visual.to(device)
        self.preprocess = preprocess

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if not freeze_encoder:
            for p in self.encoder.attnpool.parameters():
                p.requires_grad_(True)

        # ---- Mapper: feature_dim → shared trunk (optional) + two heads ----
        # mu_head:  output K values (unconstrained; softmax gives ~1/K init)
        # cov_head: output n_cov_params values (log-diagonal / L_flat)
        #
        # Initialisation strategy:
        #   mu   bias = 0       → softmax(0) = uniform(1/K) initially
        #   cov  bias = -0.5    → exp(-0.5) ≈ 0.6 std for diag;
        #                         moderate spread, not too sharp, not flat

        if mapper_hidden is None:
            # Independent linear heads — no shared trunk
            mu_lin  = nn.Linear(feature_dim, K)
            cov_lin = nn.Linear(feature_dim, self.n_cov_params)
            nn.init.xavier_uniform_(mu_lin.weight);   nn.init.zeros_(mu_lin.bias)
            nn.init.xavier_uniform_(cov_lin.weight);  nn.init.constant_(cov_lin.bias, -0.5)
            self.trunk    = None
            self.mu_head  = mu_lin.to(device)
            self.cov_head = cov_lin.to(device)
        else:
            # Shared MLP trunk → two heads
            trunk   = nn.Sequential(
                nn.Linear(feature_dim, mapper_hidden),
                nn.GELU(),
            )
            mu_lin  = nn.Linear(mapper_hidden, K)
            cov_lin = nn.Linear(mapper_hidden, self.n_cov_params)
            nn.init.xavier_uniform_(mu_lin.weight);   nn.init.zeros_(mu_lin.bias)
            nn.init.xavier_uniform_(cov_lin.weight);  nn.init.constant_(cov_lin.bias, -0.5)
            nn.init.xavier_uniform_(trunk[0].weight);  nn.init.zeros_(trunk[0].bias)
            self.trunk    = trunk.to(device)
            self.mu_head  = mu_lin.to(device)
            self.cov_head = cov_lin.to(device)

        # ---- Precomputed lower-triangle indices (for "full" mode) ----------
        if cov_type == "full":
            rows, cols = torch.tril_indices(K, K)
            self.register_buffer("_tril_rows", rows)
            self.register_buffer("_tril_cols", cols)
            # which positions are diagonal?
            self.register_buffer("_diag_mask",
                                 (rows == cols).float())  # 1 where diagonal

        # ---- Feature caches (same as DlbtAgent) ----------------------------
        self._cache:          Dict[str, torch.Tensor] = {}
        self._backbone_cache: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Cache helpers (verbatim from DlbtAgent)
    # -----------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        torch.save(self._cache, path)

    def load_cache(self, path: str) -> None:
        loaded = torch.load(path, map_location=self.device)
        self._cache.update(loaded)

    @torch.no_grad()
    def precompute_features(self, image_refs: List[ImageRef], batch_size: int = 16) -> None:
        uncached = [r for r in image_refs if r.uid not in self._cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing CLIP features (FlexAgent)", unit="batch"):
            batch = uncached[i : i + batch_size]
            feats = self._encode_fresh(batch)
            for ref, feat in zip(batch, feats):
                self._cache[ref.uid] = feat

    @torch.no_grad()
    def precompute_backbone_features(self, image_refs: List[ImageRef], batch_size: int = 16) -> None:
        uncached = [r for r in image_refs if r.uid not in self._backbone_cache]
        if not uncached:
            return
        for i in tqdm(range(0, len(uncached), batch_size),
                      desc="precomputing backbone features (FlexAgent)", unit="batch"):
            batch_refs = uncached[i : i + batch_size]
            imgs  = [Image.open(r.path).convert("RGB") for r in batch_refs]
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            feats = self._run_backbone(batch)
            for ref, feat in zip(batch_refs, feats):
                self._backbone_cache[ref.uid] = feat.cpu()

    def _run_backbone(self, batch: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        x   = batch.type(enc.conv1.weight.dtype)
        x   = enc.act1(enc.bn1(enc.conv1(x)))
        x   = enc.act2(enc.bn2(enc.conv2(x)))
        x   = enc.act3(enc.bn3(enc.conv3(x)))
        x   = enc.avgpool(x)
        x   = enc.layer1(x);  x = enc.layer2(x)
        x   = enc.layer3(x);  x = enc.layer4(x)
        return x

    def _encode_fresh(self, image_refs: List[ImageRef]) -> torch.Tensor:
        imgs  = [Image.open(r.path).convert("RGB") for r in image_refs]
        batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        ctx   = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            return self.encoder(batch).float()

    def _encode(self, image_refs: List[ImageRef]) -> torch.Tensor:
        if self.freeze_encoder and all(r.uid in self._cache for r in image_refs):
            return torch.stack([self._cache[r.uid] for r in image_refs])
        if not self.freeze_encoder and all(r.uid in self._backbone_cache for r in image_refs):
            spatial = torch.stack([self._backbone_cache[r.uid] for r in image_refs]).to(self.device)
            return self.encoder.attnpool(spatial).float()
        return self._encode_fresh(image_refs)

    # -----------------------------------------------------------------------
    # Distribution parameters
    # -----------------------------------------------------------------------

    def get_params(
        self, image_refs: List[ImageRef]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the logistic-normal parameters for a batch of images.

        Returns:
            mu:       [B, K]  — mean of the pre-softmax Gaussian.
            cov_raw:  [B, K]  for cov_type="diag" — log standard deviations.
                      [B, K(K+1)//2]  for cov_type="full" — lower-triangle
                      Cholesky flat params (diagonal entries in log-space).
        """
        feats = self._encode(image_refs)         # [B, feature_dim]
        trunk = self.trunk(feats) if self.trunk is not None else feats
        mu      = self.mu_head(trunk)            # [B, K]
        cov_raw = self.cov_head(trunk)           # [B, n_cov_params]
        return mu, cov_raw

    def _build_L(self, cov_raw: torch.Tensor) -> torch.Tensor:
        """
        For cov_type="full": construct lower-triangular Cholesky L from flat params.

        Diagonal entries are stored in log-space → exp() guarantees positivity.
        Off-diagonal entries are stored directly.

        Returns L: [B, K, K]
        """
        B  = cov_raw.shape[0]
        L  = cov_raw.new_zeros(B, self.K, self.K)
        L[:, self._tril_rows, self._tril_cols] = cov_raw
        # Exponentiate the diagonal (currently stored as raw log-diag values)
        diag_raw = L[:, self._tril_rows[self._diag_mask.bool()],
                       self._tril_cols[self._diag_mask.bool()]]  # [B, K]
        diag_idx = torch.arange(self.K, device=L.device)
        L[:, diag_idx, diag_idx] = diag_raw.exp().clamp(min=1e-4)
        return L   # [B, K, K]

    def _sample_beliefs(
        self, mu: torch.Tensor, cov_raw: torch.Tensor, N: int
    ) -> torch.Tensor:
        """
        Draw N Monte Carlo belief samples per image.

        Returns b: [N, B, K] — simplex-valued beliefs (rows sum to 1).
        """
        B = mu.shape[0]

        if self.cov_type == "diag":
            std = cov_raw.exp().clamp(min=1e-4)              # [B, K]
            eps = torch.randn(N, B, self.K, device=mu.device, dtype=mu.dtype)
            z   = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # [N, B, K]

        else:  # full Cholesky
            L   = self._build_L(cov_raw)                     # [B, K, K]
            eps = torch.randn(N, B, self.K, device=mu.device, dtype=mu.dtype)
            # z[n, b] = mu[b] + L[b] @ eps[n, b]
            z   = mu.unsqueeze(0) + torch.einsum("bkj,nbj->nbk", L, eps)

        return F.softmax(z, dim=-1)   # [N, B, K]

    # -----------------------------------------------------------------------
    # KL regularisation (Gaussian, closed-form)
    # -----------------------------------------------------------------------

    def kl_loss(
        self, image_refs: List[ImageRef], prior_std: float = 1.0
    ) -> torch.Tensor:
        """
        Mean KL( N(μ, Σ) || N(0, prior_std²·I) ) over a batch.

        Closed-form KL divergence between two Gaussians:
            KL = 0.5 * ( tr(Σ/prior²) + ‖μ‖²/prior² − K − log det(Σ/prior²) )

        For "diag":  Σ = diag(σ_k²), so:
            KL = 0.5 * Σ_k [ (σ_k/prior)² + (μ_k/prior)² − 1 − 2 log_σ_k + 2 log prior ]

        For "full":  Σ = LL^T, so det(Σ) = (Π L_kk)², and tr(Σ/p²) = ‖L‖_F²/p²:
            KL = 0.5 * ( ‖L/prior‖_F² + ‖μ/prior‖² − K − 2 Σ_k log(L_kk/prior) )

        Returns a scalar (mean over batch).
        """
        mu, cov_raw = self.get_params(image_refs)   # [B, K], [B, n_cov]
        p2 = prior_std ** 2

        if self.cov_type == "diag":
            log_std = cov_raw                        # [B, K]
            var     = (2 * log_std).exp()            # σ_k²
            kl = 0.5 * (var / p2 + mu ** 2 / p2 - 1.0 - (2 * log_std - 2 * torch.tensor(prior_std, device=mu.device).log()))
        else:
            L = self._build_L(cov_raw)               # [B, K, K]
            diag_idx = torch.arange(self.K, device=L.device)
            log_diag = L[:, diag_idx, diag_idx].log()   # [B, K]
            trace_term = (L ** 2).sum(dim=(-2, -1)) / p2
            mu_term    = (mu ** 2).sum(dim=-1) / p2
            log_det    = 2 * log_diag.sum(dim=-1)
            kl = 0.5 * (trace_term + mu_term - self.K - (log_det - self.K * torch.tensor(p2).log()))
        return kl.mean()

    # -----------------------------------------------------------------------
    # Choice probabilities (same interface as DlbtAgent)
    # -----------------------------------------------------------------------

    def choice_probs(
        self, image_refs: List[ImageRef], task: Task
    ) -> torch.Tensor:
        if self.training:
            return self._choice_probs_train(image_refs, task)
        else:
            return self._choice_probs_eval(image_refs, task)

    def _choice_probs_train(
        self, image_refs: List[ImageRef], task: Task
    ) -> torch.Tensor:
        """
        Straight-through estimator — identical logic to DlbtAgent._choice_probs_train
        but samples from the logistic-normal instead of the Dirichlet.

          1. (μ, cov_raw) = mapper(encoder(x))         [B, K], [B, ·]
          2. z ~ N(μ, Σ), b = softmax(z)               [N, B, K]
          3. logit = b · δu                             [N, B]
          4. straight-through argmax → choice indicator [N, B, 2]
          5. average over N                             [B, 2]
        """
        N = self.n_mc_samples
        mu, cov_raw = self.get_params(image_refs)
        delta_u = torch.tensor(task.delta_u, dtype=mu.dtype, device=self.device)

        b     = self._sample_beliefs(mu, cov_raw, N)                 # [N, B, K]
        logit = torch.einsum("nbk,k->nb", b, delta_u)               # [N, B]

        logits_2d  = torch.stack([-logit, logit], dim=-1)
        probs_soft = F.softmax(logits_2d, dim=-1)
        hard       = F.one_hot(logits_2d.argmax(-1), 2).float()
        st         = (hard - probs_soft).detach() + probs_soft
        return st.mean(dim=0)                                         # [B, 2]

    @torch.no_grad()
    def _choice_probs_eval(
        self, image_refs: List[ImageRef], task: Task
    ) -> torch.Tensor:
        """
        Hard MC average — identical logic to DlbtAgent._choice_probs_eval.
        """
        N = self.n_mc_samples
        mu, cov_raw = self.get_params(image_refs)
        delta_u = torch.tensor(task.delta_u, dtype=mu.dtype, device=self.device)

        b       = self._sample_beliefs(mu, cov_raw, N)               # [N, B, K]
        logit   = torch.einsum("nbk,k->nb", b, delta_u)             # [N, B]
        hard    = (logit > 0).float()                                # [N, B]
        p_right = hard.mean(dim=0)                                   # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)          # [B, 2]

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> str:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"FlexAgent(K={self.K}, cov_type={self.cov_type!r}, "
            f"n_cov_params={self.n_cov_params}, "
            f"trainable={n_train:,})"
        )
