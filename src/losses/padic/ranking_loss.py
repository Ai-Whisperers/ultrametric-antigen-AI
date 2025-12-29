# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Basic p-Adic Ranking Loss using contrastive triplets.

This module implements the original ranking-based p-adic loss
that preserves distance ORDER rather than matching absolute distances.

Single responsibility: Order-preserving ranking loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.constants import DEFAULT_N_TRIPLETS, DEFAULT_RANKING_MARGIN
from src.core import TERNARY
from src.geometry import poincare_distance


class PAdicRankingLoss(nn.Module):
    """Ranking-based p-Adic loss using contrastive triplets.

    Instead of matching absolute distances (which fails due to scale mismatch),
    this preserves the ORDER of 3-adic distances:
    If d_3(a,b) < d_3(a,c), then d_latent(a,b) should be < d_latent(a,c)

    This is more robust because:
    - Doesn't require matching absolute magnitudes
    - Naturally handles the exponential scale of 3-adic distances
    - Optimizes for Spearman correlation (ranking) rather than Pearson

    Uses margin-based triplet loss for stability.
    """

    def __init__(
        self,
        margin: float = DEFAULT_RANKING_MARGIN,
        n_triplets: int = DEFAULT_N_TRIPLETS,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize p-Adic Ranking Loss.

        Args:
            margin: Margin for triplet loss (how much closer pos should be than neg)
            n_triplets: Number of triplets to sample per batch
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.margin = margin
        self.n_triplets = n_triplets
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Compute p-Adic ranking loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            p-Adic ranking loss (scalar)
        """
        batch_size = z.size(0)

        if batch_size < 3:
            return torch.tensor(0.0, device=z.device)

        # Sample anchor, positive, negative triplets
        n_triplets = min(self.n_triplets, batch_size)
        anchor_idx = torch.randint(0, batch_size, (n_triplets,), device=z.device)
        pos_idx = torch.randint(0, batch_size, (n_triplets,), device=z.device)
        neg_idx = torch.randint(0, batch_size, (n_triplets,), device=z.device)

        # Avoid degenerate triplets (same indices)
        same_ap = anchor_idx == pos_idx
        same_an = anchor_idx == neg_idx
        pos_idx[same_ap] = (pos_idx[same_ap] + 1) % batch_size
        neg_idx[same_an] = (neg_idx[same_an] + 2) % batch_size

        # Compute 3-adic distances using TERNARY singleton
        d3_anchor_pos = TERNARY.distance(batch_indices[anchor_idx], batch_indices[pos_idx])
        d3_anchor_neg = TERNARY.distance(batch_indices[anchor_idx], batch_indices[neg_idx])

        # V5.12.2: Compute latent distances (Euclidean or Hyperbolic)
        if self.use_hyperbolic:
            d_anchor_pos = poincare_distance(z[anchor_idx], z[pos_idx], c=self.curvature)
            d_anchor_neg = poincare_distance(z[anchor_idx], z[neg_idx], c=self.curvature)
        else:
            d_anchor_pos = torch.norm(z[anchor_idx] - z[pos_idx], dim=1)
            d_anchor_neg = torch.norm(z[anchor_idx] - z[neg_idx], dim=1)

        # Select triplets where pos is 3-adically closer than neg
        # (smaller 3-adic distance = closer in p-adic metric)
        valid = d3_anchor_pos < d3_anchor_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Triplet loss: we want d_latent_pos < d_latent_neg
        # Loss = max(0, d_pos - d_neg + margin)
        loss = F.relu(d_anchor_pos[valid] - d_anchor_neg[valid] + self.margin).mean()

        return loss


__all__ = ["PAdicRankingLoss"]
