# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""p-Adic Metric Loss for distance alignment.

This module implements Phase 1A from implement.md:
Force latent distances to match 3-adic distances.

Single responsibility: Euclidean-to-p-adic distance alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.constants import DEFAULT_METRIC_LOSS_SCALE, DEFAULT_METRIC_N_PAIRS
from src.core import TERNARY
from src.geometry import poincare_distance


class PAdicMetricLoss(nn.Module):
    """p-Adic Metric Loss (Phase 1A from implement.md).

    Forces latent space distances to match 3-adic distances:
    L_padic = Σ_{i,j} (||z_i - z_j|| - C * d_3(i, j))²

    This aligns the learned manifold geometry with the ultrametric structure
    of the 3-adic integers.
    """

    def __init__(
        self,
        scale: float = DEFAULT_METRIC_LOSS_SCALE,
        n_pairs: int = DEFAULT_METRIC_N_PAIRS,
        use_hyperbolic: bool = False,
        curvature: float = 1.0,
    ):
        """Initialize p-Adic Metric Loss.

        Args:
            scale: Scaling factor C for 3-adic distances
            n_pairs: Number of pairs to sample per batch
            use_hyperbolic: If True, use poincare_distance (V5.12.2)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.scale = scale
        self.n_pairs = n_pairs
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Compute p-Adic metric loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            p-Adic metric loss (scalar)
        """
        batch_size = z.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        # Sample random pairs (more efficient than all pairs)
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        i_idx = torch.randint(0, batch_size, (n_pairs,), device=z.device)
        j_idx = torch.randint(0, batch_size, (n_pairs,), device=z.device)

        # Avoid self-pairs
        same_mask = i_idx == j_idx
        j_idx[same_mask] = (j_idx[same_mask] + 1) % batch_size

        # V5.12.2: Compute latent distances (Euclidean or Hyperbolic)
        if self.use_hyperbolic:
            d_latent = poincare_distance(z[i_idx], z[j_idx], c=self.curvature)
        else:
            d_latent = torch.norm(z[i_idx] - z[j_idx], dim=1)

        # Compute 3-adic distances using TERNARY singleton
        d_3adic = TERNARY.distance(batch_indices[i_idx], batch_indices[j_idx])

        # MSE loss: (d_latent - C * d_3adic)^2
        loss = F.mse_loss(d_latent, self.scale * d_3adic)

        return loss


__all__ = ["PAdicMetricLoss"]
