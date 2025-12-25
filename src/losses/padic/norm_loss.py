# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""p-Adic Norm Loss for MSB/LSB hierarchy enforcement.

This module implements Phase 1B from implement.md:
Enforce MSB/LSB hierarchy via p-adic norm regularization.

Single responsibility: p-Adic norm alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.constants import DEFAULT_LATENT_DIM
from src.core import TERNARY


class PAdicNormLoss(nn.Module):
    """p-Adic Norm Regularizer (Phase 1B from implement.md).

    Enforces MSB/LSB hierarchy by aligning latent p-adic norm with expected valuation:
    |z|_3 = max_i |z_i|^{1/3^i}

    L_norm = ||z|_3 - expected_valuation|²

    This fixes the MSB/LSB imbalance detected in hardest operations.
    """

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM):
        """Initialize p-Adic Norm Loss.

        Args:
            latent_dim: Dimensionality of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Precompute weights: 3^(-i) for i in [0, latent_dim)
        # Higher indices = LSB = smaller weight
        weights = torch.tensor([3.0 ** (-i) for i in range(latent_dim)])
        self.register_buffer("weights", weights)

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Compute p-Adic norm loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            p-Adic norm loss (scalar)
        """
        # Ensure weights are on same device as input
        weights = self.weights.to(z.device)

        # Compute p-adic norm of latent: |z|_3 = max_i |z_i|^{1/3^i}
        # Using log for numerical stability: log|z|_3 = max_i (log|z_i| / 3^i)
        z_abs = torch.abs(z) + 1e-8  # Avoid log(0)
        weighted_log = torch.log(z_abs) * weights.unsqueeze(0)
        z_padic_norm = torch.exp(weighted_log.max(dim=1)[0])

        # Compute expected 3-adic valuation from operation index
        expected_valuation = self._compute_expected_valuation(batch_indices)

        # MSE loss
        loss = F.mse_loss(z_padic_norm, expected_valuation)

        return loss

    def _compute_expected_valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute expected 3-adic valuation for operation indices.

        The 3-adic valuation v_3(n) = max k such that 3^k divides n.
        We normalize to [0, 1] range for stable training.

        Args:
            indices: Operation indices (batch,)

        Returns:
            Expected valuations normalized to [0, 1] (batch,)
        """
        v = TERNARY.valuation(indices).float()
        # Normalize to [0, 1] using 3^(-v)
        return torch.pow(3.0, -v)


__all__ = ["PAdicNormLoss"]
