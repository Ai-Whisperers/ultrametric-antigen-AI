"""p-Adic loss functions for 3-adic manifold coherence.

This module implements losses from implement.md Phase 1A/1B:
- PAdicMetricLoss: Force latent distances to match 3-adic distances
- PAdicNormLoss: Enforce MSB/LSB hierarchy via p-adic norm

Goal: Boost 3-adic correlation from r=0.62 to r>0.9

Single responsibility: p-Adic geometry alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import functools


@functools.lru_cache(maxsize=1)
def _get_3adic_distance_matrix(device: torch.device) -> torch.Tensor:
    """Precompute full 3-adic distance matrix for all 19,683 operations.

    The 3-adic distance between operations i and j is:
    d_3(i, j) = 3^(-v_3(i - j)) where v_3 is the 3-adic valuation

    Returns:
        Tensor of shape (19683, 19683) with 3-adic distances
    """
    n = 19683
    # Compute 3-adic valuations for all differences
    # v_3(k) = largest power of 3 dividing k
    distances = torch.zeros(n, n, device=device)

    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(i - j)
            if diff == 0:
                dist = 0.0
            else:
                # Compute 3-adic valuation
                v = 0
                while diff % 3 == 0:
                    v += 1
                    diff //= 3
                dist = 3.0 ** (-v)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def compute_3adic_distance_batch(
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute 3-adic distances for batches of index pairs.

    Args:
        idx_i: First indices (batch,)
        idx_j: Second indices (batch,)
        device: Device to compute on

    Returns:
        3-adic distances (batch,)
    """
    # Vectorized 3-adic distance computation
    diff = torch.abs(idx_i.long() - idx_j.long())

    # Handle zero differences
    distances = torch.zeros_like(diff, dtype=torch.float32)
    nonzero_mask = diff > 0

    if nonzero_mask.any():
        nonzero_diff = diff[nonzero_mask].float()

        # Compute 3-adic valuation: count factors of 3
        # v_3(n) = max k such that 3^k divides n
        v = torch.zeros_like(nonzero_diff)
        remaining = nonzero_diff.clone()

        for _ in range(9):  # Max 9 digits in base-3 for 19683
            divisible = (remaining % 3 == 0)
            if not divisible.any():
                break
            v[divisible] += 1
            remaining[divisible] = remaining[divisible] // 3

        # d_3(i, j) = 3^(-v_3(|i-j|))
        distances[nonzero_mask] = torch.pow(3.0, -v)

    return distances


class PAdicMetricLoss(nn.Module):
    """p-Adic Metric Loss (Phase 1A from implement.md).

    Forces latent space distances to match 3-adic distances:
    L_padic = Σ_{i,j} (||z_i - z_j|| - C * d_3(i, j))²

    This aligns the learned manifold geometry with the ultrametric structure
    of the 3-adic integers.
    """

    def __init__(
        self,
        scale: float = 1.0,
        n_pairs: int = 1000
    ):
        """Initialize p-Adic Metric Loss.

        Args:
            scale: Scaling factor C for 3-adic distances
            n_pairs: Number of pairs to sample per batch
        """
        super().__init__()
        self.scale = scale
        self.n_pairs = n_pairs

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
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

        # Compute latent distances (Euclidean)
        d_latent = torch.norm(z[i_idx] - z[j_idx], dim=1)

        # Compute 3-adic distances
        d_3adic = compute_3adic_distance_batch(
            batch_indices[i_idx],
            batch_indices[j_idx],
            z.device
        )

        # MSE loss: (d_latent - C * d_3adic)^2
        loss = F.mse_loss(d_latent, self.scale * d_3adic)

        return loss


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
        margin: float = 0.1,
        n_triplets: int = 500
    ):
        """Initialize p-Adic Ranking Loss.

        Args:
            margin: Margin for triplet loss (how much closer pos should be than neg)
            n_triplets: Number of triplets to sample per batch
        """
        super().__init__()
        self.margin = margin
        self.n_triplets = n_triplets

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
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

        # Compute 3-adic distances
        d3_anchor_pos = compute_3adic_distance_batch(
            batch_indices[anchor_idx], batch_indices[pos_idx], z.device
        )
        d3_anchor_neg = compute_3adic_distance_batch(
            batch_indices[anchor_idx], batch_indices[neg_idx], z.device
        )

        # Compute latent distances
        d_anchor_pos = torch.norm(z[anchor_idx] - z[pos_idx], dim=1)
        d_anchor_neg = torch.norm(z[anchor_idx] - z[neg_idx], dim=1)

        # Select triplets where pos is 3-adically closer than neg
        # (smaller 3-adic distance = closer in p-adic metric)
        valid = d3_anchor_pos < d3_anchor_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Triplet loss: we want d_latent_pos < d_latent_neg
        # Loss = max(0, d_pos - d_neg + margin)
        loss = F.relu(
            d_anchor_pos[valid] - d_anchor_neg[valid] + self.margin
        ).mean()

        return loss


class PAdicNormLoss(nn.Module):
    """p-Adic Norm Regularizer (Phase 1B from implement.md).

    Enforces MSB/LSB hierarchy by aligning latent p-adic norm with expected valuation:
    |z|_3 = max_i |z_i|^{1/3^i}

    L_norm = ||z|_3 - expected_valuation|²

    This fixes the MSB/LSB imbalance detected in hardest operations.
    """

    def __init__(self, latent_dim: int = 16):
        """Initialize p-Adic Norm Loss.

        Args:
            latent_dim: Dimensionality of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Precompute weights: 3^(-i) for i in [0, latent_dim)
        # Higher indices = LSB = smaller weight
        weights = torch.tensor([3.0 ** (-i) for i in range(latent_dim)])
        self.register_buffer('weights', weights)

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
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
        valuations = torch.zeros_like(indices, dtype=torch.float32)

        # Compute v_3(index) for each index
        nonzero_mask = indices > 0

        if nonzero_mask.any():
            remaining = indices[nonzero_mask].float()
            v = torch.zeros_like(remaining)

            for _ in range(9):  # Max 9 digits in base-3
                divisible = (remaining % 3 == 0)
                if not divisible.any():
                    break
                v[divisible] += 1
                remaining[divisible] = remaining[divisible] // 3

            # Normalize to [0, 1] using 3^(-v)
            valuations[nonzero_mask] = torch.pow(3.0, -v)

        return valuations
