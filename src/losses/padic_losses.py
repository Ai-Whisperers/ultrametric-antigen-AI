"""p-Adic loss functions for 3-adic manifold coherence.

This module implements losses from implement.md Phase 1A/1B:
- PAdicMetricLoss: Force latent distances to match 3-adic distances
- PAdicNormLoss: Enforce MSB/LSB hierarchy via p-adic norm
- PAdicRankingLossV2: Hard negative mining + hierarchical margin (v5.8)

Goal: Boost 3-adic correlation from r=0.62 to r>0.9+

Single responsibility: p-Adic geometry alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# STRUCTURAL FIX: Use core module's TERNARY singleton as single source of truth
from ..core import TERNARY


# Note: Full 19683x19683 distance matrix removed (was O(n²) dead code).
# Use TERNARY.distance() for efficient on-demand computation.


def compute_3adic_distance_batch(
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute 3-adic distances for batches of index pairs.

    Delegates to TERNARY.distance() for O(1) lookups.

    Args:
        idx_i: First indices (batch,)
        idx_j: Second indices (batch,)
        device: Device to compute on

    Returns:
        3-adic distances (batch,)
    """
    return TERNARY.distance(idx_i, idx_j)


def compute_3adic_valuation_batch(
    idx_i: torch.Tensor,
    idx_j: torch.Tensor
) -> torch.Tensor:
    """Compute 3-adic valuations for batches of index pairs.

    Delegates to TERNARY.valuation() for O(1) lookups.
    The valuation v_3(|i-j|) = max k such that 3^k divides |i-j|.

    Args:
        idx_i: First indices (batch,)
        idx_j: Second indices (batch,)

    Returns:
        3-adic valuations (batch,) - integers from 0 to 9
    """
    diff = torch.abs(idx_i.long() - idx_j.long())
    return TERNARY.valuation(diff).float()


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


class PAdicRankingLossV2(nn.Module):
    """Enhanced p-Adic Ranking Loss with Hard Negative Mining and Hierarchical Margin.

    Key improvements over PAdicRankingLoss:
    1. Hard negative mining: Focus on triplets that violate the ranking
    2. Hierarchical margin: Scale margin by valuation difference
    3. Semi-hard strategy: Select negatives that are close but wrong

    This addresses two bottlenecks:
    - Random sampling misses the hardest (most informative) triplets
    - Fixed margin ignores the multi-scale nature of 3-adic distances
    """

    def __init__(
        self,
        base_margin: float = 0.05,
        margin_scale: float = 0.15,
        n_triplets: int = 500,
        hard_negative_ratio: float = 0.5,
        semi_hard: bool = True
    ):
        """Initialize Enhanced p-Adic Ranking Loss.

        Args:
            base_margin: Minimum margin for all triplets
            margin_scale: Scale factor for valuation-based margin adjustment
            n_triplets: Number of triplets to sample per batch
            hard_negative_ratio: Fraction of triplets that should be hard negatives
            semi_hard: If True, use semi-hard negatives (close but wrong ordering)
        """
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.n_triplets = n_triplets
        self.hard_negative_ratio = hard_negative_ratio
        self.semi_hard = semi_hard

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute enhanced p-Adic ranking loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return torch.tensor(0.0, device=device), {'hard_ratio': 0.0, 'violations': 0}

        n_triplets = min(self.n_triplets, batch_size)
        n_hard = int(n_triplets * self.hard_negative_ratio)
        n_random = n_triplets - n_hard

        # === Phase 1: Hard Negative Mining ===
        if n_hard > 0:
            hard_triplets = self._mine_hard_negatives(z, batch_indices, n_hard)
        else:
            hard_triplets = None

        # === Phase 2: Random Triplets (for diversity) ===
        if n_random > 0:
            random_triplets = self._sample_random_triplets(z, batch_indices, n_random)
        else:
            random_triplets = None

        # === Combine triplets ===
        all_triplets = []
        if hard_triplets is not None and hard_triplets[0].size(0) > 0:
            all_triplets.append(hard_triplets)
        if random_triplets is not None and random_triplets[0].size(0) > 0:
            all_triplets.append(random_triplets)

        if len(all_triplets) == 0:
            return torch.tensor(0.0, device=device), {'hard_ratio': 0.0, 'violations': 0}

        # Concatenate all triplets
        anchor_idx = torch.cat([t[0] for t in all_triplets])
        pos_idx = torch.cat([t[1] for t in all_triplets])
        neg_idx = torch.cat([t[2] for t in all_triplets])
        v_pos = torch.cat([t[3] for t in all_triplets])
        v_neg = torch.cat([t[4] for t in all_triplets])

        # === Compute Hierarchical Margin ===
        # Larger valuation difference = easier distinction = larger margin
        v_diff = torch.abs(v_pos - v_neg)
        hierarchical_margin = self.base_margin + self.margin_scale * v_diff

        # === Compute Latent Distances ===
        d_anchor_pos = torch.norm(z[anchor_idx] - z[pos_idx], dim=1)
        d_anchor_neg = torch.norm(z[anchor_idx] - z[neg_idx], dim=1)

        # === Triplet Loss with Hierarchical Margin ===
        # We want d_pos < d_neg (positive is 3-adically closer)
        violations = d_anchor_pos - d_anchor_neg + hierarchical_margin
        loss = F.relu(violations).mean()

        # Compute metrics
        n_violations = (violations > 0).sum().item()
        actual_hard_ratio = n_hard / max(anchor_idx.size(0), 1)

        metrics = {
            'hard_ratio': actual_hard_ratio,
            'violations': n_violations,
            'mean_margin': hierarchical_margin.mean().item(),
            'total_triplets': anchor_idx.size(0)
        }

        return loss, metrics

    def _mine_hard_negatives(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor,
        n_hard: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hard negative triplets.

        Hard negatives are pairs where:
        - pos is 3-adically closer than neg (v_pos > v_neg)
        - BUT latent distance violates this (d_pos >= d_neg)

        Semi-hard negatives additionally require:
        - d_neg < d_pos + margin (negative is within margin)
        """
        batch_size = z.size(0)
        device = z.device

        # Sample candidate anchors
        n_candidates = min(batch_size, n_hard * 4)
        anchor_candidates = torch.randint(0, batch_size, (n_candidates,), device=device)

        # For each anchor, find pos/neg that violate ranking
        hard_anchors = []
        hard_pos = []
        hard_neg = []
        hard_v_pos = []
        hard_v_neg = []

        # Compute all pairwise latent distances for the batch
        with torch.no_grad():
            d_latent = torch.cdist(z, z, p=2)

        for anchor in anchor_candidates:
            if len(hard_anchors) >= n_hard:
                break

            anchor_idx_val = batch_indices[anchor]

            # Compute 3-adic valuations from anchor to all others
            v_to_all = compute_3adic_valuation_batch(
                anchor_idx_val.expand(batch_size),
                batch_indices
            )

            # Find candidate positives (high valuation = close in 3-adic)
            # and negatives (low valuation = far in 3-adic)
            v_sorted_idx = torch.argsort(v_to_all, descending=True)

            # Skip self
            v_sorted_idx = v_sorted_idx[v_sorted_idx != anchor]

            if len(v_sorted_idx) < 2:
                continue

            # Top half are potential positives, bottom half are potential negatives
            n_half = len(v_sorted_idx) // 2
            pos_candidates = v_sorted_idx[:max(n_half, 1)]
            neg_candidates = v_sorted_idx[max(n_half, 1):]

            if len(neg_candidates) == 0:
                continue

            # Find hard negatives: neg is latent-close but 3-adically far
            for pos in pos_candidates[:3]:  # Check top 3 positives
                d_ap = d_latent[anchor, pos]
                v_pos_val = v_to_all[pos]

                # Find negatives where d_an <= d_ap (violation) or d_an < d_ap + margin (semi-hard)
                for neg in neg_candidates:
                    d_an = d_latent[anchor, neg]
                    v_neg_val = v_to_all[neg]

                    # Must have v_pos > v_neg (pos is 3-adically closer)
                    if v_pos_val <= v_neg_val:
                        continue

                    # Check for violation or semi-hard condition
                    if self.semi_hard:
                        # Semi-hard: d_ap < d_an < d_ap + margin OR d_an < d_ap
                        is_hard = d_an < d_ap + self.base_margin + self.margin_scale * (v_pos_val - v_neg_val)
                    else:
                        # Pure hard: d_an <= d_ap (actual violation)
                        is_hard = d_an <= d_ap

                    if is_hard:
                        hard_anchors.append(anchor)
                        hard_pos.append(pos)
                        hard_neg.append(neg)
                        hard_v_pos.append(v_pos_val)
                        hard_v_neg.append(v_neg_val)

                        if len(hard_anchors) >= n_hard:
                            break
                if len(hard_anchors) >= n_hard:
                    break

        if len(hard_anchors) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device),
                torch.tensor([], device=device)
            )

        return (
            torch.stack(hard_anchors),
            torch.stack(hard_pos),
            torch.stack(hard_neg),
            torch.stack(hard_v_pos),
            torch.stack(hard_v_neg)
        )

    def _sample_random_triplets(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor,
        n_random: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random valid triplets (for diversity)."""
        batch_size = z.size(0)
        device = z.device

        # Sample random indices
        anchor_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        pos_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        neg_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)

        # Avoid degenerate triplets
        valid = (anchor_idx != pos_idx) & (anchor_idx != neg_idx) & (pos_idx != neg_idx)
        anchor_idx = anchor_idx[valid][:n_random]
        pos_idx = pos_idx[valid][:n_random]
        neg_idx = neg_idx[valid][:n_random]

        if len(anchor_idx) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device),
                torch.tensor([], device=device)
            )

        # Compute valuations
        v_pos = compute_3adic_valuation_batch(
            batch_indices[anchor_idx], batch_indices[pos_idx]
        )
        v_neg = compute_3adic_valuation_batch(
            batch_indices[anchor_idx], batch_indices[neg_idx]
        )

        # Filter to valid triplets (pos is 3-adically closer)
        valid_order = v_pos > v_neg
        anchor_idx = anchor_idx[valid_order]
        pos_idx = pos_idx[valid_order]
        neg_idx = neg_idx[valid_order]
        v_pos = v_pos[valid_order]
        v_neg = v_neg[valid_order]

        return anchor_idx, pos_idx, neg_idx, v_pos, v_neg


class PAdicRankingLossHyperbolic(nn.Module):
    """Hyperbolic p-Adic Ranking Loss using Poincaré distance.

    Key insight: 3-adic distances are ultrametric, and ultrametric spaces
    embed isometrically into hyperbolic space. By computing ranking loss
    using Poincaré distance instead of Euclidean, the VAE learns to arrange
    points in a hyperbolic-like structure that naturally captures tree geometry.

    The loss projects Euclidean latent points onto the Poincaré ball and
    measures distances using:
    d_poincare(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))

    Additionally includes radial hierarchy: points with high 3-adic valuation
    (divisible by 3^k) should be near the origin (tree root), while points
    with low valuation should be near the boundary (tree leaves).
    """

    def __init__(
        self,
        base_margin: float = 0.05,
        margin_scale: float = 0.15,
        n_triplets: int = 500,
        hard_negative_ratio: float = 0.5,
        curvature: float = 1.0,
        radial_weight: float = 0.1,
        max_norm: float = 0.95
    ):
        """Initialize Hyperbolic p-Adic Ranking Loss.

        Args:
            base_margin: Minimum margin for all triplets
            margin_scale: Scale factor for valuation-based margin adjustment
            n_triplets: Number of triplets to sample per batch
            hard_negative_ratio: Fraction of triplets that should be hard negatives
            curvature: Hyperbolic curvature (higher = more curved)
            radial_weight: Weight for radial hierarchy regularization
            max_norm: Maximum norm for Poincaré ball projection (< 1)
        """
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.n_triplets = n_triplets
        self.hard_negative_ratio = hard_negative_ratio
        self.curvature = curvature
        self.radial_weight = radial_weight
        self.max_norm = max_norm

    def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
        """Project Euclidean points onto the Poincaré ball.

        Uses the projection: z_hyp = z / (1 + ||z||) * max_norm

        This maps R^n -> B^n (open unit ball) while preserving directions.
        Points far from origin in Euclidean space map near the boundary.
        """
        norm = torch.norm(z, dim=1, keepdim=True)
        # Smooth projection that maps R^n to B^n
        z_hyp = z / (1 + norm) * self.max_norm
        return z_hyp

    def _poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré distance between points.

        d_poincare(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))

        For numerical stability, we use:
        arcosh(z) = log(z + sqrt(z² - 1)) for z >= 1
        """
        # Squared norms
        x_norm_sq = torch.sum(x ** 2, dim=1)
        y_norm_sq = torch.sum(y ** 2, dim=1)

        # Squared Euclidean distance
        diff_norm_sq = torch.sum((x - y) ** 2, dim=1)

        # Denominators (1 - ||x||²)(1 - ||y||²)
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=1e-10)  # Numerical stability

        # Argument to arcosh
        arg = 1 + 2 * diff_norm_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-7)  # arcosh requires arg >= 1

        # arcosh(z) = log(z + sqrt(z² - 1))
        distance = torch.log(arg + torch.sqrt(arg ** 2 - 1))

        return distance * self.curvature

    def _poincare_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute all pairwise Poincaré distances (vectorized).

        Args:
            z: Points in Poincaré ball (batch_size, latent_dim)

        Returns:
            Distance matrix (batch_size, batch_size)
        """
        # Squared norms for all points: (batch_size,)
        z_norm_sq = torch.sum(z ** 2, dim=1)

        # Pairwise squared Euclidean distances: (batch_size, batch_size)
        # ||z_i - z_j||² = ||z_i||² + ||z_j||² - 2 * z_i · z_j
        dot_products = torch.mm(z, z.t())  # (batch_size, batch_size)
        diff_norm_sq = (
            z_norm_sq.unsqueeze(1) + z_norm_sq.unsqueeze(0) - 2 * dot_products
        )
        diff_norm_sq = torch.clamp(diff_norm_sq, min=0.0)  # Numerical stability

        # Denominators: (1 - ||z_i||²)(1 - ||z_j||²)
        denom = (1 - z_norm_sq).unsqueeze(1) * (1 - z_norm_sq).unsqueeze(0)
        denom = torch.clamp(denom, min=1e-10)

        # Argument to arcosh
        arg = 1 + 2 * diff_norm_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-7)

        # arcosh(z) = log(z + sqrt(z² - 1))
        distance = torch.log(arg + torch.sqrt(arg ** 2 - 1))

        return distance * self.curvature

    def _compute_radial_loss(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute radial hierarchy loss.

        Points with high 3-adic valuation (divisible by large powers of 3)
        should be closer to origin. This creates tree-like hierarchy where:
        - Origin = root (indices divisible by 3^9)
        - Boundary = leaves (indices not divisible by 3)
        """
        # Compute 3-adic valuation for each index
        valuations = self._compute_valuation(batch_indices)

        # Normalize valuations to [0, 1] (0 = low valuation = boundary, 1 = high = center)
        max_val = 9.0  # Max valuation for 3^9 = 19683
        normalized_val = valuations / max_val

        # Target radius: high valuation -> small radius (near center)
        # radius = 1 - normalized_valuation (scaled by max_norm)
        target_radius = (1 - normalized_val) * self.max_norm * 0.9

        # Actual radius in Poincaré ball
        actual_radius = torch.norm(z_hyp, dim=1)

        # MSE loss on radii
        radial_loss = F.mse_loss(actual_radius, target_radius)

        return radial_loss

    def _compute_valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute 3-adic valuation for indices.

        Delegates to TERNARY.valuation() for O(1) lookups.
        """
        return TERNARY.valuation(indices).float()

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute hyperbolic p-Adic ranking loss.

        Args:
            z: Latent codes in Euclidean space (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return torch.tensor(0.0, device=device), {
                'hard_ratio': 0.0, 'violations': 0, 'poincare_dist_mean': 0.0
            }

        # Project to Poincaré ball
        z_hyp = self._project_to_poincare(z)

        # === Radial Hierarchy Loss ===
        radial_loss = self._compute_radial_loss(z_hyp, batch_indices)

        # === Triplet Ranking Loss with Poincaré Distance ===
        n_triplets = min(self.n_triplets, batch_size)
        n_hard = int(n_triplets * self.hard_negative_ratio)
        n_random = n_triplets - n_hard

        # Sample triplets (similar to V2, but we'll use Poincaré distance)
        all_triplets = []

        # Hard negative mining using Poincaré distances
        if n_hard > 0:
            hard_triplets = self._mine_hard_negatives_hyperbolic(
                z_hyp, batch_indices, n_hard
            )
            if hard_triplets[0].size(0) > 0:
                all_triplets.append(hard_triplets)

        # Random triplets for diversity
        if n_random > 0:
            random_triplets = self._sample_random_triplets(
                z_hyp, batch_indices, n_random
            )
            if random_triplets[0].size(0) > 0:
                all_triplets.append(random_triplets)

        if len(all_triplets) == 0:
            total_loss = self.radial_weight * radial_loss
            return total_loss, {
                'hard_ratio': 0.0, 'violations': 0,
                'poincare_dist_mean': 0.0, 'radial_loss': radial_loss.item()
            }

        # Concatenate all triplets
        anchor_idx = torch.cat([t[0] for t in all_triplets])
        pos_idx = torch.cat([t[1] for t in all_triplets])
        neg_idx = torch.cat([t[2] for t in all_triplets])
        v_pos = torch.cat([t[3] for t in all_triplets])
        v_neg = torch.cat([t[4] for t in all_triplets])

        # Hierarchical margin
        v_diff = torch.abs(v_pos - v_neg)
        hierarchical_margin = self.base_margin + self.margin_scale * v_diff

        # Compute POINCARÉ distances (not Euclidean!)
        d_anchor_pos = self._poincare_distance(z_hyp[anchor_idx], z_hyp[pos_idx])
        d_anchor_neg = self._poincare_distance(z_hyp[anchor_idx], z_hyp[neg_idx])

        # Triplet loss with hierarchical margin
        violations = d_anchor_pos - d_anchor_neg + hierarchical_margin
        ranking_loss = F.relu(violations).mean()

        # Total loss
        total_loss = ranking_loss + self.radial_weight * radial_loss

        # Compute metrics
        n_violations = (violations > 0).sum().item()
        actual_hard_ratio = n_hard / max(anchor_idx.size(0), 1)
        mean_poincare_dist = (d_anchor_pos.mean() + d_anchor_neg.mean()).item() / 2

        metrics = {
            'hard_ratio': actual_hard_ratio,
            'violations': n_violations,
            'mean_margin': hierarchical_margin.mean().item(),
            'total_triplets': anchor_idx.size(0),
            'poincare_dist_mean': mean_poincare_dist,
            'radial_loss': radial_loss.item(),
            'ranking_loss': ranking_loss.item()
        }

        return total_loss, metrics

    def _mine_hard_negatives_hyperbolic(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        n_hard: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hard negatives using Poincaré distances."""
        batch_size = z_hyp.size(0)
        device = z_hyp.device

        n_candidates = min(batch_size, n_hard * 4)
        anchor_candidates = torch.randint(0, batch_size, (n_candidates,), device=device)

        hard_anchors = []
        hard_pos = []
        hard_neg = []
        hard_v_pos = []
        hard_v_neg = []

        # Compute pairwise Poincaré distances (vectorized)
        with torch.no_grad():
            d_poincare_matrix = self._poincare_distance_matrix(z_hyp)

        for anchor in anchor_candidates:
            if len(hard_anchors) >= n_hard:
                break

            anchor_idx_val = batch_indices[anchor]

            # Compute valuations from anchor to all (vectorized)
            anchor_expanded = anchor_idx_val.expand(batch_size)
            v_to_all = compute_3adic_valuation_batch(anchor_expanded, batch_indices)

            v_sorted_idx = torch.argsort(v_to_all, descending=True)
            v_sorted_idx = v_sorted_idx[v_sorted_idx != anchor]

            if len(v_sorted_idx) < 2:
                continue

            n_half = len(v_sorted_idx) // 2
            pos_candidates = v_sorted_idx[:max(n_half, 1)]
            neg_candidates = v_sorted_idx[max(n_half, 1):]

            if len(neg_candidates) == 0:
                continue

            for pos in pos_candidates[:3]:
                d_ap = d_poincare_matrix[anchor, pos]
                v_pos_val = v_to_all[pos]

                for neg in neg_candidates:
                    d_an = d_poincare_matrix[anchor, neg]
                    v_neg_val = v_to_all[neg]

                    if v_pos_val <= v_neg_val:
                        continue

                    margin = self.base_margin + self.margin_scale * (v_pos_val - v_neg_val)
                    is_hard = d_an < d_ap + margin

                    if is_hard:
                        hard_anchors.append(anchor)
                        hard_pos.append(pos)
                        hard_neg.append(neg)
                        hard_v_pos.append(v_pos_val)
                        hard_v_neg.append(v_neg_val)

                        if len(hard_anchors) >= n_hard:
                            break
                if len(hard_anchors) >= n_hard:
                    break

        if len(hard_anchors) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device),
                torch.tensor([], device=device)
            )

        return (
            torch.stack(hard_anchors),
            torch.stack(hard_pos),
            torch.stack(hard_neg),
            torch.stack(hard_v_pos),
            torch.stack(hard_v_neg)
        )

    def _sample_random_triplets(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        n_random: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random valid triplets."""
        batch_size = z_hyp.size(0)
        device = z_hyp.device

        anchor_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        pos_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        neg_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)

        valid = (anchor_idx != pos_idx) & (anchor_idx != neg_idx) & (pos_idx != neg_idx)
        anchor_idx = anchor_idx[valid][:n_random]
        pos_idx = pos_idx[valid][:n_random]
        neg_idx = neg_idx[valid][:n_random]

        if len(anchor_idx) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device),
                torch.tensor([], device=device)
            )

        v_pos = compute_3adic_valuation_batch(
            batch_indices[anchor_idx], batch_indices[pos_idx]
        )
        v_neg = compute_3adic_valuation_batch(
            batch_indices[anchor_idx], batch_indices[neg_idx]
        )

        valid_order = v_pos > v_neg
        anchor_idx = anchor_idx[valid_order]
        pos_idx = pos_idx[valid_order]
        neg_idx = neg_idx[valid_order]
        v_pos = v_pos[valid_order]
        v_neg = v_neg[valid_order]

        return anchor_idx, pos_idx, neg_idx, v_pos, v_neg


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

        Delegates to TERNARY.valuation() for O(1) lookups.

        Args:
            indices: Operation indices (batch,)

        Returns:
            Expected valuations normalized to [0, 1] (batch,)
        """
        # Use TERNARY for O(1) valuation lookup
        v = TERNARY.valuation(indices).float()

        # Normalize to [0, 1] using 3^(-v)
        return torch.pow(3.0, -v)
