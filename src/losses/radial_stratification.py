"""Radial Stratification Loss for 3-adic hierarchy enforcement.

This module implements the radial component of the curriculum-based training:
- High 3-adic valuation → small radius (near origin)
- Low 3-adic valuation → large radius (near boundary)

The key insight: 3-adic tree structure maps naturally to hyperbolic geometry
where depth in the tree corresponds to distance from origin.

Single responsibility: Enforce radial hierarchy based on 3-adic valuation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# STRUCTURAL FIX: Use core module's TERNARY singleton as single source of truth
from ..core import TERNARY


def compute_single_index_valuation(indices: torch.Tensor) -> torch.Tensor:
    """Compute 3-adic valuation for single indices (not pairs).

    Delegates to TERNARY.valuation() for O(1) lookups.

    Args:
        indices: Operation indices (batch,) in range [0, 19682]

    Returns:
        valuations: 3-adic valuations (batch,) in range [0, 9]
    """
    return TERNARY.valuation(indices).float()


class RadialStratificationLoss(nn.Module):
    """Enforces radial hierarchy based on 3-adic valuation.

    Maps the 3-adic tree structure to radial position in latent space:
    - High valuation (e.g., 81 = 3^4) → small radius (near origin = tree root)
    - Low valuation (e.g., 1, 2, 4, 5) → large radius (near boundary = leaves)

    Formula:
        r_target = outer_radius - (v_3(n) / max_v) * (outer_radius - inner_radius)

    This creates concentric shells where:
    - Index 0 (v=9) targets inner_radius (tree root)
    - Indices with v=0 target outer_radius (tree leaves)

    Args:
        inner_radius: Target radius for highest valuation (default: 0.1)
        outer_radius: Target radius for lowest valuation (default: 0.85)
        max_valuation: Maximum valuation to normalize by (default: 9)
        valuation_weighting: Weight loss by valuation (high-v more important)
        loss_type: 'smooth_l1' (robust) or 'mse' (sharp gradients)
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        valuation_weighting: bool = True,
        loss_type: str = 'smooth_l1'
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.valuation_weighting = valuation_weighting
        self.loss_type = loss_type

        # Precompute radius range for efficiency
        self.radius_range = outer_radius - inner_radius

    def compute_target_radius(self, valuations: torch.Tensor) -> torch.Tensor:
        """Compute target radius from 3-adic valuations.

        Linear mapping: high valuation → small radius
        """
        normalized_v = valuations / self.max_valuation
        # Higher valuation = closer to inner (origin)
        target = self.outer_radius - normalized_v * self.radius_range
        return target

    def forward(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor,
        return_metrics: bool = False
    ) -> torch.Tensor:
        """Compute radial stratification loss.

        Args:
            z: Latent codes (batch, latent_dim)
            batch_indices: Operation indices (batch,) in range [0, 19682]
            return_metrics: If True, return dict with loss and metrics

        Returns:
            loss: Scalar loss value
            (optional) metrics: Dict with radial_correlation, mean_error, etc.
        """
        # 1. Compute 3-adic valuation for each index
        valuations = compute_single_index_valuation(batch_indices)

        # 2. Compute actual radius (Euclidean norm)
        actual_radius = torch.norm(z, dim=1)

        # 3. Compute target radius based on valuation
        target_radius = self.compute_target_radius(valuations)

        # 4. Compute loss
        if self.loss_type == 'smooth_l1':
            loss_per_sample = F.smooth_l1_loss(
                actual_radius, target_radius, reduction='none'
            )
        else:  # mse
            loss_per_sample = F.mse_loss(
                actual_radius, target_radius, reduction='none'
            )

        # 5. Apply valuation weighting if enabled
        # High-valuation points are rarer and more structurally important
        if self.valuation_weighting:
            # Weight = 1 + normalized_valuation (so high-v gets up to 2x weight)
            weights = 1.0 + (valuations / self.max_valuation)
            loss_per_sample = loss_per_sample * weights

        loss = loss_per_sample.mean()

        if return_metrics:
            with torch.no_grad():
                # Compute radial correlation (Spearman-like)
                # Higher valuation should correlate with smaller radius
                # So we correlate valuation with -radius (or equivalently, -valuation with radius)
                v_ranks = valuations.argsort().argsort().float()
                r_ranks = (-actual_radius).argsort().argsort().float()  # Negative because high-v = low radius
                n = len(v_ranks)
                correlation = 1 - 6 * ((v_ranks - r_ranks) ** 2).sum() / (n * (n**2 - 1) + 1e-8)

                metrics = {
                    'loss': loss.item(),
                    'radial_correlation': correlation.item(),
                    'mean_actual_radius': actual_radius.mean().item(),
                    'mean_target_radius': target_radius.mean().item(),
                    'mean_radius_error': (actual_radius - target_radius).abs().mean().item(),
                    'high_v_radius': actual_radius[valuations >= 4].mean().item() if (valuations >= 4).any() else 0.0,
                    'low_v_radius': actual_radius[valuations <= 1].mean().item() if (valuations <= 1).any() else 0.0,
                }
                return loss, metrics

        return loss

    def extra_repr(self) -> str:
        return (f'inner_radius={self.inner_radius}, '
                f'outer_radius={self.outer_radius}, '
                f'max_valuation={self.max_valuation}, '
                f'valuation_weighting={self.valuation_weighting}')
