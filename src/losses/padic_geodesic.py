"""Unified P-Adic Geodesic Loss for V5.11.

THE KEY V5.11 INNOVATION: Unify hierarchy and correlation in one loss.

Instead of competing losses:
  - ranking_loss: compares triplet orderings (relative)
  - radial_loss: pushes to target radii (absolute)

Use single geodesic loss:
  - For each pair (i,j): |d_poincare(z_i, z_j) - target(v_3(|i-j|))|²
  - Target maps valuation to hyperbolic distance
  - High valuation → small geodesic distance (automatically near origin)

In proper hyperbolic geometry, hierarchy IS correlation. The Poincaré metric
naturally couples them - two points near origin have smaller geodesic distance
than two at boundary.

Single responsibility: Unified p-adic geodesic alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..core import TERNARY


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-7
) -> torch.Tensor:
    """Compute Poincaré geodesic distance between points.

    d_poincare(x,y) = (1/sqrt(c)) * arcosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))

    Args:
        x: Points in Poincaré ball (batch, dim)
        y: Points in Poincaré ball (batch, dim)
        c: Curvature parameter (c > 0)
        eps: Numerical stability epsilon

    Returns:
        Geodesic distances (batch,)
    """
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_norm_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    return (1 / (c ** 0.5)) * torch.acosh(arg)


class PAdicGeodesicLoss(nn.Module):
    """Unified P-Adic Geodesic Loss.

    THE KEY V5.11 INSIGHT: In hyperbolic geometry, distance ordering and
    radial hierarchy are coupled via the metric itself.

    For pairs (i, j) with 3-adic valuation v_3(|i-j|):
    - High valuation → small target geodesic distance
    - Small geodesic distance in hyperbolic space → both points near origin

    This single loss enforces BOTH:
    - Hierarchy: High-valuation pairs close → both near origin
    - Correlation: Distance ordering matches valuation ordering

    No competing objectives - geometry handles both automatically.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_target_distance: float = 3.0,
        valuation_scale: float = 3.0,
        n_pairs: int = 2000,
        use_smooth_l1: bool = True
    ):
        """Initialize PAdicGeodesicLoss.

        Args:
            curvature: Poincaré ball curvature
            max_target_distance: Maximum target geodesic distance (for v=0)
            valuation_scale: Scale factor for valuation→distance mapping
            n_pairs: Number of pairs to sample per batch
            use_smooth_l1: Use SmoothL1 (Huber) loss instead of MSE
        """
        super().__init__()
        self.curvature = curvature
        self.max_target = max_target_distance
        self.valuation_scale = valuation_scale
        self.n_pairs = n_pairs
        self.use_smooth_l1 = use_smooth_l1

    def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
        """Map 3-adic valuation to target hyperbolic distance.

        v_3 = 0 (not divisible by 3)   → large distance (far apart)
        v_3 = 9 (divisible by 3^9)     → tiny distance (nearly same point)

        Formula: d_target = max_dist * exp(-valuation / scale)

        This exponential mapping matches the ultrametric structure of 3-adics.
        """
        return self.max_target * torch.exp(-valuation / self.valuation_scale)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute unified geodesic loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices for each sample (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z_hyp.size(0)
        device = z_hyp.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device), {'n_pairs': 0}

        # Sample random pairs
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        i_idx = torch.randint(0, batch_size, (n_pairs,), device=device)
        j_idx = torch.randint(0, batch_size, (n_pairs,), device=device)

        # Avoid self-pairs
        same_mask = i_idx == j_idx
        j_idx[same_mask] = (j_idx[same_mask] + 1) % batch_size

        # Compute actual Poincaré distance
        d_actual = poincare_distance(
            z_hyp[i_idx], z_hyp[j_idx], self.curvature
        )

        # Compute target distance from 3-adic valuation
        diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())
        valuation = TERNARY.valuation(diff).float()
        d_target = self.target_distance(valuation)

        # Loss: align actual geodesic distance with target
        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(d_actual, d_target)
        else:
            loss = F.mse_loss(d_actual, d_target)

        # Compute metrics
        with torch.no_grad():
            # Correlation between actual and target distances
            corr = torch.corrcoef(torch.stack([d_actual, d_target]))[0, 1]
            if torch.isnan(corr):
                corr = torch.tensor(0.0, device=device)

            # Mean distances by valuation level
            mean_d_low_v = d_actual[valuation < 2].mean() if (valuation < 2).any() else torch.tensor(0.0)
            mean_d_high_v = d_actual[valuation >= 4].mean() if (valuation >= 4).any() else torch.tensor(0.0)

        metrics = {
            'n_pairs': n_pairs,
            'mean_d_actual': d_actual.mean().item(),
            'mean_d_target': d_target.mean().item(),
            'distance_correlation': corr.item(),
            'mean_d_low_valuation': mean_d_low_v.item(),
            'mean_d_high_valuation': mean_d_high_v.item()
        }

        return loss, metrics


class RadialHierarchyLoss(nn.Module):
    """Radial Hierarchy Loss - direct radius enforcement.

    While PAdicGeodesicLoss handles hierarchy implicitly via geodesics,
    this loss provides explicit radial control for faster convergence.

    Target: v_3(n) high → radius small (near origin)
            v_3(n) low  → radius large (near boundary)

    V5.11.1 FIX: Added margin loss to enforce separation between valuation levels.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        valuation_weighting: bool = True,
        margin_weight: float = 1.0,
        use_margin_loss: bool = True
    ):
        """Initialize RadialHierarchyLoss.

        Args:
            inner_radius: Target radius for highest valuation (near origin)
            outer_radius: Target radius for lowest valuation (near boundary)
            max_valuation: Maximum valuation (log_3(19683) = 9)
            valuation_weighting: Weight high-valuation points more
            margin_weight: Weight for margin-based separation loss
            use_margin_loss: Enable pairwise margin loss for radial separation
        """
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.valuation_weighting = valuation_weighting
        self.margin_weight = margin_weight
        self.use_margin_loss = use_margin_loss

        # Compute radius step per valuation level
        self.radius_step = (outer_radius - inner_radius) / max_valuation

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute radial hierarchy loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = z_hyp.device
        batch_size = z_hyp.size(0)

        # Compute 3-adic valuation for each index
        valuations = TERNARY.valuation(batch_indices).float()

        # Compute actual radius
        actual_radius = torch.norm(z_hyp, dim=1)

        # Compute target radius (inverse relationship with valuation)
        normalized_v = valuations / self.max_valuation
        target_radius = self.outer_radius - normalized_v * (self.outer_radius - self.inner_radius)

        # Weighted loss (high-valuation points are rarer, more important)
        if self.valuation_weighting:
            # V5.11.2: Exponential weighting to compensate for rarity
            # v=0: weight=1, v=4: weight~5, v=7: weight~20, v=9: weight~50
            # This compensates for the fact that v=9 is ~20000x rarer than v=0
            weights = 1.0 + torch.exp(valuations * 0.4)  # exp(0.4*v)
        else:
            weights = torch.ones_like(normalized_v)

        # Primary loss: push each point to its target radius
        primary_loss = (F.mse_loss(actual_radius, target_radius, reduction='none') * weights).mean()

        # Margin loss: enforce separation between different valuation levels
        margin_loss = torch.tensor(0.0, device=device)
        if self.use_margin_loss and batch_size >= 2:
            # Sample pairs
            n_pairs = min(1000, batch_size * (batch_size - 1) // 2)
            i_idx = torch.randint(0, batch_size, (n_pairs,), device=device)
            j_idx = torch.randint(0, batch_size, (n_pairs,), device=device)

            # Avoid same index
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % batch_size

            v_i = valuations[i_idx]
            v_j = valuations[j_idx]
            r_i = actual_radius[i_idx]
            r_j = actual_radius[j_idx]

            # For pairs where v_i > v_j (i has higher valuation),
            # we want r_i < r_j (i should be closer to origin)
            higher_v_mask = v_i > v_j

            if higher_v_mask.any():
                # Expected margin: proportional to valuation difference
                v_diff = (v_i[higher_v_mask] - v_j[higher_v_mask])
                expected_margin = v_diff * self.radius_step * 0.5  # Half step as margin

                # Actual difference: r_j - r_i (should be positive)
                actual_diff = r_j[higher_v_mask] - r_i[higher_v_mask]

                # Margin violation: max(0, expected_margin - actual_diff)
                violations = F.relu(expected_margin - actual_diff)
                margin_loss = violations.mean()

        total_loss = primary_loss + self.margin_weight * margin_loss

        # Metrics
        with torch.no_grad():
            radial_corr = torch.corrcoef(torch.stack([valuations, -actual_radius]))[0, 1]
            if torch.isnan(radial_corr):
                radial_corr = torch.tensor(0.0, device=device)

            # Compute actual range
            radius_range = actual_radius.max() - actual_radius.min()

        metrics = {
            'mean_radius': actual_radius.mean().item(),
            'mean_target_radius': target_radius.mean().item(),
            'radial_hierarchy_corr': radial_corr.item(),
            'radius_min': actual_radius.min().item(),
            'radius_max': actual_radius.max().item(),
            'radius_range': radius_range.item(),
            'primary_loss': primary_loss.item(),
            'margin_loss': margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss
        }

        return total_loss, metrics


class CombinedGeodesicLoss(nn.Module):
    """Combined Geodesic + Radial Loss for V5.11.

    Wraps both losses with curriculum-based blending:
    - Early: More radial loss (establish hierarchy)
    - Late: More geodesic loss (refine correlation)

    The tau parameter controls the blend (can be learned by controller).
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_target_distance: float = 3.0,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        n_pairs: int = 2000
    ):
        super().__init__()
        self.geodesic_loss = PAdicGeodesicLoss(
            curvature=curvature,
            max_target_distance=max_target_distance,
            n_pairs=n_pairs
        )
        self.radial_loss = RadialHierarchyLoss(
            inner_radius=inner_radius,
            outer_radius=outer_radius
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        tau: float = 0.5
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss with curriculum blending.

        Args:
            z_hyp: Points in Poincaré ball
            batch_indices: Operation indices
            tau: Blend factor (0 = pure radial, 1 = pure geodesic)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        geo_loss, geo_metrics = self.geodesic_loss(z_hyp, batch_indices)
        rad_loss, rad_metrics = self.radial_loss(z_hyp, batch_indices)

        # Curriculum blend
        total_loss = (1 - tau) * rad_loss + tau * geo_loss

        # Merge metrics
        metrics = {
            'geodesic_loss': geo_loss.item(),
            'radial_loss': rad_loss.item(),
            'tau': tau,
            **{f'geo_{k}': v for k, v in geo_metrics.items()},
            **{f'rad_{k}': v for k, v in rad_metrics.items()}
        }

        return total_loss, metrics


__all__ = [
    'poincare_distance',
    'PAdicGeodesicLoss',
    'RadialHierarchyLoss',
    'CombinedGeodesicLoss'
]
