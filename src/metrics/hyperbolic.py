"""Hyperbolic geometry metrics for 3-adic structure evaluation.

This module provides metrics for evaluating how well the learned latent space
preserves 3-adic (p-adic with p=3) hierarchical structure using hyperbolic
(Poincare ball) geometry.

Key metrics:
- Ranking correlation: Concordance between 3-adic and Poincare distance orderings
- Mean radius: Distribution of points in hyperbolic space

Single responsibility: Hyperbolic geometry evaluation metrics only.

Note: Uses geoopt backend when available for numerical stability.
"""

import torch
from typing import Tuple

# Use geoopt-backed geometry module
from src.geometry import (
    project_to_poincare,
    poincare_distance,
)


def compute_3adic_valuation(diff: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
    """Compute 3-adic valuation of integer differences.

    The 3-adic valuation v_3(n) counts the highest power of 3 that divides n.
    Larger valuations indicate "closer" elements in 3-adic topology.

    Args:
        diff: Absolute differences between indices, shape (n_pairs,)
        max_depth: Maximum valuation depth to compute

    Returns:
        3-adic valuations, shape (n_pairs,)
    """
    val = torch.zeros_like(diff, dtype=torch.float32)
    remaining = diff.clone()

    for _ in range(max_depth):
        mask = (remaining % 3 == 0) & (remaining > 0)
        val[mask] += 1
        remaining[mask] = remaining[mask] // 3

    # Identical elements have "infinite" valuation (use max_depth as proxy)
    val[diff == 0] = float(max_depth)

    return val


def compute_ranking_correlation_hyperbolic(
    model: torch.nn.Module,
    device: str,
    n_samples: int = 5000,
    max_norm: float = 0.95,
    curvature: float = 2.0,
    n_triplets: int = 1000
) -> Tuple[float, float, float, float, float, float]:
    """Compute 3-adic ranking correlation using Poincare distance.

    Evaluates how well the learned hyperbolic embedding preserves the
    3-adic ultrametric structure by comparing distance orderings.

    For triplets (i, j, k), checks if:
    - 3-adic says j is closer to i than k → v_3(|i-j|) > v_3(|i-k|)
    - Hyperbolic says j is closer to i than k → d_hyp(z_i, z_j) < d_hyp(z_i, z_k)

    Concordance rate measures how often these orderings agree.

    Args:
        model: VAE model with forward() returning dict with z_A, z_B
        device: Device to run evaluation on
        n_samples: Number of samples to generate
        max_norm: Maximum norm for Poincare projection
        curvature: Hyperbolic curvature parameter
        n_triplets: Number of triplets to evaluate

    Returns:
        Tuple of (corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B)
        - corr_*_hyp: Hyperbolic correlation for VAE-A/B
        - corr_*_euc: Euclidean correlation for VAE-A/B (for comparison)
        - mean_radius_*: Mean radius in Poincare ball for VAE-A/B
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # Generate random operation indices
        indices = torch.randint(0, 19683, (n_samples,), device=device)

        # Convert indices to ternary representation
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Forward pass through model
        outputs = model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
        z_A = outputs['z_A']
        z_B = outputs['z_B']

        # Project to Poincare ball
        z_A_hyp = project_to_poincare(z_A, max_norm)
        z_B_hyp = project_to_poincare(z_B, max_norm)

        # Compute mean radius (for homeostatic monitoring)
        mean_radius_A = torch.norm(z_A_hyp, dim=1).mean().item()
        mean_radius_B = torch.norm(z_B_hyp, dim=1).mean().item()

        # Sample triplet indices
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        # Filter to distinct triplets
        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            # Not enough valid triplets
            if was_training:
                model.train()
            return 0.5, 0.5, 0.5, 0.5, mean_radius_A, mean_radius_B

        # Compute 3-adic valuations for pairs
        diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
        diff_ik = torch.abs(indices[i_idx] - indices[k_idx])

        v_ij = compute_3adic_valuation(diff_ij)
        v_ik = compute_3adic_valuation(diff_ik)

        # 3-adic ordering: larger valuation = smaller distance
        padic_closer_ij = (v_ij > v_ik).float()

        # Poincare distances
        d_A_ij = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx], curvature)
        d_A_ik = poincare_distance(z_A_hyp[i_idx], z_A_hyp[k_idx], curvature)
        d_B_ij = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx], curvature)
        d_B_ik = poincare_distance(z_B_hyp[i_idx], z_B_hyp[k_idx], curvature)

        # Euclidean distances (for comparison)
        d_A_ij_euc = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
        d_A_ik_euc = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
        d_B_ij_euc = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
        d_B_ik_euc = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

        # Hyperbolic correlations
        latent_A_closer_hyp = (d_A_ij < d_A_ik).float()
        latent_B_closer_hyp = (d_B_ij < d_B_ik).float()
        corr_A_hyp = (padic_closer_ij == latent_A_closer_hyp).float().mean().item()
        corr_B_hyp = (padic_closer_ij == latent_B_closer_hyp).float().mean().item()

        # Euclidean correlations
        latent_A_closer_euc = (d_A_ij_euc < d_A_ik_euc).float()
        latent_B_closer_euc = (d_B_ij_euc < d_B_ik_euc).float()
        corr_A_euc = (padic_closer_ij == latent_A_closer_euc).float().mean().item()
        corr_B_euc = (padic_closer_ij == latent_B_closer_euc).float().mean().item()

    if was_training:
        model.train()

    return corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B


__all__ = [
    'project_to_poincare',
    'poincare_distance',
    'compute_3adic_valuation',
    'compute_ranking_correlation_hyperbolic',
]
