#!/usr/bin/env python3
"""Viral Projection Module: Transform 3-adic codon embeddings to 13-adic viral space.

Based on rigorous statistical analysis (see research/padic_structure_analysis/):

Key Findings:
- TrainableCodonEncoder learns 3-adic structure (codon grammar)
- DENV-4 viral evolution operates in 13-adic space (R² = 0.96)
- Adelic combination (2 + 13) achieves R² = 0.99

This module provides the mathematical transformation between these spaces,
enabling p-adic codon embeddings to predict viral evolutionary distances.

Usage:
    from src.viral_projection_module import ViralProjectionModule, compute_viral_distance

    # Initialize module
    module = ViralProjectionModule()

    # Transform hyperbolic embeddings to viral space
    z_viral = module(z_hyperbolic)

    # Or compute adjusted viral distance
    dist = compute_viral_distance(seq1_embedding, seq2_embedding, module)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Empirically determined weights from adelic analysis
# See: research/padic_structure_analysis/results/adelic_decomposition_results.json
ADELIC_WEIGHTS = {
    2: 0.077,   # Binary (purine/pyrimidine) structure
    3: -0.002,  # Codon position (weak, negative)
    5: 0.006,   # AA group structure (weak)
    7: 0.0,     # Not significant
    11: 0.0,    # Not significant
    13: 0.095,  # PRIMARY - viral evolutionary structure
    17: 0.016,  # Tertiary effect
    19: 0.036,  # Secondary structure
}

DOMINANT_PRIMES = [2, 13]  # Primes with |weight| > 0.05


@dataclass
class ProjectionConfig:
    """Configuration for viral projection module."""
    input_dim: int = 16  # TrainableCodonEncoder latent dim
    hidden_dim: int = 32
    output_dim: int = 16
    dominant_primes: list = None  # Default: [2, 13]
    use_adelic: bool = True  # Use multi-prime combination
    dropout: float = 0.1

    def __post_init__(self):
        if self.dominant_primes is None:
            self.dominant_primes = DOMINANT_PRIMES


def padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation v_p(n)."""
    if n == 0:
        return 50  # Large finite value
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def padic_distance_numpy(x: int, y: int, p: int) -> float:
    """Compute p-adic distance using numpy."""
    diff = abs(x - y)
    if diff == 0:
        return 0.0
    val = padic_valuation(diff, p)
    return float(p) ** (-val)


def sequence_to_index(seq: str) -> int:
    """Convert nucleotide sequence to integer (base-4)."""
    base_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 3}
    idx = 0
    for base in seq.upper():
        if base in base_map:
            idx = idx * 4 + base_map[base]
    return idx


def compute_adelic_distance(seq1: str, seq2: str, weights: dict = None) -> float:
    """Compute adelic distance between two sequences.

    Adelic distance = weighted combination of p-adic distances:
        d_adelic = sum_p w_p * d_p(seq1, seq2)

    Args:
        seq1, seq2: Nucleotide sequences
        weights: Dict of prime -> weight (default: ADELIC_WEIGHTS)

    Returns:
        Weighted p-adic distance
    """
    if weights is None:
        weights = ADELIC_WEIGHTS

    idx1 = sequence_to_index(seq1)
    idx2 = sequence_to_index(seq2)

    total_dist = 0.0
    for p, w in weights.items():
        if abs(w) > 0.001:  # Skip negligible weights
            d_p = padic_distance_numpy(idx1, idx2, p)
            total_dist += w * d_p

    return total_dist


if HAS_TORCH:

    class ViralProjectionModule(nn.Module):
        """Neural module to transform 3-adic codon embeddings to 13-adic viral space.

        Architecture:
            Input: z_hyperbolic (batch, input_dim) - TrainableCodonEncoder output
            -> Linear projection
            -> SiLU activation
            -> Dropout
            -> Linear projection
            -> Output: z_viral (batch, output_dim)

        The learned transformation captures the 3-adic → 13-adic projection
        that maps codon grammar to viral evolutionary structure.
        """

        def __init__(self, config: Optional[ProjectionConfig] = None):
            super().__init__()
            self.config = config or ProjectionConfig()

            # Learnable transformation
            self.projection = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.output_dim)
            )

            # Learnable prime weights (initialized from empirical values)
            if self.config.use_adelic:
                initial_weights = torch.tensor([
                    ADELIC_WEIGHTS.get(p, 0.0) for p in self.config.dominant_primes
                ])
                self.prime_weights = nn.Parameter(initial_weights)
            else:
                self.prime_weights = None

            # Scale factor for output normalization
            self.output_scale = nn.Parameter(torch.ones(1))

        def forward(self, z_hyperbolic: torch.Tensor) -> torch.Tensor:
            """Transform hyperbolic embeddings to viral space.

            Args:
                z_hyperbolic: (batch, input_dim) embeddings from TrainableCodonEncoder

            Returns:
                z_viral: (batch, output_dim) embeddings in viral evolutionary space
            """
            z_projected = self.projection(z_hyperbolic)

            # Normalize to unit ball (preserve hyperbolic structure)
            norm = torch.norm(z_projected, dim=-1, keepdim=True)
            z_normalized = z_projected / (norm + 1e-8)

            # Scale to desired radius
            z_viral = z_normalized * torch.sigmoid(self.output_scale) * 0.95

            return z_viral

        def compute_viral_distance(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            use_hyperbolic: bool = True
        ) -> torch.Tensor:
            """Compute distance in viral space.

            Args:
                z1, z2: Embeddings (batch, dim)
                use_hyperbolic: If True, use hyperbolic distance; else Euclidean

            Returns:
                distances: (batch,) pairwise distances
            """
            if use_hyperbolic:
                # Poincare distance
                diff = z1 - z2
                norm_sq = torch.sum(diff ** 2, dim=-1)
                x_sq = torch.sum(z1 ** 2, dim=-1)
                y_sq = torch.sum(z2 ** 2, dim=-1)

                num = norm_sq
                denom = (1 - x_sq) * (1 - y_sq) + 1e-8
                arg = 1 + 2 * num / denom

                return torch.acosh(torch.clamp(arg, min=1.0 + 1e-8))
            else:
                return torch.norm(z1 - z2, dim=-1)

        def get_prime_weights(self) -> dict:
            """Return current prime weights as dict."""
            if self.prime_weights is not None:
                weights = self.prime_weights.detach().cpu().numpy()
                return {p: float(w) for p, w in zip(self.config.dominant_primes, weights)}
            return {}


    class ViralDistanceLoss(nn.Module):
        """Loss function for training viral projection module.

        Trains the projection to minimize difference between:
            - Projected hyperbolic distance
            - Actual viral (Hamming) distance
        """

        def __init__(self, margin: float = 0.1):
            super().__init__()
            self.margin = margin

        def forward(
            self,
            z_viral1: torch.Tensor,
            z_viral2: torch.Tensor,
            viral_dist_target: torch.Tensor
        ) -> torch.Tensor:
            """Compute loss.

            Args:
                z_viral1, z_viral2: Projected embeddings
                viral_dist_target: Target viral distances (normalized Hamming)

            Returns:
                loss: Scalar loss value
            """
            # Compute predicted distance
            diff = z_viral1 - z_viral2
            pred_dist = torch.norm(diff, dim=-1)

            # Scale target to similar range
            target_scaled = viral_dist_target * 2.0  # Empirical scaling

            # Smooth L1 loss (robust to outliers)
            loss = F.smooth_l1_loss(pred_dist, target_scaled)

            return loss


else:
    # Fallback for non-PyTorch environments
    class ViralProjectionModule:
        """Numpy-based viral projection (for inference only)."""

        def __init__(self, config: Optional[ProjectionConfig] = None):
            self.config = config or ProjectionConfig()
            self.weights = None  # Would be loaded from checkpoint

        def __call__(self, z_hyperbolic: np.ndarray) -> np.ndarray:
            """Simple linear projection fallback."""
            # Without trained weights, just return input
            return z_hyperbolic

        def compute_viral_distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
            """Compute Euclidean distance."""
            return float(np.linalg.norm(z1 - z2))


def compute_viral_distance(
    seq1: str,
    seq2: str,
    use_adelic: bool = True
) -> float:
    """Convenience function to compute viral distance between sequences.

    Args:
        seq1, seq2: Nucleotide sequences (same length preferred)
        use_adelic: If True, use learned adelic weights; else simple Hamming

    Returns:
        distance: Normalized distance in [0, 1]
    """
    if use_adelic:
        return compute_adelic_distance(seq1, seq2)
    else:
        # Simple Hamming distance
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        mismatches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
        return mismatches / min_len


# Export analysis results path
def get_analysis_results() -> dict:
    """Load p-adic structure analysis results."""
    results_dir = Path(__file__).parent.parent / "research" / "padic_structure_analysis" / "results"

    results = {}
    for name in ["adelic_decomposition_results.json", "projection_deformation_results.json",
                 "multi_prime_ultrametric_results.json", "FINAL_VERDICT.json"]:
        path = results_dir / name
        if path.exists():
            with open(path) as f:
                results[name.replace(".json", "")] = json.load(f)

    return results


if __name__ == "__main__":
    # Quick test
    print("Viral Projection Module")
    print("=" * 50)

    # Test adelic distance
    seq1 = "ATGCATGCATGCATGCATGCATGC"
    seq2 = "ATGCATGCATGCATGCATGCATGC"
    seq3 = "TTTTTTTTTTTTTTTTTTTTTTTT"

    d12 = compute_viral_distance(seq1, seq2)
    d13 = compute_viral_distance(seq1, seq3)

    print(f"Distance (identical): {d12:.6f}")
    print(f"Distance (different): {d13:.6f}")

    print("\nAdelic weights:")
    for p, w in sorted(ADELIC_WEIGHTS.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(w) > 0.001:
            print(f"  {p}-adic: {w:.4f}")

    if HAS_TORCH:
        print("\nPyTorch module available")
        module = ViralProjectionModule()
        print(f"  Parameters: {sum(p.numel() for p in module.parameters())}")

        # Test forward pass
        z_test = torch.randn(4, 16)
        z_viral = module(z_test)
        print(f"  Input shape: {z_test.shape}")
        print(f"  Output shape: {z_viral.shape}")
    else:
        print("\nPyTorch not available, using numpy fallback")
