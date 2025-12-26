# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic valuation and Goldilocks zone utilities for immunology.

Consolidates duplicated p-adic functions from:
- src/diseases/multiple_sclerosis.py (compute_padic_valuation, lines 429-437)
- src/diseases/long_covid.py (compute_padic_valuation, lines 270-279, 603-611)
- src/diseases/rheumatoid_arthritis.py (GoldilocksZoneDetector)
- src/diseases/repeat_expansion.py (_compute_goldilocks_score)

The Goldilocks zone represents the immune escape region where antigens are:
- Different enough from self to be immunogenic
- Similar enough to self to cause cross-reactive autoimmunity
"""

from typing import Union

import numpy as np
import torch

# Default p-adic base (3-adic for ternary operations)
DEFAULT_P = 3

# Infinity representation for p-adic valuation when n=0
PADIC_INFINITY = float("inf")
PADIC_INFINITY_INT = 100  # Integer representation used in some modules


def compute_padic_valuation(
    n: int,
    p: int = DEFAULT_P,
    return_infinity: bool = True,
) -> Union[int, float]:
    """Compute p-adic valuation of integer n.

    The p-adic valuation v_p(n) is the highest power of p that divides n.

    This is the unified replacement for:
    - MultipleSclerosisAnalyzer.compute_padic_valuation()
    - LongCOVIDAnalyzer.compute_padic_valuation()
    - SpikeVariantComparator.compute_padic_valuation()

    Args:
        n: Integer to compute valuation for
        p: Prime base (default: 3 for ternary)
        return_infinity: If True, return float("inf") for n=0;
                        if False, return 100 (legacy compatibility)

    Returns:
        P-adic valuation (int) or infinity for n=0
    """
    if n == 0:
        return PADIC_INFINITY if return_infinity else PADIC_INFINITY_INT

    valuation = 0
    n = abs(n)
    while n % p == 0:
        valuation += 1
        n //= p

    return valuation


def compute_padic_distance(
    a: int,
    b: int,
    p: int = DEFAULT_P,
) -> float:
    """Compute p-adic distance between two integers.

    d_p(a, b) = p^(-v_p(a - b))

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default: 3)

    Returns:
        P-adic distance (0 to 1, where 0 means equal)
    """
    if a == b:
        return 0.0

    valuation = compute_padic_valuation(a - b, p)
    if valuation == PADIC_INFINITY:
        return 0.0

    return float(p) ** (-valuation)


def compute_goldilocks_score(
    distance: float,
    center: float = 0.5,
    width: float = 0.15,
    normalize: bool = True,
) -> float:
    """Compute Goldilocks zone score for a given distance.

    The Goldilocks zone is the "just right" region for autoimmune risk:
    - Too close (distance < center - width): indistinguishable from self
    - Too far (distance > center + width): clearly foreign, no cross-reactivity
    - Just right (near center): immunogenic yet cross-reactive

    This is the unified replacement for:
    - GoldilocksZoneDetector.zone_risk_score()
    - LongCOVIDAnalyzer._compute_goldilocks_score()
    - RepeatExpansionAnalyzer._compute_goldilocks_score()

    Args:
        distance: P-adic or structural distance from self
        center: Center of Goldilocks zone (default: 0.5)
        width: Width parameter (sigma) of Gaussian (default: 0.15)
        normalize: If True, normalize distance to [0, 1] first

    Returns:
        Score between 0 and 1 (1 = in zone, 0 = outside zone)
    """
    if normalize and distance > 1.0:
        distance = distance / (1.0 + distance)  # Sigmoid-like normalization

    # Gaussian scoring centered on the Goldilocks zone
    deviation = abs(distance - center)
    score = np.exp(-(deviation**2) / (2 * width**2))

    return float(score)


def is_in_goldilocks_zone(
    distance: float,
    center: float = 0.5,
    width: float = 0.15,
    threshold: float = 0.5,
) -> bool:
    """Check if distance falls within the Goldilocks zone.

    Args:
        distance: P-adic or structural distance
        center: Center of zone
        width: Width of zone
        threshold: Minimum score to be considered "in zone"

    Returns:
        True if in Goldilocks zone
    """
    score = compute_goldilocks_score(distance, center, width)
    return score >= threshold


def compute_padic_distance_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute p-adic distance between tensor elements.

    Vectorized version for batch processing.

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)
        p: Prime base

    Returns:
        Tensor of p-adic distances
    """
    diff = (a - b).abs().long()

    # Compute valuation for each element
    valuations = torch.zeros_like(diff, dtype=torch.float)

    # Iteratively divide by p to find valuation
    remaining = diff.clone()
    for v in range(20):  # Max valuation depth
        divisible = remaining % p == 0
        valuations[divisible] = v + 1
        remaining = remaining // p
        if not divisible.any():
            break

    # Handle zeros (infinite valuation)
    zero_mask = diff == 0
    valuations[zero_mask] = float("inf")

    # Convert to distance: p^(-v)
    distances = torch.pow(float(p), -valuations)
    distances[zero_mask] = 0.0

    return distances


def compute_hierarchical_padic_embedding(
    indices: torch.Tensor,
    n_digits: int = 9,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute hierarchical p-adic embedding from indices.

    Creates a multi-scale representation where each level corresponds
    to a p-adic digit, enabling hierarchical similarity comparisons.

    Args:
        indices: Tensor of integer indices
        n_digits: Number of p-adic digits (depth)
        p: Prime base

    Returns:
        Tensor of shape (..., n_digits) with p-adic digits
    """
    shape = indices.shape
    embedding = torch.zeros(*shape, n_digits, dtype=torch.float, device=indices.device)

    remaining = indices.clone()
    for d in range(n_digits):
        embedding[..., d] = (remaining % p).float()
        remaining = remaining // p

    return embedding


__all__ = [
    "DEFAULT_P",
    "PADIC_INFINITY",
    "PADIC_INFINITY_INT",
    "compute_padic_valuation",
    "compute_padic_distance",
    "compute_goldilocks_score",
    "is_in_goldilocks_zone",
    "compute_padic_distance_tensor",
    "compute_hierarchical_padic_embedding",
]
