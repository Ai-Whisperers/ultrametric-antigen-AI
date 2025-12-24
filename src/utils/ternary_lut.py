"""Precomputed lookup tables for ternary operations.

P1 FIX: Eliminates redundant 3-adic valuation computation.
Previously 4 separate implementations with 9-iteration loops each,
called thousands of times per epoch.

Now: O(1) lookup instead of O(9) loop.

Usage:
    from src.utils.ternary_lut import VALUATION_LUT, TERNARY_LUT, get_valuation_batch

    # Single valuation lookup
    v = VALUATION_LUT[index]

    # Batch valuation lookup (tensor)
    valuations = get_valuation_batch(indices)

    # Ternary representation lookup
    ternary = TERNARY_LUT[index]  # Returns tensor of shape (9,)
"""

import torch


def _compute_valuation(n: int) -> int:
    """Compute 3-adic valuation of integer n.

    The 3-adic valuation v_3(n) is the largest k such that 3^k divides n.
    For n=0, returns 9 (maximum for 3^9 = 19683 space).

    Args:
        n: Integer in [0, 19682]

    Returns:
        3-adic valuation in [0, 9]
    """
    if n == 0:
        return 9  # Maximum depth

    v = 0
    while n % 3 == 0:
        v += 1
        n //= 3
    return v


def _compute_ternary(n: int) -> tuple:
    """Compute ternary representation of integer n.

    Converts index to 9-digit base-3 representation with values in {-1, 0, 1}.

    Args:
        n: Integer in [0, 19682]

    Returns:
        Tuple of 9 values in {-1, 0, 1}
    """
    digits = []
    for _ in range(9):
        digits.append((n % 3) - 1)  # Convert 0,1,2 to -1,0,1
        n //= 3
    return tuple(digits)


# Precompute valuation for all 19,683 possible indices
# Memory: 19,683 * 8 bytes (int64) = ~157 KB
VALUATION_LUT = torch.tensor(
    [_compute_valuation(i) for i in range(19683)],
    dtype=torch.long
)

# Precompute ternary representation for all 19,683 indices
# Memory: 19,683 * 9 * 4 bytes (float32) = ~708 KB
TERNARY_LUT = torch.tensor(
    [list(_compute_ternary(i)) for i in range(19683)],
    dtype=torch.float32
)


def get_valuation_batch(indices: torch.Tensor) -> torch.Tensor:
    """Get 3-adic valuations for a batch of indices.

    O(1) lookup per index instead of O(9) loop.

    Args:
        indices: Tensor of indices in [0, 19682], any shape

    Returns:
        Tensor of valuations, same shape as input
    """
    device = indices.device
    # Move LUT to device if needed (cached after first call)
    lut = VALUATION_LUT.to(device)
    return lut[indices]


def get_ternary_batch(indices: torch.Tensor) -> torch.Tensor:
    """Get ternary representations for a batch of indices.

    O(1) lookup per index instead of O(9) loop.

    Args:
        indices: Tensor of indices in [0, 19682], shape (N,)

    Returns:
        Tensor of ternary representations, shape (N, 9)
    """
    device = indices.device
    lut = TERNARY_LUT.to(device)
    return lut[indices]


def get_3adic_distance(i: int, j: int) -> float:
    """Compute 3-adic distance between two indices.

    d_3(i, j) = 3^(-v_3(|i - j|))

    Args:
        i, j: Indices in [0, 19682]

    Returns:
        3-adic distance in (0, 1]
    """
    if i == j:
        return 0.0
    diff = abs(i - j)
    v = VALUATION_LUT[diff].item()
    return 3.0 ** (-v)


def get_3adic_distance_batch(
    indices_i: torch.Tensor,
    indices_j: torch.Tensor
) -> torch.Tensor:
    """Compute 3-adic distances between pairs of indices.

    Args:
        indices_i, indices_j: Tensors of indices, same shape

    Returns:
        Tensor of 3-adic distances, same shape as input
    """
    diff = torch.abs(indices_i - indices_j)

    # Handle zeros (same index = distance 0)
    zero_mask = diff == 0

    # Clamp to valid range for LUT lookup
    diff = torch.clamp(diff, 0, 19682)

    valuations = get_valuation_batch(diff)
    distances = torch.pow(3.0, -valuations.float())

    # Set distance to 0 for identical indices
    distances[zero_mask] = 0.0

    return distances


__all__ = [
    'VALUATION_LUT',
    'TERNARY_LUT',
    'get_valuation_batch',
    'get_ternary_batch',
    'get_3adic_distance',
    'get_3adic_distance_batch',
]
