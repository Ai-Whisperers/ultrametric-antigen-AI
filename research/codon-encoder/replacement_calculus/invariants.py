"""Minimal Invariant Basis: Valuation, Redundancy, Symmetry Rank.

These three invariants characterize any configuration and form a partial order
that defines what transformations are valid.

Invariant Tuple: I(x) = (ν(x), ρ(x), σ(x))

Ordering: x ≤ y iff ν(x) ≤ ν(y) AND ρ(x) ≤ ρ(y) AND σ(x) ≤ σ(y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class InvariantTuple:
    """The minimal invariant basis for a configuration.

    Attributes:
        valuation: p-adic valuation (hierarchical depth)
        redundancy: coset index (degeneracy)
        symmetry_rank: automorphism dimension
    """
    valuation: int
    redundancy: int
    symmetry_rank: int

    def __ge__(self, other: InvariantTuple) -> bool:
        """Partial order: all components must be >=."""
        return (
            self.valuation >= other.valuation and
            self.redundancy >= other.redundancy and
            self.symmetry_rank >= other.symmetry_rank
        )

    def __gt__(self, other: InvariantTuple) -> bool:
        """Strict partial order."""
        return self >= other and self != other

    def dominates(self, other: InvariantTuple) -> bool:
        """Check if this tuple dominates another (Pareto dominance)."""
        return self >= other

    def distance(self, other: InvariantTuple) -> float:
        """L2 distance in invariant space."""
        return np.sqrt(
            (self.valuation - other.valuation) ** 2 +
            (self.redundancy - other.redundancy) ** 2 +
            (self.symmetry_rank - other.symmetry_rank) ** 2
        )


def valuation(x: int, p: int = 3) -> int:
    """Compute p-adic valuation ν_p(x).

    The valuation measures hierarchical depth:
    - High valuation → deep in hierarchy → hard to perturb
    - Low valuation → near boundary → easily perturbed
    - Zero valuation → maximally exposed

    Args:
        x: Integer to compute valuation for
        p: Prime base (default 3 for ternary)

    Returns:
        Largest power of p that divides x

    Examples:
        >>> valuation(9, 3)  # 9 = 3^2
        2
        >>> valuation(6, 3)  # 6 = 2 * 3
        1
        >>> valuation(5, 3)  # 5 not divisible by 3
        0
    """
    if x == 0:
        return 100  # Represents infinity

    x = abs(x)
    v = 0
    while x % p == 0:
        v += 1
        x //= p
    return v


def redundancy(
    element: str,
    equivalence_class: Dict[str, List[str]],
) -> int:
    """Compute redundancy (coset index) for an element.

    Redundancy measures degeneracy:
    - High redundancy → many equivalent representations → robust
    - Low redundancy → unique representation → fragile
    - ρ = 1 → no redundancy → crystallized

    Args:
        element: The element to compute redundancy for
        equivalence_class: Mapping from element to equivalent representations

    Returns:
        Number of equivalent representations

    Examples:
        >>> redundancy('L', AMINO_ACID_TO_CODONS)  # Leucine has 6 codons
        6
        >>> redundancy('M', AMINO_ACID_TO_CODONS)  # Methionine has 1 codon
        1
    """
    return len(equivalence_class.get(element, [element]))


def symmetry_rank(
    structure: Union[np.ndarray, List[List[float]]],
    tolerance: float = 1e-6,
) -> int:
    """Compute symmetry rank (dimension of automorphism group).

    Symmetry rank measures flexibility:
    - High symmetry → many self-transformations → flexible
    - Low symmetry → rigid structure → specialized
    - σ = 0 → asymmetric → maximally differentiated

    Args:
        structure: Adjacency matrix or distance matrix representing the group
        tolerance: Numerical tolerance for eigenvalue comparison

    Returns:
        Dimension of automorphism group (count of repeated eigenvalues)

    Examples:
        >>> # Fully symmetric (all same)
        >>> symmetry_rank(np.ones((3, 3)))
        2
        >>> # Asymmetric (random)
        >>> symmetry_rank(np.random.rand(3, 3))
        0
    """
    if isinstance(structure, list):
        structure = np.array(structure)

    # Ensure square matrix
    if structure.ndim != 2 or structure.shape[0] != structure.shape[1]:
        return 0

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(structure)
    except np.linalg.LinAlgError:
        return 0

    # Round to tolerance and count degeneracies
    rounded = np.round(eigenvalues / tolerance) * tolerance
    unique, counts = np.unique(rounded, return_counts=True)

    # Symmetry rank = number of repeated eigenvalues
    return sum(counts[counts > 1])


def symmetry_rank_from_points(
    points: np.ndarray,
    tolerance: float = 1e-6,
) -> int:
    """Compute symmetry rank from a set of points.

    Uses the pairwise distance matrix to compute structure.

    Args:
        points: Array of shape (n_points, n_dims)
        tolerance: Numerical tolerance

    Returns:
        Symmetry rank
    """
    from scipy.spatial.distance import pdist, squareform

    if len(points) < 2:
        return 0

    # Compute pairwise distance matrix
    distances = squareform(pdist(points))

    return symmetry_rank(distances, tolerance)


def invariant_tuple(
    x: int,
    equivalence_class: Dict[str, List[str]] = None,
    structure: np.ndarray = None,
    p: int = 3,
) -> InvariantTuple:
    """Compute the complete invariant tuple for a configuration.

    Args:
        x: Integer representation of configuration
        equivalence_class: Mapping for redundancy computation
        structure: Matrix for symmetry computation
        p: Prime for valuation

    Returns:
        InvariantTuple(valuation, redundancy, symmetry_rank)
    """
    v = valuation(x, p)

    if equivalence_class is not None:
        ρ = redundancy(str(x), equivalence_class)
    else:
        ρ = 1

    if structure is not None:
        σ = symmetry_rank(structure)
    else:
        σ = 0

    return InvariantTuple(valuation=v, redundancy=ρ, symmetry_rank=σ)


def compare_invariants(
    I1: InvariantTuple,
    I2: InvariantTuple,
) -> str:
    """Compare two invariant tuples.

    Returns:
        'dominates': I1 > I2 (strictly better)
        'dominated': I1 < I2 (strictly worse)
        'equal': I1 == I2
        'incomparable': Neither dominates
    """
    if I1 == I2:
        return "equal"
    elif I1 > I2:
        return "dominates"
    elif I2 > I1:
        return "dominated"
    else:
        return "incomparable"
