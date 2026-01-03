"""Group Morphisms: Structure-Preserving Transformations.

Transformations between configurations are MORPHISMS, not patches.
A morphism is valid iff:
1. Valuation never decreases
2. Invariants are preserved or generalized
3. Entropy is displaced, not deleted

Key insight: Morphisms transform without accumulating junk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .groups import LocalMinimum
from .invariants import InvariantTuple, valuation


class MorphismType(Enum):
    """Types of group morphisms."""
    HOMOMORPHISM = "homomorphism"  # Preserves structure
    QUOTIENT = "quotient"          # Removes junk constraints
    EXTENSION = "extension"        # Adds dimensions
    ISOMORPHISM = "isomorphism"    # Bijective homomorphism


@dataclass
class Morphism:
    """A structure-preserving transformation between local minima.

    Attributes:
        source: The source local minimum
        target: The target local minimum
        map_function: Function mapping source generators to target generators
        morphism_type: Type of morphism
        geodesic: The path in embedding space (for hyperbolic)
        cost: Transformation cost (lower is better)
    """
    source: LocalMinimum
    target: LocalMinimum
    map_function: Callable[[int], int]
    morphism_type: MorphismType = MorphismType.HOMOMORPHISM
    geodesic: Optional[np.ndarray] = None
    cost: float = 0.0

    def apply(self, generator: int) -> int:
        """Apply the morphism to a generator."""
        return self.map_function(generator)

    def apply_all(self) -> List[int]:
        """Apply morphism to all source generators."""
        return [self.apply(g) for g in self.source.generators]

    def is_injective(self) -> bool:
        """Check if morphism is injective (one-to-one)."""
        images = self.apply_all()
        return len(images) == len(set(images))

    def is_surjective(self) -> bool:
        """Check if morphism is surjective (onto)."""
        images = set(self.apply_all())
        targets = set(self.target.generators)
        return images == targets

    def is_isomorphism(self) -> bool:
        """Check if morphism is an isomorphism."""
        return self.is_injective() and self.is_surjective()

    def kernel(self) -> List[int]:
        """Compute the kernel (elements mapping to identity/zero)."""
        identity = self.target.generators[0] if self.target.generators else 0
        return [g for g in self.source.generators if self.apply(g) == identity]


def is_valid_morphism(
    morphism: Morphism,
    p: int = 3,
    allow_entropy_increase: bool = False,
) -> Tuple[bool, str]:
    """Check if a morphism is valid according to replacement calculus rules.

    Validity criteria:
    1. Valuation preservation: ∀x: ν(φ(x)) ≥ ν(x)
    2. Invariant preservation: I(target) ≥ I(source)
    3. Entropy displacement: entropy change must be absorbed

    Args:
        morphism: The morphism to validate
        p: Prime for valuation computation
        allow_entropy_increase: Whether to allow entropy increase

    Returns:
        (is_valid, reason)
    """
    # Check 1: Valuation preservation
    for g in morphism.source.generators:
        v_source = valuation(g, p)
        v_target = valuation(morphism.apply(g), p)
        if v_target < v_source:
            return False, f"Valuation decreased: {v_source} → {v_target} for generator {g}"

    # Check 2: Invariant preservation
    I_source = morphism.source.invariant_tuple(p)
    I_target = morphism.target.invariant_tuple(p)

    if not I_target >= I_source:
        return False, f"Invariants not preserved: {I_source} → {I_target}"

    # Check 3: Entropy displacement
    # Approximate entropy as log of redundancy
    entropy_source = np.log(I_source.redundancy + 1)
    entropy_target = np.log(I_target.redundancy + 1)
    entropy_delta = entropy_target - entropy_source

    if entropy_delta < 0 and not allow_entropy_increase:
        # Entropy decreased - this is only valid if it's absorbed elsewhere
        # For now, we flag this as suspicious
        return False, f"Entropy decreased by {-entropy_delta:.3f} without absorption"

    return True, "Valid morphism"


def compute_morphism_cost(
    morphism: Morphism,
    p: int = 3,
) -> float:
    """Compute the cost of a morphism.

    Lower cost = better transformation.
    Cost considers:
    - Valuation change (should increase)
    - Symmetry change (should increase)
    - Geodesic length (should be short)

    Args:
        morphism: The morphism
        p: Prime for valuation

    Returns:
        Transformation cost
    """
    I_source = morphism.source.invariant_tuple(p)
    I_target = morphism.target.invariant_tuple(p)

    # Valuation improvement (negative = cost)
    valuation_delta = I_target.valuation - I_source.valuation

    # Symmetry improvement (negative = cost)
    symmetry_delta = I_target.symmetry_rank - I_source.symmetry_rank

    # Geodesic length cost
    if morphism.geodesic is not None:
        geodesic_cost = np.sum(np.linalg.norm(np.diff(morphism.geodesic, axis=0), axis=1))
    else:
        geodesic_cost = 0.0

    # Combined cost (negative improvements become positive costs)
    cost = (
        -1.0 * valuation_delta +    # Want valuation to increase
        -0.5 * symmetry_delta +     # Want symmetry to increase
        0.1 * geodesic_cost         # Want short paths
    )

    return max(0.0, cost)


def create_identity_morphism(minimum: LocalMinimum) -> Morphism:
    """Create identity morphism for a local minimum."""
    return Morphism(
        source=minimum,
        target=minimum,
        map_function=lambda x: x,
        morphism_type=MorphismType.ISOMORPHISM,
        cost=0.0,
    )


def create_quotient_morphism(
    source: LocalMinimum,
    target: LocalMinimum,
    kernel_elements: List[int],
) -> Morphism:
    """Create a quotient morphism that removes kernel elements.

    Quotient morphisms remove "junk" constraints without losing invariants.
    The kernel contains redundant elements that map to identity.

    Args:
        source: Source local minimum
        target: Target local minimum (quotient group)
        kernel_elements: Elements that map to identity

    Returns:
        Quotient morphism
    """
    kernel_set = set(kernel_elements)

    def quotient_map(x: int) -> int:
        if x in kernel_set:
            return target.generators[0] if target.generators else 0
        # Find corresponding element in target
        try:
            idx = source.generators.index(x)
            if idx < len(target.generators):
                return target.generators[idx]
        except (ValueError, IndexError):
            pass
        return x

    return Morphism(
        source=source,
        target=target,
        map_function=quotient_map,
        morphism_type=MorphismType.QUOTIENT,
        cost=len(kernel_elements),  # Cost = removed elements
    )


def compose_morphisms(f: Morphism, g: Morphism) -> Optional[Morphism]:
    """Compose two morphisms: g ∘ f (apply f first, then g).

    Args:
        f: First morphism (source → intermediate)
        g: Second morphism (intermediate → target)

    Returns:
        Composed morphism, or None if composition undefined
    """
    # Check compatibility
    if f.target != g.source:
        return None

    def composed_map(x: int) -> int:
        return g.apply(f.apply(x))

    # Combine geodesics if both exist
    if f.geodesic is not None and g.geodesic is not None:
        combined_geodesic = np.vstack([f.geodesic, g.geodesic[1:]])
    else:
        combined_geodesic = None

    return Morphism(
        source=f.source,
        target=g.target,
        map_function=composed_map,
        morphism_type=MorphismType.HOMOMORPHISM,
        geodesic=combined_geodesic,
        cost=f.cost + g.cost,
    )


def find_morphisms_between(
    source: LocalMinimum,
    target: LocalMinimum,
    p: int = 3,
    max_morphisms: int = 10,
) -> List[Morphism]:
    """Find valid morphisms between two local minima.

    Uses heuristics to generate candidate morphisms and filters valid ones.

    Args:
        source: Source local minimum
        target: Target local minimum
        p: Prime for valuation
        max_morphisms: Maximum morphisms to return

    Returns:
        List of valid morphisms, sorted by cost
    """
    morphisms = []

    # Try direct generator-to-generator mapping
    if len(source.generators) <= len(target.generators):
        from itertools import permutations

        for perm in permutations(range(len(target.generators)), len(source.generators)):
            def make_map(p=perm):
                mapping = dict(zip(source.generators, [target.generators[i] for i in p]))
                return lambda x: mapping.get(x, x)

            morphism = Morphism(
                source=source,
                target=target,
                map_function=make_map(),
                morphism_type=MorphismType.HOMOMORPHISM,
            )

            is_valid, _ = is_valid_morphism(morphism, p)
            if is_valid:
                morphism.cost = compute_morphism_cost(morphism, p)
                morphisms.append(morphism)

            if len(morphisms) >= max_morphisms:
                break

    # Sort by cost
    morphisms.sort(key=lambda m: m.cost)
    return morphisms[:max_morphisms]
