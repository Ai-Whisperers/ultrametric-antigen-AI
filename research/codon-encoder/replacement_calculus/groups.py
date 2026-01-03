"""Local Minima as Algebraic Groups.

Each stable-but-suboptimal configuration is a group with:
- Generators: minimal functional primitives
- Relations: constraints maintaining stability
- Symmetry: what the configuration preserves
- Valuation: cost to break the configuration

Key insight: Bad local minima are not "wrong"—they are over-constrained groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .invariants import InvariantTuple, valuation, symmetry_rank


@dataclass
class Constraint:
    """A relation/constraint that maintains stability.

    Attributes:
        name: Human-readable name
        variables: Which generators are involved
        predicate: Function that checks if constraint is satisfied
        strength: How costly to violate (higher = more stable)
    """
    name: str
    variables: Tuple[int, ...]
    predicate: Callable[..., bool]
    strength: float = 1.0

    def is_satisfied(self, values: Dict[int, int]) -> bool:
        """Check if constraint is satisfied given generator values."""
        args = [values.get(v, 0) for v in self.variables]
        return self.predicate(*args)


@dataclass
class LocalMinimum:
    """A local minimum represented as an algebraic group.

    The group structure captures:
    - What primitives (generators) compose the configuration
    - What constraints (relations) keep it stable
    - What symmetries it possesses
    - What its invariant signature is

    Attributes:
        name: Identifier for this local minimum
        generators: Minimal functional primitives (as integers)
        relations: Constraints maintaining stability
        center: Representative point in embedding space
        members: All points belonging to this minimum
        metadata: Additional information
    """
    name: str
    generators: List[int]
    relations: List[Constraint] = field(default_factory=list)
    center: Optional[np.ndarray] = None
    members: List[np.ndarray] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    # Cached invariants
    _invariant_tuple: Optional[InvariantTuple] = field(default=None, repr=False)

    def __post_init__(self):
        """Compute cached values."""
        if self.center is None and self.members:
            self.center = np.mean(self.members, axis=0)

    @property
    def n_generators(self) -> int:
        """Number of generators."""
        return len(self.generators)

    @property
    def n_relations(self) -> int:
        """Number of relations/constraints."""
        return len(self.relations)

    @property
    def is_over_constrained(self) -> bool:
        """Check if group has too many relations relative to generators.

        Over-constrained groups are "stuck" in their local minimum
        because any change violates multiple constraints.
        """
        return self.n_relations > self.n_generators

    @property
    def constraint_ratio(self) -> float:
        """Ratio of constraints to generators."""
        if self.n_generators == 0:
            return float('inf')
        return self.n_relations / self.n_generators

    def compute_valuation(self, p: int = 3) -> int:
        """Compute mean valuation over generators."""
        if not self.generators:
            return 0
        return int(np.mean([valuation(g, p) for g in self.generators]))

    def compute_redundancy(self) -> int:
        """Compute redundancy (number of members)."""
        return max(1, len(self.members))

    def compute_symmetry_rank(self) -> int:
        """Compute symmetry rank from member point distribution."""
        if len(self.members) < 2:
            return 0

        members_array = np.array(self.members)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(members_array))
        return symmetry_rank(distances)

    def invariant_tuple(self, p: int = 3) -> InvariantTuple:
        """Compute the invariant tuple for this local minimum."""
        if self._invariant_tuple is None:
            self._invariant_tuple = InvariantTuple(
                valuation=self.compute_valuation(p),
                redundancy=self.compute_redundancy(),
                symmetry_rank=self.compute_symmetry_rank(),
            )
        return self._invariant_tuple

    def invalidate_cache(self):
        """Invalidate cached invariant tuple (call after modifications)."""
        self._invariant_tuple = None

    def total_constraint_strength(self) -> float:
        """Sum of all constraint strengths."""
        return sum(c.strength for c in self.relations)

    def check_all_constraints(self, values: Dict[int, int]) -> Tuple[int, int]:
        """Check how many constraints are satisfied.

        Returns:
            (satisfied_count, total_count)
        """
        satisfied = sum(1 for c in self.relations if c.is_satisfied(values))
        return satisfied, len(self.relations)

    def find_violated_constraints(self, values: Dict[int, int]) -> List[Constraint]:
        """Find which constraints are violated."""
        return [c for c in self.relations if not c.is_satisfied(values)]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, LocalMinimum):
            return False
        return self.name == other.name


def create_codon_local_minimum(
    amino_acid: str,
    codons: List[str],
    codon_to_index: Dict[str, int],
    center: Optional[np.ndarray] = None,
) -> LocalMinimum:
    """Create a LocalMinimum from amino acid and its codons.

    This maps the biological concept to the algebraic structure:
    - Generators: codon indices
    - Relations: synonymous codon constraints (all encode same AA)
    - Redundancy: number of codons (degeneracy)

    Args:
        amino_acid: The amino acid (e.g., 'L' for Leucine)
        codons: List of codons encoding this amino acid
        codon_to_index: Mapping from codon string to index
        center: Optional center point in embedding space

    Returns:
        LocalMinimum representing this amino acid's codon group
    """
    generators = [codon_to_index[c] for c in codons]

    # Create synonymous constraint: all codons encode same amino acid
    def synonymous_constraint(*indices):
        """All indices should map to same amino acid (trivially true by construction)."""
        return True

    relations = [
        Constraint(
            name=f"synonymous_{amino_acid}",
            variables=tuple(range(len(generators))),
            predicate=synonymous_constraint,
            strength=1.0,
        )
    ]

    return LocalMinimum(
        name=f"AA_{amino_acid}",
        generators=generators,
        relations=relations,
        center=center,
        metadata={"amino_acid": amino_acid, "codons": codons},
    )


def extract_local_minima_from_clusters(
    cluster_assignments: np.ndarray,
    embeddings: np.ndarray,
    indices: np.ndarray,
    n_clusters: int,
) -> List[LocalMinimum]:
    """Extract LocalMinimum objects from cluster assignments.

    Args:
        cluster_assignments: Cluster ID for each point
        embeddings: Embedding vectors for each point
        indices: Original indices (e.g., codon indices)
        n_clusters: Number of clusters

    Returns:
        List of LocalMinimum objects
    """
    minima = []

    for cluster_id in range(n_clusters):
        mask = cluster_assignments == cluster_id
        if not mask.any():
            continue

        cluster_indices = indices[mask].tolist()
        cluster_embeddings = embeddings[mask]
        cluster_center = np.mean(cluster_embeddings, axis=0)

        minimum = LocalMinimum(
            name=f"cluster_{cluster_id}",
            generators=cluster_indices,
            relations=[],  # No constraints by default
            center=cluster_center,
            members=[emb for emb in cluster_embeddings],
            metadata={"cluster_id": cluster_id},
        )

        minima.append(minimum)

    return minima
