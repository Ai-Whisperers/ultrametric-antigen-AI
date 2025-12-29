"""
Repository interfaces - protocols for data access.

Defines contracts for accessing HIV datasets (Stanford, LANL, CATNAP, etc.)
following the repository pattern for clean separation of data access.
"""
from typing import Protocol, TypeVar, Optional, Iterator, runtime_checkable

from ..entities.sequence import Sequence
from ..entities.mutation import Mutation
from ..entities.epitope import Epitope


T = TypeVar("T")


@runtime_checkable
class IRepository(Protocol[T]):
    """Generic repository protocol."""

    def get(self, id: str) -> Optional[T]:
        """
        Get entity by ID.

        Args:
            id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        ...

    def get_all(self) -> list[T]:
        """
        Get all entities.

        Returns:
            List of all entities
        """
        ...

    def find(self, **criteria) -> list[T]:
        """
        Find entities matching criteria.

        Args:
            **criteria: Key-value pairs to match

        Returns:
            List of matching entities
        """
        ...

    def count(self) -> int:
        """
        Count total entities.

        Returns:
            Total count
        """
        ...


@runtime_checkable
class ISequenceRepository(Protocol):
    """Repository for sequence data."""

    def get_sequence(self, id: str) -> Optional[Sequence]:
        """Get sequence by ID."""
        ...

    def get_by_protein(self, protein: str) -> list[Sequence]:
        """Get all sequences for a protein."""
        ...

    def get_by_subtype(self, subtype: str) -> list[Sequence]:
        """Get sequences by HIV subtype (e.g., 'B', 'C')."""
        ...

    def iterate(self, batch_size: int = 100) -> Iterator[list[Sequence]]:
        """Iterate over sequences in batches."""
        ...


@runtime_checkable
class IMutationRepository(Protocol):
    """Repository for mutation data (Stanford HIVDB)."""

    def get_mutations_for_drug(self, drug: str) -> list[Mutation]:
        """Get all mutations associated with a drug."""
        ...

    def get_resistance_score(self, mutations: list[Mutation], drug: str) -> float:
        """Get resistance fold-change for mutations against a drug."""
        ...

    def get_drug_class(self, drug: str) -> str:
        """Get drug class (PI, NRTI, NNRTI, INSTI)."""
        ...


@runtime_checkable
class IEpitopeRepository(Protocol):
    """Repository for epitope data (LANL CTL)."""

    def get_epitopes_for_protein(self, protein: str) -> list[Epitope]:
        """Get all epitopes in a protein."""
        ...

    def get_epitopes_for_hla(self, hla: str) -> list[Epitope]:
        """Get all epitopes restricted by an HLA allele."""
        ...

    def get_epitopes_at_position(self, protein: str, position: int) -> list[Epitope]:
        """Get epitopes covering a specific position."""
        ...


@runtime_checkable
class INeutralizationRepository(Protocol):
    """Repository for neutralization data (CATNAP)."""

    def get_ic50(self, antibody: str, virus: str) -> Optional[float]:
        """Get IC50 for antibody-virus pair."""
        ...

    def get_sensitive_viruses(self, antibody: str, threshold: float = 50.0) -> list[str]:
        """Get viruses sensitive to an antibody."""
        ...

    def get_antibody_breadth(self, antibody: str, threshold: float = 50.0) -> float:
        """Calculate antibody breadth (% viruses neutralized)."""
        ...
