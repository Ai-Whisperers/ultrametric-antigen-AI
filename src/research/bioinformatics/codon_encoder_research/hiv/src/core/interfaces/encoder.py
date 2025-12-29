"""
Encoder interfaces - protocols for sequence encoding.

Defines the contract for encoding biological sequences into
mathematical representations (p-adic, hyperbolic, etc.).
"""
from typing import Protocol, runtime_checkable

from ..entities.codon import Codon
from ..entities.sequence import Sequence
from ..entities.embedding import Embedding, SequenceEmbedding


@runtime_checkable
class ICodonEncoder(Protocol):
    """Protocol for codon-level encoding."""

    def encode_codon(self, codon: Codon) -> Embedding:
        """
        Encode a single codon to embedding.

        Args:
            codon: The codon to encode

        Returns:
            Embedding vector for the codon
        """
        ...

    def codon_distance(self, codon1: Codon, codon2: Codon) -> float:
        """
        Calculate distance between two codons in embedding space.

        Args:
            codon1: First codon
            codon2: Second codon

        Returns:
            Distance in the embedding space
        """
        ...


@runtime_checkable
class ISequenceEncoder(Protocol):
    """Protocol for sequence-level encoding."""

    def encode_sequence(self, sequence: Sequence) -> SequenceEmbedding:
        """
        Encode a full sequence to embedding.

        Args:
            sequence: The sequence to encode

        Returns:
            SequenceEmbedding containing trajectory and statistics
        """
        ...

    def sequence_distance(
        self, seq1: Sequence, seq2: Sequence
    ) -> float:
        """
        Calculate distance between two sequences in embedding space.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Distance measure between sequences
        """
        ...


@runtime_checkable
class IEncoder(ICodonEncoder, ISequenceEncoder, Protocol):
    """
    Combined encoder protocol for full encoding capability.

    Inherits from both ICodonEncoder and ISequenceEncoder.
    """

    def embedding_distance(self, emb1: Embedding, emb2: Embedding) -> float:
        """
        Calculate distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Distance in the embedding space
        """
        ...

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding space."""
        ...

    @property
    def space_type(self) -> str:
        """Get the type of embedding space (e.g., 'hyperbolic', 'padic')."""
        ...
