"""
Embedding value object - represents a vector embedding.

Embeddings are immutable vector representations of biological entities
(codons, sequences, etc.) in various metric spaces.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class EmbeddingSpace(Enum):
    """Type of embedding space."""
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"
    PADIC = "padic"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class Embedding:
    """
    Immutable embedding vector in a metric space.

    Attributes:
        vector: The embedding coordinates as immutable tuple
        space: The type of embedding space
        curvature: Curvature parameter (for hyperbolic space)
        metadata: Optional metadata
    """

    vector: tuple[float, ...]
    space: EmbeddingSpace = EmbeddingSpace.EUCLIDEAN
    curvature: float = -1.0
    metadata: Optional[tuple[tuple[str, ...], ...]] = None

    @classmethod
    def from_list(cls, values: list[float], **kwargs) -> "Embedding":
        """Create from list of values."""
        return cls(vector=tuple(values), **kwargs)

    @classmethod
    def from_array(cls, arr, **kwargs) -> "Embedding":
        """Create from numpy array."""
        return cls(vector=tuple(float(x) for x in arr.flatten()), **kwargs)

    def to_list(self) -> list[float]:
        """Convert to list."""
        return list(self.vector)

    def to_array(self):
        """Convert to numpy array."""
        import numpy as np
        return np.array(self.vector)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.vector)

    @property
    def norm(self) -> float:
        """Calculate Euclidean norm."""
        return math.sqrt(sum(x * x for x in self.vector))

    @property
    def is_valid_poincare(self) -> bool:
        """Check if valid point in Poincaré disk (norm < 1)."""
        return self.norm < 1.0

    def __len__(self) -> int:
        return self.dimension

    def __getitem__(self, index: int) -> float:
        return self.vector[index]

    def __iter__(self):
        return iter(self.vector)

    def __add__(self, other: "Embedding") -> "Embedding":
        """Element-wise addition (Euclidean only)."""
        if self.space != EmbeddingSpace.EUCLIDEAN:
            raise ValueError("Direct addition only valid for Euclidean space")
        if len(self) != len(other):
            raise ValueError("Dimension mismatch")
        return Embedding(
            vector=tuple(a + b for a, b in zip(self.vector, other.vector)),
            space=self.space,
        )

    def scale(self, scalar: float) -> "Embedding":
        """Scale embedding by scalar."""
        return Embedding(
            vector=tuple(x * scalar for x in self.vector),
            space=self.space,
            curvature=self.curvature,
        )

    def __repr__(self) -> str:
        vec_str = f"[{', '.join(f'{x:.4f}' for x in self.vector[:3])}{'...' if len(self) > 3 else ''}]"
        return f"Embedding({vec_str}, space={self.space.value})"


@dataclass(frozen=True, slots=True)
class SequenceEmbedding:
    """
    Embedding of an entire sequence (trajectory through space).

    Contains embeddings for each codon/position plus aggregate statistics.
    """

    codon_embeddings: tuple[Embedding, ...]
    centroid: Embedding
    dispersion: float
    trajectory_length: float

    @classmethod
    def from_embeddings(cls, embeddings: list[Embedding]) -> "SequenceEmbedding":
        """Create from list of codon embeddings."""
        if not embeddings:
            raise ValueError("Cannot create sequence embedding from empty list")

        n = len(embeddings)
        dim = embeddings[0].dimension

        # Calculate centroid
        centroid_vec = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb.vector):
                centroid_vec[i] += v
        centroid_vec = [v / n for v in centroid_vec]
        centroid = Embedding(
            vector=tuple(centroid_vec),
            space=embeddings[0].space,
            curvature=embeddings[0].curvature,
        )

        # Calculate dispersion (average distance from centroid)
        dispersion = 0.0
        for emb in embeddings:
            dist_sq = sum((a - b) ** 2 for a, b in zip(emb.vector, centroid_vec))
            dispersion += math.sqrt(dist_sq)
        dispersion /= n

        # Calculate trajectory length
        trajectory_length = 0.0
        for i in range(1, n):
            dist_sq = sum(
                (a - b) ** 2
                for a, b in zip(embeddings[i].vector, embeddings[i - 1].vector)
            )
            trajectory_length += math.sqrt(dist_sq)

        return cls(
            codon_embeddings=tuple(embeddings),
            centroid=centroid,
            dispersion=dispersion,
            trajectory_length=trajectory_length,
        )

    @property
    def num_codons(self) -> int:
        """Number of codons in sequence."""
        return len(self.codon_embeddings)

    def __len__(self) -> int:
        return self.num_codons
