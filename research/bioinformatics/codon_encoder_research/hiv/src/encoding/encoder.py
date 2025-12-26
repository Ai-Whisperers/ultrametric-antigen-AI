"""
P-adic Hyperbolic Encoder - Main encoder orchestrator.

Combines p-adic number theory with hyperbolic geometry to
encode biological sequences in a space that naturally
represents evolutionary relationships.
"""
from dataclasses import dataclass, field

from ..core.entities.codon import Codon
from ..core.entities.sequence import Sequence, SequenceType
from ..core.entities.embedding import Embedding, EmbeddingSpace, SequenceEmbedding

from .padic.number import PadicNumber
from .padic.distance import normalized_padic_distance
from .hyperbolic.poincare.point import PoincarePoint
from .hyperbolic.poincare.operations import exp_map
from .hyperbolic.poincare.distance import poincare_distance
from .mapping.codon_to_padic import codon_to_padic_number
from .mapping.amino_acid_properties import get_aa_vector
from .mapping.hierarchy import hierarchical_encoding


@dataclass
class EncoderConfig:
    """Configuration for the encoder."""

    prime: int = 3
    embedding_dim: int = 3
    curvature: float = -1.0
    padic_precision: int = 10
    use_amino_acid_features: bool = True
    hyperbolic_scale: float = 0.5


@dataclass
class PadicHyperbolicEncoder:
    """
    Main encoder combining p-adic and hyperbolic representations.

    This encoder:
    1. Maps codons to p-adic numbers (hierarchical structure)
    2. Projects to hyperbolic space (tree-like phylogeny)
    3. Incorporates amino acid properties (biochemistry)

    Implements IEncoder protocol.
    """

    config: EncoderConfig = field(default_factory=EncoderConfig)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.embedding_dim

    @property
    def space_type(self) -> str:
        """Get space type."""
        return "padic_hyperbolic"

    def encode_codon(self, codon: Codon) -> Embedding:
        """
        Encode a single codon to embedding.

        Combines:
        1. P-adic encoding (hierarchical distance)
        2. Hyperbolic projection (tree structure)
        3. Amino acid properties (biochemistry)

        Args:
            codon: Codon to encode

        Returns:
            Embedding in hybrid p-adic hyperbolic space
        """
        # Step 1: P-adic encoding
        padic_num = codon_to_padic_number(
            codon.nucleotides,
            prime=self.config.prime,
            precision=self.config.padic_precision,
        )

        # Step 2: Get hierarchical encodings
        codon_p, aa_p, class_p = hierarchical_encoding(codon.nucleotides)

        # Step 3: Create feature vector
        features = self._create_feature_vector(codon, padic_num, aa_p, class_p)

        # Step 4: Project to hyperbolic space
        hyperbolic_point = self._project_to_hyperbolic(features)

        # Return as Embedding
        return Embedding(
            vector=hyperbolic_point.coords,
            space=EmbeddingSpace.HYPERBOLIC,
            curvature=self.config.curvature,
            metadata=(
                ("codon", codon.nucleotides),
                ("amino_acid", codon.amino_acid),
            ),
        )

    def _create_feature_vector(
        self,
        codon: Codon,
        padic_num: PadicNumber,
        aa_padic: PadicNumber,
        class_padic: PadicNumber,
    ) -> tuple[float, ...]:
        """Create combined feature vector."""
        features = []

        # P-adic norm features
        features.append(padic_num.norm())
        features.append(aa_padic.norm())
        features.append(class_padic.norm())

        # P-adic valuation features
        features.append(float(padic_num.valuation()) / self.config.padic_precision)

        # Amino acid property features
        if self.config.use_amino_acid_features:
            aa_vec = get_aa_vector(codon.amino_acid, normalize=True)
            features.extend(aa_vec)

        return tuple(features)

    def _project_to_hyperbolic(self, features: tuple[float, ...]) -> PoincarePoint:
        """Project feature vector to Poincaré disk."""
        # Reduce to target dimension if needed
        dim = self.config.embedding_dim
        if len(features) > dim:
            # Simple projection: take first dim components
            # (In production, use PCA or learned projection)
            reduced = features[:dim]
        else:
            # Pad with zeros
            reduced = features + (0.0,) * (dim - len(features))

        # Scale for hyperbolic projection
        scaled = tuple(f * self.config.hyperbolic_scale for f in reduced)

        # Project using exp map from origin
        return exp_map(scaled)

    def encode_sequence(self, sequence: Sequence) -> SequenceEmbedding:
        """
        Encode full sequence to embedding trajectory.

        Args:
            sequence: DNA/RNA sequence to encode

        Returns:
            SequenceEmbedding with per-codon embeddings and statistics
        """
        if sequence.sequence_type == SequenceType.PROTEIN:
            raise ValueError("Cannot encode protein sequence directly")

        codon_embeddings = []
        for codon in sequence.codons():
            emb = self.encode_codon(codon)
            codon_embeddings.append(emb)

        return SequenceEmbedding.from_embeddings(codon_embeddings)

    def codon_distance(self, codon1: Codon, codon2: Codon) -> float:
        """
        Calculate distance between codons in embedding space.

        Args:
            codon1: First codon
            codon2: Second codon

        Returns:
            Hyperbolic distance
        """
        emb1 = self.encode_codon(codon1)
        emb2 = self.encode_codon(codon2)
        return self.embedding_distance(emb1, emb2)

    def embedding_distance(self, emb1: Embedding, emb2: Embedding) -> float:
        """
        Calculate distance between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Distance in hyperbolic space
        """
        p1 = PoincarePoint(coords=emb1.vector, curvature=emb1.curvature)
        p2 = PoincarePoint(coords=emb2.vector, curvature=emb2.curvature)
        return poincare_distance(p1, p2)

    def sequence_distance(self, seq1: Sequence, seq2: Sequence) -> float:
        """
        Calculate distance between sequences.

        Uses Fréchet mean distance between sequence embeddings.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Sequence-level distance
        """
        emb1 = self.encode_sequence(seq1)
        emb2 = self.encode_sequence(seq2)

        # Distance between centroids
        return self.embedding_distance(emb1.centroid, emb2.centroid)

    def padic_codon_distance(self, codon1: Codon, codon2: Codon) -> float:
        """
        Calculate pure p-adic distance between codons.

        Args:
            codon1: First codon
            codon2: Second codon

        Returns:
            Normalized p-adic distance
        """
        p1 = codon_to_padic_number(codon1.nucleotides, prime=self.config.prime)
        p2 = codon_to_padic_number(codon2.nucleotides, prime=self.config.prime)
        return normalized_padic_distance(p1, p2)


# Convenience function
def create_encoder(
    prime: int = 3,
    embedding_dim: int = 3,
    curvature: float = -1.0,
) -> PadicHyperbolicEncoder:
    """
    Create encoder with specified parameters.

    Args:
        prime: Prime for p-adic representation
        embedding_dim: Dimension of hyperbolic embedding
        curvature: Curvature of hyperbolic space

    Returns:
        Configured encoder
    """
    config = EncoderConfig(
        prime=prime,
        embedding_dim=embedding_dim,
        curvature=curvature,
    )
    return PadicHyperbolicEncoder(config=config)
