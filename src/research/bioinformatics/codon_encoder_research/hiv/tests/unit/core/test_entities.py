"""
Unit tests for core entities.
"""
import pytest
from src.core.entities.codon import Codon
from src.core.entities.sequence import Sequence, SequenceType
from src.core.entities.mutation import Mutation
from src.core.entities.epitope import Epitope, EpitopeType
from src.core.entities.embedding import Embedding, EmbeddingSpace


class TestCodon:
    """Tests for Codon entity."""

    def test_valid_codon_creation(self):
        """Test creating valid codons."""
        codon = Codon("ATG")
        assert codon.nucleotides == "ATG"
        assert codon.amino_acid == "M"

    def test_codon_normalization(self):
        """Test lowercase normalization."""
        codon = Codon("atg")
        assert codon.nucleotides == "ATG"

    def test_rna_conversion(self):
        """Test RNA U to DNA T conversion."""
        codon = Codon("AUG")
        assert codon.nucleotides == "ATG"

    def test_stop_codon(self):
        """Test stop codon detection."""
        assert Codon("TAA").is_stop
        assert Codon("TAG").is_stop
        assert Codon("TGA").is_stop
        assert not Codon("ATG").is_stop

    def test_start_codon(self):
        """Test start codon detection."""
        assert Codon("ATG").is_start
        assert not Codon("TTT").is_start

    def test_invalid_codon_length(self):
        """Test rejection of invalid length."""
        with pytest.raises(ValueError):
            Codon("AT")
        with pytest.raises(ValueError):
            Codon("ATGC")

    def test_invalid_nucleotide(self):
        """Test rejection of invalid nucleotides."""
        with pytest.raises(ValueError):
            Codon("AXG")


class TestSequence:
    """Tests for Sequence entity."""

    def test_dna_sequence_creation(self):
        """Test DNA sequence creation."""
        seq = Sequence("ATGCCC", SequenceType.DNA)
        assert len(seq) == 6
        assert seq.sequence_type == SequenceType.DNA

    def test_codon_iteration(self):
        """Test codon extraction."""
        seq = Sequence("ATGCCCGGG", SequenceType.DNA)
        codons = list(seq.codons())
        assert len(codons) == 3
        assert codons[0].nucleotides == "ATG"
        assert codons[1].nucleotides == "CCC"
        assert codons[2].nucleotides == "GGG"

    def test_translation(self):
        """Test sequence translation."""
        seq = Sequence("ATGTTT", SequenceType.DNA)
        protein = seq.translate()
        assert protein.raw == "MF"
        assert protein.sequence_type == SequenceType.PROTEIN

    def test_gc_content(self):
        """Test GC content calculation."""
        seq = Sequence("GGCCATAT", SequenceType.DNA)
        assert seq.gc_content == 0.5

    def test_protein_codon_error(self):
        """Test error when getting codons from protein."""
        seq = Sequence("MFV", SequenceType.PROTEIN)
        with pytest.raises(ValueError):
            list(seq.codons())


class TestMutation:
    """Tests for Mutation entity."""

    def test_mutation_creation(self):
        """Test mutation creation."""
        mut = Mutation(wild_type="D", position=30, mutant="N")
        assert str(mut) == "D30N"

    def test_mutation_from_string(self):
        """Test parsing mutation from string."""
        mut = Mutation.from_string("D30N")
        assert mut.wild_type == "D"
        assert mut.position == 30
        assert mut.mutant == "N"

    def test_mutation_with_protein(self):
        """Test mutation with protein prefix."""
        mut = Mutation.from_string("PR:D30N")
        assert mut.protein == "PR"
        assert str(mut) == "PR:D30N"

    def test_synonymous_mutation(self):
        """Test synonymous mutation detection."""
        mut = Mutation(wild_type="A", position=10, mutant="A")
        assert mut.is_synonymous

    def test_nonsense_mutation(self):
        """Test nonsense mutation detection."""
        mut = Mutation(wild_type="W", position=50, mutant="*")
        assert mut.is_nonsense

    def test_invalid_mutation_format(self):
        """Test rejection of invalid format."""
        with pytest.raises(ValueError):
            Mutation.from_string("invalid")


class TestEpitope:
    """Tests for Epitope entity."""

    def test_epitope_creation(self):
        """Test epitope creation."""
        epitope = Epitope(
            sequence="SLYNTVATL",
            protein="Gag",
            start_position=77,
            end_position=85,
        )
        assert epitope.length == 9
        assert len(epitope.positions) == 9

    def test_epitope_anchor_positions(self):
        """Test anchor position calculation."""
        epitope = Epitope(
            sequence="SLYNTVATL",
            protein="Gag",
            start_position=77,
            end_position=85,
            epitope_type=EpitopeType.CTL,
        )
        p2, p_omega = epitope.anchor_positions
        assert p2 == 78
        assert p_omega == 85

    def test_epitope_overlap(self):
        """Test overlap detection."""
        e1 = Epitope("SLYNTVATL", "Gag", 77, 85)
        e2 = Epitope("TVATLYCVH", "Gag", 82, 90)
        e3 = Epitope("DIFFERENT", "Gag", 100, 108)
        e4 = Epitope("DIFFERENT", "Pol", 77, 85)

        assert e1.overlaps(e2)
        assert not e1.overlaps(e3)
        assert not e1.overlaps(e4)  # Different protein


class TestEmbedding:
    """Tests for Embedding entity."""

    def test_embedding_creation(self):
        """Test embedding creation."""
        emb = Embedding(vector=(0.1, 0.2, 0.3))
        assert emb.dimension == 3
        assert emb.norm == pytest.approx(0.374, rel=0.01)

    def test_embedding_from_list(self):
        """Test creation from list."""
        emb = Embedding.from_list([0.1, 0.2, 0.3])
        assert emb.dimension == 3

    def test_hyperbolic_validity(self):
        """Test Poincaré disk validity check."""
        valid = Embedding(vector=(0.1, 0.2, 0.3), space=EmbeddingSpace.HYPERBOLIC)
        invalid = Embedding(vector=(0.8, 0.8, 0.8), space=EmbeddingSpace.HYPERBOLIC)

        assert valid.is_valid_poincare
        assert not invalid.is_valid_poincare

    def test_embedding_immutable(self):
        """Test embedding immutability."""
        emb = Embedding(vector=(0.1, 0.2))
        with pytest.raises(Exception):  # frozen dataclass
            emb.vector = (0.3, 0.4)
