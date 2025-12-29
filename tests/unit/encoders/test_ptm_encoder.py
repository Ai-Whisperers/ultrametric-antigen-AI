# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for PTM-Goldilocks Encoder."""

import pytest
import torch

from src.encoders.ptm_encoder import (GoldilocksZone, PTMDataset,
                                      PTMGoldilocksEncoder, PTMType)


# Use CPU for all tests to avoid geoopt device mismatch
@pytest.fixture
def cpu_device():
    """Force CPU for geoopt-related tests."""
    return "cpu"


class TestPTMType:
    """Tests for PTMType enumeration."""

    def test_ptm_types_are_integers(self):
        """PTM types should be integer values."""
        assert isinstance(PTMType.NONE, int)
        assert isinstance(PTMType.CITRULLINATION, int)
        assert PTMType.NONE == 0
        assert PTMType.CITRULLINATION == 1

    def test_all_ptm_types_unique(self):
        """All PTM type values should be unique."""
        values = [ptm.value for ptm in PTMType]
        assert len(values) == len(set(values))


class TestGoldilocksZone:
    """Tests for GoldilocksZone configuration."""

    def test_default_zone_boundaries(self):
        """Default zone should have validated boundaries."""
        zone = GoldilocksZone()
        assert zone.alpha == -0.1205
        assert zone.beta == 0.0495
        assert zone.alpha < zone.beta

    def test_custom_zone_boundaries(self):
        """Custom zone boundaries should be accepted."""
        zone = GoldilocksZone(alpha=-0.2, beta=0.1)
        assert zone.alpha == -0.2
        assert zone.beta == 0.1


class TestPTMGoldilocksEncoder:
    """Tests for PTMGoldilocksEncoder module."""

    @pytest.fixture
    def cpu_device(self):
        """Force CPU for geoopt-related tests."""
        return "cpu"

    @pytest.fixture
    def encoder(self, cpu_device):
        """Create encoder instance on CPU."""
        enc = PTMGoldilocksEncoder(
            embedding_dim=16,
            num_amino_acids=22,
            num_ptm_types=10,
            curvature=1.0,
        )
        return enc.to(cpu_device)

    @pytest.fixture
    def sample_input(self, cpu_device):
        """Create sample input tensors on CPU."""
        batch_size = 4
        seq_len = 10
        aa_indices = torch.randint(1, 21, (batch_size, seq_len), device=cpu_device)
        ptm_states = torch.zeros(batch_size, seq_len, dtype=torch.long, device=cpu_device)
        # Add some citrullination at arginine positions
        ptm_states[:, 3] = PTMType.CITRULLINATION
        return aa_indices, ptm_states

    def test_forward_shape(self, encoder, sample_input):
        """Forward pass should produce correct output shape."""
        aa_indices, ptm_states = sample_input
        output = encoder(aa_indices, ptm_states)

        assert output.shape == (4, 10, 16)

    def test_forward_without_ptm(self, encoder, cpu_device):
        """Forward should work without PTM states."""
        aa_indices = torch.randint(1, 21, (2, 5), device=cpu_device)
        output = encoder(aa_indices)

        assert output.shape == (2, 5, 16)

    def test_output_on_poincare_ball(self, encoder, sample_input):
        """Output should be on Poincare ball (norm < 1)."""
        aa_indices, ptm_states = sample_input
        output = encoder(aa_indices, ptm_states)

        norms = torch.norm(output, dim=-1)
        assert torch.all(norms < 1.0)

    def test_entropy_change_computation(self, encoder, cpu_device):
        """Entropy change should be computable between embeddings."""
        batch_size = 2
        seq_len = 5

        aa = torch.randint(1, 21, (batch_size, seq_len), device=cpu_device)
        no_ptm = torch.zeros(batch_size, seq_len, dtype=torch.long, device=cpu_device)
        with_ptm = torch.full((batch_size, seq_len), PTMType.CITRULLINATION, device=cpu_device)

        z_native = encoder(aa, no_ptm)
        z_modified = encoder(aa, with_ptm)

        delta_h = encoder.compute_entropy_change(z_native, z_modified)

        assert delta_h.shape == (batch_size, seq_len)
        assert not torch.isnan(delta_h).any()

    def test_goldilocks_membership(self, encoder):
        """Goldilocks membership should be computable."""
        # Test values in and out of zone
        entropy_changes = torch.tensor(
            [
                -0.05,  # In zone
                0.0,  # In zone (center)
                -0.2,  # Below zone
                0.1,  # Above zone
            ]
        )

        in_zone, zone_dist = encoder.compute_goldilocks_membership(entropy_changes, return_distance=True)

        assert in_zone[0].item() is True
        assert in_zone[1].item() is True
        assert in_zone[2].item() is False
        assert in_zone[3].item() is False
        assert zone_dist is not None

    def test_immunogenicity_score(self, encoder, cpu_device):
        """Comprehensive immunogenicity score should be computable."""
        batch_size = 2
        seq_len = 5

        aa = torch.randint(1, 21, (batch_size, seq_len), device=cpu_device)
        no_ptm = torch.zeros(batch_size, seq_len, dtype=torch.long, device=cpu_device)
        with_ptm = torch.full((batch_size, seq_len), PTMType.CITRULLINATION, device=cpu_device)

        z_native = encoder(aa, no_ptm)
        z_modified = encoder(aa, with_ptm)

        result = encoder.compute_immunogenicity_score(z_native, z_modified)

        assert "entropy_change" in result
        assert "in_goldilocks" in result
        assert "zone_distance" in result
        assert "centroid_shift" in result
        assert "immunogenicity_score" in result

        assert result["immunogenicity_score"].shape == (batch_size,)

    def test_ptm_shift_vector(self, encoder, cpu_device):
        """PTM shift vector should be computable."""
        # Test citrullination of arginine (index 15)
        shift = encoder.get_ptm_shift_vector(15, PTMType.CITRULLINATION)

        assert shift.shape == (16,)
        assert not torch.isnan(shift).any()

    def test_hierarchical_initialization(self, encoder):
        """Encoder should have hierarchical weight initialization."""
        weights = encoder.aa_embedding.weight.data

        # Arginine (15) should have positive charge encoding
        assert weights[15, 0] > 0  # Positive charge

        # Aspartic acid (3) should have negative charge
        assert weights[3, 0] < 0  # Negative charge

    def test_gradient_flow(self, encoder, sample_input):
        """Gradients should flow through the encoder."""
        aa_indices, ptm_states = sample_input
        output = encoder(aa_indices, ptm_states)

        loss = output.sum()
        loss.backward()

        assert encoder.aa_embedding.weight.grad is not None
        assert encoder.ptm_embedding.weight.grad is not None


class TestPTMDataset:
    """Tests for PTMDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        sequences = ["MARTKQTARK", "PEPTIDESEQ"]
        ptm_annotations = [
            {"modifications": [(1, PTMType.CITRULLINATION)], "immunogenic": 1},
            {"modifications": [], "immunogenic": 0},
        ]
        return PTMDataset(sequences, ptm_annotations, max_length=20)

    def test_dataset_length(self, sample_dataset):
        """Dataset should have correct length."""
        assert len(sample_dataset) == 2

    def test_dataset_item_structure(self, sample_dataset):
        """Dataset items should have correct structure."""
        item = sample_dataset[0]

        assert "aa_indices" in item
        assert "ptm_states" in item
        assert "labels" in item

        assert item["aa_indices"].shape == (20,)
        assert item["ptm_states"].shape == (20,)

    def test_ptm_state_encoding(self, sample_dataset):
        """PTM states should be correctly encoded."""
        item = sample_dataset[0]

        # Position 1 should have citrullination
        assert item["ptm_states"][1] == PTMType.CITRULLINATION

        # Other positions should be NONE
        assert item["ptm_states"][0] == PTMType.NONE
