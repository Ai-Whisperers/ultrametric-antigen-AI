# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for AlphaFold Encoder.

Tests cover:
- AlphaFoldStructure dataclass
- AlphaFoldStructureLoader
- AlphaFoldEncoder
- AlphaFoldFeatureExtractor
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.encoders.alphafold_encoder import (
    AlphaFoldEncoder,
    AlphaFoldFeatureExtractor,
    AlphaFoldStructure,
    AlphaFoldStructureLoader,
)


class TestAlphaFoldStructure:
    """Test AlphaFoldStructure dataclass."""

    def test_structure_creation(self):
        """Test structure creation."""
        structure = AlphaFoldStructure(
            coords=np.random.randn(100, 3).astype(np.float32),
            plddt=np.random.rand(100).astype(np.float32) * 100,
            sequence="M" * 100,
            uniprot_id="P00000",
        )

        assert structure.coords.shape == (100, 3)
        assert structure.plddt.shape == (100,)
        assert len(structure.sequence) == 100
        assert structure.pae is None

    def test_structure_with_pae(self):
        """Test structure with PAE matrix."""
        structure = AlphaFoldStructure(
            coords=np.random.randn(50, 3).astype(np.float32),
            plddt=np.random.rand(50).astype(np.float32) * 100,
            sequence="A" * 50,
            uniprot_id="P00001",
            pae=np.random.rand(50, 50).astype(np.float32) * 30,
        )

        assert structure.pae.shape == (50, 50)

    def test_to_tensors(self):
        """Test conversion to PyTorch tensors."""
        structure = AlphaFoldStructure(
            coords=np.random.randn(30, 3).astype(np.float32),
            plddt=np.random.rand(30).astype(np.float32) * 100,
            sequence="G" * 30,
            uniprot_id="P00002",
        )

        tensors = structure.to_tensors()

        assert isinstance(tensors["coords"], torch.Tensor)
        assert isinstance(tensors["plddt"], torch.Tensor)
        assert tensors["coords"].dtype == torch.float32

    def test_to_tensors_with_device(self):
        """Test conversion to tensors on specific device."""
        structure = AlphaFoldStructure(
            coords=np.random.randn(20, 3).astype(np.float32),
            plddt=np.random.rand(20).astype(np.float32) * 100,
            sequence="V" * 20,
            uniprot_id="P00003",
        )

        tensors = structure.to_tensors(device="cpu")

        assert tensors["coords"].device.type == "cpu"


class TestAlphaFoldStructureLoader:
    """Test AlphaFoldStructureLoader."""

    @pytest.fixture
    def loader(self):
        """Create loader with temp cache."""
        cache_dir = Path(tempfile.mkdtemp())
        return AlphaFoldStructureLoader(cache_dir=cache_dir)

    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.cache_dir.exists()
        assert loader.model_version == 4

    def test_cache_dir_creation(self):
        """Test cache directory is created."""
        cache_path = Path(tempfile.mkdtemp()) / "new_cache"
        loader = AlphaFoldStructureLoader(cache_dir=cache_path)

        assert cache_path.exists()

    def test_mock_structure(self, loader):
        """Test mock structure generation."""
        structure = loader._mock_structure("P00000", length=100)

        assert isinstance(structure, AlphaFoldStructure)
        assert structure.coords.shape == (100, 3)
        assert structure.plddt.shape == (100,)
        assert len(structure.sequence) == 100

    def test_mock_structure_plddt_range(self, loader):
        """Test mock pLDDT values are in valid range."""
        structure = loader._mock_structure("P00000", length=200)

        assert structure.plddt.min() >= 0
        assert structure.plddt.max() <= 100

    def test_get_structure_uses_cache(self, loader):
        """Test that get_structure uses cache."""
        # First call creates mock and caches
        structure1 = loader.get_structure("P00000")

        # Second call should use cache
        structure2 = loader.get_structure("P00000")

        # Should get same data (within floating point tolerance)
        np.testing.assert_array_almost_equal(structure1.coords, structure2.coords)

    def test_force_download(self, loader):
        """Test force_download bypasses cache."""
        structure1 = loader.get_structure("P00000")
        structure2 = loader.get_structure("P00000", force_download=True)

        # May or may not be equal depending on implementation
        # Main test is that it doesn't crash

    def test_list_cached_structures(self, loader):
        """Test listing cached structures."""
        # Cache some structures
        loader.get_structure("P00001")
        loader.get_structure("P00002")

        cached = loader.list_cached_structures()

        assert "P00001" in cached
        assert "P00002" in cached

    def test_parse_pdb_mock(self, loader):
        """Test PDB parsing (with mock data)."""
        # Create minimal PDB content
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 90.00           C
"""
        coords, plddt, sequence = loader._parse_pdb(pdb_content)

        assert coords.shape == (2, 3)
        assert plddt.shape == (2,)
        assert sequence == "AG"

    def test_parse_pdb_plddt_from_bfactor(self, loader):
        """Test pLDDT extracted from B-factor."""
        pdb_content = """ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 75.50           C
"""
        coords, plddt, sequence = loader._parse_pdb(pdb_content)

        assert abs(plddt[0] - 75.50) < 0.01


class TestAlphaFoldEncoder:
    """Test AlphaFoldEncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return AlphaFoldEncoder(
            embed_dim=64,
            n_layers=2,
            cutoff=10.0,
            use_plddt=True,
            use_pae=False,
        )

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embed_dim == 64
        assert encoder.cutoff == 10.0
        assert encoder.use_plddt is True
        assert len(encoder.layers) == 2

    def test_forward_shape(self, encoder):
        """Test forward pass shape."""
        coords = torch.randn(4, 100, 3)
        plddt = torch.rand(4, 100) * 100

        output = encoder(coords, plddt)

        assert output.shape == (4, 100, 64)

    def test_forward_without_plddt(self, encoder):
        """Test forward without pLDDT."""
        coords = torch.randn(4, 50, 3)

        output = encoder(coords)

        assert output.shape == (4, 50, 64)

    def test_forward_with_pae(self):
        """Test forward with PAE matrix."""
        encoder = AlphaFoldEncoder(embed_dim=32, use_pae=True)

        coords = torch.randn(2, 30, 3)
        plddt = torch.rand(2, 30) * 100
        pae = torch.rand(2, 30, 30) * 30

        output = encoder(coords, plddt, pae)

        assert output.shape == (2, 30, 32)

    def test_pool(self, encoder):
        """Test pooling method."""
        coords = torch.randn(4, 50, 3)
        h = encoder(coords)

        pooled = encoder.pool(h)

        assert pooled.shape == (4, 64)

    def test_pool_with_mask(self, encoder):
        """Test pooling with mask."""
        coords = torch.randn(4, 50, 3)
        h = encoder(coords)
        mask = torch.ones(4, 50)
        mask[:, 30:] = 0  # Mask last 20 residues

        pooled = encoder.pool(h, mask)

        assert pooled.shape == (4, 64)

    def test_plddt_weighting(self, encoder):
        """Test pLDDT confidence weighting."""
        coords = torch.randn(4, 30, 3)

        plddt_high = torch.ones(4, 30) * 95
        plddt_low = torch.ones(4, 30) * 30

        out_high = encoder(coords, plddt_high)
        out_low = encoder(coords, plddt_low)

        # Different pLDDT should give different outputs
        assert not torch.allclose(out_high, out_low)

    def test_distance_cutoff(self):
        """Test distance cutoff effect."""
        encoder = AlphaFoldEncoder(cutoff=5.0)

        # Create coords with some far apart
        coords = torch.zeros(1, 20, 3)
        coords[0, 10:, 0] = 20.0  # Second half far away

        output = encoder(coords)

        assert not torch.isnan(output).any()

    def test_gradient_flow(self, encoder):
        """Test gradient flow."""
        coords = torch.randn(2, 30, 3, requires_grad=True)
        plddt = torch.rand(2, 30) * 100

        output = encoder(coords, plddt)
        loss = output.sum()
        loss.backward()

        # Main test is no error during backward


class TestAlphaFoldFeatureExtractor:
    """Test AlphaFoldFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor fixture."""
        encoder = AlphaFoldEncoder(embed_dim=32)
        cache_dir = Path(tempfile.mkdtemp())
        loader = AlphaFoldStructureLoader(cache_dir=cache_dir)
        return AlphaFoldFeatureExtractor(encoder=encoder, loader=loader, device="cpu")

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.encoder is not None
        assert extractor.loader is not None

    def test_extract_features(self, extractor):
        """Test feature extraction."""
        features = extractor.extract_features("P00000")

        assert "residue_embeddings" in features
        assert "sequence_embedding" in features
        assert "sequence" in features

    def test_extract_features_with_structure(self, extractor):
        """Test feature extraction returning structure."""
        features = extractor.extract_features("P00000", return_structure=True)

        assert "structure" in features
        assert isinstance(features["structure"], AlphaFoldStructure)

    def test_extract_mutation_region(self, extractor):
        """Test mutation region extraction."""
        positions = [50, 100, 150]

        embeddings = extractor.extract_mutation_region("P00000", positions, window_size=5)

        assert embeddings.shape == (3, 32)  # 3 positions, 32 dim

    def test_extract_mutation_region_boundary(self, extractor):
        """Test mutation region at boundaries."""
        # Near start
        positions = [2, 5]

        embeddings = extractor.extract_mutation_region("P00000", positions, window_size=10)

        # Should handle boundary gracefully
        assert embeddings.shape[0] == 2


class TestAlphaFoldEncoderEdgeCases:
    """Test edge cases for AlphaFold encoder."""

    def test_very_short_sequence(self):
        """Test with very short sequence."""
        encoder = AlphaFoldEncoder(embed_dim=32)

        coords = torch.randn(2, 5, 3)  # Only 5 residues
        output = encoder(coords)

        assert output.shape == (2, 5, 32)

    def test_batch_size_one(self):
        """Test with batch size 1."""
        encoder = AlphaFoldEncoder(embed_dim=32)

        coords = torch.randn(1, 50, 3)
        output = encoder(coords)

        assert output.shape == (1, 50, 32)

    def test_large_cutoff(self):
        """Test with large cutoff (all connected)."""
        encoder = AlphaFoldEncoder(cutoff=1000.0)

        coords = torch.randn(2, 30, 3)
        output = encoder(coords)

        assert not torch.isnan(output).any()

    def test_small_cutoff(self):
        """Test with small cutoff (few connections)."""
        encoder = AlphaFoldEncoder(cutoff=1.0)

        coords = torch.randn(2, 30, 3) * 10  # Spread out
        output = encoder(coords)

        assert not torch.isnan(output).any()


class TestAlphaFoldIntegration:
    """Integration tests for AlphaFold components."""

    def test_end_to_end_pipeline(self):
        """Test full pipeline from structure to embedding."""
        # Create structure
        structure = AlphaFoldStructure(
            coords=np.random.randn(100, 3).astype(np.float32),
            plddt=np.random.rand(100).astype(np.float32) * 100,
            sequence="M" * 100,
            uniprot_id="P00000",
        )

        # Convert to tensors
        tensors = structure.to_tensors()

        # Encode
        encoder = AlphaFoldEncoder(embed_dim=64)
        coords = tensors["coords"].unsqueeze(0)
        plddt = tensors["plddt"].unsqueeze(0)

        embeddings = encoder(coords, plddt)
        pooled = encoder.pool(embeddings)

        assert pooled.shape == (1, 64)

    def test_multiple_structures(self):
        """Test processing multiple structures."""
        encoder = AlphaFoldEncoder(embed_dim=32)

        structures = [
            torch.randn(1, 100, 3),
            torch.randn(1, 150, 3),
            torch.randn(1, 80, 3),
        ]

        embeddings = [encoder(s) for s in structures]
        pooled = [encoder.pool(e) for e in embeddings]

        for p in pooled:
            assert p.shape == (1, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
