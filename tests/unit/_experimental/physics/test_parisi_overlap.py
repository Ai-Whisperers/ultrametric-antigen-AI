# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ParisiOverlapAnalyzer class."""

from __future__ import annotations

import pytest
import torch

from src.physics import ParisiOverlapAnalyzer


class TestParisiOverlapInit:
    """Tests for ParisiOverlapAnalyzer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        analyzer = ParisiOverlapAnalyzer()
        assert analyzer.n_bins == 50

    def test_custom_bins(self):
        """Test custom number of bins."""
        analyzer = ParisiOverlapAnalyzer(n_bins=100)
        assert analyzer.n_bins == 100


class TestOverlapComputation:
    """Tests for overlap computation."""

    def test_compute_overlap_ising(self, device):
        """Test overlap for Ising-like (0/1) spins."""
        config1 = torch.tensor([0, 1, 0, 1], device=device)
        config2 = torch.tensor([0, 1, 1, 0], device=device)
        analyzer = ParisiOverlapAnalyzer()
        overlap = analyzer.compute_overlap(config1, config2)
        # Convert to +1/-1: [−1, +1, −1, +1] and [−1, +1, +1, −1]
        # Product: [+1, +1, −1, −1], mean = 0
        assert overlap == pytest.approx(0.0)

    def test_compute_overlap_identical(self, device):
        """Test overlap of identical configurations."""
        config = torch.tensor([0, 1, 0, 1, 1], device=device)
        analyzer = ParisiOverlapAnalyzer()
        overlap = analyzer.compute_overlap(config, config)
        # Identical configs have overlap 1
        assert overlap == pytest.approx(1.0)

    def test_compute_overlap_opposite(self, device):
        """Test overlap of opposite configurations."""
        config1 = torch.tensor([0, 0, 0, 0], device=device)
        config2 = torch.tensor([1, 1, 1, 1], device=device)
        analyzer = ParisiOverlapAnalyzer()
        overlap = analyzer.compute_overlap(config1, config2)
        # Opposite configs have overlap -1
        assert overlap == pytest.approx(-1.0)

    def test_overlap_range(self, spin_samples):
        """Test overlaps are in valid range."""
        analyzer = ParisiOverlapAnalyzer()
        # Check several pairs
        for i in range(min(5, len(spin_samples))):
            for j in range(i + 1, min(5, len(spin_samples))):
                overlap = analyzer.compute_overlap(spin_samples[i], spin_samples[j])
                assert -1 <= overlap <= 1


class TestOverlapDistribution:
    """Tests for overlap distribution analysis."""

    def test_distribution_returns_tuple(self, spin_samples):
        """Test overlap_distribution returns tuple."""
        analyzer = ParisiOverlapAnalyzer(n_bins=20)
        result = analyzer.overlap_distribution(spin_samples)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_distribution_shapes(self, spin_samples):
        """Test distribution histogram shape."""
        n_bins = 20
        analyzer = ParisiOverlapAnalyzer(n_bins=n_bins)
        bin_centers, hist = analyzer.overlap_distribution(spin_samples)
        assert len(bin_centers) == n_bins
        assert len(hist) == n_bins

    def test_distribution_normalized(self, spin_samples):
        """Test histogram is normalized."""
        analyzer = ParisiOverlapAnalyzer()
        _, hist = analyzer.overlap_distribution(spin_samples)
        # Sum should be approximately 1 (normalized)
        assert hist.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestPhaseAnalysis:
    """Tests for phase analysis."""

    def test_analyze_phase_returns_dict(self, spin_samples):
        """Test analyze_phase returns dict."""
        analyzer = ParisiOverlapAnalyzer()
        result = analyzer.analyze_phase(spin_samples)
        assert isinstance(result, dict)

    def test_analyze_phase_keys(self, spin_samples):
        """Test analyze_phase has expected keys."""
        analyzer = ParisiOverlapAnalyzer()
        result = analyzer.analyze_phase(spin_samples)
        # Should have mean_overlap and phase classification
        assert "mean_overlap" in result
        assert "phase" in result

    def test_analyze_phase_classification(self, spin_samples):
        """Test phase classification is valid."""
        analyzer = ParisiOverlapAnalyzer()
        result = analyzer.analyze_phase(spin_samples)
        valid_phases = ["paramagnetic", "ferromagnetic", "ferromagnetic (symmetric)", "spin_glass"]
        assert result["phase"] in valid_phases


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_samples(self, device):
        """Test with minimum number of samples."""
        samples = torch.randint(0, 2, (2, 10), device=device)
        analyzer = ParisiOverlapAnalyzer()
        bin_centers, hist = analyzer.overlap_distribution(samples)
        # Should still work with just 2 samples (1 pair)
        assert len(hist) == analyzer.n_bins

    def test_single_site(self, device):
        """Test with single site."""
        samples = torch.randint(0, 2, (10, 1), device=device)
        analyzer = ParisiOverlapAnalyzer()
        # Overlap of single site should be ±1
        overlap = analyzer.compute_overlap(samples[0], samples[1])
        assert overlap in [-1.0, 1.0]
