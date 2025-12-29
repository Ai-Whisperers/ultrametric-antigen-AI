# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TropicalNNAnalyzer class."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.tropical import TropicalNNAnalyzer


class TestTropicalNNAnalyzerInit:
    """Tests for analyzer initialization."""

    def test_init_with_simple_network(self, simple_relu_network):
        """Test initialization with simple network."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        assert len(analyzer.layers) == 2

    def test_init_with_deep_network(self, deep_relu_network):
        """Test initialization with deeper network."""
        analyzer = TropicalNNAnalyzer(deep_relu_network)
        assert len(analyzer.layers) == 3

    def test_extract_layers(self, simple_relu_network):
        """Test layer extraction."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        W1, b1 = analyzer.layers[0]
        W2, b2 = analyzer.layers[1]

        assert W1.shape == (4, 2)
        assert b1.shape == (4,)
        assert W2.shape == (1, 4)
        assert b2.shape == (1,)


class TestLinearRegionCounting:
    """Tests for linear region counting."""

    def test_compute_linear_regions(self, simple_relu_network):
        """Test computing linear regions."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        n_regions = analyzer.compute_linear_regions(n_samples=1000)
        # Should find some regions (depends on random weights)
        assert n_regions >= 1
        # Upper bound: 2^(hidden_units) = 2^4 = 16
        assert n_regions <= 16

    def test_linear_regions_different_bounds(self, simple_relu_network):
        """Test with different input bounds."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        n_regions = analyzer.compute_linear_regions(input_bounds=bounds, n_samples=1000)
        assert n_regions >= 1

    def test_linear_regions_grid_sampling(self, simple_relu_network):
        """Test grid sampling method."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        n_regions = analyzer.compute_linear_regions(sampling_method="grid", n_samples=100)
        assert n_regions >= 1


class TestActivationPatterns:
    """Tests for activation pattern extraction."""

    def test_get_activation_pattern(self, simple_relu_network):
        """Test getting activation pattern for a point."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        x = np.array([0.5, 0.5])
        pattern = analyzer._get_activation_pattern(x)

        # Should have one tuple for hidden layer
        assert len(pattern) == 1
        # Pattern should have 4 boolean values (4 hidden units)
        assert len(pattern[0]) == 4
        assert all(isinstance(v, (bool, np.bool_)) for v in pattern[0])


class TestExpressivityAnalysis:
    """Tests for expressivity analysis."""

    def test_analyze_expressivity(self, simple_relu_network):
        """Test expressivity analysis."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        result = analyzer.analyze_expressivity()

        assert "depth" in result
        assert "widths" in result
        assert "input_dim" in result
        assert "max_linear_regions_upper_bound" in result
        assert "total_parameters" in result

    def test_expressivity_values(self, simple_relu_network):
        """Test expressivity values are correct."""
        analyzer = TropicalNNAnalyzer(simple_relu_network)
        result = analyzer.analyze_expressivity()

        assert result["depth"] == 2
        assert result["widths"] == [4, 1]
        assert result["input_dim"] == 2
        assert result["total_parameters"] > 0


class TestTropicalPolynomialExtraction:
    """Tests for tropical polynomial extraction."""

    def test_extract_single_layer(self):
        """Test extraction for single layer network."""
        model = nn.Sequential(nn.Linear(2, 3))
        analyzer = TropicalNNAnalyzer(model)

        poly = analyzer.extract_tropical_polynomial(output_idx=0)
        assert poly.n_monomials == 1

    def test_extract_deep_network_raises(self, deep_relu_network):
        """Test extraction for very deep network raises if too large."""
        # deep_relu_network has 3+3 = 6 hidden units, should work
        analyzer = TropicalNNAnalyzer(deep_relu_network)
        # This should work for small networks
        poly = analyzer.extract_tropical_polynomial(output_idx=0)
        assert poly.n_monomials >= 1
