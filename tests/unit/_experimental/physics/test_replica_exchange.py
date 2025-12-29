# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ReplicaExchange class."""

from __future__ import annotations

import pytest
import torch

from src.physics import ReplicaExchange, SpinGlassLandscape


class TestReplicaExchangeInit:
    """Tests for ReplicaExchange initialization."""

    def test_default_init(self):
        """Test default initialization."""
        rex = ReplicaExchange(n_replicas=4)
        assert rex.n_replicas == 4
        assert len(rex.temperatures) == 4

    def test_custom_temperatures(self):
        """Test custom temperature range."""
        rex = ReplicaExchange(n_replicas=3, temp_min=0.5, temp_max=2.0)
        assert rex.temperatures[0] == pytest.approx(0.5)
        assert rex.temperatures[-1] == pytest.approx(2.0)

    def test_temperature_ordering(self):
        """Test temperatures are ordered."""
        rex = ReplicaExchange(n_replicas=5)
        temps = rex.temperatures
        # Temperatures should be in increasing order
        for i in range(len(temps) - 1):
            assert temps[i] <= temps[i + 1]


class TestReplicaExchangeSampling:
    """Tests for replica exchange sampling."""

    def test_sample_returns_dict(self, small_spin_glass, device):
        """Test sample returns correct dict structure."""
        rex = ReplicaExchange(n_replicas=3, n_sweeps=5, exchange_frequency=2)
        result = rex.sample(
            small_spin_glass,
            n_samples=5,
        )
        assert isinstance(result, dict)
        assert "samples" in result
        assert "energies" in result

    def test_sample_correct_shape(self, small_spin_glass, device):
        """Test samples have correct shape."""
        rex = ReplicaExchange(n_replicas=3, n_sweeps=5, exchange_frequency=2)
        result = rex.sample(
            small_spin_glass,
            n_samples=5,
        )
        samples = result["samples"]
        assert samples.shape[0] == 5
        assert samples.shape[1] == small_spin_glass.n_sites

    def test_sample_valid_states(self, small_spin_glass, device):
        """Test samples are valid spin states."""
        rex = ReplicaExchange(n_replicas=3, n_sweeps=5, exchange_frequency=2)
        result = rex.sample(
            small_spin_glass,
            n_samples=5,
        )
        samples = result["samples"]
        # All values should be valid state indices
        assert (samples >= 0).all()
        assert (samples < small_spin_glass.n_states).all()


class TestReplicaExchangeAcceptance:
    """Tests for exchange acceptance."""

    def test_exchange_acceptance_tracked(self, small_spin_glass, device):
        """Test exchange acceptance is tracked."""
        rex = ReplicaExchange(n_replicas=4, n_sweeps=10, exchange_frequency=2)
        result = rex.sample(
            small_spin_glass,
            n_samples=5,
        )
        # Should have exchange rate in results
        assert "exchange_rate" in result
        assert 0 <= result["exchange_rate"] <= 1


class TestReplicaExchangeWithPotts:
    """Tests with Potts model (n_states > 2)."""

    def test_potts_sampling(self, device):
        """Test sampling with Potts model."""
        landscape = SpinGlassLandscape(n_sites=4, n_states=3)
        rex = ReplicaExchange(n_replicas=3, n_sweeps=5, exchange_frequency=2)
        result = rex.sample(
            landscape,
            n_samples=5,
        )
        samples = result["samples"]
        assert samples.shape == (5, 4)
        assert (samples >= 0).all()
        assert (samples < 3).all()
