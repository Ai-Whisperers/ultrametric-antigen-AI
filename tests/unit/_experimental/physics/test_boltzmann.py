# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for BoltzmannMachine class."""

from __future__ import annotations

import pytest
import torch

from src.physics import BoltzmannMachine


class TestBoltzmannInit:
    """Tests for BoltzmannMachine initialization."""

    def test_default_init(self):
        """Test default initialization."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5)
        assert rbm.n_visible == 10
        assert rbm.n_hidden == 5

    def test_weight_shape(self):
        """Test weight matrix shape."""
        rbm = BoltzmannMachine(n_visible=8, n_hidden=4)
        assert rbm.W.shape == (8, 4)

    def test_bias_shapes(self):
        """Test bias vector shapes."""
        rbm = BoltzmannMachine(n_visible=6, n_hidden=3)
        assert rbm.a.shape == (6,)  # Visible bias
        assert rbm.b.shape == (3,)  # Hidden bias


class TestBoltzmannEnergy:
    """Tests for energy computation."""

    def test_energy_shape(self, device):
        """Test energy has correct shape."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        h = torch.rand(4, 5, device=device).round()
        energy = rbm.energy(v, h)
        assert energy.shape == (4,)

    def test_energy_finite(self, device):
        """Test energy is finite."""
        rbm = BoltzmannMachine(n_visible=8, n_hidden=4).to(device)
        v = torch.rand(2, 8, device=device).round()
        h = torch.rand(2, 4, device=device).round()
        energy = rbm.energy(v, h)
        assert torch.isfinite(energy).all()


class TestBoltzmannSampling:
    """Tests for Gibbs sampling."""

    def test_sample_hidden_shapes(self, device):
        """Test sample_hidden output shapes."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        p_h, h = rbm.sample_hidden(v)
        assert p_h.shape == (4, 5)
        assert h.shape == (4, 5)

    def test_sample_hidden_probabilities(self, device):
        """Test hidden probabilities in [0, 1]."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        p_h, _ = rbm.sample_hidden(v)
        assert (p_h >= 0).all() and (p_h <= 1).all()

    def test_sample_hidden_binary(self, device):
        """Test hidden samples are binary."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        _, h = rbm.sample_hidden(v)
        assert ((h == 0) | (h == 1)).all()

    def test_sample_visible_shapes(self, device):
        """Test sample_visible output shapes."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        h = torch.rand(4, 5, device=device).round()
        p_v, v = rbm.sample_visible(h)
        assert p_v.shape == (4, 10)
        assert v.shape == (4, 10)


class TestBoltzmannFreeEnergy:
    """Tests for free energy computation."""

    def test_free_energy_shape(self, device):
        """Test free energy has correct shape."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        F = rbm.free_energy(v)
        assert F.shape == (4,)

    def test_free_energy_finite(self, device):
        """Test free energy is finite."""
        rbm = BoltzmannMachine(n_visible=8, n_hidden=4).to(device)
        v = torch.rand(2, 8, device=device).round()
        F = rbm.free_energy(v)
        assert torch.isfinite(F).all()


class TestBoltzmannCD:
    """Tests for contrastive divergence."""

    def test_cd_returns_gradients(self, device):
        """Test CD returns gradient dict."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        grads = rbm.contrastive_divergence(v, k=1)
        assert "W" in grads
        assert "a" in grads
        assert "b" in grads

    def test_cd_gradient_shapes(self, device):
        """Test CD gradients have correct shapes."""
        rbm = BoltzmannMachine(n_visible=8, n_hidden=4).to(device)
        v = torch.rand(4, 8, device=device).round()
        grads = rbm.contrastive_divergence(v, k=1)
        assert grads["W"].shape == (8, 4)
        assert grads["a"].shape == (8,)
        assert grads["b"].shape == (4,)


class TestBoltzmannForward:
    """Tests for forward pass (reconstruction)."""

    def test_forward_shape(self, device):
        """Test forward output shape."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        recon = rbm(v, n_samples=1)
        assert recon.shape == (4, 10)

    def test_forward_probabilities(self, device):
        """Test forward outputs probabilities."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5).to(device)
        v = torch.rand(4, 10, device=device).round()
        recon = rbm(v)
        assert (recon >= 0).all() and (recon <= 1).all()


class TestBoltzmannPAdicStructure:
    """Tests for p-adic structure."""

    def test_padic_structure_applied(self, device):
        """Test p-adic structure modifies weights."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5, use_padic_structure=True)
        # Weights should have some structure (not all same magnitude)
        W = rbm.W.abs()
        assert W.std() > 0  # Some variation

    def test_without_padic_structure(self, device):
        """Test without p-adic structure."""
        rbm = BoltzmannMachine(n_visible=10, n_hidden=5, use_padic_structure=False)
        assert rbm.W.shape == (10, 5)
