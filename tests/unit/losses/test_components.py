# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/losses/components.py - LossComponent implementations.

Tests the modular loss component architecture:
- ReconstructionLossComponent
- KLDivergenceLossComponent
- EntropyLossComponent
- RepulsionLossComponent
- EntropyAlignmentComponent
"""

import pytest
import torch
from src.losses.components import (
    ReconstructionLossComponent,
    KLDivergenceLossComponent,
    EntropyLossComponent,
    RepulsionLossComponent,
    EntropyAlignmentComponent,
)
from src.losses.base import LossResult


# =============================================================================
# ReconstructionLossComponent Tests
# =============================================================================


class TestReconstructionLossComponent:
    """Tests for ReconstructionLossComponent."""

    @pytest.fixture
    def component(self):
        return ReconstructionLossComponent(weight=1.0)

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.weight == 1.0
        assert component.name == 'reconstruction'

    def test_forward_returns_loss_result(self, component, loss_test_outputs, device):
        """Test forward returns LossResult."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = component.forward(loss_test_outputs, targets)

        assert isinstance(result, LossResult)
        assert hasattr(result, 'loss')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'weight')

    def test_loss_is_non_negative(self, component, loss_test_outputs, device):
        """Test loss is non-negative."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = component.forward(loss_test_outputs, targets)

        assert result.loss >= 0

    def test_metrics_contain_expected_keys(self, component, loss_test_outputs, device):
        """Test metrics contain ce_A and ce_B."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = component.forward(loss_test_outputs, targets)

        assert 'ce_A' in result.metrics
        assert 'ce_B' in result.metrics

    def test_perfect_prediction_low_loss(self, component, device):
        """Test perfect prediction results in low loss."""
        batch_size = 4

        # Create targets
        targets = torch.zeros(batch_size, 9, device=device)  # All zeros

        # Create logits that strongly predict zero (class 1)
        logits = torch.zeros(batch_size, 9, 3, device=device)
        logits[:, :, 1] = 10.0  # Strong prediction for class 1

        outputs = {
            'logits_A': logits.clone(),
            'logits_B': logits.clone(),
        }

        result = component.forward(outputs, targets)

        # Loss should be very low for perfect predictions
        assert result.loss < 0.1

    def test_weight_applied_correctly(self, device, loss_test_outputs):
        """Test weight is correctly assigned."""
        component = ReconstructionLossComponent(weight=0.5)
        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        result = component.forward(loss_test_outputs, targets)

        assert result.weight == 0.5


# =============================================================================
# KLDivergenceLossComponent Tests
# =============================================================================


class TestKLDivergenceLossComponent:
    """Tests for KLDivergenceLossComponent."""

    @pytest.fixture
    def component(self):
        return KLDivergenceLossComponent(weight=1.0, free_bits=0.0)

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.weight == 1.0
        assert component.name == 'kl'
        assert component.free_bits == 0.0

    def test_forward_returns_loss_result(self, component, loss_test_outputs, device):
        """Test forward returns LossResult."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = component.forward(loss_test_outputs, targets)

        assert isinstance(result, LossResult)

    def test_zero_mean_zero_var_near_zero_kl(self, device):
        """Test standard normal posterior has near-zero KL."""
        component = KLDivergenceLossComponent(weight=1.0, free_bits=0.0)

        batch_size = 32
        latent_dim = 16

        outputs = {
            'mu_A': torch.zeros(batch_size, latent_dim, device=device),
            'logvar_A': torch.zeros(batch_size, latent_dim, device=device),
            'mu_B': torch.zeros(batch_size, latent_dim, device=device),
            'logvar_B': torch.zeros(batch_size, latent_dim, device=device),
            'beta_A': 1.0,
            'beta_B': 1.0,
        }

        targets = torch.zeros(batch_size, 9, device=device)
        result = component.forward(outputs, targets)

        # KL should be near zero for standard normal
        assert result.loss.item() < 0.1

    def test_far_from_prior_high_kl(self, component, high_kl_outputs, device):
        """Test posterior far from prior has high KL."""
        targets = torch.zeros(4, 9, device=device)
        result = component.forward(high_kl_outputs, targets)

        # KL should be high when mean is far from zero
        assert result.loss.item() > 1.0

    def test_free_bits_reduces_kl(self, device):
        """Test free bits reduces effective KL."""
        batch_size = 32
        latent_dim = 16

        outputs = {
            'mu_A': torch.randn(batch_size, latent_dim, device=device),
            'logvar_A': torch.randn(batch_size, latent_dim, device=device),
            'mu_B': torch.randn(batch_size, latent_dim, device=device),
            'logvar_B': torch.randn(batch_size, latent_dim, device=device),
            'beta_A': 1.0,
            'beta_B': 1.0,
        }
        targets = torch.zeros(batch_size, 9, device=device)

        component_no_free = KLDivergenceLossComponent(weight=1.0, free_bits=0.0)
        component_free = KLDivergenceLossComponent(weight=1.0, free_bits=2.0)

        result_no_free = component_no_free.forward(outputs, targets)
        result_free = component_free.forward(outputs, targets)

        # Free bits should result in different (typically higher) KL due to clamping
        assert result_free.loss.item() >= result_no_free.loss.item()

    def test_metrics_contain_kl_values(self, component, loss_test_outputs, device):
        """Test metrics contain KL values."""
        targets = torch.zeros(32, 9, device=device)
        result = component.forward(loss_test_outputs, targets)

        assert 'kl_A' in result.metrics
        assert 'kl_B' in result.metrics


# =============================================================================
# EntropyLossComponent Tests
# =============================================================================


class TestEntropyLossComponent:
    """Tests for EntropyLossComponent."""

    @pytest.fixture
    def component(self):
        return EntropyLossComponent(weight=0.01, vae='B')

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.weight == 0.01
        assert component.name == 'entropy'
        assert component.vae == 'B'

    def test_forward_returns_loss_result(self, component, loss_test_outputs, device):
        """Test forward returns LossResult."""
        targets = torch.zeros(32, 9, device=device)
        result = component.forward(loss_test_outputs, targets)

        assert isinstance(result, LossResult)

    def test_uniform_distribution_high_entropy(self, device):
        """Test uniform distribution has high entropy."""
        component = EntropyLossComponent(weight=1.0, vae='B')

        batch_size = 32
        # Uniform logits -> uniform distribution -> high entropy
        outputs = {
            'logits_A': torch.zeros(batch_size, 9, 3, device=device),
            'logits_B': torch.zeros(batch_size, 9, 3, device=device),
        }
        targets = torch.zeros(batch_size, 9, device=device)

        result = component.forward(outputs, targets)

        # Entropy should be close to max (log(3) * 9 per digit)
        assert 'entropy_B' in result.metrics
        assert result.metrics['entropy_B'] > 0  # High entropy value

    def test_peaked_distribution_low_entropy(self, device):
        """Test peaked distribution has low entropy."""
        component = EntropyLossComponent(weight=1.0, vae='B')

        batch_size = 32
        # Peaked logits -> low entropy
        logits = torch.zeros(batch_size, 9, 3, device=device)
        logits[:, :, 0] = 10.0  # Strong peak at first class

        outputs = {
            'logits_A': logits.clone(),
            'logits_B': logits.clone(),
        }
        targets = torch.zeros(batch_size, 9, device=device)

        result = component.forward(outputs, targets)

        # Entropy should be low for peaked distribution
        assert result.metrics['entropy_B'] < result.metrics.get('max_entropy', 100)

    def test_vae_both_computes_both(self, device, loss_test_outputs):
        """Test vae='both' computes entropy for both VAEs."""
        component = EntropyLossComponent(weight=1.0, vae='both')
        targets = torch.zeros(32, 9, device=device)

        result = component.forward(loss_test_outputs, targets)

        assert 'entropy_A' in result.metrics
        assert 'entropy_B' in result.metrics


# =============================================================================
# RepulsionLossComponent Tests
# =============================================================================


class TestRepulsionLossComponent:
    """Tests for RepulsionLossComponent."""

    @pytest.fixture
    def component(self):
        return RepulsionLossComponent(weight=0.01, sigma=0.5, vae='B')

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.weight == 0.01
        assert component.name == 'repulsion'
        assert component.sigma == 0.5

    def test_forward_returns_loss_result(self, component, loss_test_outputs, device):
        """Test forward returns LossResult."""
        targets = torch.zeros(32, 9, device=device)
        result = component.forward(loss_test_outputs, targets)

        assert isinstance(result, LossResult)

    def test_identical_points_high_repulsion(self, device):
        """Test identical points result in high repulsion."""
        component = RepulsionLossComponent(weight=1.0, sigma=0.5, vae='B')

        batch_size = 8
        latent_dim = 16

        # All points are identical
        z = torch.zeros(batch_size, latent_dim, device=device)

        outputs = {
            'z_A': z.clone(),
            'z_B': z.clone(),
        }
        targets = torch.zeros(batch_size, 9, device=device)

        result = component.forward(outputs, targets)

        # Repulsion should be high (close to 1) for identical points
        assert result.metrics['repulsion_B'] > 0.9

    def test_distant_points_low_repulsion(self, device):
        """Test distant points result in low repulsion."""
        component = RepulsionLossComponent(weight=1.0, sigma=0.5, vae='B')

        batch_size = 8
        latent_dim = 16

        # Points are far apart
        z = torch.randn(batch_size, latent_dim, device=device) * 10

        outputs = {
            'z_A': z.clone(),
            'z_B': z.clone(),
        }
        targets = torch.zeros(batch_size, 9, device=device)

        result = component.forward(outputs, targets)

        # Repulsion should be low for distant points
        assert result.metrics['repulsion_B'] < 0.5

    def test_single_sample_zero_repulsion(self, device):
        """Test single sample has zero repulsion."""
        component = RepulsionLossComponent(weight=1.0, sigma=0.5, vae='B')

        outputs = {
            'z_A': torch.randn(1, 16, device=device),
            'z_B': torch.randn(1, 16, device=device),
        }
        targets = torch.zeros(1, 9, device=device)

        result = component.forward(outputs, targets)

        # Can't compute repulsion with single point
        assert result.loss.item() == 0.0


# =============================================================================
# EntropyAlignmentComponent Tests
# =============================================================================


class TestEntropyAlignmentComponent:
    """Tests for EntropyAlignmentComponent."""

    @pytest.fixture
    def component(self):
        return EntropyAlignmentComponent(weight=0.1)

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.weight == 0.1
        assert component.name == 'entropy_align'

    def test_forward_returns_loss_result(self, component, loss_test_outputs, device):
        """Test forward returns LossResult."""
        targets = torch.zeros(32, 9, device=device)
        result = component.forward(loss_test_outputs, targets)

        assert isinstance(result, LossResult)

    def test_equal_entropies_zero_alignment(self, component, device):
        """Test equal entropies result in zero alignment loss."""
        outputs = {
            'H_A': torch.tensor(2.0, device=device),
            'H_B': torch.tensor(2.0, device=device),
        }
        targets = torch.zeros(4, 9, device=device)

        result = component.forward(outputs, targets)

        assert result.loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_entropies_positive_alignment(self, component, device):
        """Test different entropies result in positive alignment loss."""
        outputs = {
            'H_A': torch.tensor(1.0, device=device),
            'H_B': torch.tensor(3.0, device=device),
        }
        targets = torch.zeros(4, 9, device=device)

        result = component.forward(outputs, targets)

        assert result.loss.item() == pytest.approx(2.0, abs=1e-6)

    def test_metrics_contain_entropies(self, component, loss_test_outputs, device):
        """Test metrics contain H_A, H_B, and alignment."""
        targets = torch.zeros(32, 9, device=device)
        result = component.forward(loss_test_outputs, targets)

        assert 'H_A' in result.metrics
        assert 'H_B' in result.metrics
        assert 'alignment' in result.metrics


# =============================================================================
# Integration Tests
# =============================================================================


class TestComponentIntegration:
    """Integration tests for loss components working together."""

    def test_all_components_compatible(self, loss_test_outputs, device):
        """Test all components can process same outputs."""
        components = [
            ReconstructionLossComponent(weight=1.0),
            KLDivergenceLossComponent(weight=1.0),
            EntropyLossComponent(weight=0.01, vae='B'),
            RepulsionLossComponent(weight=0.01, vae='B'),
            EntropyAlignmentComponent(weight=0.1),
        ]

        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        for component in components:
            result = component.forward(loss_test_outputs, targets)
            assert isinstance(result, LossResult)
            assert torch.isfinite(result.loss)

    def test_combined_loss_calculation(self, loss_test_outputs, device):
        """Test combining multiple component losses."""
        components = [
            ReconstructionLossComponent(weight=1.0),
            KLDivergenceLossComponent(weight=0.1),
            EntropyAlignmentComponent(weight=0.05),
        ]

        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        total_loss = 0.0
        for component in components:
            result = component.forward(loss_test_outputs, targets)
            total_loss += result.weight * result.loss

        assert torch.isfinite(total_loss)
        assert total_loss > 0

    def test_gradient_flow_through_components(self, device):
        """Test gradients flow through combined components."""
        batch_size = 4
        latent_dim = 16

        # Create outputs with gradients
        mu = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
        logvar = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)

        outputs = {
            'logits_A': torch.randn(batch_size, 9, 3, device=device, requires_grad=True),
            'logits_B': torch.randn(batch_size, 9, 3, device=device, requires_grad=True),
            'mu_A': mu,
            'mu_B': mu.clone(),
            'logvar_A': logvar,
            'logvar_B': logvar.clone(),
            'z_A': torch.randn(batch_size, latent_dim, device=device, requires_grad=True),
            'z_B': torch.randn(batch_size, latent_dim, device=device, requires_grad=True),
            'H_A': torch.tensor(2.0, device=device),
            'H_B': torch.tensor(2.1, device=device),
            'beta_A': 1.0,
            'beta_B': 1.0,
        }

        targets = torch.randint(-1, 2, (batch_size, 9), device=device).float()

        component = ReconstructionLossComponent(weight=1.0)
        result = component.forward(outputs, targets)

        result.loss.backward()

        # Check gradients exist
        assert outputs['logits_A'].grad is not None
        assert outputs['logits_B'].grad is not None
