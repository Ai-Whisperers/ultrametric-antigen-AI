# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for DualVAELoss from src/losses/dual_vae_loss.py.

Tests the combined loss function that integrates all loss components:
- Reconstruction loss (cross-entropy)
- KL divergence loss
- Entropy regularization
- Repulsion loss
- Entropy alignment
- p-Adic losses (optional)
"""

import pytest
import torch
from src.losses.dual_vae_loss import DualVAELoss


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_outputs(device):
    """Sample model outputs dictionary."""
    batch_size = 32
    latent_dim = 16
    return {
        "logits_A": torch.randn(batch_size, 9, 3, device=device),
        "logits_B": torch.randn(batch_size, 9, 3, device=device),
        "mu_A": torch.randn(batch_size, latent_dim, device=device),
        "mu_B": torch.randn(batch_size, latent_dim, device=device),
        "logvar_A": torch.randn(batch_size, latent_dim, device=device),
        "logvar_B": torch.randn(batch_size, latent_dim, device=device),
        "z_A": torch.randn(batch_size, latent_dim, device=device),
        "z_B": torch.randn(batch_size, latent_dim, device=device),
        "beta_A": 1.0,
        "beta_B": 1.0,
        "H_A": torch.tensor(1.5, device=device),
        "H_B": torch.tensor(1.5, device=device),
    }


@pytest.fixture
def sample_input(device):
    """Sample input tensor with ternary values."""
    return torch.randint(-1, 2, (32, 9), device=device).float()


@pytest.fixture
def sample_batch_indices(device):
    """Sample operation indices for p-adic losses."""
    return torch.randint(0, 19683, (32,), device=device)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDualVAELossInit:
    """Test DualVAELoss initialization."""

    def test_init_default(self):
        """Default initialization should work."""
        loss_fn = DualVAELoss()

        assert loss_fn.reconstruction_loss is not None
        assert loss_fn.kl_loss is not None
        assert loss_fn.entropy_loss is not None
        assert loss_fn.repulsion_loss is not None

    def test_init_with_free_bits(self):
        """Should accept free_bits parameter."""
        loss_fn = DualVAELoss(free_bits=2.0)
        assert loss_fn.kl_loss.free_bits == 2.0

    def test_init_with_repulsion_sigma(self):
        """Should accept repulsion_sigma parameter."""
        loss_fn = DualVAELoss(repulsion_sigma=1.0)
        assert loss_fn.repulsion_loss.sigma == 1.0

    def test_init_padic_metric_loss_disabled(self):
        """p-Adic metric loss should be disabled by default."""
        loss_fn = DualVAELoss()
        assert loss_fn.enable_metric_loss is False

    def test_init_padic_metric_loss_enabled(self):
        """p-Adic metric loss should be enabled when configured."""
        padic_config = {"enable_metric_loss": True, "metric_loss_weight": 0.1}
        loss_fn = DualVAELoss(padic_config=padic_config)

        assert loss_fn.enable_metric_loss is True
        assert loss_fn.metric_loss_weight == 0.1


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestDualVAELossForward:
    """Test DualVAELoss forward pass."""

    @pytest.fixture
    def loss_fn(self):
        return DualVAELoss()

    def test_forward_returns_dict(self, loss_fn, sample_input, sample_outputs):
        """Forward should return a dictionary."""
        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert isinstance(result, dict)

    def test_forward_contains_required_keys(self, loss_fn, sample_input, sample_outputs):
        """Result dict should contain all required keys."""
        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        required_keys = [
            "loss", "ce_A", "ce_B", "kl_A", "kl_B",
            "loss_A", "loss_B", "entropy_B", "repulsion_B",
            "entropy_align", "H_A", "H_B",
            "grad_scale_A", "grad_scale_B",
            "lambda1", "lambda2", "lambda3",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_total_loss_is_tensor(self, loss_fn, sample_input, sample_outputs):
        """Total loss should be a tensor."""
        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].dim() == 0  # Scalar

    def test_total_loss_is_finite(self, loss_fn, sample_input, sample_outputs):
        """Total loss should be finite."""
        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert torch.isfinite(result["loss"])

    def test_gradient_flows_through_total_loss(self, loss_fn, sample_input, sample_outputs):
        """Gradients should flow through total loss."""
        # Make inputs require grad
        sample_outputs["logits_A"].requires_grad_(True)
        sample_outputs["logits_B"].requires_grad_(True)
        sample_outputs["mu_A"].requires_grad_(True)
        sample_outputs["logvar_A"].requires_grad_(True)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        result["loss"].backward()

        assert sample_outputs["logits_A"].grad is not None
        assert sample_outputs["mu_A"].grad is not None


# =============================================================================
# Gradient Balance Tests
# =============================================================================


class TestDualVAELossGradientBalance:
    """Test gradient balancing in DualVAELoss."""

    @pytest.fixture
    def loss_fn(self):
        return DualVAELoss()

    def test_gradient_balance_scales_losses(self, loss_fn, sample_input, sample_outputs):
        """Gradient balancing should scale losses."""
        grad_norm_A = torch.tensor(1.0)
        grad_norm_B = torch.tensor(2.0)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=grad_norm_A,
            grad_norm_B_ema=grad_norm_B,
            gradient_balance=True,
            training=True,
        )

        assert result["grad_scale_A"] == pytest.approx(2.0, abs=0.01)
        assert result["grad_scale_B"] == pytest.approx(0.5, abs=0.01)

    def test_gradient_balance_clamped(self, loss_fn, sample_input, sample_outputs):
        """Gradient scales should be clamped to [0.5, 2.0]."""
        # Extreme ratio
        grad_norm_A = torch.tensor(0.1)
        grad_norm_B = torch.tensor(10.0)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=grad_norm_A,
            grad_norm_B_ema=grad_norm_B,
            gradient_balance=True,
            training=True,
        )

        assert 0.5 <= result["grad_scale_A"] <= 2.0
        assert 0.5 <= result["grad_scale_B"] <= 2.0

    def test_gradient_balance_disabled_when_not_training(self, loss_fn, sample_input, sample_outputs):
        """Gradient balancing should be disabled during inference."""
        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(0.1),
            grad_norm_B_ema=torch.tensor(10.0),
            gradient_balance=True,
            training=False,
        )

        assert result["grad_scale_A"] == 1.0
        assert result["grad_scale_B"] == 1.0


# =============================================================================
# Entropy Alignment Tests
# =============================================================================


class TestDualVAELossEntropyAlignment:
    """Test entropy alignment in DualVAELoss."""

    @pytest.fixture
    def loss_fn(self):
        return DualVAELoss()

    def test_entropy_alignment_zero_when_equal(self, loss_fn, sample_input, sample_outputs):
        """Entropy alignment should be 0 when H_A == H_B."""
        sample_outputs["H_A"] = torch.tensor(1.5)
        sample_outputs["H_B"] = torch.tensor(1.5)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=1.0,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert result["entropy_align"].item() == pytest.approx(0.0, abs=1e-6)

    def test_entropy_alignment_increases_with_difference(self, loss_fn, sample_input, sample_outputs):
        """Entropy alignment should increase when H_A != H_B."""
        sample_outputs["H_A"] = torch.tensor(1.0)
        sample_outputs["H_B"] = torch.tensor(2.0)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=1.0,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert result["entropy_align"].item() == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Lambda Weight Tests
# =============================================================================


class TestDualVAELossLambdaWeights:
    """Test lambda weight effects in DualVAELoss."""

    @pytest.fixture
    def loss_fn(self):
        return DualVAELoss()

    def test_lambda1_scales_vae_a_loss(self, loss_fn, sample_input, sample_outputs):
        """Lambda1 should scale VAE-A contribution."""
        result_low = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=0.1,
            lambda2=1.0,
            lambda3=0.0,
            entropy_weight_B=0.0,
            repulsion_weight_B=0.0,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        result_high = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.0,
            entropy_weight_B=0.0,
            repulsion_weight_B=0.0,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert result_high["loss"] >= result_low["loss"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDualVAELossEdgeCases:
    """Test edge cases for DualVAELoss."""

    @pytest.fixture
    def loss_fn(self):
        return DualVAELoss()

    def test_handles_single_sample(self, loss_fn, device):
        """Should handle batch size of 1."""
        latent_dim = 16
        x = torch.randint(-1, 2, (1, 9), device=device).float()
        outputs = {
            "logits_A": torch.randn(1, 9, 3, device=device),
            "logits_B": torch.randn(1, 9, 3, device=device),
            "mu_A": torch.randn(1, latent_dim, device=device),
            "mu_B": torch.randn(1, latent_dim, device=device),
            "logvar_A": torch.randn(1, latent_dim, device=device),
            "logvar_B": torch.randn(1, latent_dim, device=device),
            "z_A": torch.randn(1, latent_dim, device=device),
            "z_B": torch.randn(1, latent_dim, device=device),
            "beta_A": 1.0,
            "beta_B": 1.0,
            "H_A": torch.tensor(1.5, device=device),
            "H_B": torch.tensor(1.5, device=device),
        }

        result = loss_fn(
            x=x,
            outputs=outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert torch.isfinite(result["loss"])

    def test_handles_zero_beta(self, loss_fn, sample_input, sample_outputs):
        """Should handle beta = 0 (no KL regularization)."""
        sample_outputs["beta_A"] = 0.0
        sample_outputs["beta_B"] = 0.0

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
        )

        assert torch.isfinite(result["loss"])


# =============================================================================
# p-Adic Loss Tests
# =============================================================================


class TestDualVAELossWithPAdicLosses:
    """Test DualVAELoss with p-Adic losses enabled."""

    def test_metric_loss_included(self, sample_input, sample_outputs, sample_batch_indices):
        """p-Adic metric loss should be included when enabled."""
        padic_config = {"enable_metric_loss": True, "metric_loss_weight": 0.1}
        loss_fn = DualVAELoss(padic_config=padic_config)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
            batch_indices=sample_batch_indices,
        )

        assert "padic_metric_A" in result
        assert "padic_metric_B" in result
        assert result["padic_metric_A"] >= 0
        assert result["padic_metric_B"] >= 0

    def test_ranking_loss_included(self, sample_input, sample_outputs, sample_batch_indices):
        """p-Adic ranking loss should be included when enabled."""
        padic_config = {"enable_ranking_loss": True, "ranking_loss_weight": 0.5}
        loss_fn = DualVAELoss(padic_config=padic_config)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
            batch_indices=sample_batch_indices,
        )

        assert "padic_ranking_A" in result
        assert "padic_ranking_B" in result

    def test_no_padic_loss_without_batch_indices(self, sample_input, sample_outputs):
        """p-Adic losses should not be computed without batch_indices."""
        padic_config = {
            "enable_metric_loss": True,
            "enable_ranking_loss": True,
            "enable_norm_loss": True,
        }
        loss_fn = DualVAELoss(padic_config=padic_config)

        result = loss_fn(
            x=sample_input,
            outputs=sample_outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0),
            grad_norm_B_ema=torch.tensor(1.0),
            gradient_balance=False,
            training=True,
            batch_indices=None,
        )

        assert result["padic_metric_A"] == 0.0
        assert result["padic_metric_B"] == 0.0


# =============================================================================
# CUDA Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDualVAELossCUDA:
    """Test DualVAELoss on CUDA."""

    def test_forward_on_cuda(self):
        """Should work on CUDA."""
        device = torch.device("cuda")
        batch_size, latent_dim = 32, 16
        loss_fn = DualVAELoss()

        x = torch.randint(-1, 2, (batch_size, 9), device=device).float()
        outputs = {
            "logits_A": torch.randn(batch_size, 9, 3, device=device),
            "logits_B": torch.randn(batch_size, 9, 3, device=device),
            "mu_A": torch.randn(batch_size, latent_dim, device=device),
            "mu_B": torch.randn(batch_size, latent_dim, device=device),
            "logvar_A": torch.randn(batch_size, latent_dim, device=device),
            "logvar_B": torch.randn(batch_size, latent_dim, device=device),
            "z_A": torch.randn(batch_size, latent_dim, device=device),
            "z_B": torch.randn(batch_size, latent_dim, device=device),
            "beta_A": 1.0,
            "beta_B": 1.0,
            "H_A": torch.tensor(1.5, device=device),
            "H_B": torch.tensor(1.5, device=device),
        }

        result = loss_fn(
            x=x,
            outputs=outputs,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=0.1,
            entropy_weight_B=0.01,
            repulsion_weight_B=0.1,
            grad_norm_A_ema=torch.tensor(1.0, device=device),
            grad_norm_B_ema=torch.tensor(1.0, device=device),
            gradient_balance=False,
            training=True,
        )

        assert result["loss"].device.type == "cuda"
        assert torch.isfinite(result["loss"])
