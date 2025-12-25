# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/losses/registry.py - LossRegistry and LossGroup.

Tests the dynamic loss composition system:
- LossRegistry: Dynamic registration and composition
- LossGroup: Logical grouping of losses
- Factory function for config-based creation
"""

import pytest
import torch

from src.losses.base import LossResult
from src.losses.components import (EntropyLossComponent,
                                   KLDivergenceLossComponent,
                                   ReconstructionLossComponent)
from src.losses.registry import (LossGroup, LossRegistry,
                                 create_registry_from_config)

# =============================================================================
# LossRegistry Tests
# =============================================================================


class TestLossRegistry:
    """Tests for LossRegistry."""

    @pytest.fixture
    def registry(self):
        """Create empty registry."""
        return LossRegistry()

    @pytest.fixture
    def populated_registry(self):
        """Create registry with components."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        registry.register("kl", KLDivergenceLossComponent(weight=0.1))
        return registry

    def test_initialization(self, registry):
        """Test registry initializes correctly."""
        assert len(registry.list_losses()) == 0

    def test_register_component(self, registry):
        """Test registering a component."""
        component = ReconstructionLossComponent(weight=1.0)
        registry.register("recon", component)

        assert "recon" in registry.list_losses()
        assert len(registry.list_losses()) == 1

    def test_register_multiple_components(self, registry):
        """Test registering multiple components."""
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        registry.register("kl", KLDivergenceLossComponent(weight=0.1))
        registry.register("entropy", EntropyLossComponent(weight=0.01))

        assert len(registry.list_losses()) == 3
        assert "recon" in registry.list_losses()
        assert "kl" in registry.list_losses()
        assert "entropy" in registry.list_losses()

    def test_get_loss(self, populated_registry):
        """Test getting a registered component."""
        component = populated_registry.get_loss("recon")
        assert isinstance(component, ReconstructionLossComponent)

    def test_get_nonexistent_component(self, registry):
        """Test getting non-existent component returns None."""
        component = registry.get_loss("nonexistent")
        assert component is None

    def test_unregister_component(self, populated_registry):
        """Test removing a component."""
        populated_registry.unregister("recon")
        assert "recon" not in populated_registry.list_losses()
        assert len(populated_registry.list_losses()) == 1

    def test_list_losses(self, populated_registry):
        """Test listing registered components."""
        names = populated_registry.list_losses()
        assert set(names) == {"recon", "kl"}

    def test_register_duplicate_raises(self, registry):
        """Test registering duplicate name raises error."""
        registry.register("recon", ReconstructionLossComponent(weight=1.0))

        with pytest.raises(ValueError):
            registry.register("recon", ReconstructionLossComponent(weight=0.5))

    def test_set_weight(self, populated_registry):
        """Test setting weight override."""
        populated_registry.set_weight("recon", 2.0)
        assert populated_registry.get_weight("recon") == 2.0

    def test_reset_weight(self, populated_registry):
        """Test resetting weight override."""
        original_weight = populated_registry.get_weight("recon")
        populated_registry.set_weight("recon", 5.0)
        populated_registry.reset_weight("recon")

        assert populated_registry.get_weight("recon") == original_weight

    def test_set_enabled(self, populated_registry):
        """Test enabling/disabling losses."""
        populated_registry.set_enabled("kl", False)
        assert "kl" not in populated_registry.list_enabled()

        populated_registry.set_enabled("kl", True)
        assert "kl" in populated_registry.list_enabled()


class TestLossRegistryCompose:
    """Tests for LossRegistry.compose method."""

    @pytest.fixture
    def registry(self):
        """Create registry with components."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        registry.register("kl", KLDivergenceLossComponent(weight=0.1))
        return registry

    def test_compose_returns_loss_result(self, registry, loss_test_outputs, device):
        """Test compose returns LossResult."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = registry.compose(loss_test_outputs, targets)

        assert isinstance(result, LossResult)

    def test_compose_has_total_loss(self, registry, loss_test_outputs, device):
        """Test compose result has total loss."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = registry.compose(loss_test_outputs, targets)

        assert hasattr(result, "loss")
        assert isinstance(result.loss, torch.Tensor)

    def test_compose_has_metrics(self, registry, loss_test_outputs, device):
        """Test compose result has metrics from components."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = registry.compose(loss_test_outputs, targets)

        assert hasattr(result, "metrics")
        assert isinstance(result.metrics, dict)

        # Should have prefixed keys for each component
        assert "recon/loss" in result.metrics
        assert "kl/loss" in result.metrics

    def test_compose_total_is_weighted_sum(self, registry, loss_test_outputs, device):
        """Test total loss is weighted sum of components."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = registry.compose(loss_test_outputs, targets)

        # Get individual losses from metrics
        recon_loss = result.metrics["recon/loss"]
        kl_loss = result.metrics["kl/loss"]

        # Get weights
        recon_weight = registry.get_weight("recon")
        kl_weight = registry.get_weight("kl")

        # Calculate expected total
        expected_total = recon_weight * recon_loss + kl_weight * kl_loss

        assert torch.allclose(result.loss, torch.tensor(expected_total), atol=1e-4)

    def test_compose_empty_registry(self, device, loss_test_outputs):
        """Test compose with empty registry."""
        registry = LossRegistry()
        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        result = registry.compose(loss_test_outputs, targets)

        assert result.loss.item() == 0.0

    def test_compose_respects_enabled_flag(self, registry, loss_test_outputs, device):
        """Test compose skips disabled losses."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        # Get loss with KL enabled
        result_with_kl = registry.compose(loss_test_outputs, targets)

        # Disable KL
        registry.set_enabled("kl", False)
        result_without_kl = registry.compose(loss_test_outputs, targets)

        # Loss should be smaller without KL
        # (assuming positive KL contribution)
        assert result_without_kl.loss < result_with_kl.loss


class TestLossRegistryWithKwargs:
    """Tests for registry compose with additional kwargs."""

    @pytest.fixture
    def registry(self):
        """Create registry with components."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        return registry

    def test_compose_with_batch_indices(self, registry, loss_test_outputs, device):
        """Test compose passes batch_indices to components."""
        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        batch_indices = torch.randint(0, 19683, (32,), device=device)

        result = registry.compose(loss_test_outputs, targets, batch_indices=batch_indices)

        assert hasattr(result, "loss")
        assert torch.isfinite(result.loss)


# =============================================================================
# LossGroup Tests
# =============================================================================


class TestLossGroup:
    """Tests for LossGroup."""

    @pytest.fixture
    def group(self):
        """Create a loss group."""
        return LossGroup(name="vae_losses")

    def test_initialization(self, group):
        """Test group initializes correctly."""
        assert group.name == "vae_losses"
        assert len(group.losses) == 0

    def test_add_component(self, group):
        """Test adding component to group."""
        group.add("recon", ReconstructionLossComponent(weight=1.0))
        assert "recon" in group.losses
        assert len(group.losses) == 1

    def test_add_multiple_components(self, group):
        """Test adding multiple components."""
        group.add("recon", ReconstructionLossComponent(weight=1.0))
        group.add("kl", KLDivergenceLossComponent(weight=0.1))

        assert len(group.losses) == 2

    def test_chaining(self, group):
        """Test method chaining."""
        result = group.add("recon", ReconstructionLossComponent(weight=1.0))
        assert result is group

    def test_losses_property(self, group):
        """Test losses property returns dictionary."""
        group.add("recon", ReconstructionLossComponent(weight=1.0))

        losses = group.losses
        assert isinstance(losses, dict)
        assert "recon" in losses


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateRegistryFromConfig:
    """Tests for create_registry_from_config factory function."""

    def test_creates_registry_from_basic_config(self):
        """Test creating registry from basic config."""
        config = {
            "losses": {
                "reconstruction": {"enabled": True, "weight": 1.0},
                "kl": {"enabled": True, "weight": 0.1, "free_bits": 0.0},
            }
        }

        registry = create_registry_from_config(config)

        assert isinstance(registry, LossRegistry)
        assert len(registry.list_losses()) == 2

    def test_creates_registry_from_empty_config(self):
        """Test creating registry from empty config."""
        config = {}

        registry = create_registry_from_config(config)

        assert isinstance(registry, LossRegistry)
        assert len(registry.list_losses()) == 0

    def test_only_enables_requested_losses(self):
        """Test only enabled losses are registered."""
        config = {
            "losses": {
                "reconstruction": {"enabled": True, "weight": 1.0},
                "kl": {"enabled": False, "weight": 0.1},
            }
        }

        registry = create_registry_from_config(config)

        assert "reconstruction" in registry.list_losses()
        assert "kl" not in registry.list_losses()

    def test_uses_config_weights(self):
        """Test registry uses weights from config."""
        config = {
            "losses": {
                "reconstruction": {"enabled": True, "weight": 2.5},
            }
        }

        registry = create_registry_from_config(config)

        assert registry.get_weight("reconstruction") == 2.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Integration tests for the registry system."""

    def test_full_vae_loss_pipeline(self, loss_test_outputs, device):
        """Test complete VAE loss computation pipeline."""
        # Create registry with all standard losses
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        registry.register("kl", KLDivergenceLossComponent(weight=0.1))
        registry.register("entropy", EntropyLossComponent(weight=0.01, vae="B"))

        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        # Compute losses
        result = registry.compose(loss_test_outputs, targets)

        # Verify structure
        assert hasattr(result, "loss")
        assert "recon/loss" in result.metrics
        assert "kl/loss" in result.metrics
        assert "entropy/loss" in result.metrics

        # Verify total is sum
        total = sum(registry.get_weight(name) * result.metrics[f"{name}/loss"] for name in registry.list_losses())
        assert torch.allclose(result.loss, torch.tensor(total), atol=1e-4)

    def test_gradient_flow_through_registry(self, device):
        """Test gradients flow through registry compose."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))

        batch_size = 4
        logits_A = torch.randn(batch_size, 9, 3, device=device, requires_grad=True)
        logits_B = torch.randn(batch_size, 9, 3, device=device, requires_grad=True)

        outputs = {
            "logits_A": logits_A,
            "logits_B": logits_B,
        }

        targets = torch.randint(-1, 2, (batch_size, 9), device=device).float()

        result = registry.compose(outputs, targets)
        result.loss.backward()

        assert logits_A.grad is not None
        assert logits_B.grad is not None

    def test_registry_with_disabled_components(self, loss_test_outputs, device):
        """Test registry handles disabled components."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))
        registry.register("kl", KLDivergenceLossComponent(weight=0.1))

        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        # Disable KL
        registry.set_enabled("kl", False)

        result = registry.compose(loss_test_outputs, targets)

        # Total should equal just reconstruction (KL is disabled)
        recon_component = registry.get_loss("recon")
        recon_result = recon_component.forward(loss_test_outputs, targets)

        assert torch.allclose(result.loss, recon_result.loss, atol=1e-4)

    def test_weight_override_affects_compose(self, loss_test_outputs, device):
        """Test weight overrides affect compose result."""
        registry = LossRegistry()
        registry.register("recon", ReconstructionLossComponent(weight=1.0))

        targets = torch.randint(-1, 2, (32, 9), device=device).float()

        # Get baseline result
        result1 = registry.compose(loss_test_outputs, targets)

        # Override weight to 2x
        registry.set_weight("recon", 2.0)
        result2 = registry.compose(loss_test_outputs, targets)

        # Result should be ~2x larger
        assert torch.allclose(result2.loss, result1.loss * 2, atol=1e-4)
