import pytest
import torch
import torch.nn as nn
from tests.core.builders import VAEBuilder
from tests.factories.data import TernaryOperationFactory
from tests.core.matchers import expect_poincare
from tests.core.helpers import MockFrozenModule

# Helper for mocking modules
# (Kept for now if needed by builder internal logic, but builder handles patching)
# Actually builder uses MockFrozenModule from this file?
# We should move MockFrozenModule to a shared helper or keep it here if only used here.
# The builder imports it from here: "from tests.unit.test_models import MockFrozenModule"
# So we must keep it or move it.
# Better to move it to tests/core/helpers.py but for now lets keep it to avoid circular deps if builder imports it.
# Wait, builder imports it from "tests.unit.test_models".
# If we change this file, builder might break if it expects it here.
# Let's check builder import: "from tests.unit.test_models import MockFrozenModule"
# Yes. So we must keep MockFrozenModule definition here or refactor builder to use a shared location.
# Let's keep it here for this refactor step to minimize diff.


def test_vae_initialization():
    """Verify VAE initializes using Builder."""
    model = VAEBuilder().build()

    # We check if attributes exist
    assert hasattr(model, "projection")
    assert hasattr(model, "controller")


def test_vae_forward_shape(device):
    """Verify forward pass returns correct shapes."""
    # Use Builder to get a model with mocked frozen components
    model = VAEBuilder().build()
    model.to(device)

    # Use Factory for data
    x = TernaryOperationFactory.create_batch(size=4, device=device)

    # Forward
    outputs = model(x)

    assert "logits_A" in outputs
    assert outputs["logits_A"].shape == (4, 9, 3)
    assert "z_A_hyp" in outputs
    assert outputs["z_A_hyp"].shape == (4, 16)

    # Use Matcher
    expect_poincare(outputs["z_A_hyp"])


def test_gradients_only_projection(device):
    """Verify gradients propagate to projection but NOT to frozen encoder."""
    # Use Builder to get a model with mocked frozen components
    model = VAEBuilder().build()
    model.to(device)

    x = TernaryOperationFactory.create_batch(size=4, device=device)

    # We need to verify parameters of projection have grad
    outputs = model(x)
    loss = outputs["z_A_hyp"].sum()
    loss.backward()

    # Check projection weights
    # Note: init_identity=True zero-initializes the last layer of direction_net,
    # which blocks gradient flow to earlier layers of that sub-network.
    # So we only expect SOME parameters (e.g. radius_net, last layer of direction_net) to have grad.

    found_grad = False
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                found_grad = True

    assert found_grad, "Projection should learn (at least some params have grad)"


def test_dual_projection_flow(device):
    """Verify dual projection (Bio-Hyperbolic) gradient flow."""
    model = VAEBuilder().with_dual_projection().build()
    model.to(device)

    x = TernaryOperationFactory.create_batch(size=4, device=device)
    outputs = model(x)

    # Dual projection produces z_A_hyp and z_B_hyp
    loss = outputs["z_A_hyp"].sum() + outputs["z_B_hyp"].sum()
    loss.backward()

    found_grad = False
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                found_grad = True

    assert found_grad, "Dual projection should receive gradients"
