# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Ternary VAE models.

Tests cover:
- Model instantiation
- Forward pass shapes
- Hyperbolic projection
- Loss computation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root
root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root))


class TestHyperbolicProjection:
    """Tests for hyperbolic projection module."""

    def test_import(self):
        """Test that HyperbolicProjection can be imported."""
        from src.models.hyperbolic_projection import HyperbolicProjection
        assert HyperbolicProjection is not None

    def test_instantiation(self):
        """Test HyperbolicProjection instantiation."""
        import torch
        from src.models.hyperbolic_projection import HyperbolicProjection

        proj = HyperbolicProjection(latent_dim=16)
        assert proj is not None
        assert isinstance(proj, torch.nn.Module)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        import torch
        from src.models.hyperbolic_projection import HyperbolicProjection

        proj = HyperbolicProjection(latent_dim=16)
        x = torch.randn(32, 16)  # batch_size=32, dim=16
        z_hyp = proj(x)

        assert z_hyp.shape == (32, 16)

    def test_poincare_constraint(self):
        """Test that output is within Poincaré ball (norm < 1)."""
        import torch
        from src.models.hyperbolic_projection import HyperbolicProjection

        proj = HyperbolicProjection(latent_dim=16, max_radius=0.95)
        x = torch.randn(100, 16) * 10  # Large input
        z_hyp = proj(x)

        norms = torch.norm(z_hyp, dim=-1)
        assert (norms < 1.0).all(), "All points should be within unit ball"

    def test_gradient_flow(self):
        """Test that gradients flow through projection."""
        import torch
        from src.models.hyperbolic_projection import HyperbolicProjection

        proj = HyperbolicProjection(latent_dim=16)
        x = torch.randn(8, 16, requires_grad=True)
        z_hyp = proj(x)

        loss = z_hyp.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"


class TestSimpleVAE:
    """Tests for SimpleVAE model."""

    def test_import(self):
        """Test that SimpleVAE can be imported."""
        from src.models.simple_vae import SimpleVAE
        assert SimpleVAE is not None

    def test_instantiation(self):
        """Test SimpleVAE instantiation."""
        import torch
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        import torch
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])
        x = torch.randn(8, 100)

        output = model(x)

        assert "recon" in output or "x_recon" in output or "logits" in output
        assert "mu" in output
        assert "logvar" in output or "log_var" in output

    def test_encode_decode(self):
        """Test encode and decode methods."""
        import torch
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])
        x = torch.randn(8, 100)

        # Encode via encoder network (returns mu, logvar tuple)
        mu, logvar = model.encoder(x)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

        # Encode method returns just mu
        mu_only = model.encode(x)
        assert mu_only.shape == (8, 16)

        # Decode
        z = mu  # Use mean for deterministic test
        x_recon = model.decode(z)
        # Decoder returns logits of shape (batch, input_dim, 3) for ternary classification
        assert x_recon.shape == (8, 100, 3)

    def test_reparameterize(self):
        """Test reparameterization trick."""
        import torch
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])

        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)

        z = model.reparameterize(mu, logvar)
        assert z.shape == (8, 16)

        # With zero variance, z should approximately equal mu
        z_deterministic = model.reparameterize(mu, torch.full_like(logvar, -100))
        assert torch.allclose(z_deterministic, mu, atol=1e-3)


class TestPadicNetworks:
    """Tests for p-adic network layers."""

    def test_import(self):
        """Test that p-adic networks can be imported."""
        from src.models.padic_networks import HierarchicalPAdicMLP, PAdicEmbedding
        assert HierarchicalPAdicMLP is not None
        assert PAdicEmbedding is not None

    def test_mlp_instantiation(self):
        """Test HierarchicalPAdicMLP instantiation."""
        import torch
        from src.models.padic_networks import HierarchicalPAdicMLP

        mlp = HierarchicalPAdicMLP(input_dim=32, hidden_dims=[64], output_dim=16, p=3)
        assert mlp is not None

    def test_mlp_forward(self):
        """Test HierarchicalPAdicMLP forward pass."""
        import torch
        from src.models.padic_networks import HierarchicalPAdicMLP

        mlp = HierarchicalPAdicMLP(input_dim=32, hidden_dims=[64], output_dim=16, p=3)
        x = torch.randn(8, 32)
        indices = torch.randint(0, 27, (8,))  # p^3 = 27 possible indices

        output = mlp(x, indices)
        assert output is not None
        assert output.shape == (8, 16)


class TestResistanceTransformer:
    """Tests for Resistance Transformer model."""

    def test_import(self):
        """Test that ResistanceTransformer can be imported."""
        try:
            from src.models.resistance_transformer import ResistanceTransformer
            assert ResistanceTransformer is not None
        except ImportError:
            pytest.skip("ResistanceTransformer not available")

    def test_instantiation(self):
        """Test ResistanceTransformer instantiation."""
        import torch
        try:
            from src.models.resistance_transformer import ResistanceTransformer, TransformerConfig

            cfg = TransformerConfig(
                n_aa=22,  # 20 AA + gap + unknown
                d_model=64,
                n_heads=4,
                n_layers=2,
            )
            model = ResistanceTransformer(cfg)
            assert model is not None
        except ImportError:
            pytest.skip("ResistanceTransformer not available")


class TestModelUtilities:
    """Tests for model utility functions."""

    def test_checkpoint_loading(self):
        """Test checkpoint compatibility loader."""
        from src.utils.checkpoint import load_checkpoint_compat
        assert load_checkpoint_compat is not None

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        import torch

        # Standard normal KL divergence
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)

        # KL(N(0,1) || N(0,1)) = 0
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-5)

        # KL with non-zero mean
        mu = torch.ones(8, 16)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        assert kl > 0, "KL should be positive with non-zero mean"


class TestModelIntegration:
    """Integration tests for model training workflow."""

    def test_training_step(self):
        """Test a single training step."""
        import torch
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create ternary data in {-1, 0, 1}
        x = torch.randint(-1, 2, (8, 100)).float()

        # Forward pass
        model.train()
        output = model(x)

        # Compute loss (reconstruction + KL)
        if "recon" in output:
            recon = output["recon"]
            recon_loss = torch.nn.functional.mse_loss(recon, x)
        elif "x_recon" in output:
            recon = output["x_recon"]
            recon_loss = torch.nn.functional.mse_loss(recon, x)
        else:
            # SimpleVAE returns logits of shape (batch, input_dim, 3)
            logits = output.get("logits", x)
            if logits.dim() == 3 and logits.shape[-1] == 3:
                # Use cross-entropy loss for ternary classification
                # Convert x from {-1, 0, 1} to class indices {0, 1, 2}
                target_classes = (x + 1).long()
                recon_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, 3), target_classes.view(-1)
                )
            else:
                recon_loss = torch.nn.functional.mse_loss(logits, x)

        mu = output["mu"]
        logvar = output.get("logvar", output.get("log_var", torch.zeros_like(mu)))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + 0.1 * kl_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0, "Loss should be non-negative"

    def test_model_save_load(self):
        """Test model save and load."""
        import torch
        import tempfile
        from src.models.simple_vae import SimpleVAE

        model = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            # Load into new model
            model2 = SimpleVAE(input_dim=100, latent_dim=16, hidden_dims=[64, 32])
            model2.load_state_dict(torch.load(f.name, weights_only=True))

            # Check weights are equal
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                assert torch.equal(p1, p2), f"Parameter {n1} mismatch"
