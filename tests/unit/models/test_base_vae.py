# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for BaseVAE abstraction.

Tests cover:
- VAEConfig dataclass
- VAEOutput container
- BaseVAE abstract methods
- HyperbolicBaseVAE projections
- ConditionalBaseVAE conditioning
- Parameter counting
- Loss computation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base_vae import (
    BaseVAE,
    ConditionalBaseVAE,
    HyperbolicBaseVAE,
    VAEConfig,
    VAEOutput,
)


class SimpleTestVAE(BaseVAE):
    """Simple VAE for testing BaseVAE functionality."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[-1]),
        )
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], self.input_dim * 3),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, self.input_dim, 3)


class TestVAEConfig:
    """Test VAEConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VAEConfig()
        assert config.input_dim == 9
        assert config.latent_dim == 16
        assert config.hidden_dims == [64, 32]
        assert config.dropout == 0.0
        assert config.activation == "relu"
        assert config.beta_vae_weight == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = VAEConfig(
            input_dim=128,
            latent_dim=32,
            hidden_dims=[256, 128],
            dropout=0.2,
            activation="gelu",
        )
        assert config.input_dim == 128
        assert config.latent_dim == 32
        assert config.hidden_dims == [256, 128]
        assert config.dropout == 0.2

    def test_config_curvature(self):
        """Test hyperbolic curvature configuration."""
        config = VAEConfig(curvature=0.5)
        assert config.curvature == 0.5


class TestVAEOutput:
    """Test VAEOutput container."""

    def test_output_creation(self):
        """Test VAEOutput creation."""
        logits = torch.randn(4, 9, 3)
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)
        z = torch.randn(4, 16)

        output = VAEOutput(logits=logits, mu=mu, logvar=logvar, z=z)

        assert output.logits.shape == (4, 9, 3)
        assert output.mu.shape == (4, 16)
        assert output.z_hyp is None

    def test_output_to_dict(self):
        """Test conversion to dictionary."""
        logits = torch.randn(4, 9, 3)
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)
        z = torch.randn(4, 16)

        output = VAEOutput(logits=logits, mu=mu, logvar=logvar, z=z)
        d = output.to_dict()

        assert "logits" in d
        assert "mu" in d
        assert "logvar" in d
        assert "z" in d

    def test_output_with_extras(self):
        """Test output with extra fields."""
        output = VAEOutput(
            logits=torch.randn(4, 9, 3),
            mu=torch.randn(4, 16),
            logvar=torch.randn(4, 16),
            z=torch.randn(4, 16),
            extras={"custom_field": torch.randn(4, 8)},
        )

        d = output.to_dict()
        assert "custom_field" in d


class TestBaseVAE:
    """Test BaseVAE abstract base class."""

    @pytest.fixture
    def vae(self):
        """Create test VAE."""
        return SimpleTestVAE()

    @pytest.fixture
    def batch(self):
        """Create test batch."""
        return torch.randn(8, 9)

    def test_initialization_default(self, vae):
        """Test default initialization."""
        assert vae.input_dim == 9
        assert vae.latent_dim == 16
        assert vae.hidden_dims == [64, 32]

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SimpleTestVAE(config=config)
        assert vae.input_dim == 32
        assert vae.latent_dim == 8

    def test_initialization_with_kwargs(self):
        """Test initialization with kwargs."""
        vae = SimpleTestVAE(input_dim=64, latent_dim=24)
        assert vae.input_dim == 64
        assert vae.latent_dim == 24

    def test_reparameterize_training(self, vae):
        """Test reparameterization during training."""
        vae.train()
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)  # std = 1

        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)

        # Should be different (stochastic)
        assert not torch.allclose(z1, z2)

    def test_reparameterize_eval(self, vae):
        """Test reparameterization during evaluation."""
        vae.eval()
        mu = torch.randn(8, 16)
        logvar = torch.zeros(8, 16)

        z = vae.reparameterize(mu, logvar)

        # Should equal mu (deterministic)
        assert torch.allclose(z, mu)

    def test_forward_shape(self, vae, batch):
        """Test forward pass output shapes."""
        outputs = vae(batch)

        assert "logits" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs

        assert outputs["logits"].shape == (8, 9, 3)
        assert outputs["mu"].shape == (8, 16)
        assert outputs["logvar"].shape == (8, 16)
        assert outputs["z"].shape == (8, 16)

    def test_encode(self, vae, batch):
        """Test encode method."""
        mu, logvar = vae.encode(batch)

        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

    def test_decode(self, vae):
        """Test decode method."""
        z = torch.randn(8, 16)
        logits = vae.decode(z)

        assert logits.shape == (8, 9, 3)

    def test_encode_mean(self, vae, batch):
        """Test encode_mean method."""
        mu = vae.encode_mean(batch)
        assert mu.shape == (8, 16)

    def test_kl_divergence(self, vae):
        """Test KL divergence computation."""
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)

        kl = vae.kl_divergence(mu, logvar)

        # KL(N(0,1) || N(0,1)) = 0
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_kl_divergence_nonzero(self, vae):
        """Test KL divergence with non-standard parameters."""
        mu = torch.ones(8, 16)
        logvar = torch.zeros(8, 16)

        kl = vae.kl_divergence(mu, logvar)

        # Should be positive
        assert kl > 0

    def test_reconstruction_loss_cross_entropy(self, vae, batch):
        """Test cross-entropy reconstruction loss."""
        outputs = vae(batch)
        targets = torch.randint(0, 3, (8, 9)) - 1  # {-1, 0, 1}

        loss = vae.reconstruction_loss(outputs["logits"], targets.float(), "cross_entropy")

        assert loss.dim() == 0  # Scalar
        assert loss > 0

    def test_compute_loss(self, vae, batch):
        """Test combined loss computation."""
        losses = vae.compute_loss(batch)

        assert "total" in losses
        assert "recon" in losses
        assert "kl" in losses
        assert losses["total"] == losses["recon"] + losses["kl"]

    def test_compute_loss_with_beta(self, vae, batch):
        """Test loss with custom beta."""
        # Use eval mode for deterministic behavior
        vae.eval()
        losses1 = vae.compute_loss(batch, beta=1.0)
        losses2 = vae.compute_loss(batch, beta=0.1)
        vae.train()

        # Same reconstruction (in eval mode), different total due to beta
        assert torch.isclose(losses1["recon"], losses2["recon"])

    def test_reconstruct(self, vae, batch):
        """Test reconstruction."""
        recon = vae.reconstruct(batch)

        assert recon.shape == (8, 9)
        # Values should be in {-1, 0, 1}
        assert set(recon.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_sample(self, vae):
        """Test sampling from prior."""
        samples = vae.sample(16)

        assert samples.shape[0] == 16

    def test_interpolate(self, vae, batch):
        """Test latent space interpolation."""
        x1 = batch[:1]
        x2 = batch[1:2]

        interpolations = vae.interpolate(x1, x2, n_steps=5)

        assert len(interpolations) == 5

    def test_count_parameters(self, vae):
        """Test parameter counting."""
        params = vae.count_parameters()

        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params
        assert params["total"] == params["trainable"] + params["frozen"]
        assert params["total"] > 0

    def test_get_latent_dim(self, vae):
        """Test get_latent_dim method."""
        assert vae.get_latent_dim() == 16

    def test_activation_functions(self):
        """Test different activation functions."""
        for act in ["relu", "gelu", "silu", "tanh"]:
            vae = SimpleTestVAE(activation=act)
            assert vae.activation is not None


class TestHyperbolicBaseVAE:
    """Test HyperbolicBaseVAE class."""

    class SimpleHyperbolicVAE(HyperbolicBaseVAE):
        """Simple hyperbolic VAE for testing."""

        def __init__(self, config=None, **kwargs):
            super().__init__(config, **kwargs)
            self.encoder = nn.Linear(self.input_dim, self.latent_dim * 2)
            self.decoder = nn.Linear(self.latent_dim, self.input_dim)

        def encode(self, x):
            h = self.encoder(x)
            return h[:, : self.latent_dim], h[:, self.latent_dim :]

        def decode(self, z):
            return self.decoder(z)

    @pytest.fixture
    def vae(self):
        """Create hyperbolic VAE."""
        return self.SimpleHyperbolicVAE()

    def test_exp_map(self, vae):
        """Test exponential map."""
        v = torch.randn(8, 16)
        y = vae.exp_map(v)

        # Result should be in Poincare ball (norm < 1)
        norms = torch.norm(y, dim=-1)
        assert (norms < 1).all()

    def test_log_map(self, vae):
        """Test logarithmic map."""
        v = torch.randn(8, 16) * 0.1  # Small to stay in ball
        y = vae.exp_map(v)
        v_recovered = vae.log_map(y)

        # Should approximately recover original
        assert torch.allclose(v, v_recovered, atol=0.1)

    def test_hyperbolic_distance(self, vae):
        """Test hyperbolic distance."""
        x = torch.randn(8, 16) * 0.1
        y = torch.randn(8, 16) * 0.1

        dist = vae.hyperbolic_distance(x, y)

        # Distance should be non-negative
        assert (dist >= 0).all()

    def test_forward_with_hyperbolic(self, vae):
        """Test forward pass returns hyperbolic latent."""
        x = torch.randn(8, 9)
        outputs = vae(x)

        assert "z_hyp" in outputs
        assert outputs["z_euc"].shape == outputs["z_hyp"].shape

        # z_hyp should be in Poincare ball
        norms = torch.norm(outputs["z_hyp"], dim=-1)
        assert (norms < 1).all()


class TestConditionalBaseVAE:
    """Test ConditionalBaseVAE class."""

    class SimpleConditionalVAE(ConditionalBaseVAE):
        """Simple conditional VAE for testing."""

        def __init__(self, config=None, condition_dim=8, n_conditions=5, **kwargs):
            super().__init__(config, condition_dim=condition_dim, n_conditions=n_conditions, **kwargs)
            total_dim = self.input_dim + condition_dim
            self.encoder = nn.Linear(total_dim, self.latent_dim * 2)
            self.decoder = nn.Linear(self.latent_dim + condition_dim, self.input_dim)

        def encode(self, x, condition=None):
            if condition is not None:
                x = torch.cat([x, condition], dim=-1)
            else:
                x = torch.cat([x, torch.zeros(x.shape[0], self.condition_dim, device=x.device)], dim=-1)
            h = self.encoder(x)
            return h[:, : self.latent_dim], h[:, self.latent_dim :]

        def decode(self, z, condition=None):
            if condition is not None:
                z = torch.cat([z, condition], dim=-1)
            else:
                z = torch.cat([z, torch.zeros(z.shape[0], self.condition_dim, device=z.device)], dim=-1)
            return self.decoder(z)

        def forward(self, x, condition=None, **kwargs):
            if condition is not None:
                cond_emb = self.get_condition_embedding(condition)
            else:
                cond_emb = None

            mu, logvar = self.encode(x, cond_emb)
            z = self.reparameterize(mu, logvar)
            logits = self.decode(z, cond_emb)

            return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    @pytest.fixture
    def vae(self):
        """Create conditional VAE."""
        return self.SimpleConditionalVAE()

    def test_condition_embedding(self, vae):
        """Test condition embedding."""
        condition = torch.tensor([0, 1, 2, 3])
        emb = vae.get_condition_embedding(condition)

        assert emb.shape == (4, 8)

    def test_forward_with_condition(self, vae):
        """Test forward with condition."""
        x = torch.randn(4, 9)
        condition = torch.tensor([0, 1, 2, 3])

        outputs = vae(x, condition=condition)

        assert "logits" in outputs
        assert outputs["logits"].shape[0] == 4

    def test_forward_without_condition(self, vae):
        """Test forward without condition."""
        x = torch.randn(4, 9)

        outputs = vae(x)

        assert "logits" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
