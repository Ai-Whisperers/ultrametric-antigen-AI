# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for SwarmVAE multi-agent architecture."""

import pytest
import torch

from src.models.swarm_vae import (
    AgentConfig,
    AgentRole,
    PheromoneField,
    SwarmAgent,
    SwarmVAE,
)


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_creation(self):
        """Test default config creation."""
        config = AgentConfig(role=AgentRole.EXPLORER)
        assert config.role == AgentRole.EXPLORER
        assert config.temperature == 1.0
        assert config.exploration_rate == 0.5

    def test_default_explorer(self):
        """Test default explorer configuration."""
        config = AgentConfig.default_explorer()
        assert config.role == AgentRole.EXPLORER
        assert config.temperature == 1.5
        assert config.exploration_rate == 0.8

    def test_default_exploiter(self):
        """Test default exploiter configuration."""
        config = AgentConfig.default_exploiter()
        assert config.role == AgentRole.EXPLOITER
        assert config.temperature == 0.5
        assert config.exploration_rate == 0.2

    def test_default_validator(self):
        """Test default validator configuration."""
        config = AgentConfig.default_validator()
        assert config.role == AgentRole.VALIDATOR
        assert config.temperature == 1.0

    def test_default_integrator(self):
        """Test default integrator configuration."""
        config = AgentConfig.default_integrator()
        assert config.role == AgentRole.INTEGRATOR
        assert config.temperature == 0.8


class TestPheromoneField:
    """Tests for PheromoneField pheromone tracking."""

    def test_creation(self):
        """Test field creation."""
        field = PheromoneField(
            positions=torch.zeros(0, 16),
            strengths=torch.zeros(0),
            decay_rate=0.95,
        )
        assert field.positions.shape[0] == 0
        assert field.decay_rate == 0.95

    def test_deposit(self):
        """Test pheromone deposit."""
        field = PheromoneField(
            positions=torch.zeros(0, 16),
            strengths=torch.zeros(0),
        )

        position = torch.randn(16)
        field.deposit(position, strength=1.0)

        assert field.positions.shape[0] == 1
        assert field.strengths.shape[0] == 1
        assert field.strengths[0] == 1.0

    def test_deposit_batch(self):
        """Test batch pheromone deposit."""
        field = PheromoneField(
            positions=torch.zeros(0, 16),
            strengths=torch.zeros(0),
        )

        positions = torch.randn(5, 16)
        field.deposit(positions, strength=0.5)

        assert field.positions.shape[0] == 5
        assert (field.strengths == 0.5).all()

    def test_decay(self):
        """Test pheromone decay."""
        field = PheromoneField(
            positions=torch.randn(3, 16),
            strengths=torch.ones(3),
            decay_rate=0.5,
        )

        initial_strength = field.strengths.clone()
        field.decay()

        assert (field.strengths < initial_strength).all()
        assert field.strengths[0] == pytest.approx(0.5)

    def test_query(self):
        """Test pheromone query."""
        # Create field with one pheromone at origin
        field = PheromoneField(
            positions=torch.zeros(1, 16),
            strengths=torch.ones(1),
        )

        # Query at origin should have high influence
        origin_influence = field.query(torch.zeros(16), radius=0.3)

        # Query far away should have low influence
        far_influence = field.query(torch.ones(16) * 10, radius=0.3)

        assert origin_influence > far_influence

    def test_query_empty(self):
        """Test query on empty field."""
        field = PheromoneField(
            positions=torch.zeros(0, 16),
            strengths=torch.zeros(0),
        )

        influence = field.query(torch.randn(16))
        assert influence.item() == 0.0

    def test_pruning(self):
        """Test automatic pruning of old pheromones."""
        field = PheromoneField(
            positions=torch.zeros(0, 8),
            strengths=torch.zeros(0),
            max_positions=10,
        )

        # Add more than max
        for i in range(15):
            field.deposit(torch.randn(8), strength=float(i))

        # Should be pruned to max
        assert field.positions.shape[0] == 10
        # Should keep strongest (highest indices)
        assert field.strengths.min() >= 5.0


class TestSwarmAgent:
    """Tests for SwarmAgent individual agent."""

    def test_creation(self):
        """Test agent creation."""
        config = AgentConfig.default_explorer()
        agent = SwarmAgent(config, latent_dim=16, hidden_dim=64)

        assert agent.config.role == AgentRole.EXPLORER
        assert agent.latent_dim == 16

    def test_encode(self):
        """Test agent encoding."""
        config = AgentConfig(role=AgentRole.EXPLORER, temperature=1.5)
        agent = SwarmAgent(config, latent_dim=16, hidden_dim=64)

        h = torch.randn(4, 64)
        mu, logvar = agent.encode(h)

        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_reparameterize(self):
        """Test reparameterization."""
        config = AgentConfig.default_exploiter()
        agent = SwarmAgent(config, latent_dim=16, hidden_dim=64)

        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)

        z = agent.reparameterize(mu, logvar)

        assert z.shape == (4, 16)

    def test_create_message(self):
        """Test message creation."""
        config = AgentConfig.default_integrator()
        agent = SwarmAgent(config, latent_dim=16, hidden_dim=64)

        z = torch.randn(4, 16)
        context = torch.randn(4, 16)

        message = agent.create_message(z, context)

        assert message.shape == (4, 16)


class TestSwarmVAE:
    """Tests for SwarmVAE multi-agent VAE."""

    def test_creation(self):
        """Test SwarmVAE creation."""
        model = SwarmVAE(input_dim=64, latent_dim=16, n_agents=4)

        assert model.input_dim == 64
        assert model.latent_dim == 16
        assert len(model.agents) == 4

    def test_creation_custom_configs(self):
        """Test creation with custom agent configs."""
        configs = [
            AgentConfig.default_explorer(),
            AgentConfig.default_exploiter(),
        ]

        model = SwarmVAE(
            input_dim=64,
            latent_dim=16,
            n_agents=2,
            agent_configs=configs,
        )

        assert len(model.agents) == 2
        assert model.agents[0].config.role == AgentRole.EXPLORER
        assert model.agents[1].config.role == AgentRole.EXPLOITER

    def test_encode(self):
        """Test encoding."""
        model = SwarmVAE(input_dim=64, latent_dim=16, n_agents=4)
        x = torch.randn(8, 64)

        encoding = model.encode(x)

        assert "agent_mus" in encoding
        assert "agent_zs" in encoding
        assert "z_consensus" in encoding
        assert len(encoding["agent_mus"]) == 4
        assert encoding["z_consensus"].shape == (8, 16)

    def test_decode(self):
        """Test decoding."""
        model = SwarmVAE(input_dim=64, latent_dim=16)
        z = torch.randn(8, 16)

        x_recon = model.decode(z)

        assert x_recon.shape == (8, 64)

    def test_forward(self):
        """Test forward pass."""
        model = SwarmVAE(input_dim=64, latent_dim=16, n_agents=4)
        x = torch.randn(8, 64)

        outputs = model.forward(x)

        assert "x_recon" in outputs
        assert "agent_recons" in outputs
        assert "z_consensus" in outputs
        assert outputs["x_recon"].shape == (8, 64)
        assert len(outputs["agent_recons"]) == 4

    def test_compute_loss(self):
        """Test loss computation."""
        model = SwarmVAE(input_dim=64, latent_dim=16)
        x = torch.randn(8, 64)

        outputs = model.forward(x)
        losses = model.compute_loss(x, outputs)

        assert "total_loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        assert "diversity_loss" in losses
        assert losses["total_loss"].requires_grad

    def test_swarm_step(self):
        """Test full swarm step."""
        model = SwarmVAE(input_dim=64, latent_dim=16)
        x = torch.randn(8, 64)

        result = model.swarm_step(x)

        assert "x_recon" in result
        assert "total_loss" in result
        assert "exploration_biases" in result
        assert len(result["exploration_biases"]) == 4

    def test_update_pheromones(self):
        """Test pheromone update."""
        model = SwarmVAE(input_dim=64, latent_dim=16)
        x = torch.randn(8, 64)

        # Run a step to populate pheromones
        model.swarm_step(x)

        # Check some pheromones exist
        total_pheromones = sum(
            f.positions.shape[0] for f in model.pheromone_fields.values()
        )
        assert total_pheromones > 0

    def test_get_exploration_bias(self):
        """Test exploration bias computation."""
        model = SwarmVAE(input_dim=64, latent_dim=16)

        # Initialize pheromone field with some data
        z = torch.randn(8, 16)
        bias = model.get_exploration_bias(z)

        assert bias.shape == (8,)

    def test_get_agent_statistics(self):
        """Test agent statistics retrieval."""
        model = SwarmVAE(input_dim=64, latent_dim=16, n_agents=4)

        # Run a step first
        x = torch.randn(4, 64)
        model.swarm_step(x)

        stats = model.get_agent_statistics()

        assert "agent_0_explorer" in stats
        assert "temperature" in stats["agent_0_explorer"]

    def test_diversity_loss(self):
        """Test diversity loss encourages spread."""
        model = SwarmVAE(input_dim=64, latent_dim=16)

        # Similar agents
        similar_zs = [torch.zeros(8, 16) for _ in range(4)]
        loss_similar = model._compute_diversity_loss(similar_zs)

        # Diverse agents
        diverse_zs = [torch.randn(8, 16) * i for i in range(4)]
        loss_diverse = model._compute_diversity_loss(diverse_zs)

        # Similar agents should have higher loss
        assert loss_similar > loss_diverse

    def test_gradient_flow(self):
        """Test gradients flow through main model components."""
        model = SwarmVAE(input_dim=64, latent_dim=16)
        x = torch.randn(8, 64)

        outputs = model.forward(x)
        losses = model.compute_loss(x, outputs)

        losses["total_loss"].backward()

        # Check gradients exist for key components
        # Encoder backbone should have gradients
        encoder_has_grad = any(
            p.grad is not None for p in model.encoder_backbone.parameters()
        )
        assert encoder_has_grad, "Encoder backbone should have gradients"

        # Decoder should have gradients
        decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())
        assert decoder_has_grad, "Decoder should have gradients"

        # Consensus layer should have gradients
        assert model.consensus_layer.weight.grad is not None


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert AgentRole.EXPLORER.value == "explorer"
        assert AgentRole.EXPLOITER.value == "exploiter"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.INTEGRATOR.value == "integrator"
