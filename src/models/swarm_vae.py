# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-Agent Swarm VAE Architecture.

This module implements a swarm-based VAE architecture inspired by collective
behavior systems (e.g., spider colonies, ant colonies). The key insight is
that local rules between agents can create emergent global optimization.

Key Concepts:
    - Multiple VAE agents with specialized roles (explorer, exploiter, validator, integrator)
    - Local communication rules between agents (stigmergy-like)
    - Emergent coverage of the ternary operation space
    - Pheromone-like influence fields in latent space

Research Reference:
    RESEARCH_PROPOSALS/06_SWARM_VAE_ARCHITECTURE.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentRole(Enum):
    """Roles for swarm agents."""

    EXPLORER = "explorer"  # High temperature, discovers new regions
    EXPLOITER = "exploiter"  # Low temperature, refines known regions
    VALIDATOR = "validator"  # Checks consistency between regions
    INTEGRATOR = "integrator"  # Combines information from all agents


@dataclass
class AgentConfig:
    """Configuration for a swarm agent."""

    role: AgentRole
    temperature: float = 1.0
    exploration_rate: float = 0.5
    communication_radius: float = 0.3
    influence_strength: float = 0.1

    @classmethod
    def default_explorer(cls) -> "AgentConfig":
        """Create default explorer configuration."""
        return cls(
            role=AgentRole.EXPLORER,
            temperature=1.5,
            exploration_rate=0.8,
            communication_radius=0.5,
            influence_strength=0.15,
        )

    @classmethod
    def default_exploiter(cls) -> "AgentConfig":
        """Create default exploiter configuration."""
        return cls(
            role=AgentRole.EXPLOITER,
            temperature=0.5,
            exploration_rate=0.2,
            communication_radius=0.2,
            influence_strength=0.3,
        )

    @classmethod
    def default_validator(cls) -> "AgentConfig":
        """Create default validator configuration."""
        return cls(
            role=AgentRole.VALIDATOR,
            temperature=1.0,
            exploration_rate=0.5,
            communication_radius=0.4,
            influence_strength=0.1,
        )

    @classmethod
    def default_integrator(cls) -> "AgentConfig":
        """Create default integrator configuration."""
        return cls(
            role=AgentRole.INTEGRATOR,
            temperature=0.8,
            exploration_rate=0.3,
            communication_radius=0.6,
            influence_strength=0.2,
        )


@dataclass
class PheromoneField:
    """Pheromone-like influence field in latent space.

    Represents accumulated "knowledge" about explored regions.
    """

    positions: torch.Tensor  # (N, latent_dim) - explored positions
    strengths: torch.Tensor  # (N,) - pheromone strength at each position
    decay_rate: float = 0.95  # Per-step decay
    max_positions: int = 1000  # Maximum stored positions

    def deposit(self, position: torch.Tensor, strength: float = 1.0) -> None:
        """Deposit pheromone at a position.

        Args:
            position: Position in latent space (latent_dim,) or (B, latent_dim)
            strength: Strength of pheromone deposit
        """
        if position.dim() == 1:
            position = position.unsqueeze(0)

        # Add new positions
        self.positions = torch.cat([self.positions, position], dim=0)
        new_strengths = torch.full((position.shape[0],), strength, device=position.device)
        self.strengths = torch.cat([self.strengths, new_strengths])

        # Prune if too many
        if self.positions.shape[0] > self.max_positions:
            # Keep strongest positions
            _, indices = torch.topk(self.strengths, self.max_positions)
            self.positions = self.positions[indices]
            self.strengths = self.strengths[indices]

    def decay(self) -> None:
        """Apply decay to all pheromones."""
        self.strengths = self.strengths * self.decay_rate

        # Remove very weak pheromones
        mask = self.strengths > 0.01
        self.positions = self.positions[mask]
        self.strengths = self.strengths[mask]

    def query(self, position: torch.Tensor, radius: float = 0.3) -> torch.Tensor:
        """Query pheromone influence at a position.

        Args:
            position: Query position (latent_dim,) or (B, latent_dim)
            radius: Radius of influence

        Returns:
            Total pheromone influence at position
        """
        if self.positions.shape[0] == 0:
            return torch.tensor(0.0, device=position.device)

        if position.dim() == 1:
            position = position.unsqueeze(0)

        # Compute distances
        dists = torch.cdist(position, self.positions)  # (B, N)

        # Gaussian kernel influence
        influence = torch.exp(-(dists**2) / (2 * radius**2))  # (B, N)
        weighted = influence * self.strengths.unsqueeze(0)  # (B, N)

        return weighted.sum(dim=1)  # (B,)


class SwarmAgent(nn.Module):
    """Individual agent in the swarm VAE.

    Each agent has its own encoder/decoder heads but shares a common backbone.
    """

    def __init__(
        self,
        config: AgentConfig,
        latent_dim: int = 16,
        hidden_dim: int = 64,
    ):
        """Initialize swarm agent.

        Args:
            config: Agent configuration
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for agent-specific layers
        """
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim

        # Agent-specific projection layers
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Role-specific attention for combining with other agents
        self.role_query = nn.Linear(latent_dim, latent_dim)
        self.role_key = nn.Linear(latent_dim, latent_dim)

        # Communication layer
        self.message_encoder = nn.Linear(latent_dim * 2, latent_dim)

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode shared hidden state to agent-specific latent distribution.

        Args:
            h: Hidden state from shared backbone (B, hidden_dim)

        Returns:
            mu, logvar for this agent's latent distribution
        """
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Apply temperature scaling
        logvar = logvar + torch.log(torch.tensor(self.config.temperature))

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with temperature.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * self.config.temperature

    def create_message(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Create message to send to other agents.

        Args:
            z: This agent's latent representation
            context: Contextual information (e.g., reconstruction error)

        Returns:
            Message tensor to broadcast
        """
        combined = torch.cat([z, context], dim=-1)
        return self.message_encoder(combined)


class SwarmVAE(nn.Module):
    """Multi-Agent Swarm VAE Architecture.

    Implements collective behavior-inspired VAE with multiple specialized agents
    that communicate and coordinate to explore the latent space.
    """

    def __init__(
        self,
        input_dim: int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        n_agents: int = 4,
        agent_configs: Optional[List[AgentConfig]] = None,
        pheromone_decay: float = 0.95,
    ):
        """Initialize SwarmVAE.

        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            n_agents: Number of agents in swarm
            agent_configs: Optional custom agent configurations
            pheromone_decay: Decay rate for pheromone field
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # Default agent configurations
        if agent_configs is None:
            agent_configs = [
                AgentConfig.default_explorer(),
                AgentConfig.default_exploiter(),
                AgentConfig.default_validator(),
                AgentConfig.default_integrator(),
            ]
            # Extend if more agents needed
            while len(agent_configs) < n_agents:
                agent_configs.append(
                    AgentConfig(
                        role=AgentRole.EXPLORER,
                        temperature=1.0 + 0.2 * len(agent_configs),
                    )
                )

        # Shared encoder backbone
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Individual agents
        self.agents = nn.ModuleList(
            [SwarmAgent(config, latent_dim, hidden_dim) for config in agent_configs[:n_agents]]
        )

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Agent communication attention
        self.agent_attention = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)

        # Consensus layer for combining agent outputs
        self.consensus_layer = nn.Linear(latent_dim * n_agents, latent_dim)

        # Pheromone fields (one per role type)
        self._init_pheromone_fields(latent_dim, pheromone_decay)

    def _init_pheromone_fields(self, latent_dim: int, decay_rate: float) -> None:
        """Initialize pheromone fields for each role."""
        self.pheromone_fields: Dict[AgentRole, PheromoneField] = {}
        for role in AgentRole:
            self.pheromone_fields[role] = PheromoneField(
                positions=torch.zeros(0, latent_dim),
                strengths=torch.zeros(0),
                decay_rate=decay_rate,
            )

    def encode(self, x: torch.Tensor) -> Dict[str, Any]:
        """Encode input through all agents.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Dictionary with agent-specific encodings and consensus
        """
        # Shared backbone
        h = self.encoder_backbone(x)

        # Each agent encodes
        agent_mus = []
        agent_logvars = []
        agent_zs = []

        for agent in self.agents:
            mu, logvar = agent.encode(h)
            z = agent.reparameterize(mu, logvar)
            agent_mus.append(mu)
            agent_logvars.append(logvar)
            agent_zs.append(z)

        # Stack for attention
        z_stack = torch.stack(agent_zs, dim=1)  # (B, n_agents, latent_dim)

        # Agent communication via attention
        z_attended, attention_weights = self.agent_attention(z_stack, z_stack, z_stack)

        # Consensus latent
        z_flat = z_attended.reshape(x.shape[0], -1)  # (B, n_agents * latent_dim)
        z_consensus = self.consensus_layer(z_flat)  # (B, latent_dim)

        return {
            "agent_mus": agent_mus,
            "agent_logvars": agent_logvars,
            "agent_zs": agent_zs,
            "z_attended": z_attended,
            "z_consensus": z_consensus,
            "attention_weights": attention_weights,
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: Latent tensor (B, latent_dim)

        Returns:
            Reconstruction (B, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through SwarmVAE.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Dictionary with all outputs and metrics
        """
        # Encode
        encoding = self.encode(x)

        # Decode consensus
        x_recon = self.decode(encoding["z_consensus"])

        # Also decode each agent for diversity loss
        agent_recons = [self.decode(z) for z in encoding["agent_zs"]]

        return {
            "x_recon": x_recon,
            "agent_recons": agent_recons,
            **encoding,
        }

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute swarm VAE loss components.

        Args:
            x: Input tensor
            outputs: Forward pass outputs

        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (consensus)
        recon_loss = F.mse_loss(outputs["x_recon"], x, reduction="mean")

        # KL divergence (average over agents)
        kl_loss = torch.tensor(0.0, device=x.device)
        for mu, logvar in zip(outputs["agent_mus"], outputs["agent_logvars"]):
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss + kl.mean()
        kl_loss = kl_loss / len(outputs["agent_mus"])

        # Diversity loss: encourage agents to explore different regions
        diversity_loss = self._compute_diversity_loss(outputs["agent_zs"])

        # Consistency loss: ensure consensus is consistent with agents
        consistency_loss = self._compute_consistency_loss(
            outputs["z_consensus"], outputs["agent_zs"]
        )

        # Coverage loss: penalize unexplored regions
        coverage_loss = self._compute_coverage_loss(outputs["agent_zs"])

        total_loss = recon_loss + 0.1 * kl_loss + 0.05 * diversity_loss + 0.02 * consistency_loss + 0.01 * coverage_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "diversity_loss": diversity_loss,
            "consistency_loss": consistency_loss,
            "coverage_loss": coverage_loss,
        }

    def _compute_diversity_loss(self, agent_zs: List[torch.Tensor]) -> torch.Tensor:
        """Encourage agents to explore different regions.

        Args:
            agent_zs: List of agent latent vectors

        Returns:
            Diversity loss (higher when agents are too similar)
        """
        n = len(agent_zs)
        if n < 2:
            return torch.tensor(0.0, device=agent_zs[0].device)

        # Compute pairwise distances between agents
        total_dist = torch.tensor(0.0, device=agent_zs[0].device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = F.pairwise_distance(agent_zs[i], agent_zs[j])
                total_dist = total_dist + dist.mean()
                count += 1

        # We want to maximize distance, so return negative (as loss to minimize)
        avg_dist = total_dist / count
        return torch.exp(-avg_dist)  # Close agents = high loss

    def _compute_consistency_loss(
        self, z_consensus: torch.Tensor, agent_zs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Ensure consensus is consistent with agent opinions.

        Args:
            z_consensus: Consensus latent vector
            agent_zs: List of agent latent vectors

        Returns:
            Consistency loss
        """
        # Consensus should be close to weighted mean of agents
        agent_mean = torch.stack(agent_zs, dim=0).mean(dim=0)
        return F.mse_loss(z_consensus, agent_mean)

    def _compute_coverage_loss(self, agent_zs: List[torch.Tensor]) -> torch.Tensor:
        """Penalize poor coverage of latent space.

        Args:
            agent_zs: List of agent latent vectors

        Returns:
            Coverage loss
        """
        # Stack all agent positions
        all_z = torch.cat(agent_zs, dim=0)  # (B * n_agents, latent_dim)

        # Variance across latent dimensions (want high variance = good coverage)
        variance = all_z.var(dim=0).mean()

        # Return negative variance as loss (maximize variance)
        return 1.0 / (variance + 1e-6)

    def update_pheromones(
        self,
        agent_zs: List[torch.Tensor],
        recon_errors: List[torch.Tensor],
    ) -> None:
        """Update pheromone fields based on exploration results.

        Args:
            agent_zs: List of agent latent positions
            recon_errors: List of reconstruction errors per agent
        """
        for i, (agent, z, error) in enumerate(
            zip(self.agents, agent_zs, recon_errors)
        ):
            swarm_agent = cast(SwarmAgent, agent)
            role = swarm_agent.config.role
            field = self.pheromone_fields[role]

            # Deposit strength inversely proportional to error
            strength = 1.0 / (error.mean().item() + 1e-6)
            strength = min(strength, 10.0)  # Cap strength

            # Move field to correct device
            if field.positions.device != z.device:
                field.positions = field.positions.to(z.device)
                field.strengths = field.strengths.to(z.device)

            field.deposit(z.detach(), strength)

        # Decay all fields
        for field in self.pheromone_fields.values():
            field.decay()

    def get_exploration_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Get exploration bias based on pheromone landscape.

        Encourages exploration of low-pheromone regions.

        Args:
            z: Current latent position

        Returns:
            Bias vector pointing toward unexplored regions
        """
        total_influence = torch.zeros(z.shape[0], device=z.device)

        for field in self.pheromone_fields.values():
            if field.positions.device != z.device:
                field.positions = field.positions.to(z.device)
                field.strengths = field.strengths.to(z.device)
            influence = field.query(z, radius=0.3)
            total_influence = total_influence + influence

        # Normalize and invert (high influence = should move away)
        max_influence = total_influence.max() + 1e-6
        exploration_signal = 1.0 - total_influence / max_influence

        return exploration_signal

    def swarm_step(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """Perform one swarm optimization step.

        This includes forward pass, loss computation, and pheromone update.

        Args:
            x: Input batch

        Returns:
            Dictionary with outputs and losses
        """
        # Forward pass
        outputs = self.forward(x)

        # Compute losses
        losses = self.compute_loss(x, outputs)

        # Compute per-agent reconstruction errors
        agent_errors = [
            F.mse_loss(recon, x, reduction="none").mean(dim=1)
            for recon in outputs["agent_recons"]
        ]

        # Update pheromones
        self.update_pheromones(outputs["agent_zs"], agent_errors)

        # Get exploration biases
        exploration_biases = [
            self.get_exploration_bias(z) for z in outputs["agent_zs"]
        ]

        return {
            **outputs,
            **losses,
            "agent_errors": agent_errors,
            "exploration_biases": exploration_biases,
        }

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent behavior.

        Returns:
            Dictionary with agent statistics
        """
        stats: Dict[str, Any] = {}

        for i, agent in enumerate(self.agents):
            swarm_agent = cast(SwarmAgent, agent)
            role = swarm_agent.config.role.value
            field = self.pheromone_fields[swarm_agent.config.role]

            stats[f"agent_{i}_{role}"] = {
                "temperature": swarm_agent.config.temperature,
                "exploration_rate": swarm_agent.config.exploration_rate,
                "pheromone_count": field.positions.shape[0],
                "avg_pheromone_strength": (
                    field.strengths.mean().item() if field.strengths.numel() > 0 else 0.0
                ),
            }

        return stats
