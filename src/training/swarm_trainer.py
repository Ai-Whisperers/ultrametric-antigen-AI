# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

import copy

import torch
import torch.nn as nn


class SwarmTrainer:
    """Manages distributed consensus training for a swarm of VAE agents.

    Implements a Federated Averaging (FedAvg) style algorithms where agents:
    1. Train locally on their own data subset.
    2. Periodically synchronize weights (consensus).
    3. Update with the aggregated 'swarm wisdom'.
    """

    def __init__(self, model_template: nn.Module, n_agents: int = 3):
        self.model_template = model_template
        self.n_agents = n_agents

        # Initialize swarm
        self.agents = []
        for _ in range(n_agents):
            # Deep copy to ensure independent instances
            self.agents.append(copy.deepcopy(model_template))

    def train_epoch_local(
        self,
        agent_idx: int,
        data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn,
    ):
        """Mock training step for a single agent."""
        model = self.agents[agent_idx]
        model.train()

        optimizer.zero_grad()
        # Mock forward pass logic - typically would call model forward
        # and loss computation.
        # For this class structure we mostly care about weight manipulation.
        pass

    def perform_consensus(self):
        """Aggregate weights from all agents (Federated Averaging)."""

        # 1. Collect state dicts
        state_dicts = [agent.state_dict() for agent in self.agents]

        # 2. Average weights
        avg_state_dict = copy.deepcopy(state_dicts[0])

        for key in avg_state_dict.keys():
            # Check if consistent type (skip non-tensor stats if any, though usually in buffer)
            if isinstance(avg_state_dict[key], torch.Tensor):
                stack = torch.stack([sd[key] for sd in state_dicts])
                # Check for float/integer
                if stack.is_floating_point():
                    avg_state_dict[key] = torch.mean(stack, dim=0)
                else:
                    # For int buffers (like num_batches_tracked), median or mode?
                    # Usually take mode or just the first one. Rounding mean for now.
                    avg_state_dict[key] = torch.round(torch.mean(stack.float(), dim=0)).long()

        # 3. Redistribute to swarm
        for agent in self.agents:
            agent.load_state_dict(avg_state_dict)

        return avg_state_dict

    def get_swarm_consensus_model(self):
        """Return a single model representing the swarm consensus."""
        consensus_model = copy.deepcopy(self.model_template)
        consensus_state = self.perform_consensus()
        consensus_model.load_state_dict(consensus_state)
        return consensus_model
