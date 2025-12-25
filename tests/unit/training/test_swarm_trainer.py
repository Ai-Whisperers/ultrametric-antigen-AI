import torch
import torch.nn as nn

from src.training.swarm_trainer import SwarmTrainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)


def test_swarm_initialization():
    template = SimpleModel()
    swarm = SwarmTrainer(template, n_agents=3)
    assert len(swarm.agents) == 3
    # Check they are different objects
    assert swarm.agents[0] is not swarm.agents[1]


def test_consensus_averaging():
    template = SimpleModel()
    # Fix weights for deterministic test
    # Agent 0: all ones
    # Agent 1: all threes
    # Agent 2: all fives
    # Average should be 3

    swarm = SwarmTrainer(template, n_agents=3)

    with torch.no_grad():
        swarm.agents[0].fc.weight.fill_(1.0)
        swarm.agents[0].fc.bias.fill_(1.0)

        swarm.agents[1].fc.weight.fill_(3.0)
        swarm.agents[1].fc.bias.fill_(3.0)

        swarm.agents[2].fc.weight.fill_(5.0)
        swarm.agents[2].fc.bias.fill_(5.0)

    avg_state = swarm.perform_consensus()

    expected_weight = 3.0
    assert torch.allclose(avg_state["fc.weight"], torch.tensor(expected_weight))

    # Check that agents are updated
    for agent in swarm.agents:
        assert torch.allclose(agent.fc.weight, torch.tensor(expected_weight))
