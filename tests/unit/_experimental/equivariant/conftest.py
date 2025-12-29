# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for equivariant neural network tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def random_vectors(device):
    """Create random 3D vectors."""
    return torch.randn(10, 3, device=device)


@pytest.fixture
def random_positions(device):
    """Create random 3D positions for nodes."""
    return torch.randn(20, 3, device=device)


@pytest.fixture
def simple_edge_index(device):
    """Create simple edge index for testing."""
    # Create edges for a small graph
    src = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    dst = torch.tensor([1, 2, 0, 2, 0, 3, 1, 2], device=device)
    return torch.stack([src, dst])


@pytest.fixture
def graph_data(device):
    """Create a simple graph for testing."""
    n_nodes = 10
    n_edges = 30

    # Random node features
    x = torch.randn(n_nodes, 16, device=device)

    # Random positions
    pos = torch.randn(n_nodes, 3, device=device)

    # Random edges
    src = torch.randint(0, n_nodes, (n_edges,), device=device)
    dst = torch.randint(0, n_nodes, (n_edges,), device=device)
    edge_index = torch.stack([src, dst])

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "n_nodes": n_nodes,
    }


@pytest.fixture
def codon_sequence(device):
    """Create a sample codon sequence."""
    # Random codon indices (0-63)
    return torch.randint(0, 64, (4, 50), device=device)  # batch=4, seq_len=50


@pytest.fixture
def rotation_matrix(device):
    """Create a random 3D rotation matrix."""
    # Generate random Euler angles
    angles = torch.rand(3, device=device) * 2 * 3.14159

    # Rotation matrices for each axis
    c1, s1 = torch.cos(angles[0]), torch.sin(angles[0])
    c2, s2 = torch.cos(angles[1]), torch.sin(angles[1])
    c3, s3 = torch.cos(angles[2]), torch.sin(angles[2])

    Rz = torch.tensor([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]], device=device)
    Ry = torch.tensor([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]], device=device)
    Rx = torch.tensor([[1, 0, 0], [0, c3, -s3], [0, s3, c3]], device=device)

    return Rz @ Ry @ Rx
