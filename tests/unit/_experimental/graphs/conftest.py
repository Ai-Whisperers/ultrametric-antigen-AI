# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for graph neural network tests.

Provides fixtures for:
- Poincare operations
- Lorentz operations
- Graph data (nodes, edges)
- Hyperbolic points and batches
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

from src.graphs import (
    HyboWaveNet,
    HyperbolicGraphConv,
    HyperbolicLinear,
    LorentzMLP,
    LorentzOperations,
    PoincareOperations,
    SpectralWavelet,
)


# =============================================================================
# Device Override for Graph Tests
# Override the session-scoped device fixture to always use CPU for graph tests.
# This avoids geoopt device mismatch issues between manifold k tensors and inputs.
# =============================================================================


@pytest.fixture
def device() -> str:
    """Override device to CPU for all graph tests to avoid geoopt device issues."""
    return "cpu"


# =============================================================================
# Operation Fixtures
# =============================================================================


@pytest.fixture
def poincare_ops() -> PoincareOperations:
    """Returns a PoincareOperations instance with default curvature."""
    return PoincareOperations(curvature=1.0)


@pytest.fixture
def poincare_ops_c2() -> PoincareOperations:
    """Returns a PoincareOperations instance with curvature 2.0."""
    return PoincareOperations(curvature=2.0)


@pytest.fixture
def lorentz_ops() -> LorentzOperations:
    """Returns a LorentzOperations instance with default curvature."""
    return LorentzOperations(curvature=1.0)


@pytest.fixture
def lorentz_ops_c2() -> LorentzOperations:
    """Returns a LorentzOperations instance with curvature 2.0."""
    return LorentzOperations(curvature=2.0)


# =============================================================================
# Point Fixtures
# Note: Using cpu_device for all geoopt-related fixtures to avoid device mismatch
# =============================================================================


@pytest.fixture
def poincare_point(cpu_device) -> torch.Tensor:
    """Returns a single valid point in the Poincare ball."""
    return torch.tensor([[0.3, 0.4, 0.0, 0.0]], device=cpu_device)


@pytest.fixture
def poincare_points(cpu_device) -> torch.Tensor:
    """Returns a batch of valid points in the Poincare ball."""
    batch_size = 32
    dim = 16
    # Generate points with small norm (inside ball)
    points = torch.randn(batch_size, dim, device=cpu_device) * 0.3
    return points


@pytest.fixture
def poincare_points_small(cpu_device) -> torch.Tensor:
    """Returns a small batch of points for quick tests."""
    return torch.randn(4, 8, device=cpu_device) * 0.2


@pytest.fixture
def lorentz_point(lorentz_ops, cpu_device) -> torch.Tensor:
    """Returns a single valid point on the hyperboloid."""
    x = torch.randn(1, 9, device=cpu_device)  # 8D space + 1 time
    return lorentz_ops.project_to_hyperboloid(x)


@pytest.fixture
def lorentz_points(lorentz_ops, cpu_device) -> torch.Tensor:
    """Returns a batch of valid points on the hyperboloid."""
    x = torch.randn(32, 17, device=cpu_device)  # 16D space + 1 time
    return lorentz_ops.project_to_hyperboloid(x)


# =============================================================================
# Graph Data Fixtures
# Note: Using cpu_device for all geoopt-related fixtures to avoid device mismatch
# =============================================================================


@pytest.fixture
def small_graph(cpu_device) -> Dict[str, torch.Tensor]:
    """Returns a small graph with 10 nodes and simple connectivity."""
    n_nodes = 10
    n_features = 16

    # Node features (inside Poincare ball)
    x = torch.randn(n_nodes, n_features, device=cpu_device) * 0.2

    # Simple edge list (chain + some cross edges)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])  # Undirected
    # Add some cross edges
    edges.append([0, 5])
    edges.append([5, 0])
    edges.append([3, 8])
    edges.append([8, 3])

    edge_index = torch.tensor(edges, device=cpu_device).t().contiguous()

    return {"x": x, "edge_index": edge_index, "n_nodes": n_nodes}


@pytest.fixture
def medium_graph(cpu_device) -> Dict[str, torch.Tensor]:
    """Returns a medium graph with 50 nodes."""
    n_nodes = 50
    n_features = 32

    x = torch.randn(n_nodes, n_features, device=cpu_device) * 0.2

    # Random edges
    n_edges = 200
    src = torch.randint(0, n_nodes, (n_edges,), device=cpu_device)
    dst = torch.randint(0, n_nodes, (n_edges,), device=cpu_device)
    edge_index = torch.stack([src, dst], dim=0)

    return {"x": x, "edge_index": edge_index, "n_nodes": n_nodes}


@pytest.fixture
def batched_graphs(cpu_device) -> Dict[str, torch.Tensor]:
    """Returns multiple graphs batched together."""
    # Graph 1: 10 nodes
    x1 = torch.randn(10, 16, device=cpu_device) * 0.2
    edges1 = torch.randint(0, 10, (2, 30), device=cpu_device)

    # Graph 2: 15 nodes
    x2 = torch.randn(15, 16, device=cpu_device) * 0.2
    edges2 = torch.randint(0, 15, (2, 45), device=cpu_device) + 10  # Offset

    # Graph 3: 8 nodes
    x3 = torch.randn(8, 16, device=cpu_device) * 0.2
    edges3 = torch.randint(0, 8, (2, 24), device=cpu_device) + 25  # Offset

    # Combine
    x = torch.cat([x1, x2, x3], dim=0)
    edge_index = torch.cat([edges1, edges2, edges3], dim=1)
    batch = torch.cat([
        torch.zeros(10, device=cpu_device, dtype=torch.long),
        torch.ones(15, device=cpu_device, dtype=torch.long),
        torch.full((8,), 2, device=cpu_device, dtype=torch.long),
    ])

    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch,
        "n_graphs": 3,
    }


# =============================================================================
# Layer Fixtures
# =============================================================================


@pytest.fixture
def hyperbolic_linear() -> HyperbolicLinear:
    """Returns a HyperbolicLinear layer."""
    return HyperbolicLinear(in_features=16, out_features=8, curvature=1.0)


@pytest.fixture
def hyperbolic_graph_conv() -> HyperbolicGraphConv:
    """Returns a HyperbolicGraphConv layer."""
    return HyperbolicGraphConv(
        in_channels=16,
        out_channels=16,
        curvature=1.0,
        use_attention=False,
    )


@pytest.fixture
def hyperbolic_graph_conv_attention() -> HyperbolicGraphConv:
    """Returns a HyperbolicGraphConv layer with attention."""
    return HyperbolicGraphConv(
        in_channels=16,
        out_channels=16,
        curvature=1.0,
        use_attention=True,
        heads=4,
    )


@pytest.fixture
def lorentz_mlp() -> LorentzMLP:
    """Returns a LorentzMLP instance."""
    return LorentzMLP(
        in_features=16,
        hidden_features=32,
        out_features=8,
        n_layers=2,
    )


@pytest.fixture
def spectral_wavelet() -> SpectralWavelet:
    """Returns a SpectralWavelet instance."""
    return SpectralWavelet(n_scales=4)


@pytest.fixture
def hybowave_net() -> HyboWaveNet:
    """Returns a HyboWaveNet model."""
    return HyboWaveNet(
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        n_scales=3,
        n_layers=2,
        curvature=1.0,
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def graph_config() -> Dict:
    """Returns a standard graph model configuration."""
    return {
        "in_channels": 16,
        "hidden_channels": 32,
        "out_channels": 8,
        "n_scales": 4,
        "n_layers": 2,
        "curvature": 1.0,
        "dropout": 0.1,
        "use_attention": True,
    }


@pytest.fixture
def minimal_graph_config() -> Dict:
    """Returns a minimal configuration for fast tests."""
    return {
        "in_channels": 8,
        "hidden_channels": 16,
        "out_channels": 4,
        "n_scales": 2,
        "n_layers": 1,
        "curvature": 1.0,
        "dropout": 0.0,
        "use_attention": False,
    }
