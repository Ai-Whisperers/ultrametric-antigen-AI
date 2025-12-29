# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for topology tests.

Provides fixtures for:
- Point cloud data
- Persistence diagrams
- Topological fingerprints
- Filtrations
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch

from src.topology import (
    PAdicFiltration,
    PersistenceDiagram,
    PersistenceVectorizer,
    ProteinTopologyEncoder,
    RipsFiltration,
    TopologicalFingerprint,
)


# =============================================================================
# Point Cloud Fixtures
# =============================================================================


@pytest.fixture
def simple_point_cloud() -> np.ndarray:
    """Returns a simple 2D point cloud (square)."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)


@pytest.fixture
def circle_point_cloud() -> np.ndarray:
    """Returns points on a circle (has H1 feature)."""
    n_points = 20
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


@pytest.fixture
def random_point_cloud() -> np.ndarray:
    """Returns random 3D point cloud."""
    np.random.seed(42)
    return np.random.randn(50, 3).astype(np.float32)


@pytest.fixture
def protein_like_cloud() -> np.ndarray:
    """Returns a point cloud resembling protein backbone."""
    np.random.seed(42)
    # Generate spiral-like structure
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.cos(t) + np.random.randn(100) * 0.1
    y = np.sin(t) + np.random.randn(100) * 0.1
    z = t / (2 * np.pi) + np.random.randn(100) * 0.1
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.fixture
def point_cloud_tensor(simple_point_cloud) -> torch.Tensor:
    """Returns point cloud as tensor."""
    return torch.from_numpy(simple_point_cloud)


@pytest.fixture
def batch_coordinates(device) -> torch.Tensor:
    """Returns batch of coordinate tensors."""
    batch_size = 4
    n_points = 20
    return torch.randn(batch_size, n_points, 3, device=device)


# =============================================================================
# Persistence Diagram Fixtures
# =============================================================================


@pytest.fixture
def simple_diagram() -> PersistenceDiagram:
    """Returns a simple H0 persistence diagram."""
    return PersistenceDiagram(
        dimension=0,
        birth=np.array([0.0, 0.0, 0.0]),
        death=np.array([0.5, 0.8, 1.2]),
    )


@pytest.fixture
def h1_diagram() -> PersistenceDiagram:
    """Returns a simple H1 persistence diagram."""
    return PersistenceDiagram(
        dimension=1,
        birth=np.array([0.3, 0.5]),
        death=np.array([0.9, 1.5]),
    )


@pytest.fixture
def empty_diagram() -> PersistenceDiagram:
    """Returns an empty persistence diagram."""
    return PersistenceDiagram.empty(dimension=0)


@pytest.fixture
def multi_dim_fingerprint(simple_diagram, h1_diagram) -> TopologicalFingerprint:
    """Returns fingerprint with multiple dimensions."""
    return TopologicalFingerprint(
        diagrams={0: simple_diagram, 1: h1_diagram},
        metadata={"test": True},
    )


# =============================================================================
# Filtration Fixtures
# =============================================================================


@pytest.fixture
def rips_filtration() -> RipsFiltration:
    """Returns a RipsFiltration instance."""
    return RipsFiltration(max_dimension=1)


@pytest.fixture
def rips_filtration_h2() -> RipsFiltration:
    """Returns a RipsFiltration with H2."""
    return RipsFiltration(max_dimension=2)


@pytest.fixture
def padic_filtration() -> PAdicFiltration:
    """Returns a PAdicFiltration instance."""
    return PAdicFiltration(prime=3, max_dimension=1)


# =============================================================================
# Vectorizer Fixtures
# =============================================================================


@pytest.fixture
def statistics_vectorizer() -> PersistenceVectorizer:
    """Returns a statistics vectorizer."""
    return PersistenceVectorizer(method="statistics", dimensions=[0, 1])


@pytest.fixture
def landscape_vectorizer() -> PersistenceVectorizer:
    """Returns a landscape vectorizer."""
    return PersistenceVectorizer(method="landscape", resolution=20, dimensions=[0, 1])


@pytest.fixture
def image_vectorizer() -> PersistenceVectorizer:
    """Returns a persistence image vectorizer."""
    return PersistenceVectorizer(method="image", resolution=10, dimensions=[0, 1])


# =============================================================================
# Encoder Fixtures
# =============================================================================


@pytest.fixture
def topology_encoder() -> ProteinTopologyEncoder:
    """Returns a ProteinTopologyEncoder instance."""
    return ProteinTopologyEncoder(
        output_dim=64,
        hidden_dims=[128],
        max_dimension=1,
        vectorization="statistics",
    )


@pytest.fixture
def topology_encoder_padic() -> ProteinTopologyEncoder:
    """Returns a ProteinTopologyEncoder with p-adic."""
    return ProteinTopologyEncoder(
        output_dim=64,
        hidden_dims=[128],
        max_dimension=1,
        vectorization="statistics",
        use_padic=True,
        prime=3,
    )


# =============================================================================
# P-adic Indices Fixtures
# =============================================================================


@pytest.fixture
def padic_indices() -> np.ndarray:
    """Returns p-adic test indices."""
    return np.array([0, 1, 3, 9, 27, 81], dtype=np.int64)


@pytest.fixture
def padic_indices_tensor(padic_indices) -> torch.Tensor:
    """Returns p-adic indices as tensor."""
    return torch.from_numpy(padic_indices)
