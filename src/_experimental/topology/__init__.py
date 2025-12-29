# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Topological Data Analysis (TDA) module.

This module provides persistent homology and related topological methods for
biological structure analysis. It implements multi-scale structural feature
extraction via filtrations and persistence diagrams.

Key Components:
    PersistenceDiagram: Representation of birth/death pairs
    TopologicalFingerprint: Multi-dimensional topological summary
    RipsFiltration: Vietoris-Rips complex construction
    PAdicFiltration: P-adic hierarchical filtration
    PersistenceVectorizer: Convert diagrams to ML-ready vectors
    ProteinTopologyEncoder: Neural network using TDA features

Supported Backends:
    - ripser: Fast Vietoris-Rips computation (recommended)
    - gudhi: Full-featured TDA library
    - numpy: Basic fallback (H0 only)

Example:
    >>> from src.topology import RipsFiltration, PersistenceVectorizer
    >>>
    >>> # Compute topology of point cloud
    >>> filt = RipsFiltration(max_dimension=2)
    >>> fingerprint = filt.build(coordinates)
    >>>
    >>> # Vectorize for ML
    >>> vectorizer = PersistenceVectorizer(method="statistics")
    >>> features = vectorizer.transform(fingerprint)
    >>>
    >>> # Neural network encoder
    >>> from src.topology import ProteinTopologyEncoder
    >>> encoder = ProteinTopologyEncoder(output_dim=128)
    >>> embeddings = encoder(coordinates_batch)

References:
    - Edelsbrunner & Harer (2010): Computational Topology
    - Carlsson (2009): Topology and Data
    - Adams et al. (2017): Persistence Images
"""

from .persistent_homology import (
    PAdicFiltration,
    PersistenceDiagram,
    PersistenceVectorizer,
    ProteinTopologyEncoder,
    RipsFiltration,
    TopologicalFingerprint,
)

__all__ = [
    # Data Structures
    "PersistenceDiagram",
    "TopologicalFingerprint",
    # Filtrations
    "RipsFiltration",
    "PAdicFiltration",
    # Vectorization
    "PersistenceVectorizer",
    # Neural Networks
    "ProteinTopologyEncoder",
]

__version__ = "1.0.0"
