# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Topological Data Analysis (TDA) module.

Provides persistent homology and related topological methods for
biological structure analysis.

Key Components:
- PersistenceDiagram: Representation of topological features
- RipsFiltration: Vietoris-Rips complex construction
- PAdicFiltration: P-adic based filtration
- ProteinTopologyEncoder: Neural network encoder using TDA

Example:
    from src.topology import RipsFiltration, ProteinTopologyEncoder

    # Compute protein topology
    filt = RipsFiltration(max_dimension=2)
    fingerprint = filt.build(protein_coordinates)

    # Use in neural network
    encoder = ProteinTopologyEncoder(output_dim=128)
    embedding = encoder(coordinates_batch)
"""

# Lazy imports to avoid requiring optional dependencies
def __getattr__(name):
    if name in [
        "PersistenceDiagram",
        "TopologicalFingerprint",
        "RipsFiltration",
        "PAdicFiltration",
        "PersistenceVectorizer",
        "ProteinTopologyEncoder",
    ]:
        from src.topology.persistent_homology import (
            PersistenceDiagram,
            TopologicalFingerprint,
            RipsFiltration,
            PAdicFiltration,
            PersistenceVectorizer,
            ProteinTopologyEncoder,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PersistenceDiagram",
    "TopologicalFingerprint",
    "RipsFiltration",
    "PAdicFiltration",
    "PersistenceVectorizer",
    "ProteinTopologyEncoder",
]
