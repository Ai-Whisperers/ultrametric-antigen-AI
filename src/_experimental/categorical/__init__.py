# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Category Theory module.

Provides categorical abstractions for compositional neural network
design and formal verification.

Key Components:
- CategoricalLayer: Layers as morphisms with type safety
- Functor: Structure-preserving maps between architectures
- NaturalTransformation: Layer-to-layer mappings
- ParametricLens: Categorical formulation of backprop
- Optic: Bidirectional data flow (residuals, attention)

Example:
    from src.categorical import CategoricalLayer, TensorType

    input_type = TensorType((128,))
    hidden_type = TensorType((64,))
    layer1 = CategoricalLayer(input_type, hidden_type, name="encoder")
    layer2 = CategoricalLayer(hidden_type, input_type, name="decoder")
    autoencoder = layer1 >> layer2
"""

from src._experimental.categorical.category_theory import (
    AttentionOptic,
    CategoricalLayer,
    CategoricalNetwork,
    Functor,
    LinearLens,
    MonoidalCategory,
    Morphism,
    NaturalTransformation,
    Optic,
    ParametricLens,
    ProductCategory,
    ResidualOptic,
    StringDiagram,
    TensorType,
)

__all__ = [
    "TensorType",
    "Morphism",
    "CategoricalLayer",
    "Functor",
    "NaturalTransformation",
    "ParametricLens",
    "LinearLens",
    "ProductCategory",
    "MonoidalCategory",
    "StringDiagram",
    "Optic",
    "ResidualOptic",
    "AttentionOptic",
    "CategoricalNetwork",
]
