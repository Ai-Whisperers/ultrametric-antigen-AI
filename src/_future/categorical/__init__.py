# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Category Theory module.

Provides categorical abstractions for compositional neural network
design and formal verification.

Key Components:
- CategoricalLayer: Layers as morphisms
- Functor: Structure-preserving maps
- NaturalTransformation: Layer-to-layer mappings

Example:
    from src.categorical import CategoricalLayer, Functor

    layer1 = CategoricalLayer(input_type, hidden_type, transform1)
    layer2 = CategoricalLayer(hidden_type, output_type, transform2)
    composed = layer1.compose(layer2)
"""

__all__ = [
    "CategoricalLayer",
    "Functor",
    "NaturalTransformation",
]
