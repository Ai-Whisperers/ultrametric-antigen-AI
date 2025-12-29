# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Diffusion Models module.

Provides score-based generative models for biological sequences
and structures.

**Noise Schedules:**
- `NoiseScheduler`: Continuous diffusion noise schedules (linear, cosine, etc.)
- `DiscreteNoiseScheduler`: Discrete diffusion for token sequences

**Codon Diffusion Models:**
- `CodonDiffusion`: Discrete diffusion model for codon sequences
- `ConditionalCodonDiffusion`: Codon diffusion with context conditioning
- `TransformerDenoiser`: Transformer-based denoising network

**Structure-Conditioned Generation:**
- `StructureConditionedGen`: Generate codons conditioned on protein structure
- `StructureEncoder`: Encode protein backbone coordinates
- `MultiObjectiveDesigner`: Multi-objective sequence design

Example:
    >>> from src.diffusion import CodonDiffusion, StructureConditionedGen
    >>>
    >>> # Unconditional codon generation
    >>> model = CodonDiffusion(n_steps=1000, vocab_size=64)
    >>> samples = model.sample(n_samples=10, seq_length=100)
    >>>
    >>> # Structure-conditioned generation
    >>> gen = StructureConditionedGen(hidden_dim=256)
    >>> coords = torch.randn(1, 100, 3)  # Backbone Ca coords
    >>> sequences = gen.design(coords, n_designs=5)
"""

from .codon_diffusion import (
    CodonDiffusion,
    ConditionalCodonDiffusion,
    PositionalEncoding,
    TimestepEmbedding,
    TransformerDenoiser,
)
from .noise_schedule import (
    DiscreteNoiseScheduler,
    NoiseScheduler,
)
from .structure_gen import (
    MultiObjectiveDesigner,
    RadialBasisEmbedding,
    StructureConditionedGen,
    StructureEncoder,
    StructureGNNLayer,
)

__all__ = [
    # Noise schedules
    "NoiseScheduler",
    "DiscreteNoiseScheduler",
    # Codon diffusion
    "CodonDiffusion",
    "ConditionalCodonDiffusion",
    "TransformerDenoiser",
    "PositionalEncoding",
    "TimestepEmbedding",
    # Structure-conditioned generation
    "StructureConditionedGen",
    "StructureEncoder",
    "StructureGNNLayer",
    "RadialBasisEmbedding",
    "MultiObjectiveDesigner",
]
