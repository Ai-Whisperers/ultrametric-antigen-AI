# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Diffusion Models module.

Provides score-based generative models for biological sequences
and structures.

Key Components:
- CodonDiffusion: Discrete diffusion for codon sequences
- StructureConditionedGen: Structure-to-sequence generation
- NoiseScheduler: Various noise schedules (cosine, linear)

Example:
    from src.diffusion import CodonDiffusion

    model = CodonDiffusion(n_steps=1000)
    samples = model.sample(n_samples=10)
"""

__all__ = [
    "CodonDiffusion",
    "StructureConditionedGen",
    "NoiseScheduler",
]
