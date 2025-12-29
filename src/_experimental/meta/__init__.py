# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Meta-Learning module.

Provides meta-learning algorithms for few-shot adaptation
to new diseases and biological domains.

Key Components:
- MAML: Model-Agnostic Meta-Learning
- Reptile: Simpler meta-learning algorithm
- PAdicTaskSampler: Sample tasks by p-adic similarity
- FewShotAdapter: Quick adaptation module

Example:
    from src.meta import MAML, PAdicTaskSampler

    sampler = PAdicTaskSampler(data_x, data_y, padic_indices)
    tasks = sampler.sample_batch(n_tasks=16)
    maml = MAML(base_model, inner_lr=0.01)
"""

from src._experimental.meta.meta_learning import (
    MAML,
    FewShotAdapter,
    PAdicTaskSampler,
    Reptile,
    Task,
)

__all__ = [
    "Task",
    "MAML",
    "PAdicTaskSampler",
    "FewShotAdapter",
    "Reptile",
]
