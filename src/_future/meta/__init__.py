# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Meta-Learning module.

Provides meta-learning algorithms for few-shot adaptation
to new diseases and biological domains.

Key Components:
- MAML: Model-Agnostic Meta-Learning
- PAdicTaskSampler: Sample tasks by p-adic similarity
- FewShotAdapter: Quick adaptation module

Example:
    from src.meta import MAML, PAdicTaskSampler

    sampler = PAdicTaskSampler(n_tasks=100)
    maml = MAML(base_model, inner_lr=0.01)
    maml.train(sampler)
"""

__all__ = [
    "MAML",
    "PAdicTaskSampler",
    "FewShotAdapter",
]
