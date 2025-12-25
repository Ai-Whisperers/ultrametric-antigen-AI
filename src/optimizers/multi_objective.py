# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

from typing import Tuple

import torch


class ParetoFrontOptimizer:
    """Multi-objective optimizer using Pareto dominance.

    Used to select optimal latent candidates that balance multiple conflicting objectives:
    e.g., Minimize Reconstruction Loss vs Minimize Autoimmunity Risk.
    """

    def __init__(self):
        pass

    def is_dominated(self, candidate_scores: torch.Tensor, population_scores: torch.Tensor) -> bool:
        """Check if a candidate solution is dominated by any in the population.

        Args:
            candidate_scores: (N_Objectives,) tensor where LOWER is BETTER.
            population_scores: (Pop_Size, N_Objectives) tensor.

        Returns:
            True if dominated, False otherwise.
        """
        # A dominates B if A <= B for all objs AND A < B for at least one.
        # We assume minimization for all objectives.

        # Iterate efficiently?
        # Check if ANY individual in population dominates candidate

        # expand candidate to broadcast
        c = candidate_scores.unsqueeze(0)  # (1, N_obj)

        # domination_check: (Pop, N_obj) bools
        # better_or_equal: all(pop <= cand)
        better_or_equal = (population_scores <= c).all(dim=1)

        # strictly_better: any(pop < cand)
        strictly_better = (population_scores < c).any(dim=1)

        # dominated if there exists an individual that is better_or_equal AND strictly_better
        is_dominated = (better_or_equal & strictly_better).any()

        return bool(is_dominated.item())

    def identify_pareto_front(self, candidates: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify the non-dominated set (Pareto Front) from a batch of candidates.

        Args:
            candidates: (Batch, Dim) - The actual latent vectors or solutions.
            scores: (Batch, N_Objectives) - The objective scores (LOWER is BETTER).

        Returns:
            (front_candidates, front_scores) containing only the non-dominated solutions.
        """
        batch_size = candidates.shape[0]
        is_dominated_mask = torch.zeros(batch_size, dtype=torch.bool, device=candidates.device)

        # Naive O(N^2) pairwise comparison
        # For large batches, NSGA-II fast non-dominated sorting is better, but this suffices for
        # typical inference batches (e.g. 512-2048).

        for i in range(batch_size):
            # Compare i against all others
            # We can vectorize the specific check for 'i' vs 'all others'

            c_scores = scores[i].unsqueeze(0)  # (1, Obj)

            # Others: excluding i? Or just include i (it won't dominate itself strictly)
            # Let's check against all

            better_or_equal = (scores <= c_scores).all(dim=1)
            strictly_better = (scores < c_scores).any(dim=1)
            dominators = better_or_equal & strictly_better

            if dominators.any():
                is_dominated_mask[i] = True

        # Filter
        non_dominated_indices = ~is_dominated_mask

        return candidates[non_dominated_indices], scores[non_dominated_indices]
