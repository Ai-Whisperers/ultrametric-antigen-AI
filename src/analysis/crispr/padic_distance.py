# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""p-Adic sequence distance for CRISPR analysis.

This module implements p-adic distance computation between DNA sequences,
capturing the hierarchical importance of different sequence positions
for CRISPR targeting specificity.

Single responsibility: p-Adic distance computation.
"""

import torch

# Position weights for mismatch tolerance (empirical from literature)
# Seed region (11-20 from PAM) is more critical than non-seed (1-10)
POSITION_WEIGHTS = torch.tensor([
    0.1, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # Non-seed (1-10)
    0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # Seed (11-20)
])


class PAdicSequenceDistance:
    """Computes p-adic distances between DNA sequences.

    Uses position-weighted p-adic norm to capture the
    hierarchical importance of different sequence positions.

    Attributes:
        p: Prime base for p-adic calculations
        seed_start: Position where seed region begins (from PAM)
    """

    def __init__(self, p: int = 3, seed_start: int = 12):
        """Initialize distance calculator.

        Args:
            p: Prime for p-adic calculations
            seed_start: Position where seed region begins (from PAM)
        """
        self.p = p
        self.seed_start = seed_start

    def mismatch_positions(
        self,
        seq1: str,
        seq2: str,
    ) -> list[int]:
        """Find mismatch positions between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            List of mismatch positions (0-indexed)
        """
        seq1 = seq1.upper()
        seq2 = seq2.upper()
        return [i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b]

    def padic_valuation(self, position: int) -> int:
        """Compute p-adic valuation for a position.

        Higher valuation = position is more divisible by p = less critical.
        Lower valuation = position is less divisible = more critical.

        Args:
            position: Sequence position (1-indexed)

        Returns:
            p-adic valuation
        """
        if position == 0:
            return 100  # Infinity for position 0

        v = 0
        n = position
        while n % self.p == 0:
            v += 1
            n //= self.p
        return v

    def compute_distance(
        self,
        target: str,
        offtarget: str,
        position_weighted: bool = True,
    ) -> float:
        """Compute p-adic distance between target and off-target.

        Args:
            target: Target sequence (guide RNA target)
            offtarget: Potential off-target sequence
            position_weighted: Whether to weight by position importance

        Returns:
            p-adic distance (0 = identical, higher = more different)
        """
        mismatches = self.mismatch_positions(target, offtarget)

        if not mismatches:
            return 0.0

        # Compute p-adic contribution from each mismatch
        total_distance = 0.0
        for pos in mismatches:
            # Base p-adic contribution
            v = self.padic_valuation(pos + 1)  # 1-indexed
            padic_contrib = self.p ** (-v)

            # Apply position weight if requested
            if position_weighted and pos < len(POSITION_WEIGHTS):
                padic_contrib *= POSITION_WEIGHTS[pos].item()

            total_distance += padic_contrib

        return total_distance


__all__ = ["PAdicSequenceDistance", "POSITION_WEIGHTS"]
