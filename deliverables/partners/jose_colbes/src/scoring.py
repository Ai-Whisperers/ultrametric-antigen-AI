# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Geometric scoring functions for protein stability analysis.

This module provides geometric stability scoring using hyperbolic geometry
and p-adic valuation from the core src.core.padic_math module.

Dependencies:
    - src.core.padic_math: P-adic valuation functions
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.padic_math import padic_valuation as core_padic_valuation


class GeometricScorer:
    """Computes geometric stability scores for rotamers.

    Uses hyperbolic distance in Poincare ball combined with
    3-adic valuation from src.core.padic_math.
    """

    def __init__(self, rare_threshold: float = 0.8, p: int = 3):
        """Initialize scorer.

        Args:
            rare_threshold: Threshold for flagging rare conformations
            p: Prime base for p-adic valuation (default 3 for ternary)
        """
        self.rare_threshold = rare_threshold
        self.p = p

    def hyperbolic_distance(self, chi_angles: list[float]) -> float:
        """Compute distance to origin in Poincare ball.

        Args:
            chi_angles: List of chi dihedral angles in radians

        Returns:
            Hyperbolic distance from origin
        """
        # Map angles to [-1, 1] via tanh
        valid_angles = [c for c in chi_angles if c is not None and not np.isnan(c)]
        if len(valid_angles) == 0:
            return 0.0

        coords = np.array([np.tanh(c / np.pi) for c in valid_angles])
        r = np.linalg.norm(coords)
        if r >= 1.0:
            r = 0.999

        return 2 * np.arctanh(r)

    def padic_valuation(self, chi_angles: list[float]) -> int:
        """Compute p-adic valuation of the angle configuration.

        Uses src.core.padic_math.padic_valuation for computation.

        Args:
            chi_angles: List of chi dihedral angles in radians

        Returns:
            P-adic valuation of discretized angle configuration
        """
        valid_angles = [c for c in chi_angles if c is not None and not np.isnan(c)]
        if len(valid_angles) == 0:
            return 0

        # Discretize angles into bins
        bins = 36
        indices = []
        for c in valid_angles[:4]:
            # Normalize to [0, 2pi] then to bin index
            normalized = (c + np.pi) % (2 * np.pi)
            idx = int(normalized / (2 * np.pi) * bins)
            indices.append(idx)

        # Combine indices into single integer
        combined = sum(idx * (bins ** i) for i, idx in enumerate(indices))

        # Use core p-adic valuation
        if combined <= 0:
            return 0
        return core_padic_valuation(combined, p=self.p)

    def compute_stability(self, chi_angles: list[float]) -> float:
        """Compute composite stability score.

        Combines hyperbolic distance with p-adic valuation.
        Higher stability = closer to common conformations.

        Args:
            chi_angles: List of chi dihedral angles in radians

        Returns:
            Stability score in [0, 1] range
        """
        d_hyp = self.hyperbolic_distance(chi_angles)
        v_p = self.padic_valuation(chi_angles)

        # Higher valuation = more structured = more stable
        # Higher hyperbolic distance = further from origin = less stable
        adjusted = d_hyp - v_p * 0.1

        return 1.0 / (1.0 + max(0, adjusted))

    def is_rare(self, chi_angles: list[float]) -> bool:
        """Check if conformation is rare (potentially unstable).

        Args:
            chi_angles: List of chi dihedral angles in radians

        Returns:
            True if stability score is below threshold
        """
        return self.compute_stability(chi_angles) < self.rare_threshold
