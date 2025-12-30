import numpy as np


class GeometricScorer:
    """Computes geometric stability scores for rotamers."""

    def __init__(self, rare_threshold: float = 0.8):
        self.rare_threshold = rare_threshold

    def hyperbolic_distance(self, chi_angles: list[float]) -> float:
        """Compute distance to origin in Poincare ball."""
        # Map angles to [-1, 1] via tanh
        coords = np.array([np.tanh(c / np.pi) for c in chi_angles if c is not None])
        if len(coords) == 0:
            return 0.0

        r = np.linalg.norm(coords)
        if r >= 1.0:
            r = 0.999

        return 2 * np.arctanh(r)

    def padic_valuation(self, chi_angles: list[float], p: int = 3) -> int:
        """
        Compute p-adic valuation of the angle configuration.
        Replaces rigid binning with a more robust mapping if needed.
        """
        # Placeholder logic
        return 0

    def compute_stability(self, chi_angles: list[float]) -> float:
        """Composite stability score."""
        d_hyp = self.hyperbolic_distance(chi_angles)
        return 1.0 / (1.0 + d_hyp)
