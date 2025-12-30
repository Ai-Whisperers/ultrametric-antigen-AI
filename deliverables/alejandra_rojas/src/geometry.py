import numpy as np

try:
    from geomstats.geometry.hyperbolic import Hyperbolic

    HAS_GEOMSTATS = True
except ImportError:
    HAS_GEOMSTATS = False


class HyperbolicSpace:
    """Wrapper for hyperbolic geometry operations in the Poincare Ball."""

    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        if HAS_GEOMSTATS:
            self.manifold = Hyperbolic(dim=dimension, default_coords_type="poincare")
        else:
            self.manifold = None

    def exponential_map(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        """Map a tangent vector to the manifold (Riemannian 'step')."""
        if not HAS_GEOMSTATS:
            # Fallback: Euclidean step (linear approximation) - ONLY for fallback
            return base_point + tangent_vec

        return self.manifold.metric.exp(tangent_vec, base_point=base_point)

    def parallel_transport(self, tangent_vec: np.ndarray, base_point: np.ndarray, end_point: np.ndarray):
        """Transport a vector along the geodesic from base to end."""
        if not HAS_GEOMSTATS:
            return tangent_vec

        return self.manifold.metric.parallel_transport(tangent_vec, base_point=base_point, end_point=end_point)
