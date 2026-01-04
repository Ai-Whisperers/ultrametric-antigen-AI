# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic geometry utilities for arbovirus trajectory analysis.

This module provides hyperbolic geometry operations for embedding viral genomes
into the Poincare ball model. Uses src.geometry for proper geoopt-backed
implementations per V5.12.2 audit requirements.

V5.12.2 COMPLIANT: Uses poincare_distance instead of Euclidean .norm()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np

# Add project root for src imports
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Try to import proper hyperbolic geometry from src.geometry
try:
    import torch
    from src.geometry import (
        poincare_distance,
        exp_map_zero,
        project_to_poincare,
        get_manifold,
    )
    HAS_SRC_GEOMETRY = True
except ImportError:
    HAS_SRC_GEOMETRY = False
    torch = None

# Fallback to geomstats if src.geometry unavailable
try:
    from geomstats.geometry.hyperbolic import Hyperbolic
    HAS_GEOMSTATS = True
except ImportError:
    HAS_GEOMSTATS = False


def _to_tensor(x: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
    """Convert numpy array to torch tensor if needed."""
    if torch is None:
        raise ImportError("PyTorch required for hyperbolic geometry")
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def _to_numpy(x: "torch.Tensor") -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


class HyperbolicSpace:
    """Wrapper for hyperbolic geometry operations in the Poincare Ball.

    V5.12.2 COMPLIANT: Uses src.geometry.poincare_distance for proper
    hyperbolic distance computation instead of Euclidean .norm() fallback.

    Attributes:
        dimension: Dimension of the Poincare ball
        curvature: Curvature parameter c (default 1.0)
        backend: 'src.geometry', 'geomstats', or 'numpy' (fallback)
    """

    def __init__(self, dimension: int = 2, curvature: float = 1.0):
        """Initialize hyperbolic space.

        Args:
            dimension: Dimension of the Poincare ball
            curvature: Curvature parameter c (default 1.0)
        """
        self.dimension = dimension
        self.curvature = curvature

        # Determine backend
        if HAS_SRC_GEOMETRY:
            self.backend = "src.geometry"
            self.manifold = get_manifold(curvature)
        elif HAS_GEOMSTATS:
            self.backend = "geomstats"
            self.manifold = Hyperbolic(dim=dimension, default_coords_type="poincare")
        else:
            self.backend = "numpy"
            self.manifold = None

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute hyperbolic distance between points.

        V5.12.2: Uses poincare_distance, NOT Euclidean norm.

        Args:
            x: First point(s), shape (..., dim)
            y: Second point(s), shape (..., dim)

        Returns:
            Hyperbolic distances, shape (...)
        """
        if self.backend == "src.geometry":
            x_t = _to_tensor(x)
            y_t = _to_tensor(y)
            dist = poincare_distance(x_t, y_t, c=self.curvature)
            return _to_numpy(dist)

        elif self.backend == "geomstats":
            return self.manifold.metric.dist(x, y)

        else:
            # Numpy fallback: proper Poincare distance formula
            # d(x,y) = arcosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)

            diff_sq = np.sum((x - y) ** 2, axis=-1)
            x_sq = np.sum(x ** 2, axis=-1)
            y_sq = np.sum(y ** 2, axis=-1)

            # Clamp to avoid numerical issues
            x_sq = np.clip(x_sq, 0, 0.999)
            y_sq = np.clip(y_sq, 0, 0.999)

            denom = (1 - x_sq) * (1 - y_sq)
            arg = 1 + 2 * diff_sq / (denom + 1e-10)
            arg = np.clip(arg, 1.0, None)  # arcosh domain

            return np.arccosh(arg) / np.sqrt(self.curvature)

    def distance_to_origin(self, x: np.ndarray) -> np.ndarray:
        """Compute hyperbolic distance from origin (radial coordinate).

        V5.12.2: This is the correct way to compute 'radius' in hyperbolic space.

        Args:
            x: Points on Poincare ball, shape (..., dim)

        Returns:
            Hyperbolic distances from origin, shape (...)
        """
        origin = np.zeros_like(x)
        return self.distance(x, origin)

    def exponential_map(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        """Map a tangent vector to the manifold (Riemannian 'step').

        Args:
            base_point: Base point on manifold, shape (..., dim)
            tangent_vec: Tangent vector at base point, shape (..., dim)

        Returns:
            Point on manifold after exponential map
        """
        if self.backend == "src.geometry":
            base_t = _to_tensor(base_point)
            vec_t = _to_tensor(tangent_vec)
            # Use geoopt expmap
            result = self.manifold.expmap(base_t, vec_t)
            return _to_numpy(result)

        elif self.backend == "geomstats":
            return self.manifold.metric.exp(tangent_vec, base_point=base_point)

        else:
            # Numpy fallback: Mobius addition approximation
            # For small tangent vectors, exp_p(v) ≈ p ⊕ v
            return self._mobius_add(base_point, tangent_vec)

    def exponential_map_zero(self, tangent_vec: np.ndarray) -> np.ndarray:
        """Exponential map from tangent space at origin.

        Args:
            tangent_vec: Tangent vector at origin, shape (..., dim)

        Returns:
            Point on Poincare ball
        """
        if self.backend == "src.geometry":
            vec_t = _to_tensor(tangent_vec)
            result = exp_map_zero(vec_t, c=self.curvature)
            return _to_numpy(result)

        elif self.backend == "geomstats":
            origin = np.zeros_like(tangent_vec)
            return self.manifold.metric.exp(tangent_vec, base_point=origin)

        else:
            # exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
            c = self.curvature
            norm = np.linalg.norm(tangent_vec, axis=-1, keepdims=True)
            norm = np.maximum(norm, 1e-10)
            sqrt_c = np.sqrt(c)
            return np.tanh(sqrt_c * norm) * tangent_vec / (sqrt_c * norm)

    def parallel_transport(
        self,
        tangent_vec: np.ndarray,
        base_point: np.ndarray,
        end_point: np.ndarray
    ) -> np.ndarray:
        """Transport a vector along the geodesic from base to end.

        Args:
            tangent_vec: Vector to transport, shape (..., dim)
            base_point: Starting point, shape (..., dim)
            end_point: Ending point, shape (..., dim)

        Returns:
            Transported vector at end_point
        """
        if self.backend == "src.geometry":
            vec_t = _to_tensor(tangent_vec)
            base_t = _to_tensor(base_point)
            end_t = _to_tensor(end_point)
            result = self.manifold.transp(base_t, end_t, vec_t)
            return _to_numpy(result)

        elif self.backend == "geomstats":
            return self.manifold.metric.parallel_transport(
                tangent_vec, base_point=base_point, end_point=end_point
            )

        else:
            # Fallback: identity transport (approximation)
            return tangent_vec

    def project(self, x: np.ndarray, max_norm: float = 0.95) -> np.ndarray:
        """Project points onto the Poincare ball.

        Args:
            x: Points to project, shape (..., dim)
            max_norm: Maximum norm (< 1)

        Returns:
            Projected points on Poincare ball
        """
        if self.backend == "src.geometry":
            x_t = _to_tensor(x)
            result = project_to_poincare(x_t, max_norm=max_norm, c=self.curvature)
            return _to_numpy(result)

        else:
            # Numpy fallback: clamp norm
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            scale = np.where(norm > max_norm, max_norm / (norm + 1e-10), 1.0)
            return x * scale

    def _mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mobius addition in Poincare ball.

        x ⊕ y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
                (1 + 2c<x,y> + c^2||x||^2||y||^2)
        """
        c = self.curvature
        xy = np.sum(x * y, axis=-1, keepdims=True)
        x_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        y_sq = np.sum(y ** 2, axis=-1, keepdims=True)

        num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
        denom = 1 + 2*c*xy + c**2 * x_sq * y_sq

        return num / (denom + 1e-10)


# Convenience functions for trajectory analysis
def compute_hyperbolic_centroid(
    points: np.ndarray,
    curvature: float = 1.0,
    max_iter: int = 100
) -> np.ndarray:
    """Compute Frechet mean (centroid) in hyperbolic space.

    Uses iterative algorithm for proper hyperbolic averaging.

    Args:
        points: Points on Poincare ball, shape (n, dim)
        curvature: Curvature parameter
        max_iter: Maximum iterations

    Returns:
        Centroid point, shape (dim,)
    """
    if len(points) == 0:
        raise ValueError("Cannot compute centroid of empty set")
    if len(points) == 1:
        return points[0]

    space = HyperbolicSpace(dimension=points.shape[-1], curvature=curvature)

    # Initialize with Euclidean mean projected to ball
    centroid = space.project(np.mean(points, axis=0))

    # Iterative refinement (gradient descent on Frechet objective)
    lr = 0.1
    for _ in range(max_iter):
        # Compute gradient (sum of log maps)
        gradient = np.zeros_like(centroid)
        for p in points:
            # Log map approximation: direction from centroid to p
            diff = p - centroid
            gradient += diff
        gradient /= len(points)

        # Update via exponential map
        new_centroid = space.exponential_map(centroid, lr * gradient)
        new_centroid = space.project(new_centroid)

        # Check convergence
        if np.linalg.norm(new_centroid - centroid) < 1e-6:
            break
        centroid = new_centroid

    return centroid


def compute_hyperbolic_velocity(
    trajectory: list[np.ndarray],
    curvature: float = 1.0
) -> tuple[np.ndarray, float]:
    """Compute velocity vector from trajectory points.

    Args:
        trajectory: List of points on Poincare ball
        curvature: Curvature parameter

    Returns:
        Tuple of (direction_vector, magnitude)
    """
    if len(trajectory) < 2:
        return np.zeros(trajectory[0].shape), 0.0

    space = HyperbolicSpace(dimension=trajectory[0].shape[-1], curvature=curvature)

    # Compute displacement
    start = trajectory[0]
    end = trajectory[-1]

    # Direction in tangent space (log map approximation)
    direction = end - start

    # Magnitude is hyperbolic distance
    magnitude = float(space.distance(start, end))

    # Normalize direction
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    return direction, magnitude
