"""
Poincaré disk point representation.

Points in the Poincaré disk must have norm < 1.
The boundary (norm = 1) represents infinity.
"""
from dataclasses import dataclass
import math


# Numerical stability constant
EPS = 1e-7
MAX_NORM = 1.0 - EPS


@dataclass(slots=True)
class PoincarePoint:
    """
    A point in the Poincaré disk.

    The Poincaré disk is the open unit ball {x : ||x|| < 1}.

    Attributes:
        coords: Coordinates in the disk
        curvature: Negative curvature (default -1)
    """

    coords: tuple[float, ...]
    curvature: float = -1.0

    def __post_init__(self) -> None:
        """Validate point is inside disk."""
        if isinstance(self.coords, list):
            self.coords = tuple(self.coords)

        norm = self.norm
        if norm >= 1.0:
            # Project back into disk
            scale = MAX_NORM / (norm + EPS)
            self.coords = tuple(c * scale for c in self.coords)

    @classmethod
    def from_array(cls, arr, curvature: float = -1.0) -> "PoincarePoint":
        """Create from numpy array."""
        return cls(coords=tuple(float(x) for x in arr), curvature=curvature)

    @classmethod
    def origin(cls, dim: int = 2, curvature: float = -1.0) -> "PoincarePoint":
        """Create origin point."""
        return cls(coords=tuple(0.0 for _ in range(dim)), curvature=curvature)

    @property
    def dimension(self) -> int:
        """Get dimension."""
        return len(self.coords)

    @property
    def norm(self) -> float:
        """Calculate Euclidean norm."""
        return math.sqrt(sum(x * x for x in self.coords))

    @property
    def norm_squared(self) -> float:
        """Calculate squared norm (faster)."""
        return sum(x * x for x in self.coords)

    @property
    def is_valid(self) -> bool:
        """Check if point is inside disk."""
        return self.norm < 1.0

    @property
    def radial_distance(self) -> float:
        """
        Distance from origin in hyperbolic metric.

        d(0, x) = 2 * arctanh(||x||)
        """
        r = min(self.norm, MAX_NORM)
        return 2.0 * math.atanh(r)

    def to_array(self):
        """Convert to numpy array."""
        import numpy as np
        return np.array(self.coords)

    def to_list(self) -> list[float]:
        """Convert to list."""
        return list(self.coords)

    def scale(self, factor: float) -> "PoincarePoint":
        """Scale coordinates (Euclidean)."""
        new_coords = tuple(c * factor for c in self.coords)
        return PoincarePoint(coords=new_coords, curvature=self.curvature)

    def __getitem__(self, index: int) -> float:
        return self.coords[index]

    def __len__(self) -> int:
        return self.dimension

    def __iter__(self):
        return iter(self.coords)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoincarePoint):
            return False
        return all(
            abs(a - b) < EPS for a, b in zip(self.coords, other.coords)
        )

    def __repr__(self) -> str:
        coords_str = ", ".join(f"{c:.4f}" for c in self.coords)
        return f"PoincarePoint([{coords_str}])"
