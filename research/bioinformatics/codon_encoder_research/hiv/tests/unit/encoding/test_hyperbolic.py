"""
Unit tests for hyperbolic geometry.
"""
import pytest
import math
from src.encoding.hyperbolic.poincare.point import PoincarePoint
from src.encoding.hyperbolic.poincare.operations import (
    mobius_add, exp_map, log_map, mobius_scalar_mult
)
from src.encoding.hyperbolic.poincare.distance import (
    poincare_distance, distance_from_origin
)
from src.encoding.hyperbolic.poincare.geodesic import (
    geodesic_midpoint, geodesic_path, frechet_mean
)


class TestPoincarePoint:
    """Tests for PoincarePoint class."""

    def test_point_creation(self):
        """Test point creation."""
        p = PoincarePoint(coords=(0.3, 0.4))
        assert p.dimension == 2
        assert p.norm == pytest.approx(0.5)

    def test_origin(self):
        """Test origin creation."""
        origin = PoincarePoint.origin(dim=3)
        assert origin.norm == 0.0
        assert origin.dimension == 3

    def test_validity_inside(self):
        """Test valid point detection."""
        p = PoincarePoint(coords=(0.3, 0.4))
        assert p.is_valid

    def test_projection_outside(self):
        """Test projection when outside disk."""
        p = PoincarePoint(coords=(0.8, 0.8))
        # Should be projected inside
        assert p.is_valid
        assert p.norm < 1.0

    def test_radial_distance(self):
        """Test radial distance calculation."""
        p = PoincarePoint(coords=(0.5, 0.0))
        expected = 2 * math.atanh(0.5)
        assert p.radial_distance == pytest.approx(expected)


class TestMobiusOperations:
    """Tests for Möbius operations."""

    def test_mobius_add_origin(self):
        """Adding origin is identity."""
        p = PoincarePoint(coords=(0.3, 0.4))
        origin = PoincarePoint.origin(2)
        result = mobius_add(origin, p)
        assert result.coords[0] == pytest.approx(p.coords[0])
        assert result.coords[1] == pytest.approx(p.coords[1])

    def test_mobius_add_inverse(self):
        """Point + (-point) = origin."""
        p = PoincarePoint(coords=(0.3, 0.4))
        neg_p = PoincarePoint(coords=(-0.3, -0.4))
        result = mobius_add(p, neg_p)
        assert result.norm < 0.01  # Should be near origin

    def test_exp_log_inverse(self):
        """Log is inverse of exp."""
        v = (0.2, 0.3)
        p = exp_map(v)
        v_back = log_map(p)
        assert v_back[0] == pytest.approx(v[0], rel=0.01)
        assert v_back[1] == pytest.approx(v[1], rel=0.01)

    def test_scalar_mult_identity(self):
        """1 ⊗ x = x."""
        p = PoincarePoint(coords=(0.3, 0.4))
        result = mobius_scalar_mult(1.0, p)
        assert result.coords[0] == pytest.approx(p.coords[0])
        assert result.coords[1] == pytest.approx(p.coords[1])


class TestPoincareDistance:
    """Tests for Poincaré distance."""

    def test_distance_to_self(self):
        """Distance to self is 0."""
        p = PoincarePoint(coords=(0.3, 0.4))
        assert poincare_distance(p, p) == pytest.approx(0.0)

    def test_distance_symmetry(self):
        """Distance is symmetric."""
        a = PoincarePoint(coords=(0.2, 0.3))
        b = PoincarePoint(coords=(0.5, 0.1))
        assert poincare_distance(a, b) == pytest.approx(poincare_distance(b, a))

    def test_triangle_inequality(self):
        """Triangle inequality holds."""
        a = PoincarePoint(coords=(0.1, 0.2))
        b = PoincarePoint(coords=(0.4, 0.3))
        c = PoincarePoint(coords=(0.2, 0.5))

        d_ac = poincare_distance(a, c)
        d_ab = poincare_distance(a, b)
        d_bc = poincare_distance(b, c)

        assert d_ac <= d_ab + d_bc + 1e-10

    def test_distance_from_origin(self):
        """Test distance from origin formula."""
        p = PoincarePoint(coords=(0.5, 0.0))
        origin = PoincarePoint.origin(2)

        d1 = poincare_distance(origin, p)
        d2 = distance_from_origin(p)

        assert d1 == pytest.approx(d2)


class TestGeodesic:
    """Tests for geodesic operations."""

    def test_geodesic_midpoint(self):
        """Midpoint is equidistant from endpoints."""
        a = PoincarePoint(coords=(0.2, 0.0))
        b = PoincarePoint(coords=(0.6, 0.0))

        mid = geodesic_midpoint(a, b)
        d_a = poincare_distance(a, mid)
        d_b = poincare_distance(b, mid)

        assert d_a == pytest.approx(d_b, rel=0.01)

    def test_geodesic_path_endpoints(self):
        """Geodesic path starts and ends at given points."""
        a = PoincarePoint(coords=(0.1, 0.2))
        b = PoincarePoint(coords=(0.5, 0.3))

        path = geodesic_path(a, b, steps=10)

        assert len(path) == 10
        assert path[0].coords[0] == pytest.approx(a.coords[0], rel=0.01)
        assert path[-1].coords[0] == pytest.approx(b.coords[0], rel=0.01)

    def test_frechet_mean_single(self):
        """Fréchet mean of single point is itself."""
        p = PoincarePoint(coords=(0.3, 0.4))
        mean = frechet_mean([p])
        assert mean.coords[0] == pytest.approx(p.coords[0])
        assert mean.coords[1] == pytest.approx(p.coords[1])

    def test_frechet_mean_symmetric(self):
        """Fréchet mean of symmetric points is near origin."""
        p1 = PoincarePoint(coords=(0.5, 0.0))
        p2 = PoincarePoint(coords=(-0.5, 0.0))
        mean = frechet_mean([p1, p2])
        assert mean.norm < 0.1  # Should be near origin
