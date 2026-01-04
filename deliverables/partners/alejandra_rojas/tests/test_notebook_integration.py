# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Integration tests for the Rojas serotype forecast notebook.

Tests that all notebook components can be imported and executed
without matplotlib rendering (headless validation).

Run with:
    python -m pytest deliverables/partners/alejandra_rojas/tests/test_notebook_integration.py -v

Or directly:
    python deliverables/partners/alejandra_rojas/tests/test_notebook_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

# Add paths
project_root = Path(__file__).resolve().parents[4]
deliverables_path = project_root / "deliverables"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(deliverables_path))

import numpy as np


class TestSharedImports(unittest.TestCase):
    """Test that shared module imports work correctly."""

    def test_primer_designer_import(self):
        """Test PrimerDesigner can be imported."""
        from shared import PrimerDesigner
        designer = PrimerDesigner()
        self.assertIsNotNone(designer)

    def test_peptide_utils_import(self):
        """Test peptide utilities can be imported."""
        from shared import compute_peptide_properties, validate_sequence
        self.assertTrue(callable(compute_peptide_properties))
        self.assertTrue(callable(validate_sequence))

    def test_config_import(self):
        """Test configuration can be imported."""
        from shared import get_config
        config = get_config()
        self.assertIsNotNone(config)


class TestHyperbolicGeometry(unittest.TestCase):
    """Test hyperbolic geometry utilities from notebook."""

    def test_poincare_distance(self):
        """Test Poincare distance computation."""
        # Simplified HyperbolicPoint (from notebook)
        class HyperbolicPoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @property
            def coords(self):
                return np.array([self.x, self.y])

            @property
            def radius(self):
                return np.sqrt(self.x**2 + self.y**2)

        def poincare_distance(p1, p2):
            u, v = p1.coords, p2.coords
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u >= 1 or norm_v >= 1:
                return float('inf')
            diff = u - v
            norm_diff_sq = np.dot(diff, diff)
            denom = (1 - norm_u**2) * (1 - norm_v**2)
            cosh_dist = 1 + 2 * norm_diff_sq / denom
            return np.arccosh(max(1.0, cosh_dist))

        # Test same point has zero distance
        p1 = HyperbolicPoint(0.3, 0.4)
        p2 = HyperbolicPoint(0.3, 0.4)
        self.assertAlmostEqual(poincare_distance(p1, p2), 0.0, places=5)

        # Test origin distance
        origin = HyperbolicPoint(0.0, 0.0)
        point = HyperbolicPoint(0.5, 0.0)
        dist = poincare_distance(origin, point)
        self.assertGreater(dist, 0)
        self.assertLess(dist, 10)  # Reasonable bound

    def test_mobius_addition(self):
        """Test Mobius addition."""
        def mobius_addition(u, v, c=1.0):
            u_sq = np.dot(u, u)
            v_sq = np.dot(v, v)
            uv = np.dot(u, v)
            num = (1 + 2*c*uv + c*v_sq) * u + (1 - c*u_sq) * v
            denom = 1 + 2*c*uv + c**2 * u_sq * v_sq
            return num / denom

        # Identity: u + 0 = u
        u = np.array([0.3, 0.4])
        zero = np.array([0.0, 0.0])
        result = mobius_addition(u, zero)
        np.testing.assert_array_almost_equal(result, u)


class TestTrajectoryGeneration(unittest.TestCase):
    """Test trajectory generation from notebook."""

    def test_generate_trajectories(self):
        """Test trajectory generation produces valid data."""
        np.random.seed(42)
        years = list(range(2015, 2025))
        n_years = len(years)

        # Simplified trajectory generation (from notebook)
        denv2_cases = 20 + 15 * np.arange(n_years) + np.random.randn(n_years) * 15
        denv2_cases = np.maximum(denv2_cases, 5)

        self.assertEqual(len(denv2_cases), n_years)
        self.assertTrue(np.all(denv2_cases >= 5))

    def test_hyperbolic_momentum(self):
        """Test momentum computation."""
        # Mock trajectory data
        positions = np.array([
            [0.3, 0.4],
            [0.32, 0.42],
            [0.35, 0.45],
        ])

        # Compute velocity
        velocities = np.diff(positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        magnitude = np.linalg.norm(avg_velocity)

        self.assertGreater(magnitude, 0)
        self.assertEqual(len(avg_velocity), 2)


class TestPrimerDesign(unittest.TestCase):
    """Test primer design functionality."""

    def test_design_for_peptide(self):
        """Test primer design for peptide sequence."""
        from shared import PrimerDesigner

        designer = PrimerDesigner()

        # Test peptide sequence
        peptide = "MGKREKKLGEFGKAKGSRAIWYM"
        result = designer.design_for_peptide(
            peptide,
            codon_optimization='ecoli',
            add_start_codon=True,
            add_stop_codon=False
        )

        self.assertIsNotNone(result.forward)
        self.assertIsNotNone(result.reverse)
        self.assertGreater(len(result.forward), 0)
        self.assertGreater(len(result.reverse), 0)

    def test_tm_calculation(self):
        """Test melting temperature calculation."""
        from shared.primer_design import calculate_tm

        # Known primer
        primer = "ATGCGATCGATCGATCGATC"
        tm = calculate_tm(primer)

        self.assertGreater(tm, 40)
        self.assertLess(tm, 80)

    def test_gc_calculation(self):
        """Test GC content calculation."""
        from shared.primer_design import calculate_gc

        # 50% GC primer
        primer = "ATGCATGC"
        gc = calculate_gc(primer)
        self.assertAlmostEqual(gc, 50.0, places=1)

        # 100% GC primer
        primer_gc = "GCGCGCGC"
        gc = calculate_gc(primer_gc)
        self.assertAlmostEqual(gc, 100.0, places=1)


class TestRiskAssessment(unittest.TestCase):
    """Test risk assessment computation."""

    def test_risk_score_range(self):
        """Test that risk scores are in valid range."""
        # Simplified risk computation
        components = {
            'divergence': 0.5,
            'momentum': 0.3,
            'virulence': 0.7,
            'case_trend': 0.4,
            'uncertainty': 0.2,
        }

        weights = {
            'divergence': 0.20,
            'momentum': 0.25,
            'virulence': 0.25,
            'case_trend': 0.20,
            'uncertainty': 0.10,
        }

        risk_score = sum(components[k] * weights[k] for k in weights)

        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)

    def test_risk_level_classification(self):
        """Test risk level classification."""
        def classify_risk(score):
            if score >= 0.7:
                return 'Critical'
            elif score >= 0.5:
                return 'High'
            elif score >= 0.3:
                return 'Moderate'
            else:
                return 'Low'

        self.assertEqual(classify_risk(0.8), 'Critical')
        self.assertEqual(classify_risk(0.6), 'High')
        self.assertEqual(classify_risk(0.4), 'Moderate')
        self.assertEqual(classify_risk(0.2), 'Low')


class TestGeometryModule(unittest.TestCase):
    """Test the updated geometry.py module."""

    def test_hyperbolic_space_creation(self):
        """Test HyperbolicSpace can be created."""
        from deliverables.partners.alejandra_rojas.src.geometry import HyperbolicSpace

        space = HyperbolicSpace(dimension=2, curvature=1.0)
        self.assertIsNotNone(space)
        self.assertIn(space.backend, ['src.geometry', 'geomstats', 'numpy'])

    def test_hyperbolic_distance(self):
        """Test hyperbolic distance computation."""
        from deliverables.partners.alejandra_rojas.src.geometry import HyperbolicSpace

        space = HyperbolicSpace(dimension=2, curvature=1.0)

        x = np.array([0.3, 0.4])
        y = np.array([0.5, 0.2])

        dist = space.distance(x, y)
        self.assertGreater(dist, 0)

    def test_distance_to_origin(self):
        """Test distance to origin (radial coordinate)."""
        from deliverables.partners.alejandra_rojas.src.geometry import HyperbolicSpace

        space = HyperbolicSpace(dimension=2, curvature=1.0)

        # Point at origin should have zero distance
        origin = np.array([0.0, 0.0])
        dist = space.distance_to_origin(origin)
        self.assertAlmostEqual(float(dist), 0.0, places=5)

        # Point away from origin should have positive distance
        point = np.array([0.5, 0.0])
        dist = space.distance_to_origin(point)
        self.assertGreater(float(dist), 0)


class TestNCBILoader(unittest.TestCase):
    """Test NCBI loader functionality."""

    def test_demo_sequence_generation(self):
        """Test that demo sequences can be generated."""
        from deliverables.partners.alejandra_rojas.scripts.ncbi_arbovirus_loader import (
            NCBIArbovirusLoader,
            VirusSequence,
        )

        loader = NCBIArbovirusLoader()
        sequences = loader._generate_demo_sequences("DENV-1", n=5)

        self.assertEqual(len(sequences), 5)
        for seq in sequences:
            self.assertIsInstance(seq, VirusSequence)
            self.assertEqual(seq.virus, "DENV-1")
            self.assertGreater(len(seq.sequence), 1000)


def run_all_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSharedImports))
    suite.addTests(loader.loadTestsFromTestCase(TestHyperbolicGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPrimerDesign))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskAssessment))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometryModule))
    suite.addTests(loader.loadTestsFromTestCase(TestNCBILoader))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Rojas Serotype Forecast Notebook Integration Tests")
    print("=" * 70)
    print()

    result = run_all_tests()

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}, ERRORS: {len(result.errors)}")
    print("=" * 70)
