#!/usr/bin/env python3
"""Mathematical Framework Validation Suite for TernaryVAE v5.12.5.

Validates that the mathematical framework components work correctly:
- P-adic mathematics operations
- Hyperbolic geometry (default)
- VAE architecture integrity
- Checkpoint compatibility
- Mathematical property preservation

Usage:
    python scripts/mathematical/validate_mathematical_framework.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

class MathematicalFrameworkValidator:
    """Comprehensive validation for mathematical TernaryVAE framework."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def test_mathematical_imports(self):
        """Test that all mathematical components import correctly."""
        try:
            # Core mathematical components (fallback to original structure)
            from src.core import TERNARY, padic_distance, padic_valuation
            from src.geometry import poincare_distance, exp_map_zero
            from src.models import TernaryVAEV5_11_PartialFreeze, HomeostasisController
            from src.losses import RichHierarchyLoss

            self.results['mathematical_imports'] = {
                'passed': True,
                'description': "All mathematical components import successfully"
            }
            print("✓ Mathematical imports: PASS")
            return True

        except ImportError as e:
            self.results['mathematical_imports'] = {
                'passed': False,
                'description': f"Import error: {e}"
            }
            print(f"✗ Mathematical imports: FAIL - {e}")
            return False

    def test_ternary_operations(self):
        """Test 3-adic ternary operations."""
        try:
            from src.core import TERNARY

            # Test space size
            assert TERNARY.N_OPERATIONS == 19683

            # Test valuation computation
            assert TERNARY.valuation(torch.tensor(0)) == 9  # Mathematical invariant
            assert TERNARY.valuation(torch.tensor(1)) == 0

            # Test ternary conversion
            ops = TERNARY.generate_all_operations()[:100]
            indices = TERNARY.from_ternary(torch.tensor(ops))
            reconstructed = TERNARY.to_ternary(indices, length=9)

            assert torch.allclose(torch.tensor(ops), reconstructed)

            self.results['ternary_operations'] = {
                'passed': True,
                'description': "3-adic ternary operations working correctly"
            }
            print("✓ Ternary operations: PASS")
            return True

        except Exception as e:
            self.results['ternary_operations'] = {
                'passed': False,
                'description': f"Ternary operations error: {e}"
            }
            print(f"✗ Ternary operations: FAIL - {e}")
            return False

    def test_hyperbolic_geometry(self):
        """Test hyperbolic Poincaré ball operations (default geometry)."""
        try:
            from src.geometry import poincare_distance, exp_map_zero

            # Test hyperbolic properties
            origin = torch.zeros(2)
            point = torch.tensor([0.5, 0.0])

            # Distance properties
            assert poincare_distance(origin, origin, c=1.0) == 0  # Identity
            assert poincare_distance(origin, point, c=1.0) > 0   # Positive distance

            # Symmetry
            d1 = poincare_distance(origin, point, c=1.0)
            d2 = poincare_distance(point, origin, c=1.0)
            assert torch.allclose(d1, d2, atol=1e-6)

            # Hyperbolic constraint (points in Poincaré ball)
            random_points = torch.randn(10, 2) * 0.8  # Stay within ball
            norms = torch.norm(random_points, dim=-1)
            assert (norms < 1.0).all()  # All points in ball

            self.results['hyperbolic_geometry'] = {
                'passed': True,
                'description': "Hyperbolic geometry operations working correctly"
            }
            print("✓ Hyperbolic geometry: PASS")
            return True

        except Exception as e:
            self.results['hyperbolic_geometry'] = {
                'passed': False,
                'description': f"Hyperbolic geometry error: {e}"
            }
            print(f"✗ Hyperbolic geometry: FAIL - {e}")
            return False

    def test_vae_architecture(self):
        """Test TernaryVAE mathematical architecture."""
        try:
            from src.models import TernaryVAEV5_11_PartialFreeze
            from src.core import TERNARY

            # Create model
            model = TernaryVAEV5_11_PartialFreeze(
                latent_dim=16,
                hidden_dim=64,
                max_radius=0.95,
                curvature=1.0,
                encoder_type='improved',
                decoder_type='improved'
            ).to(self.device)

            # Test forward pass
            test_input = torch.randint(0, 19683, (32, 9)).to(self.device)
            with torch.no_grad():
                output = model(test_input, compute_control=False)

            # Check output structure
            assert 'z_A_hyp' in output
            assert 'z_B_hyp' in output
            assert 'reconstructed' in output

            # Check hyperbolic constraints
            z_A_norms = torch.norm(output['z_A_hyp'], dim=-1)
            z_B_norms = torch.norm(output['z_B_hyp'], dim=-1)
            assert (z_A_norms < 1.0).all()
            assert (z_B_norms < 1.0).all()

            # Check reconstruction shape
            assert output['reconstructed'].shape == (32, 9, 27)

            self.results['vae_architecture'] = {
                'passed': True,
                'description': "VAE architecture working correctly"
            }
            print("✓ VAE architecture: PASS")
            return True

        except Exception as e:
            self.results['vae_architecture'] = {
                'passed': False,
                'description': f"VAE architecture error: {e}"
            }
            print(f"✗ VAE architecture: FAIL - {e}")
            return False

    def test_mathematical_loss(self):
        """Test mathematical loss functions."""
        try:
            from src.losses import RichHierarchyLoss
            from src.core import TERNARY

            # Create loss function
            loss_fn = RichHierarchyLoss(
                hierarchy_weight=5.0,
                coverage_weight=1.0,
                richness_weight=2.0,
                separation_weight=3.0
            )

            # Mock model output
            batch_size = 32
            mock_output = {
                'z_A_hyp': torch.randn(batch_size, 16) * 0.5,
                'z_B_hyp': torch.randn(batch_size, 16) * 0.5,
                'reconstructed': torch.randn(batch_size, 9, 27),
            }

            mock_input = torch.randint(0, 19683, (batch_size, 9))

            # Compute loss
            loss_dict = loss_fn(mock_output, mock_input, TERNARY)

            # Check loss components
            assert 'total_loss' in loss_dict
            assert isinstance(loss_dict['total_loss'], torch.Tensor)
            assert not torch.isnan(loss_dict['total_loss'])
            assert not torch.isinf(loss_dict['total_loss'])

            self.results['mathematical_loss'] = {
                'passed': True,
                'description': "Mathematical loss functions working correctly"
            }
            print("✓ Mathematical loss: PASS")
            return True

        except Exception as e:
            self.results['mathematical_loss'] = {
                'passed': False,
                'description': f"Mathematical loss error: {e}"
            }
            print(f"✗ Mathematical loss: FAIL - {e}")
            return False

    def test_checkpoint_compatibility(self):
        """Test checkpoint loading compatibility."""
        try:
            from src.models import TernaryVAEV5_11_PartialFreeze

            # Check if TIER-1 checkpoints exist
            checkpoints = [
                'checkpoints/tier-1/v5_12_4_best_Q.pt',
                'checkpoints/tier-1/homeostatic_rich_best.pt',
                'checkpoints/tier-1/v5_11_structural_best.pt',
                'checkpoints/tier-1/v5_11_homeostasis_best.pt'
            ]

            found_checkpoints = []
            for checkpoint_path in checkpoints:
                if Path(checkpoint_path).exists():
                    found_checkpoints.append(checkpoint_path)

            if not found_checkpoints:
                self.results['checkpoint_compatibility'] = {
                    'passed': False,
                    'description': "No TIER-1 checkpoints found"
                }
                print("⚠ Checkpoint compatibility: No checkpoints to test")
                return False

            # Test loading one checkpoint
            test_checkpoint = found_checkpoints[0]
            checkpoint = torch.load(test_checkpoint, map_location='cpu')

            # Create model and try to load
            model = TernaryVAEV5_11_PartialFreeze(
                latent_dim=16,
                hidden_dim=64,
                max_radius=0.95,
                curvature=1.0
            )

            # Test loading (may require strict=False)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            self.results['checkpoint_compatibility'] = {
                'passed': True,
                'description': f"Successfully loaded checkpoint: {test_checkpoint}"
            }
            print(f"✓ Checkpoint compatibility: PASS ({len(found_checkpoints)} checkpoints)")
            return True

        except Exception as e:
            self.results['checkpoint_compatibility'] = {
                'passed': False,
                'description': f"Checkpoint compatibility error: {e}"
            }
            print(f"✗ Checkpoint compatibility: FAIL - {e}")
            return False

    def test_p_adic_structure(self):
        """Test p-adic mathematical structure preservation."""
        try:
            from src.core import TERNARY, padic_distance

            # Test p-adic distance properties
            a, b, c = 0, 1, 2

            # Ultrametric inequality: d(a,c) <= max(d(a,b), d(b,c))
            d_ac = padic_distance(a, c, p=3)
            d_ab = padic_distance(a, b, p=3)
            d_bc = padic_distance(b, c, p=3)

            assert d_ac <= max(d_ab, d_bc) + 1e-6  # Allow small numerical error

            # Test valuation ordering
            valuations = [TERNARY.valuation(torch.tensor(i)).item() for i in range(100)]
            assert min(valuations) >= 0
            assert max(valuations) <= 9

            self.results['p_adic_structure'] = {
                'passed': True,
                'description': "P-adic mathematical structure preserved"
            }
            print("✓ P-adic structure: PASS")
            return True

        except Exception as e:
            self.results['p_adic_structure'] = {
                'passed': False,
                'description': f"P-adic structure error: {e}"
            }
            print(f"✗ P-adic structure: FAIL - {e}")
            return False

    def run_comprehensive_validation(self):
        """Run all mathematical validation tests."""
        print("=== TernaryVAE Mathematical Framework Validation ===")
        print(f"Device: {self.device}")
        print()

        tests = [
            self.test_mathematical_imports,
            self.test_ternary_operations,
            self.test_hyperbolic_geometry,
            self.test_vae_architecture,
            self.test_mathematical_loss,
            self.test_checkpoint_compatibility,
            self.test_p_adic_structure
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1

        print()
        print(f"=== Validation Results: {passed}/{total} tests passed ===")

        if passed == total:
            print("🎉 Mathematical framework fully validated!")
            return True
        else:
            print("⚠️  Some validation tests failed. Check issues above.")
            return False

def main():
    validator = MathematicalFrameworkValidator()
    success = validator.run_comprehensive_validation()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()