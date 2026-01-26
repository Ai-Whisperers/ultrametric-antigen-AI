#!/usr/bin/env python3
"""Verify Carlos Brizuela Package Path Configuration.

This script verifies that all paths are correctly configured for the
self-contained package structure using local checkpoints_definitive/.

Usage:
    python verify_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths (same as other scripts)
_script_dir = Path(__file__).resolve().parent
_deliverables_dir = _script_dir.parent.parent
_repo_root = _deliverables_dir.parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_repo_root))


def verify_directory_structure():
    """Verify that all expected directories exist."""
    print("🔍 Verifying Directory Structure")
    print("=" * 50)

    # Key directories that should exist (local to package)
    directories = [
        _script_dir / "checkpoints_definitive",
        _script_dir / "training",
        _script_dir / "scripts",
        _script_dir / "results",
        _script_dir / "data",  # Will be created by data loader
    ]

    all_exist = True
    for directory in directories:
        if directory.exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} (missing)")
            all_exist = False

    return all_exist


def verify_imports():
    """Verify that critical imports work."""
    print("\n🔗 Verifying Imports")
    print("=" * 50)

    try:
        from src.encoders.peptide_encoder import PeptideVAE
        print("✅ PeptideVAE import")
    except ImportError as e:
        print(f"❌ PeptideVAE import: {e}")
        return False

    try:
        from src.losses.peptide_losses import PeptideLossManager
        print("✅ PeptideLossManager import")
    except ImportError as e:
        print(f"❌ PeptideLossManager import: {e}")
        return False

    try:
        from training.dataset import create_stratified_dataloaders
        print("✅ Dataset import")
    except ImportError as e:
        print(f"❌ Dataset import: {e}")
        return False

    try:
        from scripts.dramp_activity_loader import DRAMPLoader
        print("✅ DRAMPLoader import")
    except ImportError as e:
        print(f"❌ DRAMPLoader import: {e}")
        return False

    return True


def verify_checkpoint_paths():
    """Verify checkpoint path configuration."""
    print("\n📁 Verifying Checkpoint Paths")
    print("=" * 50)

    # Expected checkpoint directory (local to package)
    checkpoint_dir = _script_dir / "checkpoints_definitive"

    print(f"📂 Checkpoint directory: {checkpoint_dir}")
    if checkpoint_dir.exists():
        print("✅ Checkpoint directory exists")
    else:
        print("❌ Checkpoint directory missing")
        return False

    # Check for required checkpoints
    best_checkpoint = checkpoint_dir / "best_production.pt"
    if best_checkpoint.exists():
        print(f"✅ best_production.pt ({best_checkpoint.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("❌ best_production.pt missing")
        return False

    # Check for fold checkpoints
    fold_checkpoints = list(checkpoint_dir.glob("fold_*_definitive.pt"))
    if fold_checkpoints:
        print(f"📦 Found {len(fold_checkpoints)} fold checkpoint(s):")
        for ckpt in sorted(fold_checkpoints):
            print(f"   - {ckpt.name}")
    else:
        print("⚠️  No fold checkpoints found (optional)")

    # Check for CV results
    cv_results = checkpoint_dir / "cv_results_definitive.json"
    if cv_results.exists():
        print("✅ cv_results_definitive.json exists")
    else:
        print("⚠️  cv_results_definitive.json missing (optional)")

    return True


def verify_scripts():
    """Verify that key scripts have correct path references."""
    print("\n📝 Verifying Script Configuration")
    print("=" * 50)

    # Check predict_mic.py
    try:
        from scripts.predict_mic import DEFAULT_CHECKPOINT
        print(f"🎯 predict_mic.py DEFAULT_CHECKPOINT: {DEFAULT_CHECKPOINT}")

        # Check that it points to local checkpoints_definitive
        expected_path = _script_dir / "checkpoints_definitive" / "best_production.pt"

        if DEFAULT_CHECKPOINT.resolve() == expected_path.resolve():
            print("✅ predict_mic.py uses correct local path")
        else:
            print(f"⚠️  predict_mic.py path: {DEFAULT_CHECKPOINT}")
            print(f"   Expected: {expected_path}")
            # Not a failure - may be intentional override
    except Exception as e:
        print(f"❌ Error checking predict_mic.py: {e}")
        return False

    # Check training script config
    try:
        from training.train_definitive import DefinitiveConfig
        config = DefinitiveConfig()
        print(f"🎯 train_definitive.py checkpoint_dir: {config.checkpoint_dir}")

        if "checkpoints_definitive" in str(config.checkpoint_dir):
            print("✅ train_definitive.py uses local checkpoint dir")
        else:
            print(f"⚠️  train_definitive.py checkpoint_dir: {config.checkpoint_dir}")
    except Exception as e:
        print(f"❌ Error checking train_definitive.py: {e}")
        return False

    return True


def main():
    """Main verification routine."""
    print("Carlos Brizuela Package Path Verification")
    print("="*50)
    print(f"Package directory: {_script_dir}")
    print(f"Repository root: {_repo_root}")
    print()

    # Run all verification steps
    steps = [
        ("Directory Structure", verify_directory_structure),
        ("Python Imports", verify_imports),
        ("Checkpoint Paths", verify_checkpoint_paths),
        ("Script Configuration", verify_scripts),
    ]

    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            results[step_name] = False

    # Summary
    print("\n" + "="*50)
    print("📊 VERIFICATION SUMMARY")
    print("="*50)

    all_passed = True
    for step_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<10} {step_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All verifications passed! Package is ready to use.")
        print()
        print("Next steps:")
        print("1. Train models: cd training && python train_definitive.py")
        print("2. Test prediction: python scripts/predict_mic.py \"KLAKLAKKLAKLAK\"")
        print("3. Run optimization: python scripts/B1_pathogen_specific_design.py")
    else:
        print("⚠️  Some verifications failed. Check the issues above.")
        print()
        print("Common fixes:")
        print("1. Make sure you're running from the carlos_brizuela package directory")
        print("2. Check that src/ directory exists in repository root")
        print("3. Create missing directories if needed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)