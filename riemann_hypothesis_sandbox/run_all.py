"""
run_all.py - Run complete Riemann Hypothesis analysis pipeline

This script orchestrates the full spectral analysis:
1. Extract hyperbolic embeddings from trained model
2. Compute graph Laplacian spectrum
3. Compare to Riemann zeta zeros

Usage:
    python run_all.py [--checkpoint PATH] [--output DIR]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SANDBOX_DIR = Path(__file__).parent


def run_script(script_name: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    script_path = SANDBOX_DIR / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Run Riemann Hypothesis analysis pipeline')
    parser.add_argument('--checkpoint', type=str,
                       default='sandbox-training/checkpoints/v5_5/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                       default='riemann_hypothesis_sandbox/results',
                       help='Output directory for results')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Kernel bandwidth for Laplacian')
    parser.add_argument('--n-eigenvalues', type=int, default=1000,
                       help='Number of eigenvalues to compute')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip step 1 if embeddings already exist')
    parser.add_argument('--skip-spectrum', action='store_true',
                       help='Skip step 2 if eigenvalues already computed')
    args = parser.parse_args()

    print("=" * 60)
    print("RIEMANN HYPOTHESIS SPECTRAL ANALYSIS")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")

    # Check for existing results
    embeddings_path = PROJECT_ROOT / 'riemann_hypothesis_sandbox' / 'embeddings' / 'embeddings.pt'
    eigenvalues_path = PROJECT_ROOT / args.output / 'eigenvalues.npy'

    # Step 1: Extract embeddings
    if args.skip_extraction and embeddings_path.exists():
        print(f"\nSkipping extraction (embeddings exist at {embeddings_path})")
    else:
        success = run_script('01_extract_embeddings.py', [
            '--checkpoint', args.checkpoint,
            '--output', 'riemann_hypothesis_sandbox/embeddings'
        ])
        if not success:
            print("\nPipeline failed at step 1: Extract embeddings")
            return 1

    # Step 2: Compute spectrum
    if args.skip_spectrum and eigenvalues_path.exists():
        print(f"\nSkipping spectrum computation (eigenvalues exist at {eigenvalues_path})")
    else:
        success = run_script('02_compute_spectrum.py', [
            '--embeddings', 'riemann_hypothesis_sandbox/embeddings/embeddings.pt',
            '--output', args.output,
            '--sigma', str(args.sigma),
            '--n-eigenvalues', str(args.n_eigenvalues)
        ])
        if not success:
            print("\nPipeline failed at step 2: Compute spectrum")
            return 1

    # Step 3: Compare to zeta
    success = run_script('03_compare_zeta.py', [
        '--results-dir', args.output
    ])
    if not success:
        print("\nPipeline failed at step 3: Compare to zeta")
        return 1

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {PROJECT_ROOT / args.output}")
    print("\nKey output files:")
    print("  - spacing_distribution.png: Eigenvalue spacing histogram vs GUE/GOE/Poisson")
    print("  - zeta_comparison.png: Comparison with Riemann zeta zeros")
    print("  - pair_correlation.png: Montgomery pair correlation analysis")
    print("  - zeta_analysis_*.json: Full statistical results")

    return 0


if __name__ == '__main__':
    sys.exit(main())
