"""
02_compute_spectrum_fast.py - Fast spectral analysis with sampling

Computes eigenvalue spectrum using a random sample of operations
for quick results. Use full version for rigorous analysis.

Usage:
    python 02_compute_spectrum_fast.py [--n-samples 2000]
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def poincare_distance_batch(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute all pairwise Poincaré distances efficiently."""
    n = z.shape[0]

    # Precompute norms
    norm_sq = torch.sum(z ** 2, dim=-1)  # (n,)
    norm_sq = torch.clamp(norm_sq, max=1.0 - eps)

    # Compute ||u - v||^2 using broadcasting
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u·v
    dot_products = torch.mm(z, z.t())  # (n, n)
    diff_norm_sq = norm_sq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * dot_products
    diff_norm_sq = torch.clamp(diff_norm_sq, min=0)  # Numerical safety

    # Denominators: (1 - ||u||^2)(1 - ||v||^2)
    denom = (1 - norm_sq.unsqueeze(1)) * (1 - norm_sq.unsqueeze(0)) + eps

    # arcosh argument
    arg = 1 + 2 * diff_norm_sq / denom

    # arcosh(x) = log(x + sqrt(x^2 - 1))
    distances = torch.log(arg + torch.sqrt(torch.clamp(arg ** 2 - 1, min=eps)))

    return distances


def gue_distribution(s: np.ndarray) -> np.ndarray:
    """GUE spacing distribution (what zeta zeros follow)."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def goe_distribution(s: np.ndarray) -> np.ndarray:
    """GOE spacing distribution."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def poisson_distribution(s: np.ndarray) -> np.ndarray:
    """Poisson spacing distribution (uncorrelated)."""
    return np.exp(-s)


def sample_gue(n: int) -> np.ndarray:
    """Sample from GUE distribution via rejection sampling."""
    samples = []
    max_p = gue_distribution(np.sqrt(np.pi/8)) * 1.1

    while len(samples) < n:
        s = np.random.uniform(0, 5, size=n - len(samples))
        u = np.random.uniform(0, max_p, size=n - len(samples))
        accepted = s[u < gue_distribution(s)]
        samples.extend(accepted.tolist())

    return np.array(samples[:n])


def main():
    parser = argparse.ArgumentParser(description='Fast spectral analysis')
    parser.add_argument('--embeddings', type=str,
                       default='riemann_hypothesis_sandbox/embeddings/embeddings.pt',
                       help='Path to embeddings')
    parser.add_argument('--output', type=str,
                       default='riemann_hypothesis_sandbox/results',
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of operations to sample')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Kernel bandwidth')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings...")
    data = torch.load(PROJECT_ROOT / args.embeddings)
    z_hyp = data['z_hyperbolic']
    n_total = z_hyp.shape[0]

    print(f"Total operations: {n_total}")
    print(f"Sampling {args.n_samples} operations for fast analysis")

    # Random sample
    torch.manual_seed(42)
    indices = torch.randperm(n_total)[:args.n_samples]
    z_sample = z_hyp[indices]

    print(f"Sample shape: {z_sample.shape}")

    # Compute pairwise distances (vectorized)
    print("Computing pairwise Poincaré distances...")
    distances = poincare_distance_batch(z_sample)

    print(f"Distance stats: min={distances.min():.4f}, max={distances.max():.4f}, mean={distances.mean():.4f}")

    # Build graph Laplacian
    print(f"Building graph Laplacian (sigma={args.sigma})...")
    W = torch.exp(-distances ** 2 / (2 * args.sigma ** 2))
    W.fill_diagonal_(0)
    D = torch.diag(W.sum(dim=1))
    L = D - W

    # Compute eigenvalues
    print("Computing eigenvalues...")
    eigenvalues = torch.linalg.eigvalsh(L).numpy()
    eigenvalues = np.sort(eigenvalues)

    # Skip near-zero eigenvalues (null space)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]

    print(f"Computed {len(eigenvalues)} non-zero eigenvalues")
    print(f"Range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

    # Compute spacings
    spacings = np.diff(eigenvalues)
    spacings_norm = spacings / spacings.mean()
    spacings_norm = spacings_norm[spacings_norm > 1e-10]

    # Statistical tests
    print("\nComparing to random matrix distributions...")

    gue_samples = sample_gue(len(spacings_norm))
    poisson_samples = np.random.exponential(1.0, len(spacings_norm))

    ks_gue, p_gue = stats.ks_2samp(spacings_norm, gue_samples)
    ks_poisson, p_poisson = stats.ks_2samp(spacings_norm, poisson_samples)

    print(f"\n{'='*60}")
    print("SPECTRAL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Sample size: {args.n_samples} operations")
    print(f"Eigenvalues: {len(eigenvalues)}")
    print(f"Spacings analyzed: {len(spacings_norm)}")
    print(f"\nKS test vs GUE: D={ks_gue:.4f}, p={p_gue:.4f}")
    print(f"KS test vs Poisson: D={ks_poisson:.4f}, p={p_poisson:.4f}")
    print(f"\nCloser to: {'GUE (zeta-like!)' if ks_gue < ks_poisson else 'Poisson (uncorrelated)'}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s_range = np.linspace(0, 4, 200)

    # Spacing distribution
    ax1 = axes[0]
    ax1.hist(spacings_norm, bins=40, density=True, alpha=0.7,
             label='Observed', color='steelblue', edgecolor='black')
    ax1.plot(s_range, gue_distribution(s_range), 'r-', lw=2, label='GUE (Riemann zeros)')
    ax1.plot(s_range, goe_distribution(s_range), 'g--', lw=2, label='GOE')
    ax1.plot(s_range, poisson_distribution(s_range), 'b:', lw=2, label='Poisson')
    ax1.set_xlabel('Normalized spacing', fontsize=12)
    ax1.set_ylabel('P(s)', fontsize=12)
    ax1.set_title(f'Eigenvalue Spacing Distribution\n(KS vs GUE: {ks_gue:.3f})', fontsize=12)
    ax1.legend()
    ax1.set_xlim(0, 4)
    ax1.grid(True, alpha=0.3)

    # Eigenvalue spectrum
    ax2 = axes[1]
    ax2.plot(eigenvalues, 'b-', lw=0.5)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Laplacian Eigenvalue Spectrum', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'spacing_distribution.png', dpi=150)
    plt.savefig(output_dir / 'spacing_distribution.pdf')
    plt.close()

    print(f"\nSaved plot to {output_dir}/spacing_distribution.png")

    # Save results
    np.save(output_dir / 'eigenvalues.npy', eigenvalues)
    np.save(output_dir / 'spacings.npy', spacings)

    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_samples': args.n_samples,
        'n_eigenvalues': len(eigenvalues),
        'ks_gue': float(ks_gue),
        'p_gue': float(p_gue),
        'ks_poisson': float(ks_poisson),
        'p_poisson': float(p_poisson),
        'closer_to': 'GUE' if ks_gue < ks_poisson else 'Poisson',
        'mean_spacing': float(spacings.mean()),
        'std_spacing': float(spacings.std())
    }

    with open(output_dir / 'spectrum_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if ks_gue < 0.1:
        print("\n*** STRONG GUE FIT ***")
        print("Eigenvalue spacings closely follow GUE statistics!")
        print("This suggests a deep connection to Riemann zeta zeros.")
    elif ks_gue < ks_poisson:
        print("\n** MODERATE GUE FIT **")
        print("Spacings are closer to GUE than Poisson.")
        print("Suggestive of random matrix universality.")
    else:
        print("\nSpacings closer to Poisson (uncorrelated).")
        print("May need different sigma or more samples.")

    return eigenvalues, spacings, results


if __name__ == '__main__':
    main()
