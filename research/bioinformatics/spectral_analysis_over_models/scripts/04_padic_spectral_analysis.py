"""
04_padic_spectral_analysis.py - Analyze embedding structure via 3-adic metric

Instead of random sampling, this uses the natural 3-adic structure:
- Operations indexed 0 to 19682 have 3-adic valuation v_3(|i-j|)
- The model learned to embed respecting this ultrametric
- We analyze the spectral structure of this learned mapping

Usage:
    python 04_padic_spectral_analysis.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def v3(n: int) -> int:
    """Compute 3-adic valuation of n (highest power of 3 dividing n)."""
    if n == 0:
        return float("inf")
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def padic_distance_3(i: int, j: int) -> float:
    """3-adic distance: d(i,j) = 3^(-v_3(|i-j|))."""
    if i == j:
        return 0.0
    return 3.0 ** (-v3(abs(i - j)))


def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> float:
    """Poincaré distance between two points."""
    u_norm_sq = torch.sum(u**2)
    v_norm_sq = torch.sum(v**2)
    diff_norm_sq = torch.sum((u - v) ** 2)

    u_norm_sq = torch.clamp(u_norm_sq, max=1.0 - eps)
    v_norm_sq = torch.clamp(v_norm_sq, max=1.0 - eps)

    arg = 1 + 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + eps)
    distance = torch.log(arg + torch.sqrt(torch.clamp(arg**2 - 1, min=eps)))

    return distance.item()


def gue_distribution(s: np.ndarray) -> np.ndarray:
    """GUE spacing distribution."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def sample_gue(n: int) -> np.ndarray:
    """Sample from GUE distribution."""
    samples = []
    max_p = gue_distribution(np.sqrt(np.pi / 8)) * 1.1
    while len(samples) < n:
        s = np.random.uniform(0, 5, size=n - len(samples))
        u = np.random.uniform(0, max_p, size=n - len(samples))
        accepted = s[u < gue_distribution(s)]
        samples.extend(accepted.tolist())
    return np.array(samples[:n])


def main():
    output_dir = PROJECT_ROOT / "research/spectral_analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL embeddings (the full 3-adic structure)
    print("Loading full embedding space...")
    data = torch.load(
        PROJECT_ROOT / "research/spectral_analysis" / "embeddings" / "embeddings.pt",
        weights_only=False,
    )
    z_hyp = data["z_hyperbolic"]
    n_ops = z_hyp.shape[0]

    print(f"Full embedding space: {n_ops} operations in {z_hyp.shape[1]}D Poincaré ball")

    # Analyze radial distribution by 3-adic level
    print("\n=== Radial Structure by 3-adic Level ===")
    radii = torch.norm(z_hyp, dim=-1).numpy()

    # Group operations by their 3-adic valuation structure
    # v_3(i) for i = 0, 1, ..., 19682
    valuations = np.array([v3(i) if i > 0 else 9 for i in range(n_ops)])

    print("\n3-adic valuation distribution:")
    for v in range(10):
        count = np.sum(valuations == v)
        if count > 0:
            mean_radius = radii[valuations == v].mean()
            print(f"  v_3 = {v}: {count:5d} ops, mean radius = {mean_radius:.4f}")

    # Build 3-adic Laplacian: connect operations by 3-adic distance
    print("\n=== Building 3-adic Weighted Laplacian ===")

    # Use smaller sample for eigenvalue computation but preserve 3-adic structure
    # Sample one operation from each 3-adic "ball" at various levels
    n_sample = 729  # 3^6 - manageable but structured
    step = n_ops // n_sample
    indices = np.arange(0, n_ops, step)[:n_sample]
    z_sample = z_hyp[indices]

    print(f"Sampling {len(indices)} operations with 3-adic structure preserved")

    # Weight matrix: W_ij = exp(-α * d_poincare) where edges exist by 3-adic distance
    # Use 3-adic distance to determine edge WEIGHTS, Poincaré for geometric structure
    n = len(indices)
    W = np.zeros((n, n))

    print("Computing 3-adic weighted adjacency matrix...")
    for i in range(n):
        for j in range(i + 1, n):
            idx_i, idx_j = indices[i], indices[j]

            # 3-adic weight (closer in 3-adic sense = stronger connection)
            padic_d = padic_distance_3(idx_i, idx_j)
            padic_weight = 1.0 / (padic_d + 0.01)  # Inverse 3-adic distance

            # Poincaré distance for geometric scaling
            poincare_d = poincare_distance(z_sample[i], z_sample[j])

            # Combined weight: 3-adic structure + learned geometry
            W[i, j] = padic_weight * np.exp(-poincare_d)
            W[j, i] = W[i, j]

    # Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    print(f"Laplacian shape: {L.shape}")

    # Eigenvalues
    print("Computing eigenvalues...")
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]  # Remove null space

    print(f"Non-zero eigenvalues: {len(eigenvalues)}")
    print(f"Range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

    # Spacings
    spacings = np.diff(eigenvalues)
    spacings_norm = spacings / spacings.mean()
    spacings_norm = spacings_norm[spacings_norm > 1e-10]

    # Statistical tests
    gue_samples = sample_gue(len(spacings_norm))
    poisson_samples = np.random.exponential(1.0, len(spacings_norm))

    ks_gue, p_gue = stats.ks_2samp(spacings_norm, gue_samples)
    ks_poisson, p_poisson = stats.ks_2samp(spacings_norm, poisson_samples)

    print(f"\n{'='*60}")
    print("3-ADIC SPECTRAL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"KS vs GUE: D={ks_gue:.4f}")
    print(f"KS vs Poisson: D={ks_poisson:.4f}")
    print(f"Closer to: {'GUE (zeta-like!)' if ks_gue < ks_poisson else 'Poisson'}")

    # Analyze correlation: does 3-adic distance predict Poincaré distance?
    print("\n=== 3-adic vs Poincaré Distance Correlation ===")

    padic_dists = []
    poincare_dists = []

    # Sample pairs
    np.random.seed(42)
    n_pairs = 5000
    for _ in range(n_pairs):
        i, j = np.random.randint(0, n_ops, size=2)
        if i != j:
            padic_dists.append(padic_distance_3(i, j))
            poincare_dists.append(poincare_distance(z_hyp[i], z_hyp[j]))

    padic_dists = np.array(padic_dists)
    poincare_dists = np.array(poincare_dists)

    # Correlation
    corr, p_val = stats.spearmanr(padic_dists, poincare_dists)
    print(f"Spearman correlation (3-adic vs Poincaré): ρ = {corr:.4f}, p = {p_val:.2e}")

    # Group by 3-adic level
    print("\nMean Poincaré distance by 3-adic distance level:")
    unique_padic = np.unique(padic_dists)
    for pd in sorted(unique_padic, reverse=True)[:6]:
        mask = padic_dists == pd
        mean_poincare = poincare_dists[mask].mean()
        v = -int(np.log(pd) / np.log(3)) if pd > 0 else "inf"
        print(f"  3^(-{v}) = {pd:.4f}: mean Poincaré = {mean_poincare:.4f} ({mask.sum()} pairs)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Spacing distribution
    ax1 = axes[0, 0]
    s_range = np.linspace(0, 4, 200)
    ax1.hist(
        spacings_norm,
        bins=40,
        density=True,
        alpha=0.7,
        label="3-adic Laplacian",
        color="steelblue",
        edgecolor="black",
    )
    ax1.plot(s_range, gue_distribution(s_range), "r-", lw=2, label="GUE")
    ax1.plot(s_range, np.exp(-s_range), "b:", lw=2, label="Poisson")
    ax1.set_xlabel("Normalized spacing")
    ax1.set_ylabel("P(s)")
    ax1.set_title(f"3-adic Laplacian Spacing (KS_GUE={ks_gue:.3f})")
    ax1.legend()
    ax1.set_xlim(0, 4)
    ax1.grid(True, alpha=0.3)

    # 2. 3-adic vs Poincaré scatter
    ax2 = axes[0, 1]
    ax2.scatter(padic_dists, poincare_dists, alpha=0.3, s=5)
    ax2.set_xlabel("3-adic distance")
    ax2.set_ylabel("Poincaré distance")
    ax2.set_title(f"3-adic vs Learned Geometry (ρ={corr:.3f})")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    # 3. Radial distribution by 3-adic level
    ax3 = axes[1, 0]
    for v in range(7):
        mask = valuations == v
        if mask.sum() > 0:
            ax3.hist(radii[mask], bins=30, alpha=0.5, label=f"v₃={v}", density=True)
    ax3.set_xlabel("Poincaré ball radius")
    ax3.set_ylabel("Density")
    ax3.set_title("Radial Distribution by 3-adic Valuation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Eigenvalue spectrum
    ax4 = axes[1, 1]
    ax4.plot(eigenvalues, "b-", lw=0.5)
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Eigenvalue")
    ax4.set_title("3-adic Laplacian Spectrum")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "padic_spectral_analysis.png", dpi=150)
    plt.close()

    print(f"\nSaved plot to {output_dir}/padic_spectral_analysis.png")

    # Save results
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_operations": n_ops,
        "n_sample": n,
        "ks_gue": float(ks_gue),
        "ks_poisson": float(ks_poisson),
        "closer_to": "GUE" if ks_gue < ks_poisson else "Poisson",
        "padic_poincare_correlation": float(corr),
        "padic_poincare_pvalue": float(p_val),
    }

    with open(output_dir / "padic_spectral_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if corr > 0.5:
        print(f"\nSTRONG 3-adic/Poincaré correlation (ρ={corr:.3f})!")
        print("The model has learned geometry that respects 3-adic structure.")
    elif corr > 0.2:
        print(f"\nMODERATE correlation (ρ={corr:.3f}).")
        print("Partial alignment between 3-adic and learned geometry.")
    else:
        print(f"\nWEAK correlation (ρ={corr:.3f}).")
        print("Learned geometry diverges from pure 3-adic structure.")

    if ks_gue < ks_poisson:
        print("\n*** 3-adic Laplacian shows GUE-like behavior! ***")

    return eigenvalues, spacings_norm, results


if __name__ == "__main__":
    main()
