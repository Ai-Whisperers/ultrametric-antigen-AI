"""
02_compute_spectrum.py - Compute spectral properties of hyperbolic Laplacian

This script builds the graph Laplacian from Poincaré distances and computes
its eigenvalue spectrum for comparison with Riemann zeta zeros.

Usage:
    python 02_compute_spectrum.py [--embeddings PATH] [--output DIR]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm


def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Poincaré ball distance between points u and v.

    d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    """
    u_norm_sq = torch.sum(u**2, dim=-1)
    v_norm_sq = torch.sum(v**2, dim=-1)
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)

    # Clamp to avoid numerical issues at boundary
    u_norm_sq = torch.clamp(u_norm_sq, max=1.0 - eps)
    v_norm_sq = torch.clamp(v_norm_sq, max=1.0 - eps)

    # Compute argument of arcosh
    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq) + eps

    arg = 1 + numerator / denominator

    # arcosh(x) = log(x + sqrt(x^2 - 1))
    distance = torch.log(arg + torch.sqrt(torch.clamp(arg**2 - 1, min=eps)))

    return distance


def compute_pairwise_distances(z: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    """Compute pairwise Poincaré distances (memory-efficient batched version)."""
    n = z.shape[0]
    distances = torch.zeros(n, n, dtype=z.dtype)

    print(f"Computing {n}x{n} = {n*n:,} pairwise distances...")

    for i in tqdm(range(0, n, batch_size), desc="Computing distances"):
        i_end = min(i + batch_size, n)
        z_batch = z[i:i_end]  # (batch, dim)

        for j in range(0, n, batch_size):
            j_end = min(j + batch_size, n)
            z_other = z[j:j_end]  # (batch2, dim)

            # Broadcast: (batch, 1, dim) vs (1, batch2, dim)
            d = poincare_distance(z_batch.unsqueeze(1), z_other.unsqueeze(0))
            distances[i:i_end, j:j_end] = d

    return distances


def build_graph_laplacian(distances: torch.Tensor, sigma: float = 1.0, k_neighbors: int = None) -> torch.Tensor:
    """Build graph Laplacian from distance matrix.

    L = D - W where W_ij = exp(-d_ij^2 / (2*sigma^2))

    Args:
        distances: Pairwise distance matrix (n, n)
        sigma: Kernel bandwidth
        k_neighbors: If set, use k-NN graph instead of full graph

    Returns:
        Laplacian matrix (n, n)
    """
    n = distances.shape[0]

    # Compute affinity/weight matrix
    W = torch.exp(-(distances**2) / (2 * sigma**2))

    # Zero out diagonal (no self-loops)
    W.fill_diagonal_(0)

    if k_neighbors is not None:
        # Sparsify to k-NN graph
        print(f"Sparsifying to {k_neighbors}-NN graph...")
        _, indices = torch.topk(W, k=k_neighbors, dim=1)
        mask = torch.zeros_like(W)
        for i in range(n):
            mask[i, indices[i]] = 1
        # Symmetrize
        mask = torch.max(mask, mask.t())
        W = W * mask

    # Degree matrix
    D = torch.diag(W.sum(dim=1))

    # Laplacian
    L = D - W

    return L, W, D


def compute_eigenvalues(L: torch.Tensor, k: int = None, use_sparse: bool = False):
    """Compute eigenvalues of Laplacian.

    Args:
        L: Laplacian matrix
        k: Number of eigenvalues to compute (None = all)
        use_sparse: Use sparse solver (faster for large matrices)

    Returns:
        eigenvalues: Sorted eigenvalues
    """
    n = L.shape[0]

    if k is None:
        k = n

    if use_sparse and k < n:
        print(f"Computing {k} smallest eigenvalues (sparse solver)...")
        L_sparse = sparse.csr_matrix(L.numpy())
        eigenvalues, _ = eigsh(L_sparse, k=k, which="SM")
        eigenvalues = np.sort(eigenvalues)
    else:
        print(f"Computing all {n} eigenvalues (dense solver)...")
        eigenvalues = torch.linalg.eigvalsh(L)
        eigenvalues = eigenvalues.numpy()
        eigenvalues = np.sort(eigenvalues)

        if k < n:
            eigenvalues = eigenvalues[:k]

    return eigenvalues


def compute_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute consecutive eigenvalue spacings."""
    spacings = np.diff(eigenvalues)
    return spacings


def normalize_spacings(spacings: np.ndarray) -> np.ndarray:
    """Normalize spacings to mean 1 (unfolding)."""
    mean_spacing = np.mean(spacings)
    if mean_spacing > 0:
        return spacings / mean_spacing
    return spacings


def gue_distribution(s: np.ndarray) -> np.ndarray:
    """GUE (Gaussian Unitary Ensemble) spacing distribution.

    P(s) = (32/pi^2) * s^2 * exp(-4s^2/pi)

    This is what Riemann zeta zeros follow (Montgomery-Odlyzko).
    """
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def poisson_distribution(s: np.ndarray) -> np.ndarray:
    """Poisson spacing distribution (uncorrelated eigenvalues).

    P(s) = exp(-s)
    """
    return np.exp(-s)


def goe_distribution(s: np.ndarray) -> np.ndarray:
    """GOE (Gaussian Orthogonal Ensemble) spacing distribution.

    P(s) = (pi/2) * s * exp(-pi*s^2/4)
    """
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def analyze_spacing_distribution(spacings: np.ndarray, output_dir: Path):
    """Analyze spacing distribution and compare to RMT predictions."""
    spacings_norm = normalize_spacings(spacings)

    # Remove zeros and very small values
    spacings_norm = spacings_norm[spacings_norm > 1e-10]

    # Histogram
    s_range = np.linspace(0, 4, 200)

    plt.figure(figsize=(12, 8))

    # Plot histogram
    plt.hist(
        spacings_norm,
        bins=50,
        density=True,
        alpha=0.7,
        label="Observed spacings",
        color="steelblue",
        edgecolor="black",
    )

    # Plot theoretical distributions
    plt.plot(
        s_range,
        gue_distribution(s_range),
        "r-",
        linewidth=2,
        label="GUE (Riemann zeta zeros)",
    )
    plt.plot(s_range, goe_distribution(s_range), "g--", linewidth=2, label="GOE")
    plt.plot(
        s_range,
        poisson_distribution(s_range),
        "b:",
        linewidth=2,
        label="Poisson (uncorrelated)",
    )

    plt.xlabel("Normalized spacing s", fontsize=12)
    plt.ylabel("P(s)", fontsize=12)
    plt.title("Eigenvalue Spacing Distribution vs Random Matrix Theory", fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, 4)
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "spacing_distribution.png", dpi=150)
    plt.savefig(output_dir / "spacing_distribution.pdf")
    plt.close()

    print(f"Saved spacing distribution plot to {output_dir}")

    # Compute fit statistics
    from scipy import stats

    # KS test against GUE
    # Sample from GUE distribution for comparison
    gue_samples = sample_gue(len(spacings_norm))
    ks_stat_gue, ks_pval_gue = stats.ks_2samp(spacings_norm, gue_samples)

    # KS test against Poisson
    poisson_samples = np.random.exponential(1.0, len(spacings_norm))
    ks_stat_poisson, ks_pval_poisson = stats.ks_2samp(spacings_norm, poisson_samples)

    results = {
        "n_spacings": len(spacings_norm),
        "mean_spacing": float(np.mean(spacings_norm)),
        "std_spacing": float(np.std(spacings_norm)),
        "ks_stat_gue": float(ks_stat_gue),
        "ks_pval_gue": float(ks_pval_gue),
        "ks_stat_poisson": float(ks_stat_poisson),
        "ks_pval_poisson": float(ks_pval_poisson),
        "closer_to": "GUE" if ks_stat_gue < ks_stat_poisson else "Poisson",
    }

    return results


def sample_gue(n: int) -> np.ndarray:
    """Sample spacings from GUE distribution using rejection sampling."""
    samples = []
    max_p = gue_distribution(np.sqrt(np.pi / 8))  # Mode of GUE

    while len(samples) < n:
        s = np.random.uniform(0, 5)
        u = np.random.uniform(0, max_p * 1.1)
        if u < gue_distribution(s):
            samples.append(s)

    return np.array(samples)


def main():
    parser = argparse.ArgumentParser(description="Compute Laplacian spectrum")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="research/spectral_analysis/embeddings/embeddings.pt",
        help="Path to embeddings file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research/spectral_analysis/results",
        help="Output directory",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Kernel bandwidth for Laplacian",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Use k-NN graph (None = full graph)",
    )
    parser.add_argument(
        "--n-eigenvalues",
        type=int,
        default=1000,
        help="Number of eigenvalues to compute",
    )
    args = parser.parse_args()

    # Paths
    embeddings_path = PROJECT_ROOT / args.embeddings
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from: {embeddings_path}")
    data = torch.load(embeddings_path, weights_only=False)
    z_hyp = data["z_hyperbolic"]
    n_ops = data["n_operations"]

    print(f"Loaded {n_ops} embeddings with dimension {z_hyp.shape[1]}")

    # Compute pairwise distances
    distances = compute_pairwise_distances(z_hyp)

    print("\nDistance statistics:")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")
    print(f"  Mean: {distances.mean():.4f}")

    # Build Laplacian
    print(f"\nBuilding graph Laplacian (sigma={args.sigma})...")
    L, W, D = build_graph_laplacian(distances, sigma=args.sigma, k_neighbors=args.k_neighbors)

    print(f"Laplacian shape: {L.shape}")
    print(f"Laplacian sparsity: {(L == 0).sum().item() / L.numel():.2%}")

    # Compute eigenvalues
    k = min(args.n_eigenvalues, n_ops)
    eigenvalues = compute_eigenvalues(L, k=k, use_sparse=(k < n_ops // 2))

    print(f"\nComputed {len(eigenvalues)} eigenvalues")
    print(f"  Min: {eigenvalues.min():.6f}")
    print(f"  Max: {eigenvalues.max():.6f}")

    # Compute spacings
    spacings = compute_spacings(eigenvalues)

    print(f"\nComputed {len(spacings)} spacings")

    # Analyze distribution
    print("\nAnalyzing spacing distribution...")
    stats_results = analyze_spacing_distribution(spacings, output_dir)

    print("\n=== Spacing Analysis Results ===")
    print(f"KS statistic (vs GUE): {stats_results['ks_stat_gue']:.4f}")
    print(f"KS p-value (vs GUE): {stats_results['ks_pval_gue']:.4f}")
    print(f"KS statistic (vs Poisson): {stats_results['ks_stat_poisson']:.4f}")
    print(f"KS p-value (vs Poisson): {stats_results['ks_pval_poisson']:.4f}")
    print(f"Distribution closer to: {stats_results['closer_to']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "n_operations": n_ops,
        "latent_dim": z_hyp.shape[1],
        "sigma": args.sigma,
        "k_neighbors": args.k_neighbors,
        "n_eigenvalues": len(eigenvalues),
        "eigenvalues_min": float(eigenvalues.min()),
        "eigenvalues_max": float(eigenvalues.max()),
        "spacing_stats": stats_results,
    }

    results_file = output_dir / f"spectrum_analysis_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_file}")

    # Save eigenvalues
    np.save(output_dir / "eigenvalues.npy", eigenvalues)
    np.save(output_dir / "spacings.npy", spacings)
    np.save(output_dir / "distance_matrix.npy", distances.numpy())

    print(f"Saved eigenvalues and spacings to: {output_dir}")

    # Interpretation
    print("\n=== INTERPRETATION ===")
    if stats_results["ks_stat_gue"] < 0.1:
        print("STRONG GUE FIT: Eigenvalue spacings closely follow GUE statistics!")
        print("This suggests a deep connection to Riemann zeta zeros.")
    elif stats_results["closer_to"] == "GUE":
        print("MODERATE GUE FIT: Spacings are closer to GUE than Poisson.")
        print("This is suggestive but not conclusive.")
    else:
        print("WEAK/NO GUE FIT: Spacings do not follow GUE statistics.")
        print("May need different kernel bandwidth or more data.")

    return eigenvalues, spacings, results


if __name__ == "__main__":
    main()
