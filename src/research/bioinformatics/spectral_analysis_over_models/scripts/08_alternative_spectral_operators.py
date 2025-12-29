"""
08_alternative_spectral_operators.py - Test alternative spectral operators for GUE

The graph Laplacian produces Poisson statistics. But the Riemann zeta zeros
follow GUE (Gaussian Unitary Ensemble). This script tests alternative operators
that might produce GUE from the learned 3-adic structure:

1. Heat kernel: Tr(e^{-tL}) - spectral zeta function
2. Selberg-like zeta: Π(1 - e^{-sλ}) - product over eigenvalues
3. Weighted Laplacian: L with 3-adic weights on edges
4. Hyperbolic Laplacian: Using Poincaré metric
5. Multiplicative operator: Based on operation structure

Usage:
    python 08_alternative_spectral_operators.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.linalg import eigvalsh
from scipy.spatial.distance import pdist, squareform


def v3_exact(n: int) -> int:
    """Compute exact 3-adic valuation."""
    if n == 0:
        return 9
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def hyperbolic_radius_np(embeddings: np.ndarray, c: float = 1.0) -> np.ndarray:
    """V5.12.2: Compute hyperbolic distance from origin in Poincare ball."""
    sqrt_c = np.sqrt(c)
    euclidean_norms = np.linalg.norm(embeddings, axis=-1)
    clamped = np.clip(euclidean_norms * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


def poincare_distance_np(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Compute hyperbolic distance between two points."""
    x_norm_sq = np.clip(np.sum(x**2), 0, 0.999)
    y_norm_sq = np.clip(np.sum(y**2), 0, 0.999)
    diff_norm_sq = np.sum((x - y) ** 2)
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    arg = 1 + 2 * c * diff_norm_sq / (denom + 1e-10)
    return float(np.arccosh(np.clip(arg, 1.0, 1e10)))


def gue_spacing_pdf(s):
    """GUE Wigner surmise: P(s) = (32/π²) s² exp(-4s²/π)"""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def gue_spacing_cdf(s):
    """CDF of GUE Wigner surmise."""
    return 1 - np.exp(-4 * s**2 / np.pi) * (1 + 4 * s**2 / np.pi)


def poisson_spacing_cdf(s):
    """CDF of Poisson (exponential) distribution."""
    return 1 - np.exp(-s)


def compute_spacing_statistics(eigenvalues):
    """Compute normalized spacings and KS statistics."""
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 10:
        return None

    spacings = np.diff(eigenvalues)
    mean_spacing = np.mean(spacings)
    normalized = spacings / mean_spacing

    # KS tests
    ks_gue, p_gue = stats.kstest(normalized, gue_spacing_cdf)
    ks_poisson, p_poisson = stats.kstest(normalized, "expon")

    return {
        "n_eigenvalues": len(eigenvalues),
        "mean_spacing": float(mean_spacing),
        "ks_gue": float(ks_gue),
        "ks_poisson": float(ks_poisson),
        "normalized_spacings": normalized[:100].tolist(),
    }


def operator_1_weighted_laplacian(embeddings, n_samples=500):
    """Laplacian with 3-adic valuation weights on edges.

    W_ij = exp(-d_emb) × 3^{-v_3(|i-j|)}

    This combines embedding proximity with 3-adic structure.
    """
    print("\n--- Operator 1: 3-Adic Weighted Laplacian ---")

    z_B = embeddings["z_B"]
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]
    n = len(z_sample)

    # Build weight matrix
    W = np.zeros((n, n))
    sigma = 0.15

    for i in range(n):
        for j in range(i + 1, n):
            # V5.12.2: Use hyperbolic distance for Poincare ball embeddings
            emb_dist = poincare_distance_np(z_sample[i], z_sample[j])
            diff = abs(indices[i] - indices[j])
            v3 = v3_exact(diff) if diff > 0 else 9

            # Weight: embedding similarity × 3-adic weight
            padic_weight = 3.0 ** (-v3 / 2)  # Square root for symmetry
            W[i, j] = np.exp(-(emb_dist**2) / (2 * sigma**2)) * padic_weight
            W[j, i] = W[i, j]

    D = np.diag(W.sum(axis=1))
    L = D - W

    eigenvalues = eigvalsh(L)
    results = compute_spacing_statistics(eigenvalues)

    if results:
        print(f"  Eigenvalues: {results['n_eigenvalues']}")
        print(f"  KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  KS vs Poisson: {results['ks_poisson']:.4f}")
        verdict = "GUE-like" if results["ks_gue"] < results["ks_poisson"] else "Poisson-like"
        print(f"  Verdict: {verdict}")

    return results


def operator_2_hyperbolic_laplacian(embeddings, n_samples=500):
    """Laplacian using Poincaré (hyperbolic) distance.

    d_hyp(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
    """
    print("\n--- Operator 2: Hyperbolic (Poincaré) Laplacian ---")

    z_B = embeddings["z_B"]
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]
    n = len(z_sample)

    # Compute Poincaré distances
    def poincare_distance(x, y):
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        diff_norm_sq = np.sum((x - y) ** 2)

        # Clamp to avoid numerical issues
        x_norm_sq = min(x_norm_sq, 0.9999)
        y_norm_sq = min(y_norm_sq, 0.9999)

        arg = 1 + 2 * diff_norm_sq / ((1 - x_norm_sq) * (1 - y_norm_sq))
        return np.arccosh(max(arg, 1.0))

    # Build weight matrix using hyperbolic distance
    W = np.zeros((n, n))
    sigma = 2.0  # Hyperbolic distances are larger

    for i in range(n):
        for j in range(i + 1, n):
            d_hyp = poincare_distance(z_sample[i], z_sample[j])
            W[i, j] = np.exp(-(d_hyp**2) / (2 * sigma**2))
            W[j, i] = W[i, j]

    D = np.diag(W.sum(axis=1))
    L = D - W

    eigenvalues = eigvalsh(L)
    results = compute_spacing_statistics(eigenvalues)

    if results:
        print(f"  Eigenvalues: {results['n_eigenvalues']}")
        print(f"  KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  KS vs Poisson: {results['ks_poisson']:.4f}")
        verdict = "GUE-like" if results["ks_gue"] < results["ks_poisson"] else "Poisson-like"
        print(f"  Verdict: {verdict}")

    return results


def operator_3_radial_operator(embeddings, n_samples=500):
    """Diagonal operator based on radial position.

    H_ii = f(r_i) where r_i is the radius of point i.

    The radial distribution encodes 3-adic structure.
    """
    print("\n--- Operator 3: Radial Diagonal Operator ---")

    z_B = embeddings["z_B"]
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]
    # V5.12.2: Use hyperbolic radius for Poincare ball embeddings
    radii = hyperbolic_radius_np(z_sample)

    # Create diagonal operator: H_ii = -log(1 - r_i²)
    # This maps radius to hyperbolic depth
    H_diag = -np.log(1 - radii**2 + 1e-10)

    # Add small random off-diagonal for non-trivial spectrum
    np.random.seed(123)
    n = len(radii)
    noise = np.random.randn(n, n) * 0.01
    noise = (noise + noise.T) / 2  # Symmetrize

    H = np.diag(H_diag) + noise

    eigenvalues = eigvalsh(H)
    results = compute_spacing_statistics(eigenvalues)

    if results:
        print(f"  Eigenvalues: {results['n_eigenvalues']}")
        print(f"  KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  KS vs Poisson: {results['ks_poisson']:.4f}")
        verdict = "GUE-like" if results["ks_gue"] < results["ks_poisson"] else "Poisson-like"
        print(f"  Verdict: {verdict}")

    return results


def operator_4_multiplication_table(embeddings, n_samples=500):
    """Matrix based on ternary multiplication structure.

    M_ij = <z_i, z_j> × indicator(i*j < N)

    This encodes the multiplicative structure of integers.
    """
    print("\n--- Operator 4: Multiplicative Structure Matrix ---")

    z_B = embeddings["z_B"]
    n_ops = len(z_B)

    np.random.seed(42)
    indices = np.random.choice(n_ops, min(n_samples, n_ops), replace=False)
    z_sample = z_B[indices]
    n = len(indices)

    # Build multiplication-aware matrix
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            idx_i, idx_j = indices[i], indices[j]

            # Check if product is in range
            if idx_i > 0 and idx_j > 0:
                product = idx_i * idx_j
                if product < n_ops:
                    # Weight by embedding similarity of factors and product
                    z_product = z_B[product]
                    sim = np.dot(z_sample[i], z_sample[j])
                    M[i, j] = sim
                    M[j, i] = sim

    # Make it positive semi-definite by adding identity
    M = M + np.eye(n) * (np.abs(M).max() + 0.1)

    eigenvalues = eigvalsh(M)
    results = compute_spacing_statistics(eigenvalues)

    if results:
        print(f"  Eigenvalues: {results['n_eigenvalues']}")
        print(f"  KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  KS vs Poisson: {results['ks_poisson']:.4f}")
        verdict = "GUE-like" if results["ks_gue"] < results["ks_poisson"] else "Poisson-like"
        print(f"  Verdict: {verdict}")

    return results


def operator_5_heat_kernel_trace(embeddings, n_samples=500):
    """Heat kernel spectral analysis.

    Compute eigenvalues of e^{-tL} for various t, analyze traces.
    """
    print("\n--- Operator 5: Heat Kernel Analysis ---")

    z_B = embeddings["z_B"]
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]
    n = len(z_sample)

    # Build standard Laplacian
    dists = squareform(pdist(z_sample))
    sigma = np.median(dists)
    W = np.exp(-(dists**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Get Laplacian eigenvalues
    laplacian_eigs = eigvalsh(L)
    laplacian_eigs = laplacian_eigs[laplacian_eigs > 1e-10]

    # Heat kernel trace: Tr(e^{-tL}) = Σ e^{-t λ_i}
    t_values = np.logspace(-2, 2, 50)
    traces = []

    for t in t_values:
        trace = np.sum(np.exp(-t * laplacian_eigs))
        traces.append(trace)

    traces = np.array(traces)

    # The heat trace should follow Weyl law asymptotically
    # Tr(e^{-tL}) ~ (4πt)^{-d/2} Vol(M) for small t

    # Fit power law to small-t behavior
    small_t_mask = t_values < 0.5
    log_t = np.log(t_values[small_t_mask])
    log_trace = np.log(traces[small_t_mask])

    slope, intercept = np.polyfit(log_t, log_trace, 1)
    effective_dim = -2 * slope  # d/2 from Weyl law

    print(f"  Heat trace power law slope: {slope:.3f}")
    print(f"  Effective dimension: {effective_dim:.2f}")

    # Return Laplacian stats for comparison
    results = compute_spacing_statistics(laplacian_eigs)
    if results:
        results["heat_trace_slope"] = float(slope)
        results["effective_dimension"] = float(effective_dim)
        print(f"  Laplacian KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  Laplacian KS vs Poisson: {results['ks_poisson']:.4f}")

    return results


def operator_6_normalized_laplacian(embeddings, n_samples=500):
    """Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}

    Eigenvalues bounded in [0, 2], different spacing distribution.
    """
    print("\n--- Operator 6: Normalized Laplacian ---")

    z_B = embeddings["z_B"]
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]
    n = len(z_sample)

    # Build standard Laplacian
    dists = squareform(pdist(z_sample))
    sigma = np.median(dists)
    W = np.exp(-(dists**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))

    L = D - W
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigenvalues = eigvalsh(L_norm)
    results = compute_spacing_statistics(eigenvalues)

    if results:
        print(f"  Eigenvalues: {results['n_eigenvalues']}")
        print(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        print(f"  KS vs GUE: {results['ks_gue']:.4f}")
        print(f"  KS vs Poisson: {results['ks_poisson']:.4f}")
        verdict = "GUE-like" if results["ks_gue"] < results["ks_poisson"] else "Poisson-like"
        print(f"  Verdict: {verdict}")

    return results


def visualize_spacing_comparison(all_results, output_dir):
    """Compare spacing distributions across operators."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    s_range = np.linspace(0, 4, 100)
    gue_pdf = gue_spacing_pdf(s_range)
    poisson_pdf = np.exp(-s_range)

    for idx, (name, results) in enumerate(all_results.items()):
        if idx >= 6 or results is None:
            continue

        ax = axes[idx]
        spacings = np.array(results.get("normalized_spacings", []))

        if len(spacings) > 0:
            ax.hist(spacings, bins=30, density=True, alpha=0.7, label="Observed")
            ax.plot(s_range, gue_pdf, "r-", lw=2, label="GUE")
            ax.plot(s_range, poisson_pdf, "g--", lw=2, label="Poisson")

        ax.set_xlabel("Normalized spacing s")
        ax.set_ylabel("P(s)")
        ax.set_title(f'{name}\nKS_GUE={results["ks_gue"]:.3f}, KS_Poi={results["ks_poisson"]:.3f}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 4)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Eigenvalue Spacing Distributions: Alternative Operators")
    plt.tight_layout()
    plt.savefig(output_dir / "alternative_operators_comparison.png", dpi=150)
    plt.close()

    print(f"\n  Saved comparison to {output_dir}/alternative_operators_comparison.png")


def main():
    output_dir = PROJECT_ROOT / "research/spectral_analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    data = torch.load(
        PROJECT_ROOT / "research/spectral_analysis" / "embeddings" / "embeddings.pt",
        weights_only=False,
    )

    embeddings = {
        "z_B": (
            data.get("z_B_hyp", data.get("z_hyperbolic")).numpy()
            if torch.is_tensor(data.get("z_B_hyp", data.get("z_hyperbolic")))
            else data.get("z_B_hyp", data.get("z_hyperbolic"))
        ),
    }

    print(f"Loaded embeddings: shape = {embeddings['z_B'].shape}")

    print("\n" + "=" * 60)
    print("ALTERNATIVE SPECTRAL OPERATORS")
    print("=" * 60)
    print("Testing different operators for GUE statistics...")

    all_results = {}

    # Test each operator
    all_results["1_weighted_laplacian"] = operator_1_weighted_laplacian(embeddings)
    all_results["2_hyperbolic_laplacian"] = operator_2_hyperbolic_laplacian(embeddings)
    all_results["3_radial_operator"] = operator_3_radial_operator(embeddings)
    all_results["4_multiplicative"] = operator_4_multiplication_table(embeddings)
    all_results["5_heat_kernel"] = operator_5_heat_kernel_trace(embeddings)
    all_results["6_normalized_laplacian"] = operator_6_normalized_laplacian(embeddings)

    # Visualization
    visualize_spacing_comparison(all_results, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("OPERATOR COMPARISON SUMMARY")
    print("=" * 60)

    print("\n  Operator                    | KS_GUE | KS_Poisson | Verdict")
    print("  " + "-" * 65)

    best_gue = None
    best_gue_score = 1.0

    for name, results in all_results.items():
        if results is None:
            continue
        ks_gue = results["ks_gue"]
        ks_poi = results["ks_poisson"]
        verdict = "GUE-like" if ks_gue < ks_poi else "Poisson-like"
        print(f"  {name:28s} | {ks_gue:.4f} |   {ks_poi:.4f}   | {verdict}")

        if ks_gue < best_gue_score:
            best_gue_score = ks_gue
            best_gue = name

    print(f"\n  BEST GUE MATCH: {best_gue} (KS = {best_gue_score:.4f})")

    if best_gue_score < 0.2:
        print("\n  FINDING: Operator shows GUE-like statistics!")
    elif best_gue_score < 0.4:
        print("\n  FINDING: Some GUE character, but not definitive.")
    else:
        print("\n  FINDING: All operators remain Poisson-like.")
        print("  Possible reasons:")
        print("  - Single-prime embedding insufficient for GUE")
        print("  - Need different operator construction")
        print("  - True GUE requires adelic (multi-prime) structure")

    # Save results
    results_file = output_dir / "alternative_operators_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    save_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "operators": convert_numpy(all_results),
        "best_gue_operator": best_gue,
        "best_gue_score": best_gue_score,
    }

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    return all_results


if __name__ == "__main__":
    main()
