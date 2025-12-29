"""
07_adelic_analysis.py - Multi-prime (adelic) structure analysis

The Riemann zeta function is fundamentally adelic:
    ζ(s) = Π_p (1 - p^(-s))^(-1)

This script explores whether combining multiple p-adic valuations
in the learned embedding space reveals deeper structure.

Hypothesis: The single-prime (p=3) embedding captures partial structure.
Multi-prime analysis might reveal:
1. Hidden correlations with other primes
2. Adelic distance patterns
3. GUE statistics emerging from prime product

Usage:
    python 07_adelic_analysis.py
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


def v_p(n: int, p: int) -> int:
    """Compute p-adic valuation of n."""
    if n == 0:
        return float("inf")
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def compute_multi_prime_valuations(n_ops: int = 19683):
    """Compute p-adic valuations for multiple primes."""
    primes = [2, 3, 5, 7, 11, 13]

    valuations = {p: [] for p in primes}
    for i in range(n_ops):
        for p in primes:
            valuations[p].append(v_p(i, p) if i > 0 else 0)

    return {p: np.array(v) for p, v in valuations.items()}


def analyze_prime_correlations(embeddings, valuations):
    """Analyze how the 3-adic embedding correlates with other prime valuations."""
    print("\n" + "=" * 60)
    print("MULTI-PRIME VALUATION ANALYSIS")
    print("=" * 60)

    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)

    results = {}

    print("\n  Correlation of radius with p-adic valuations:")
    for p, v in sorted(valuations.items()):
        # Spearman correlation
        corr, pval = stats.spearmanr(radii, v)
        results[f"corr_v{p}"] = float(corr)
        results[f"pval_v{p}"] = float(pval)

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    v_{p} (p={p}): r = {corr:.4f} {sig}")

    # The model was trained on 3-adic, so v_3 should dominate
    print("\n  Expected: v_3 dominates (model trained on 3-adic structure)")

    return results


def compute_adelic_distance(i, j, primes, weights=None):
    """Compute adelic distance between two integers.

    Adelic distance combines all p-adic distances:
    d_adelic(i,j) = Π_p |i-j|_p^w_p

    where |x|_p = p^(-v_p(x)) is the p-adic norm.
    """
    if i == j:
        return 0.0

    diff = abs(i - j)
    if weights is None:
        weights = {p: 1.0 for p in primes}

    log_dist = 0.0
    for p in primes:
        v = v_p(diff, p)
        # |diff|_p = p^(-v)
        # log|diff|_p = -v * log(p)
        log_dist += weights[p] * (-v * np.log(p))

    return np.exp(log_dist)


def analyze_adelic_structure(embeddings, n_samples=2000):
    """Compare embedding distances to adelic distances."""
    print("\n" + "=" * 60)
    print("ADELIC DISTANCE ANALYSIS")
    print("=" * 60)

    z_B = embeddings["z_B"]
    n_ops = len(z_B)

    # Sample pairs
    np.random.seed(42)
    idx_i = np.random.randint(1, n_ops, n_samples)  # Start from 1 to avoid 0
    idx_j = np.random.randint(1, n_ops, n_samples)

    primes = [2, 3, 5, 7]

    # Compute distances
    emb_dists = []
    p3_dists = []
    adelic_dists = []

    for i, j in zip(idx_i, idx_j):
        if i != j:
            # Embedding distance
            emb_dists.append(np.linalg.norm(z_B[i] - z_B[j]))

            # 3-adic distance only
            diff = abs(i - j)
            v3 = v_p(diff, 3)
            p3_dists.append(3.0 ** (-v3))

            # Adelic distance (all primes)
            adelic_dists.append(compute_adelic_distance(i, j, primes))

    emb_dists = np.array(emb_dists)
    p3_dists = np.array(p3_dists)
    adelic_dists = np.array(adelic_dists)

    # Correlations
    corr_p3, p_p3 = stats.spearmanr(emb_dists, p3_dists)
    corr_adelic, p_adelic = stats.spearmanr(emb_dists, adelic_dists)

    print("\n  Embedding distance correlations:")
    print(f"    vs 3-adic distance:  r = {corr_p3:.4f} (p = {p_p3:.2e})")
    print(f"    vs adelic distance:  r = {corr_adelic:.4f} (p = {p_adelic:.2e})")

    # Does adelic explain MORE variance than 3-adic alone?
    improvement = (corr_adelic - corr_p3) / abs(corr_p3) * 100
    print(f"\n  Adelic vs 3-adic improvement: {improvement:.1f}%")

    if corr_adelic > corr_p3:
        print("  FINDING: Adelic distance explains MORE variance than 3-adic alone!")
    else:
        print("  FINDING: 3-adic distance is sufficient (no adelic benefit)")

    return {
        "corr_p3": float(corr_p3),
        "corr_adelic": float(corr_adelic),
        "improvement_pct": float(improvement),
    }


def compute_adelic_laplacian(embeddings, primes=[2, 3, 5], n_samples=500):
    """Construct Laplacian using adelic-weighted edges."""
    print("\n" + "=" * 60)
    print("ADELIC LAPLACIAN SPECTRUM")
    print("=" * 60)

    z_B = embeddings["z_B"]

    # Sample points for tractable computation
    np.random.seed(42)
    indices = np.random.choice(len(z_B), min(n_samples, len(z_B)), replace=False)
    z_sample = z_B[indices]

    # Construct weight matrix using adelic kernel
    n = len(z_sample)
    W = np.zeros((n, n))

    sigma = 0.1  # Kernel bandwidth

    for i in range(n):
        for j in range(i + 1, n):
            # Euclidean distance in embedding
            emb_dist = np.linalg.norm(z_sample[i] - z_sample[j])

            # Adelic distance
            adelic_dist = compute_adelic_distance(indices[i], indices[j], primes)

            # Combined kernel: weight by adelic proximity
            if adelic_dist > 0:
                # Higher weight for adelic-close points
                adelic_weight = np.exp(-adelic_dist)
                W[i, j] = np.exp(-(emb_dist**2) / (2 * sigma**2)) * adelic_weight
                W[j, i] = W[i, j]

    # Compute Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Eigenvalues
    eigenvalues = eigvalsh(L)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros

    # Compute spacings
    eigenvalues = np.sort(eigenvalues)
    spacings = np.diff(eigenvalues)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing

    # Compare to GUE and Poisson
    from scipy.stats import kstest

    # GUE Wigner surmise: P(s) = (32/π²) s² exp(-4s²/π)
    def gue_cdf(s):
        return 1 - np.exp(-4 * s**2 / np.pi)

    # Poisson: P(s) = exp(-s)
    def poisson_cdf(s):
        return 1 - np.exp(-s)

    ks_gue, p_gue = kstest(normalized_spacings, gue_cdf)
    ks_poisson, p_poisson = kstest(normalized_spacings, "expon")

    print("\n  Adelic Laplacian eigenvalue spacing statistics:")
    print(f"    Number of eigenvalues: {len(eigenvalues)}")
    print(f"    Mean spacing: {mean_spacing:.6f}")
    print(f"\n  KS test vs GUE: D = {ks_gue:.4f}")
    print(f"  KS test vs Poisson: D = {ks_poisson:.4f}")

    if ks_gue < ks_poisson:
        print("\n  FINDING: Adelic Laplacian is MORE GUE-like than Poisson!")
    else:
        print("\n  FINDING: Adelic Laplacian remains Poisson-like")

    return {
        "n_eigenvalues": len(eigenvalues),
        "mean_spacing": float(mean_spacing),
        "ks_gue": float(ks_gue),
        "ks_poisson": float(ks_poisson),
        "eigenvalues": eigenvalues.tolist()[:100],  # Save first 100
        "spacings": normalized_spacings.tolist()[:100],
    }


def analyze_prime_residues(embeddings, valuations):
    """Analyze how embedding organizes by prime residue classes."""
    print("\n" + "=" * 60)
    print("PRIME RESIDUE CLASS STRUCTURE")
    print("=" * 60)

    z_B = embeddings["z_B"]
    n_ops = len(z_B)

    results = {}

    for p in [2, 3, 5, 7]:
        # Group by residue class mod p
        residues = np.array([i % p for i in range(n_ops)])

        # Compute mean radius by residue class
        print(f"\n  Mod {p} residue classes:")
        class_radii = {}
        for r in range(p):
            mask = residues == r
            radii = np.linalg.norm(z_B[mask], axis=1)
            class_radii[r] = radii.mean()
            print(f"    r ≡ {r} (mod {p}): mean radius = {radii.mean():.4f}")

        # ANOVA test
        groups = [np.linalg.norm(z_B[residues == r], axis=1) for r in range(p)]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"    ANOVA: F = {f_stat:.2f}, p = {p_val:.2e}")

        results[f"mod_{p}"] = {
            "class_radii": class_radii,
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        }

    return results


def visualize_adelic_structure(embeddings, valuations, output_dir):
    """Visualize multi-prime structure in embedding."""
    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    primes = [2, 3, 5, 7, 11, 13]

    for idx, p in enumerate(primes):
        ax = axes[idx // 3, idx % 3]

        v = valuations[p]

        # Scatter: valuation vs radius
        ax.scatter(v, radii, alpha=0.1, s=1)

        # Mean by valuation
        unique_v = sorted(set(v))
        mean_r = [radii[v == uv].mean() for uv in unique_v]
        ax.plot(unique_v, mean_r, "r-", lw=2, label="Mean")

        ax.set_xlabel(f"v_{p} (valuation)")
        ax.set_ylabel("Radius")
        ax.set_title(f"p = {p}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Embedding Radius vs p-adic Valuations (Multiple Primes)")
    plt.tight_layout()
    plt.savefig(output_dir / "adelic_structure.png", dpi=150)
    plt.close()

    print(f"\n  Saved visualization to {output_dir}/adelic_structure.png")


def euler_product_test(embeddings, valuations):
    """Test if embedding encodes Euler product structure.

    The Euler product: ζ(s) = Π_p 1/(1-p^{-s})

    We test if the embedding partition function factorizes similarly.
    """
    print("\n" + "=" * 60)
    print("EULER PRODUCT STRUCTURE TEST")
    print("=" * 60)

    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)

    # Define partition function
    def Z(beta, mask=None):
        r = radii if mask is None else radii[mask]
        return np.sum(np.exp(-beta * r))

    # Full partition function
    betas = np.linspace(0.5, 5, 20)
    Z_full = np.array([Z(b) for b in betas])

    # Product of partition functions restricted to coprime residue classes
    primes = [2, 3, 5]

    print("\n  Testing factorization: Z_full ≈ Π_p Z_p")

    for beta in [1.0, 2.0, 3.0]:
        Z_full_val = Z(beta)

        # For each prime p, compute Z restricted to p-coprime integers
        Z_product = 1.0
        for p in primes:
            mask = np.array([i % p != 0 for i in range(len(radii))])
            Z_p = Z(beta, mask)
            # Normalize
            Z_product *= Z_p / Z_full_val

        # The factorization should give something related to Euler product
        ratio = Z_product

        print(f"    β = {beta}: Z_full = {Z_full_val:.2f}, Product ratio = {ratio:.4f}")

    return {"tested": True}


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

    # Compute multi-prime valuations
    print("Computing multi-prime valuations...")
    valuations = compute_multi_prime_valuations(len(embeddings["z_B"]))

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "analysis": "adelic_multi_prime",
    }

    # Analysis 1: Prime correlations
    corr_results = analyze_prime_correlations(embeddings, valuations)
    results["prime_correlations"] = corr_results

    # Analysis 2: Adelic distance
    adelic_results = analyze_adelic_structure(embeddings)
    results["adelic_distance"] = adelic_results

    # Analysis 3: Adelic Laplacian spectrum
    laplacian_results = compute_adelic_laplacian(embeddings)
    results["adelic_laplacian"] = laplacian_results

    # Analysis 4: Prime residue classes
    residue_results = analyze_prime_residues(embeddings, valuations)
    results["prime_residues"] = residue_results

    # Analysis 5: Euler product test
    euler_results = euler_product_test(embeddings, valuations)
    results["euler_product"] = euler_results

    # Visualization
    visualize_adelic_structure(embeddings, valuations, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("ADELIC ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"""
Key Findings:

1. PRIME CORRELATIONS:
   - v_3 correlation: r = {corr_results.get('corr_v3', 'N/A'):.4f} (model trained on this)
   - v_2 correlation: r = {corr_results.get('corr_v2', 'N/A'):.4f}
   - v_5 correlation: r = {corr_results.get('corr_v5', 'N/A'):.4f}

2. ADELIC vs 3-ADIC DISTANCE:
   - 3-adic only: r = {adelic_results['corr_p3']:.4f}
   - Adelic (multi-prime): r = {adelic_results['corr_adelic']:.4f}
   - Improvement: {adelic_results['improvement_pct']:.1f}%

3. ADELIC LAPLACIAN SPECTRUM:
   - KS vs GUE: {laplacian_results['ks_gue']:.4f}
   - KS vs Poisson: {laplacian_results['ks_poisson']:.4f}
   - Verdict: {'MORE GUE-like' if laplacian_results['ks_gue'] < laplacian_results['ks_poisson'] else 'Still Poisson-like'}

INTERPRETATION:
{'The adelic structure improves upon single-prime analysis!' if adelic_results['improvement_pct'] > 5 else 'Single-prime (3-adic) structure is dominant in this embedding.'}
{'GUE statistics may emerge with full multi-prime training.' if laplacian_results['ks_gue'] < laplacian_results['ks_poisson'] else 'Full adelic VAE training needed to test GUE hypothesis.'}
"""
    )

    # Save results
    results_file = output_dir / "adelic_results.json"

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

    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    main()
