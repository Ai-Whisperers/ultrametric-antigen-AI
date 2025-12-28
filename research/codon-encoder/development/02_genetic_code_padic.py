"""
07_genetic_code_padic.py - Test if genetic code forms p-adic error-correcting structure

HYPOTHESIS: The 64→20 codon degeneracy is a p-adic error-correcting code,
and synonymous codons form geodesic balls in hyperbolic space.

Phase 1 Tests:
1.1 Compute pairwise GEODESIC (Poincare) distances for all 64 codons
1.2 Test if synonymous codons form p-adic balls (epsilon_within < epsilon_between)
1.3 Measure ultrametric property for codon triplets
1.4 Correlate amino acid radius with degeneracy

Usage:
    python 07_genetic_code_padic.py
"""

import json
import sys
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# GENETIC CODE DEFINITIONS
# =============================================================================

GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Amino acid degeneracy (number of codons)
AA_DEGENERACY = {}
for codon, aa in GENETIC_CODE.items():
    AA_DEGENERACY[aa] = AA_DEGENERACY.get(aa, 0) + 1

# =============================================================================
# CODON TO TERNARY MAPPING
# =============================================================================


def codon_to_ternary_index(codon):
    """Map codon to index in 3^9 ternary space.

    Encoding: Each nucleotide → 2 ternary digits (covers 4 bases with 3^2=9 > 4)
    A=(0,0), C=(0,1), G=(1,0), T/U=(1,1)

    3 nucleotides × 2 trits = 6 trits, padded to 9.
    """
    nuc_encoding = {
        "A": (0, 0),
        "C": (0, 1),
        "G": (1, 0),
        "T": (1, 1),
        "U": (1, 1),  # RNA
    }

    trits = []
    for nuc in codon:
        trits.extend(nuc_encoding[nuc])

    # Pad to 9 trits
    while len(trits) < 9:
        trits.append(0)

    # Convert to index in 3^9 space
    result = 0
    for i, t in enumerate(trits):
        result += t * (3 ** (8 - i))

    return result


# =============================================================================
# POINCARE GEODESIC DISTANCE
# =============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-7):
    """Compute Poincare geodesic distance between points.

    d(x,y) = (1/sqrt(c)) * arcosh(1 + 2c||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))

    Args:
        x: Points (n, dim) or (dim,)
        y: Points (n, dim) or (dim,)
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Geodesic distances
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    x_norm_sq = np.sum(x**2, axis=-1)
    y_norm_sq = np.sum(y**2, axis=-1)
    diff_norm_sq = np.sum((x - y) ** 2, axis=-1)

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = np.clip(denom, eps, None)

    arg = 1 + 2 * c * diff_norm_sq / denom
    arg = np.clip(arg, 1.0 + eps, None)

    return (1 / np.sqrt(c)) * np.arccosh(arg)


def compute_pairwise_geodesic(embeddings):
    """Compute full pairwise geodesic distance matrix."""
    n = len(embeddings)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(embeddings[i], embeddings[j])[0]
            distances[i, j] = d
            distances[j, i] = d

    return distances


# =============================================================================
# PHASE 1.1: GEODESIC DISTANCE MATRIX FOR 64 CODONS
# =============================================================================


def phase_1_1_geodesic_matrix(embeddings, codons, ternary_indices):
    """Compute 64x64 geodesic distance matrix for codons."""
    print("\n" + "=" * 70)
    print("PHASE 1.1: GEODESIC DISTANCE MATRIX")
    print("=" * 70)

    # Get embeddings for codons
    codon_embeddings = embeddings[ternary_indices]

    print(f"\n  Codon embeddings shape: {codon_embeddings.shape}")
    print("  Using Poincare geodesic distance (curvature c=1.0)")

    # Compute pairwise geodesic distances
    geodesic_matrix = compute_pairwise_geodesic(codon_embeddings)

    # Also compute Euclidean for comparison
    from scipy.spatial.distance import cdist

    euclidean_matrix = cdist(codon_embeddings, codon_embeddings, metric="euclidean")

    # Statistics
    upper_tri = np.triu_indices(64, k=1)
    geo_flat = geodesic_matrix[upper_tri]
    euc_flat = euclidean_matrix[upper_tri]

    print("\n  Geodesic distances:")
    print(f"    Mean: {geo_flat.mean():.4f}")
    print(f"    Std:  {geo_flat.std():.4f}")
    print(f"    Min:  {geo_flat.min():.4f}")
    print(f"    Max:  {geo_flat.max():.4f}")

    print("\n  Euclidean distances (for comparison):")
    print(f"    Mean: {euc_flat.mean():.4f}")
    print(f"    Std:  {euc_flat.std():.4f}")

    # Correlation between geodesic and Euclidean
    corr, p = stats.spearmanr(geo_flat, euc_flat)
    print(f"\n  Geodesic vs Euclidean correlation: r = {corr:.4f}")

    return {
        "geodesic_matrix": geodesic_matrix,
        "euclidean_matrix": euclidean_matrix,
        "codon_embeddings": codon_embeddings,
        "geo_mean": float(geo_flat.mean()),
        "geo_std": float(geo_flat.std()),
        "geo_min": float(geo_flat.min()),
        "geo_max": float(geo_flat.max()),
    }


# =============================================================================
# PHASE 1.2: P-ADIC BALL TEST FOR SYNONYMOUS CODONS
# =============================================================================


def phase_1_2_padic_balls(geodesic_matrix, codons, amino_acids):
    """Test if synonymous codons form p-adic balls.

    For each amino acid with k codons:
    - epsilon_within = max distance between any two codons for this AA
    - epsilon_between = min distance from any codon of this AA to any other AA

    SUCCESS: epsilon_within < epsilon_between (synonymous codons are closer)
    """
    print("\n" + "=" * 70)
    print("PHASE 1.2: P-ADIC BALL TEST FOR SYNONYMOUS CODONS")
    print("=" * 70)

    # Group codon indices by amino acid
    aa_to_indices = {}
    for i, (codon, aa) in enumerate(zip(codons, amino_acids)):
        if aa not in aa_to_indices:
            aa_to_indices[aa] = []
        aa_to_indices[aa].append(i)

    results = {}
    balls_valid = 0
    balls_total = 0

    print("\n  Testing p-adic ball property for each amino acid:")
    print("  (epsilon_within < epsilon_between = valid ball)")
    print()

    for aa in sorted(aa_to_indices.keys()):
        indices = aa_to_indices[aa]
        n_codons = len(indices)

        if n_codons < 2:
            # Single codon AAs (Met, Trp) - trivially a ball
            results[aa] = {
                "n_codons": n_codons,
                "epsilon_within": 0.0,
                "epsilon_between": None,
                "is_valid_ball": True,
                "margin": None,
            }
            continue

        # Compute epsilon_within: max intra-AA distance
        within_distances = []
        for i, j in combinations(indices, 2):
            within_distances.append(geodesic_matrix[i, j])
        epsilon_within = max(within_distances)

        # Compute epsilon_between: min inter-AA distance
        other_indices = [i for i in range(64) if amino_acids[i] != aa]
        between_distances = []
        for idx in indices:
            for other in other_indices:
                between_distances.append(geodesic_matrix[idx, other])
        epsilon_between = min(between_distances)

        # Is it a valid p-adic ball?
        is_valid = epsilon_within < epsilon_between
        margin = epsilon_between - epsilon_within

        if n_codons >= 2:
            balls_total += 1
            if is_valid:
                balls_valid += 1

        results[aa] = {
            "n_codons": n_codons,
            "epsilon_within": float(epsilon_within),
            "epsilon_between": float(epsilon_between),
            "is_valid_ball": bool(is_valid),
            "margin": float(margin),
        }

        status = "BALL" if is_valid else "FAIL"
        print(f"    {aa} ({n_codons} codons): eps_in={epsilon_within:.4f}, " f"eps_out={epsilon_between:.4f}, margin={margin:+.4f} [{status}]")

    # Summary
    success_rate = balls_valid / balls_total if balls_total > 0 else 0

    print("\n  SUMMARY:")
    print(f"    Valid p-adic balls: {balls_valid}/{balls_total} ({success_rate*100:.1f}%)")

    # Statistical test: is margin > 0 on average?
    margins = [r["margin"] for r in results.values() if r["margin"] is not None]
    if len(margins) >= 3:
        t_stat, p_value = stats.ttest_1samp(margins, 0)
        print(f"    Mean margin: {np.mean(margins):.4f}")
        print(f"    T-test (margin > 0): t={t_stat:.3f}, p={p_value/2:.2e} (one-tailed)")

        if p_value / 2 < 0.05 and np.mean(margins) > 0:
            print("\n  *** SIGNIFICANT: Synonymous codons form p-adic balls! ***")

    return {
        "by_aa": results,
        "balls_valid": balls_valid,
        "balls_total": balls_total,
        "success_rate": float(success_rate),
        "mean_margin": float(np.mean(margins)) if margins else None,
        "p_value": float(p_value / 2) if len(margins) >= 3 else None,
    }


# =============================================================================
# PHASE 1.3: ULTRAMETRIC PROPERTY TEST
# =============================================================================


def phase_1_3_ultrametric(geodesic_matrix, codons, n_samples=10000):
    """Test ultrametric property for codon triplets.

    Ultrametric: d(x,z) <= max(d(x,y), d(y,z)) for all x,y,z

    In a perfect p-adic space, this holds exactly.
    We sample random triplets and count violations.
    """
    print("\n" + "=" * 70)
    print("PHASE 1.3: ULTRAMETRIC PROPERTY TEST")
    print("=" * 70)

    n_codons = len(codons)
    violations = 0
    total_tested = 0
    violation_magnitudes = []

    # Sample random triplets
    np.random.seed(42)
    for _ in range(n_samples):
        i, j, k = np.random.choice(n_codons, 3, replace=False)

        d_ij = geodesic_matrix[i, j]
        d_jk = geodesic_matrix[j, k]
        d_ik = geodesic_matrix[i, k]

        # Ultrametric: d(i,k) <= max(d(i,j), d(j,k))
        max_adjacent = max(d_ij, d_jk)

        if d_ik > max_adjacent + 1e-6:  # Small tolerance
            violations += 1
            violation_magnitudes.append(d_ik - max_adjacent)

        total_tested += 1

    violation_rate = violations / total_tested

    print(f"\n  Tested {total_tested} random codon triplets")
    print(f"  Ultrametric violations: {violations} ({violation_rate*100:.2f}%)")

    if violations > 0:
        print(f"  Mean violation magnitude: {np.mean(violation_magnitudes):.4f}")
        print(f"  Max violation magnitude: {np.max(violation_magnitudes):.4f}")

    # Compare to random baseline (uniform random points would give ~33% violations)
    print("\n  Random baseline would give ~33% violations")
    print(f"  Our rate: {violation_rate*100:.2f}%")

    if violation_rate < 0.05:
        print("\n  *** STRONG ULTRAMETRIC STRUCTURE (< 5% violations) ***")
    elif violation_rate < 0.15:
        print("\n  ** Moderate ultrametric structure (< 15% violations) **")

    return {
        "violations": violations,
        "total_tested": total_tested,
        "violation_rate": float(violation_rate),
        "mean_violation_magnitude": (float(np.mean(violation_magnitudes)) if violations > 0 else 0),
        "is_strongly_ultrametric": violation_rate < 0.05,
    }


# =============================================================================
# PHASE 1.4: DEGENERACY VS RADIUS CORRELATION
# =============================================================================


def phase_1_4_degeneracy_radius(codon_embeddings, amino_acids):
    """Test if amino acid degeneracy correlates with radius.

    PREDICTION: High degeneracy (more codons) → closer to origin (more fundamental)
    """
    print("\n" + "=" * 70)
    print("PHASE 1.4: DEGENERACY VS RADIUS CORRELATION")
    print("=" * 70)

    # Compute radius for each codon
    radii = np.linalg.norm(codon_embeddings, axis=1)

    # Get degeneracy for each codon's amino acid
    degeneracies = np.array([AA_DEGENERACY[aa] for aa in amino_acids])

    # Correlation
    corr, p_value = stats.spearmanr(degeneracies, radii)

    print("\n  Degeneracy vs Radius correlation:")
    print(f"    Spearman r = {corr:.4f}")
    print(f"    p-value = {p_value:.2e}")

    # Mean radius by degeneracy level
    print("\n  Mean radius by degeneracy:")
    for deg in sorted(set(degeneracies)):
        mask = degeneracies == deg
        mean_r = radii[mask].mean()
        std_r = radii[mask].std()
        count = mask.sum()
        print(f"    Degeneracy {deg}: radius = {mean_r:.4f} +/- {std_r:.4f} (n={count})")

    # Interpretation
    if corr < -0.1 and p_value < 0.05:
        print("\n  *** CONFIRMED: High degeneracy AAs are closer to origin ***")
        print("  This supports the 'fundamental operations' hypothesis!")
    elif corr > 0.1 and p_value < 0.05:
        print("\n  OPPOSITE: High degeneracy AAs are at boundary")
    else:
        print("\n  No significant correlation found")

    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "prediction_confirmed": corr < -0.1 and p_value < 0.05,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_padic_balls(geodesic_matrix, codons, amino_acids, codon_embeddings, output_dir):
    """Visualize p-adic ball structure."""
    from sklearn.manifold import MDS

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. MDS projection of geodesic distances
    ax1 = axes[0, 0]
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    coords_2d = mds.fit_transform(geodesic_matrix)

    # Color by amino acid
    unique_aa = sorted(set(amino_acids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_aa)))
    aa_to_color = {aa: colors[i] for i, aa in enumerate(unique_aa)}

    for i, (x, y) in enumerate(coords_2d):
        aa = amino_acids[i]
        ax1.scatter(
            x,
            y,
            c=[aa_to_color[aa]],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.annotate(codons[i], (x, y), fontsize=5, alpha=0.6, ha="center", va="bottom")

    ax1.set_title("MDS of Geodesic Distances (colored by AA)", fontsize=12)
    ax1.set_xlabel("MDS 1")
    ax1.set_ylabel("MDS 2")

    # 2. Radius distribution by degeneracy
    ax2 = axes[0, 1]
    radii = np.linalg.norm(codon_embeddings, axis=1)
    degeneracies = np.array([AA_DEGENERACY[aa] for aa in amino_acids])

    for deg in sorted(set(degeneracies)):
        mask = degeneracies == deg
        ax2.scatter(
            [deg] * mask.sum(),
            radii[mask],
            alpha=0.6,
            s=50,
            label=f"Deg {deg}",
        )

    ax2.set_xlabel("Degeneracy (# codons for AA)")
    ax2.set_ylabel("Hyperbolic Radius")
    ax2.set_title("Radius vs Degeneracy", fontsize=12)
    ax2.set_xticks([1, 2, 3, 4, 6])

    # 3. Geodesic distance heatmap (sorted by AA)
    ax3 = axes[1, 0]

    # Sort codons by amino acid
    sorted_indices = sorted(range(64), key=lambda i: (amino_acids[i], codons[i]))
    sorted_matrix = geodesic_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_labels = [f"{codons[i]}({amino_acids[i]})" for i in sorted_indices]

    im = ax3.imshow(sorted_matrix, cmap="viridis", aspect="auto")
    ax3.set_title("Geodesic Distance Matrix (sorted by AA)", fontsize=12)
    plt.colorbar(im, ax=ax3, label="Geodesic Distance")

    # Add AA boundaries
    aa_counts = {}
    for i in sorted_indices:
        aa = amino_acids[i]
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    pos = 0
    for aa in sorted(aa_counts.keys()):
        pos += aa_counts[aa]
        ax3.axhline(pos - 0.5, color="white", linewidth=0.5, alpha=0.5)
        ax3.axvline(pos - 0.5, color="white", linewidth=0.5, alpha=0.5)

    # 4. Within vs Between distances histogram
    ax4 = axes[1, 1]

    aa_to_indices = {}
    for i, aa in enumerate(amino_acids):
        if aa not in aa_to_indices:
            aa_to_indices[aa] = []
        aa_to_indices[aa].append(i)

    within_distances = []
    between_distances = []

    for aa, indices in aa_to_indices.items():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            within_distances.append(geodesic_matrix[i, j])

        other_indices = [i for i in range(64) if amino_acids[i] != aa]
        for idx in indices:
            for other in other_indices[:10]:  # Sample to avoid too many
                between_distances.append(geodesic_matrix[idx, other])

    ax4.hist(
        within_distances,
        bins=30,
        alpha=0.7,
        label=f"Within AA (n={len(within_distances)})",
        density=True,
    )
    ax4.hist(
        between_distances,
        bins=30,
        alpha=0.7,
        label=f"Between AA (n={len(between_distances)})",
        density=True,
    )
    ax4.axvline(
        np.mean(within_distances),
        color="blue",
        linestyle="--",
        label=f"Within mean: {np.mean(within_distances):.3f}",
    )
    ax4.axvline(
        np.mean(between_distances),
        color="orange",
        linestyle="--",
        label=f"Between mean: {np.mean(between_distances):.3f}",
    )
    ax4.set_xlabel("Geodesic Distance")
    ax4.set_ylabel("Density")
    ax4.set_title("Within-AA vs Between-AA Distances", fontsize=12)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "genetic_code_padic_balls.png", dpi=150)
    plt.close()

    print(f"\n  Saved visualization to {output_dir}/genetic_code_padic_balls.png")


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Use local data directory instead of deprecated riemann_hypothesis_sandbox
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENETIC CODE P-ADIC STRUCTURE TEST")
    print("Phase 1: Core Hypothesis - Do synonymous codons form p-adic balls?")
    print("=" * 70)

    # Load embeddings (from 3-adic hyperbolic extraction)
    print("\nLoading hyperbolic embeddings...")
    embeddings_path = data_dir / "v5_11_3_embeddings.pt"
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        print("Run 07_extract_v5_11_3_embeddings.py first")
        return
    data = torch.load(embeddings_path, weights_only=False)

    # Use VAE-B embeddings (better hierarchy)
    z_B = data.get("z_B_hyp", data.get("z_hyperbolic"))
    if torch.is_tensor(z_B):
        z_B = z_B.numpy()

    print(f"Loaded embeddings: shape = {z_B.shape}")

    # Prepare codon data
    codons = list(GENETIC_CODE.keys())
    amino_acids = [GENETIC_CODE[c] for c in codons]
    ternary_indices = [codon_to_ternary_index(c) for c in codons]

    print(f"Mapped 64 codons to ternary indices: {min(ternary_indices)} - {max(ternary_indices)}")

    # Results container
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "genetic_code_padic_phase1",
        "model_version": "v1.1.0",
        "n_codons": 64,
        "n_amino_acids": len(set(amino_acids)),
    }

    # Phase 1.1: Geodesic distance matrix
    phase_1_1_results = phase_1_1_geodesic_matrix(z_B, codons, ternary_indices)
    results["phase_1_1"] = {
        "geo_mean": phase_1_1_results["geo_mean"],
        "geo_std": phase_1_1_results["geo_std"],
        "geo_min": phase_1_1_results["geo_min"],
        "geo_max": phase_1_1_results["geo_max"],
    }

    # Phase 1.2: P-adic ball test
    phase_1_2_results = phase_1_2_padic_balls(phase_1_1_results["geodesic_matrix"], codons, amino_acids)
    results["phase_1_2"] = {
        "balls_valid": phase_1_2_results["balls_valid"],
        "balls_total": phase_1_2_results["balls_total"],
        "success_rate": phase_1_2_results["success_rate"],
        "mean_margin": phase_1_2_results["mean_margin"],
        "p_value": phase_1_2_results["p_value"],
        "by_aa": phase_1_2_results["by_aa"],
    }

    # Phase 1.3: Ultrametric test
    phase_1_3_results = phase_1_3_ultrametric(phase_1_1_results["geodesic_matrix"], codons)
    results["phase_1_3"] = phase_1_3_results

    # Phase 1.4: Degeneracy vs radius
    phase_1_4_results = phase_1_4_degeneracy_radius(phase_1_1_results["codon_embeddings"], amino_acids)
    results["phase_1_4"] = phase_1_4_results

    # Visualization
    visualize_padic_balls(
        phase_1_1_results["geodesic_matrix"],
        codons,
        amino_acids,
        phase_1_1_results["codon_embeddings"],
        output_dir,
    )

    # Final Summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    print(
        f"""
    1.1 GEODESIC DISTANCES:
        Mean: {phase_1_1_results['geo_mean']:.4f}
        Range: [{phase_1_1_results['geo_min']:.4f}, {phase_1_1_results['geo_max']:.4f}]

    1.2 P-ADIC BALLS:
        Valid balls: {phase_1_2_results['balls_valid']}/{phase_1_2_results['balls_total']} ({phase_1_2_results['success_rate']*100:.1f}%)
        Mean margin: {phase_1_2_results['mean_margin']:.4f}
        P-value: {phase_1_2_results['p_value']:.2e}
        VERDICT: {'*** SYNONYMOUS CODONS FORM P-ADIC BALLS ***' if phase_1_2_results['p_value'] < 0.05 and phase_1_2_results['mean_margin'] > 0 else 'No significant ball structure'}

    1.3 ULTRAMETRIC PROPERTY:
        Violations: {phase_1_3_results['violations']}/{phase_1_3_results['total_tested']} ({phase_1_3_results['violation_rate']*100:.2f}%)
        VERDICT: {'*** STRONG ULTRAMETRIC ***' if phase_1_3_results['is_strongly_ultrametric'] else 'Weak/no ultrametric structure'}

    1.4 DEGENERACY-RADIUS:
        Correlation: r = {phase_1_4_results['correlation']:.4f}
        P-value: {phase_1_4_results['p_value']:.2e}
        VERDICT: {'*** HIGH DEGENERACY = NEAR ORIGIN ***' if phase_1_4_results['prediction_confirmed'] else 'No degeneracy-radius relationship'}
    """
    )

    # Overall conclusion
    n_positive = sum(
        [
            (phase_1_2_results["p_value"] < 0.05 if phase_1_2_results["p_value"] else False),
            phase_1_3_results["is_strongly_ultrametric"],
            phase_1_4_results["prediction_confirmed"],
        ]
    )

    if n_positive >= 2:
        print("=" * 70)
        print("*** OVERALL: STRONG EVIDENCE FOR P-ADIC GENETIC CODE STRUCTURE ***")
        print("=" * 70)
        results["overall_conclusion"] = "STRONG_EVIDENCE"
    elif n_positive == 1:
        print("OVERALL: Partial evidence for p-adic structure")
        results["overall_conclusion"] = "PARTIAL_EVIDENCE"
    else:
        print("OVERALL: No clear p-adic structure detected")
        results["overall_conclusion"] = "NO_EVIDENCE"

    # Save results (convert numpy/bool types)
    def convert_types(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    results_file = output_dir / "genetic_code_padic_phase1.json"
    with open(results_file, "w") as f:
        json.dump(convert_types(results), f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    main()
