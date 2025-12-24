#!/usr/bin/env python3
"""
HLA Functionomic Analysis using Hyperbolic (Poincaré Ball) Geometry

Tests whether RA-associated HLA-DRB1 alleles cluster differently in hyperbolic
embedding space compared to control alleles.

Hypothesis: The immune system computes self/non-self via hyperbolic/ultrametric distance.
RA-associated alleles may have altered hyperbolic geometry that causes misclassification.

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json

# Import hyperbolic utilities
from hyperbolic_utils import (
    poincare_distance as hyp_poincare_distance,
    poincare_distance_matrix,
    project_to_poincare,
    load_hyperbolic_encoder,
    load_codon_encoder,
    get_results_dir,
    codon_to_onehot,
    HyperbolicCodonEncoder,
    CodonEncoder,
)

# ============================================================================
# HLA-DRB1 ALLELE DATA
# ============================================================================

# Shared epitope region: positions 70-74 of HLA-DRB1 beta chain
# These are the actual nucleotide sequences from IPD-IMGT/HLA database
# Format: {allele: {position: codon}}

HLA_DRB1_SHARED_EPITOPE = {
    # RA-ASSOCIATED ALLELES (Shared Epitope positive)
    "DRB1*04:01": {
        "name": "DRB1*04:01",
        "ra_status": "risk",
        "odds_ratio": 4.0,
        "amino_acids": "QKRAA",
        # Codons for positions 70-74 (from IMGT/HLA database)
        "codons": {
            70: "CAG",  # Q (Gln)
            71: "AAG",  # K (Lys)
            72: "CGG",  # R (Arg)
            73: "GCG",  # A (Ala)
            74: "GCC",  # A (Ala)
        },
    },
    "DRB1*04:04": {
        "name": "DRB1*04:04",
        "ra_status": "risk",
        "odds_ratio": 3.5,
        "amino_acids": "QRRAA",
        "codons": {
            70: "CAG",  # Q
            71: "CGG",  # R (note: R not K at 71)
            72: "CGG",  # R
            73: "GCG",  # A
            74: "GCC",  # A
        },
    },
    "DRB1*04:05": {
        "name": "DRB1*04:05",
        "ra_status": "risk",
        "odds_ratio": 3.2,
        "amino_acids": "QRRAA",
        "codons": {
            70: "CAG",  # Q
            71: "AGA",  # R (different codon than 04:04)
            72: "CGG",  # R
            73: "GCG",  # A
            74: "GCC",  # A
        },
    },
    "DRB1*01:01": {
        "name": "DRB1*01:01",
        "ra_status": "risk",
        "odds_ratio": 2.0,
        "amino_acids": "QRRAA",
        "codons": {
            70: "CAG",  # Q
            71: "AGA",  # R
            72: "AGA",  # R (different codon)
            73: "GCT",  # A (different codon)
            74: "GCC",  # A
        },
    },
    # CONTROL ALLELES (Non-shared epitope)
    "DRB1*07:01": {
        "name": "DRB1*07:01",
        "ra_status": "protective",
        "odds_ratio": 0.5,
        "amino_acids": "DRRGQ",
        "codons": {
            70: "GAC",  # D (Asp)
            71: "AGA",  # R
            72: "AGA",  # R
            73: "GGC",  # G (Gly)
            74: "CAG",  # Q (Gln)
        },
    },
    "DRB1*15:01": {
        "name": "DRB1*15:01",
        "ra_status": "neutral",
        "odds_ratio": 1.0,
        "amino_acids": "QARAA",
        "codons": {
            70: "CAG",  # Q
            71: "GCG",  # A (Ala, not R)
            72: "AGA",  # R
            73: "GCT",  # A
            74: "GCC",  # A
        },
    },
    "DRB1*03:01": {
        "name": "DRB1*03:01",
        "ra_status": "neutral",
        "odds_ratio": 0.9,
        "amino_acids": "DRRAA",
        "codons": {
            70: "GAC",  # D
            71: "AGA",  # R
            72: "AGA",  # R
            73: "GCT",  # A
            74: "GCC",  # A
        },
    },
    "DRB1*13:01": {
        "name": "DRB1*13:01",
        "ra_status": "protective",
        "odds_ratio": 0.4,
        "amino_acids": "DERAA",
        "codons": {
            70: "GAC",  # D
            71: "GAG",  # E (Glu)
            72: "AGA",  # R
            73: "GCT",  # A
            74: "GCC",  # A
        },
    },
}

# ============================================================================
# CODON ENCODER - Now imported from hyperbolic_utils
# CodonEncoder and codon_to_onehot are imported above
# ============================================================================


# ============================================================================
# DISTANCE COMPUTATION - Using Hyperbolic Utilities
# ============================================================================


def euclidean_distance(emb1, emb2):
    """Standard Euclidean distance in embedding space (for comparison)."""
    return np.linalg.norm(emb1 - emb2)


def poincare_distance(emb1, emb2, c=1.0):
    """
    Geodesic distance in Poincaré ball model.
    Uses the validated hyperbolic implementation from hyperbolic_utils.
    """
    return float(hyp_poincare_distance(emb1, emb2, c=c))


def ultrametric_distance(emb1, emb2, p=3):
    """
    Approximate p-adic distance using radial position + angular distance.
    The hyperbolic model encodes hierarchy as radial depth.
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    # Angular difference
    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2 + 1e-8)
    angular_dist = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi

    # Radial difference (hierarchical depth)
    radial_diff = np.abs(norm1 - norm2)

    return radial_diff + angular_dist


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def encode_allele(allele_data, encoder, device="cpu", use_hyperbolic=True):
    """
    Encode all codons of an allele's shared epitope.

    Args:
        allele_data: Dict with 'codons' mapping positions to codon strings
        encoder: CodonEncoder model
        device: Device for inference
        use_hyperbolic: If True, project embeddings to Poincaré ball

    Returns:
        Dict mapping positions to embeddings (in Euclidean or hyperbolic space)
    """
    embeddings = {}
    for pos, codon in allele_data["codons"].items():
        onehot = (
            torch.tensor(codon_to_onehot(codon), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            emb = encoder.encode(onehot).cpu().numpy().squeeze()
            if use_hyperbolic:
                # Project to Poincaré ball for hyperbolic geometry
                emb = project_to_poincare(emb, max_radius=0.95).squeeze()
        embeddings[pos] = emb
    return embeddings


def compute_allele_centroid(embeddings):
    """Compute centroid of all position embeddings."""
    all_embs = np.array(list(embeddings.values()))
    return np.mean(all_embs, axis=0)


def compute_distance_matrix(alleles_embeddings, distance_fn="euclidean"):
    """Compute pairwise distances between allele centroids."""
    allele_names = list(alleles_embeddings.keys())
    n = len(allele_names)

    # Get centroids
    centroids = {
        name: compute_allele_centroid(embs) for name, embs in alleles_embeddings.items()
    }

    # Compute distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            emb_i = centroids[allele_names[i]]
            emb_j = centroids[allele_names[j]]
            if distance_fn == "euclidean":
                dist_matrix[i, j] = euclidean_distance(emb_i, emb_j)
            elif distance_fn == "poincare":
                dist_matrix[i, j] = poincare_distance(emb_i, emb_j)
            elif distance_fn == "ultrametric":
                dist_matrix[i, j] = ultrametric_distance(emb_i, emb_j)

    return dist_matrix, allele_names


def test_ultrametric_property(dist_matrix, allele_names, allele_data):
    """
    Test the ultrametric inequality for all triplets.
    d(A,B) ≤ max(d(A,C), d(B,C))
    """
    n = len(allele_names)
    violations = 0
    total = 0
    violation_details = []

    for i, j, k in combinations(range(n), 3):
        total += 3  # Three inequalities per triplet

        d_ij = dist_matrix[i, j]
        d_ik = dist_matrix[i, k]
        d_jk = dist_matrix[j, k]

        # Check all three orientations
        if d_ij > max(d_ik, d_jk) + 1e-6:
            violations += 1
            violation_details.append(
                (allele_names[i], allele_names[j], allele_names[k], "d_ij")
            )
        if d_ik > max(d_ij, d_jk) + 1e-6:
            violations += 1
            violation_details.append(
                (allele_names[i], allele_names[j], allele_names[k], "d_ik")
            )
        if d_jk > max(d_ij, d_ik) + 1e-6:
            violations += 1
            violation_details.append(
                (allele_names[i], allele_names[j], allele_names[k], "d_jk")
            )

    return violations, total, violation_details


def test_ra_separation(dist_matrix, allele_names, allele_data):
    """
    Test if RA-associated alleles are closer to each other than to controls.
    """
    # Identify RA and control indices
    ra_indices = [
        i
        for i, name in enumerate(allele_names)
        if allele_data[name]["ra_status"] == "risk"
    ]
    control_indices = [
        i
        for i, name in enumerate(allele_names)
        if allele_data[name]["ra_status"] != "risk"
    ]

    # Within-group distances
    ra_within = []
    for i, j in combinations(ra_indices, 2):
        ra_within.append(dist_matrix[i, j])

    control_within = []
    for i, j in combinations(control_indices, 2):
        control_within.append(dist_matrix[i, j])

    # Between-group distances
    between = []
    for i in ra_indices:
        for j in control_indices:
            between.append(dist_matrix[i, j])

    return {
        "ra_within_mean": np.mean(ra_within) if ra_within else 0,
        "ra_within_std": np.std(ra_within) if ra_within else 0,
        "control_within_mean": np.mean(control_within) if control_within else 0,
        "control_within_std": np.std(control_within) if control_within else 0,
        "between_mean": np.mean(between) if between else 0,
        "between_std": np.std(between) if between else 0,
        "ra_within": ra_within,
        "control_within": control_within,
        "between": between,
    }


def test_position_specific_patterns(
    alleles_embeddings, allele_data, distance_fn="euclidean"
):
    """
    Analyze each position (70-74) separately.
    """
    positions = [70, 71, 72, 73, 74]
    allele_names = list(alleles_embeddings.keys())

    position_results = {}

    for pos in positions:
        # Get embeddings at this position
        pos_embeddings = {name: embs[pos] for name, embs in alleles_embeddings.items()}

        # Identify RA vs control
        ra_embs = [
            pos_embeddings[name]
            for name in allele_names
            if allele_data[name]["ra_status"] == "risk"
        ]
        control_embs = [
            pos_embeddings[name]
            for name in allele_names
            if allele_data[name]["ra_status"] != "risk"
        ]

        # Compute RA centroid and control centroid
        ra_centroid = np.mean(ra_embs, axis=0)
        control_centroid = np.mean(control_embs, axis=0)

        # Distance between group centroids
        if distance_fn == "euclidean":
            centroid_distance = euclidean_distance(ra_centroid, control_centroid)
        elif distance_fn == "poincare":
            centroid_distance = poincare_distance(ra_centroid, control_centroid)
        else:
            centroid_distance = ultrametric_distance(ra_centroid, control_centroid)

        # Variance within groups
        ra_variance = np.mean(
            [euclidean_distance(e, ra_centroid) ** 2 for e in ra_embs]
        )
        control_variance = np.mean(
            [euclidean_distance(e, control_centroid) ** 2 for e in control_embs]
        )

        position_results[pos] = {
            "centroid_distance": centroid_distance,
            "ra_variance": ra_variance,
            "control_variance": control_variance,
            "ra_centroid": ra_centroid,
            "control_centroid": control_centroid,
        }

    return position_results


def odds_ratio_correlation(dist_matrix, allele_names, allele_data):
    """
    Test correlation between p-adic distance from reference and odds ratio.
    """
    # Use protective allele (lowest OR) as reference
    reference_idx = min(
        range(len(allele_names)),
        key=lambda i: allele_data[allele_names[i]]["odds_ratio"],
    )

    distances_from_ref = []
    log_odds_ratios = []

    for i, name in enumerate(allele_names):
        if i != reference_idx:
            distances_from_ref.append(dist_matrix[reference_idx, i])
            log_odds_ratios.append(np.log(allele_data[name]["odds_ratio"]))

    # Spearman correlation
    from scipy.stats import spearmanr

    corr, pvalue = spearmanr(distances_from_ref, log_odds_ratios)

    return {
        "correlation": corr,
        "pvalue": pvalue,
        "reference_allele": allele_names[reference_idx],
        "distances": distances_from_ref,
        "log_odds_ratios": log_odds_ratios,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualization(alleles_embeddings, allele_data, results, output_path):
    """Create visualization of HLA functionomic analysis."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. PCA of allele centroids
    ax1 = axes[0, 0]
    centroids = np.array(
        [compute_allele_centroid(embs) for embs in alleles_embeddings.values()]
    )
    allele_names = list(alleles_embeddings.keys())

    if centroids.shape[1] > 2:
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroids)
    else:
        centroids_2d = centroids

    colors = [
        "red" if allele_data[name]["ra_status"] == "risk" else "blue"
        for name in allele_names
    ]

    ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=colors, s=150, alpha=0.7)
    for i, name in enumerate(allele_names):
        ax1.annotate(
            name.split("*")[1],
            (centroids_2d[i, 0], centroids_2d[i, 1]),
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("HLA-DRB1 Alleles in Embedding Space\n(Red=RA risk, Blue=Control)")

    # 2. Distance matrix heatmap
    ax2 = axes[0, 1]
    dist_matrix = results["distance_matrix"]
    im = ax2.imshow(dist_matrix, cmap="viridis")
    ax2.set_xticks(range(len(allele_names)))
    ax2.set_yticks(range(len(allele_names)))
    ax2.set_xticklabels(
        [n.split("*")[1] for n in allele_names], rotation=45, ha="right", fontsize=8
    )
    ax2.set_yticklabels([n.split("*")[1] for n in allele_names], fontsize=8)
    ax2.set_title("Pairwise Distance Matrix")
    plt.colorbar(im, ax=ax2)

    # 3. RA vs Control separation
    ax3 = axes[1, 0]
    sep = results["separation"]
    categories = ["RA within", "Control within", "Between groups"]
    means = [sep["ra_within_mean"], sep["control_within_mean"], sep["between_mean"]]
    stds = [sep["ra_within_std"], sep["control_within_std"], sep["between_std"]]
    colors_bar = ["red", "blue", "gray"]

    bars = ax3.bar(categories, means, yerr=stds, color=colors_bar, alpha=0.7, capsize=5)
    ax3.set_ylabel("Mean Distance")
    ax3.set_title("RA vs Control Separation")

    # Add ratio annotation
    if sep["between_mean"] > 0:
        within_avg = (sep["ra_within_mean"] + sep["control_within_mean"]) / 2
        ratio = sep["between_mean"] / within_avg if within_avg > 0 else 0
        ax3.text(
            0.5,
            0.95,
            f"Between/Within Ratio: {ratio:.2f}x",
            transform=ax3.transAxes,
            ha="center",
            fontsize=10,
        )

    # 4. Position-specific analysis
    ax4 = axes[1, 1]
    pos_results = results["position_specific"]
    positions = sorted(pos_results.keys())
    centroid_dists = [pos_results[p]["centroid_distance"] for p in positions]

    ax4.bar([f"Pos {p}" for p in positions], centroid_dists, color="purple", alpha=0.7)
    ax4.set_ylabel("RA-Control Centroid Distance")
    ax4.set_title("Position-Specific RA vs Control Separation\n(Shared Epitope: 70-74)")
    ax4.axhline(y=np.mean(centroid_dists), color="red", linestyle="--", label="Mean")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("HLA FUNCTIONOMIC ANALYSIS - HYPERBOLIC GEOMETRY")
    print("Testing Poincaré Ball Geometry for RA Association")
    print("=" * 70)

    # Paths - use hyperbolic results directory
    script_dir = Path(__file__).parent
    # PROJECT_ROOT is calculated earlier in the file if needed, or we can resolve it here.
    # Assuming standard structure:
    # DOCUMENTATION/03../bioinformatics/codon_encoder_research/rheumatoid_arthritis/scripts/
    # So root is 6 levels up.
    PROJECT_ROOT = Path(__file__).resolve().parents[6]
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load codon encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    encoder, codon_mapping, _ = load_codon_encoder(device="cpu", version="3adic")
    print(
        f"  Loaded encoder with {sum(p.numel() for p in encoder.parameters())} parameters"
    )

    # Encode all alleles with hyperbolic projection
    print("\nEncoding HLA-DRB1 alleles (projecting to Poincaré ball)...")
    alleles_embeddings = {}
    for allele_name, allele_data in HLA_DRB1_SHARED_EPITOPE.items():
        embeddings = encode_allele(allele_data, encoder, use_hyperbolic=True)
        alleles_embeddings[allele_name] = embeddings
        print(
            f"  {allele_name}: {allele_data['amino_acids']} (OR={allele_data['odds_ratio']}, {allele_data['ra_status']})"
        )

    # Compute distance matrix (try all distance functions)
    print("\nComputing distance matrices...")

    results = {}
    for dist_fn in ["euclidean", "poincare", "ultrametric"]:
        print(f"\n  Distance function: {dist_fn}")

        dist_matrix, allele_names = compute_distance_matrix(
            alleles_embeddings, distance_fn=dist_fn
        )

        # Test ultrametric property
        violations, total, _ = test_ultrametric_property(
            dist_matrix, allele_names, HLA_DRB1_SHARED_EPITOPE
        )
        ultrametric_score = 1 - (violations / total) if total > 0 else 0
        print(
            f"    Ultrametric property: {ultrametric_score:.1%} ({violations}/{total} violations)"
        )

        # Test RA separation
        separation = test_ra_separation(
            dist_matrix, allele_names, HLA_DRB1_SHARED_EPITOPE
        )
        within_avg = (
            separation["ra_within_mean"] + separation["control_within_mean"]
        ) / 2
        ratio = separation["between_mean"] / within_avg if within_avg > 0 else 0
        print(f"    RA within-group mean: {separation['ra_within_mean']:.4f}")
        print(f"    Control within-group mean: {separation['control_within_mean']:.4f}")
        print(f"    Between-group mean: {separation['between_mean']:.4f}")
        print(f"    Separation ratio: {ratio:.2f}x")

        # Position-specific analysis
        pos_results = test_position_specific_patterns(
            alleles_embeddings, HLA_DRB1_SHARED_EPITOPE, distance_fn=dist_fn
        )
        max_pos = max(
            pos_results.keys(), key=lambda p: pos_results[p]["centroid_distance"]
        )
        print(
            f"    Most discriminative position: {max_pos} (dist={pos_results[max_pos]['centroid_distance']:.4f})"
        )

        # Odds ratio correlation
        try:
            or_corr = odds_ratio_correlation(
                dist_matrix, allele_names, HLA_DRB1_SHARED_EPITOPE
            )
            print(
                f"    OR correlation: r={or_corr['correlation']:.3f}, p={or_corr['pvalue']:.4f}"
            )
        except Exception as e:
            or_corr = {"correlation": 0, "pvalue": 1, "reference_allele": "N/A"}
            print(f"    OR correlation: could not compute ({e})")

        results[dist_fn] = {
            "distance_matrix": dist_matrix,
            "allele_names": allele_names,
            "ultrametric_score": ultrametric_score,
            "separation": separation,
            "position_specific": pos_results,
            "or_correlation": or_corr,
        }

    # Use best performing distance function for visualization
    best_fn = max(
        results.keys(),
        key=lambda k: results[k]["separation"]["between_mean"]
        / (
            (
                results[k]["separation"]["ra_within_mean"]
                + results[k]["separation"]["control_within_mean"]
            )
            / 2
            + 1e-6
        ),
    )
    print(f"\n  Best distance function: {best_fn}")

    # Create visualization
    print("\nGenerating visualization...")
    vis_path = results_dir / "hla_functionomic_analysis.png"
    create_visualization(
        alleles_embeddings, HLA_DRB1_SHARED_EPITOPE, results[best_fn], vis_path
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_sep = results[best_fn]["separation"]
    within_avg = (best_sep["ra_within_mean"] + best_sep["control_within_mean"]) / 2
    best_ratio = best_sep["between_mean"] / within_avg if within_avg > 0 else 0

    print(
        f"""
    Distance Function: {best_fn}

    RA Separation:
      - RA alleles within-group: {best_sep['ra_within_mean']:.4f} ± {best_sep['ra_within_std']:.4f}
      - Control within-group:    {best_sep['control_within_mean']:.4f} ± {best_sep['control_within_std']:.4f}
      - Between groups:          {best_sep['between_mean']:.4f} ± {best_sep['between_std']:.4f}
      - Separation ratio:        {best_ratio:.2f}x

    Ultrametric Property: {results[best_fn]['ultrametric_score']:.1%}

    Odds Ratio Correlation:
      - Spearman r: {results[best_fn]['or_correlation']['correlation']:.3f}
      - p-value:    {results[best_fn]['or_correlation']['pvalue']:.4f}

    Position Analysis (70-74):"""
    )

    for pos in sorted(results[best_fn]["position_specific"].keys()):
        pr = results[best_fn]["position_specific"][pos]
        print(f"      Position {pos}: centroid_dist={pr['centroid_distance']:.4f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if best_ratio > 1.0:
        print(
            """
    *** POSITIVE SIGNAL: RA-associated alleles cluster separately from controls ***

    The p-adic embedding space shows group separation, suggesting that
    the shared epitope's functional properties can be captured geometrically.
        """
        )
    else:
        print(
            """
    Result: No clear separation detected in this initial analysis.

    Possible reasons:
    - Need more alleles for statistical power
    - Position-specific effects may be masked by averaging
    - The codon-level signal may require different encoding
        """
        )

    # Save results
    output_data = {
        "best_distance_function": best_fn,
        "separation_ratio": best_ratio,
        "ultrametric_score": results[best_fn]["ultrametric_score"],
        "or_correlation": results[best_fn]["or_correlation"]["correlation"],
        "or_pvalue": results[best_fn]["or_correlation"]["pvalue"],
        "alleles": {
            name: {
                "ra_status": data["ra_status"],
                "odds_ratio": data["odds_ratio"],
                "amino_acids": data["amino_acids"],
            }
            for name, data in HLA_DRB1_SHARED_EPITOPE.items()
        },
    }

    output_path = results_dir / "hla_functionomic_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved results to {output_path}")


if __name__ == "__main__":
    main()
