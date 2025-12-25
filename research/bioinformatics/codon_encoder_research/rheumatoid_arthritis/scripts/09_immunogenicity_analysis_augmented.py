"""
Immunogenicity Analysis with Augmented Epitope Database

Uses the codon-encoder-3-adic (trained on V5.11.3 native hyperbolic embeddings)
to analyze 57 RA epitopes and establish statistical signatures of immunodominance.

Key metrics analyzed:
1. Embedding norm (distance from ball origin)
2. Cluster homogeneity (within-epitope cluster consistency)
3. Boundary crossing potential (distance to nearest cluster boundary)
4. JS divergence upon citrullination (distribution shift)
5. Entropy change upon citrullination
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# Load augmented database
import importlib.util

# Import from local modules
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, get_results_dir,
                              load_codon_encoder, poincare_distance)

spec = importlib.util.spec_from_file_location("augmented_db", Path(__file__).parent / "08_augmented_epitope_database.py")
augmented_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmented_db)
RA_AUTOANTIGENS_AUGMENTED = augmented_db.RA_AUTOANTIGENS_AUGMENTED


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def encode_sequence(sequence: str, encoder, device="cpu") -> tuple:
    """Encode amino acid sequence to embeddings and clusters."""
    embeddings = []
    clusters = []
    codons = []

    for aa in sequence:
        codon = AA_TO_CODON.get(aa, "NNN")
        if codon == "NNN":
            continue
        codons.append(codon)

        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode(onehot).cpu().numpy().squeeze()
            cluster_id, _ = encoder.get_cluster(onehot)
            cluster_id = cluster_id.item()

        embeddings.append(emb)
        clusters.append(cluster_id)

    return np.array(embeddings), clusters, codons


def compute_embedding_norm(embeddings: np.ndarray) -> float:
    """Mean embedding norm (distance from origin in Poincare ball)."""
    norms = np.linalg.norm(embeddings, axis=1)
    return float(np.mean(norms))


def compute_cluster_homogeneity(clusters: list) -> float:
    """Fraction of positions sharing the majority cluster."""
    if not clusters:
        return 0.0
    from collections import Counter

    counts = Counter(clusters)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(clusters)


def compute_mean_neighbor_distance(embeddings: np.ndarray) -> float:
    """Mean Poincare distance between adjacent positions."""
    if len(embeddings) < 2:
        return 0.0
    distances = []
    for i in range(len(embeddings) - 1):
        d = poincare_distance(embeddings[i], embeddings[i + 1])
        distances.append(d)
    return float(np.mean(distances))


def compute_boundary_potential(embeddings: np.ndarray, cluster_centers: np.ndarray, clusters: list) -> float:
    """Mean distance to nearest different-cluster center."""
    if len(embeddings) == 0:
        return 0.0

    boundary_distances = []
    for i, (emb, cluster) in enumerate(zip(embeddings, clusters)):
        # Distance to each cluster center
        dists_to_centers = []
        for j, center in enumerate(cluster_centers):
            if j != cluster:  # Exclude own cluster
                d = poincare_distance(emb, center)
                dists_to_centers.append(d)
        if dists_to_centers:
            boundary_distances.append(min(dists_to_centers))

    return float(np.mean(boundary_distances)) if boundary_distances else 0.0


def compute_citrullination_shift(sequence: str, arg_positions: list, encoder, device="cpu") -> dict:
    """Compute embedding shift when arginine is replaced (simulating citrullination)."""
    if not arg_positions:
        return None

    # Original encoding
    orig_emb, orig_clusters, orig_codons = encode_sequence(sequence, encoder, device)
    if len(orig_emb) == 0:
        return None

    # Compute original cluster distribution
    n_clusters = 21
    orig_cluster_dist = np.zeros(n_clusters)
    for c in orig_clusters:
        if 0 <= c < n_clusters:
            orig_cluster_dist[c] += 1
    orig_cluster_dist /= orig_cluster_dist.sum() + 1e-10

    # Original centroid and entropy
    orig_centroid = np.mean(orig_emb, axis=0)
    orig_entropy = -np.sum(orig_cluster_dist * np.log(orig_cluster_dist + 1e-10))

    shifts = []
    for r_pos in arg_positions:
        if r_pos >= len(sequence):
            continue

        # Replace R with Q (glutamine - similar size, uncharged)
        modified_seq = list(sequence)
        modified_seq[r_pos] = "Q"
        modified_seq = "".join(modified_seq)

        mod_emb, mod_clusters, _ = encode_sequence(modified_seq, encoder, device)
        if len(mod_emb) == 0:
            continue

        # Modified cluster distribution
        mod_cluster_dist = np.zeros(n_clusters)
        for c in mod_clusters:
            if 0 <= c < n_clusters:
                mod_cluster_dist[c] += 1
        mod_cluster_dist /= mod_cluster_dist.sum() + 1e-10

        # Modified centroid and entropy
        mod_centroid = np.mean(mod_emb, axis=0)
        mod_entropy = -np.sum(mod_cluster_dist * np.log(mod_cluster_dist + 1e-10))

        # Compute shift metrics
        centroid_shift = poincare_distance(orig_centroid, mod_centroid)
        js_div = jensenshannon(orig_cluster_dist, mod_cluster_dist) ** 2
        entropy_change = mod_entropy - orig_entropy

        shifts.append(
            {
                "position": r_pos,
                "centroid_shift": float(centroid_shift),
                "js_divergence": float(js_div),
                "entropy_change": float(entropy_change),
            }
        )

    if not shifts:
        return None

    # Aggregate across all R positions
    return {
        "mean_centroid_shift": np.mean([s["centroid_shift"] for s in shifts]),
        "mean_js_divergence": np.mean([s["js_divergence"] for s in shifts]),
        "mean_entropy_change": np.mean([s["entropy_change"] for s in shifts]),
        "per_position": shifts,
    }


def analyze_epitope(
    epitope: dict,
    protein_info: dict,
    encoder,
    cluster_centers: np.ndarray,
    device="cpu",
) -> dict:
    """Compute all metrics for a single epitope."""
    sequence = epitope["sequence"]
    embeddings, clusters, codons = encode_sequence(sequence, encoder, device)

    if len(embeddings) == 0:
        return None

    # Basic metrics
    embedding_norm = compute_embedding_norm(embeddings)
    cluster_homogeneity = compute_cluster_homogeneity(clusters)
    mean_neighbor_dist = compute_mean_neighbor_distance(embeddings)
    boundary_potential = compute_boundary_potential(embeddings, cluster_centers, clusters)

    # Citrullination shift (only for epitopes with arginine)
    cit_shift = compute_citrullination_shift(sequence, epitope.get("arg_positions", []), encoder, device)

    return {
        "epitope_id": epitope["id"],
        "protein_id": protein_info["gene"],
        "sequence": sequence,
        "immunodominant": epitope["immunodominant"],
        "acpa_reactivity": epitope.get("acpa_reactivity", 0),
        "has_arginine": bool(epitope.get("arg_positions")),
        "n_arginines": len(epitope.get("arg_positions", [])),
        "metrics": {
            "embedding_norm": embedding_norm,
            "cluster_homogeneity": cluster_homogeneity,
            "mean_neighbor_distance": mean_neighbor_dist,
            "boundary_potential": boundary_potential,
        },
        "citrullination": cit_shift,
    }


def statistical_comparison(immunodominant: list, silent: list, metric_name: str) -> dict:
    """Compare metric between immunodominant and silent epitopes."""
    if len(immunodominant) < 2 or len(silent) < 2:
        return None

    imm_values = np.array(immunodominant)
    sil_values = np.array(silent)

    # t-test
    t_stat, p_value = stats.ttest_ind(imm_values, sil_values)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(imm_values) - 1) * np.var(imm_values, ddof=1) + (len(sil_values) - 1) * np.var(sil_values, ddof=1))
        / (len(imm_values) + len(sil_values) - 2)
    )
    cohens_d = (np.mean(imm_values) - np.mean(sil_values)) / (pooled_std + 1e-10)

    # Mann-Whitney U (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(imm_values, sil_values, alternative="two-sided")

    return {
        "metric": metric_name,
        "immunodominant_mean": float(np.mean(imm_values)),
        "immunodominant_std": float(np.std(imm_values)),
        "immunodominant_n": len(imm_values),
        "silent_mean": float(np.mean(sil_values)),
        "silent_std": float(np.std(sil_values)),
        "silent_n": len(sil_values),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_pvalue),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    print("=" * 80)
    print("IMMUNOGENICITY ANALYSIS - AUGMENTED DATASET")
    print("Using codon-encoder-3-adic (V5.11.3 native hyperbolic)")
    print("=" * 80)

    # Setup
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, mapping, _ = load_codon_encoder(device=device, version="3adic")
    cluster_centers = encoder.cluster_centers.detach().cpu().numpy()
    print(f"  Loaded encoder with {len(cluster_centers)} cluster centers")

    # =========================================================================
    # ANALYZE ALL EPITOPES
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYZING EPITOPES")
    print("=" * 80)

    all_analyses = []
    for protein_id, protein in RA_AUTOANTIGENS_AUGMENTED.items():
        print(f"\n{protein_id}: {protein['name']}")
        for epitope in protein["epitopes"]:
            analysis = analyze_epitope(epitope, protein, encoder, cluster_centers, device)
            if analysis:
                all_analyses.append(analysis)
                status = "IMM" if analysis["immunodominant"] else "SIL"
                has_r = "R+" if analysis["has_arginine"] else "R-"
                print(f"  [{status}] {epitope['id']}: {epitope['sequence'][:15]}... {has_r}")

    print(f"\n  Total epitopes analyzed: {len(all_analyses)}")

    # =========================================================================
    # STATISTICAL COMPARISONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS: IMMUNODOMINANT vs SILENT")
    print("=" * 80)

    # Separate by immunodominance
    imm_epitopes = [a for a in all_analyses if a["immunodominant"]]
    sil_epitopes = [a for a in all_analyses if not a["immunodominant"]]

    print(f"\n  Immunodominant: {len(imm_epitopes)}")
    print(f"  Silent/control: {len(sil_epitopes)}")

    # Basic metrics comparison
    basic_metrics = [
        "embedding_norm",
        "cluster_homogeneity",
        "mean_neighbor_distance",
        "boundary_potential",
    ]
    comparisons = {}

    print("\n" + "-" * 80)
    print("BASIC EMBEDDING METRICS")
    print("-" * 80)

    for metric in basic_metrics:
        imm_values = [a["metrics"][metric] for a in imm_epitopes]
        sil_values = [a["metrics"][metric] for a in sil_epitopes]

        comparison = statistical_comparison(imm_values, sil_values, metric)
        if comparison:
            comparisons[metric] = comparison
            sig = "**" if comparison["significant_001"] else ("*" if comparison["significant_005"] else "")
            print(f"\n{metric.upper()}:")
            print(
                f"  Immunodominant: {comparison['immunodominant_mean']:.4f} +/- {comparison['immunodominant_std']:.4f} (n={comparison['immunodominant_n']})"
            )
            print(f"  Silent:         {comparison['silent_mean']:.4f} +/- {comparison['silent_std']:.4f} (n={comparison['silent_n']})")
            print(f"  t = {comparison['t_statistic']:.3f}, p = {comparison['p_value']:.4f} {sig}")
            print(f"  Cohen's d = {comparison['cohens_d']:.3f}")
            print(f"  Mann-Whitney p = {comparison['mann_whitney_p']:.4f}")

    # Citrullination metrics (only for epitopes with arginine)
    print("\n" + "-" * 80)
    print("CITRULLINATION SHIFT METRICS (R+ epitopes only)")
    print("-" * 80)

    imm_with_r = [a for a in imm_epitopes if a["citrullination"] is not None]
    sil_with_r = [a for a in sil_epitopes if a["citrullination"] is not None]

    print(f"\n  Immunodominant with R: {len(imm_with_r)}")
    print(f"  Silent with R: {len(sil_with_r)}")

    cit_metrics = [
        "mean_centroid_shift",
        "mean_js_divergence",
        "mean_entropy_change",
    ]

    for metric in cit_metrics:
        imm_values = [a["citrullination"][metric] for a in imm_with_r]
        sil_values = [a["citrullination"][metric] for a in sil_with_r]

        comparison = statistical_comparison(imm_values, sil_values, metric)
        if comparison:
            comparisons[f"cit_{metric}"] = comparison
            sig = "**" if comparison["significant_001"] else ("*" if comparison["significant_005"] else "")
            print(f"\n{metric.upper()}:")
            print(
                f"  Immunodominant: {comparison['immunodominant_mean']:.4f} +/- {comparison['immunodominant_std']:.4f} (n={comparison['immunodominant_n']})"
            )
            print(f"  Silent:         {comparison['silent_mean']:.4f} +/- {comparison['silent_std']:.4f} (n={comparison['silent_n']})")
            print(f"  t = {comparison['t_statistic']:.3f}, p = {comparison['p_value']:.4f} {sig}")
            print(f"  Cohen's d = {comparison['cohens_d']:.3f}")

    # =========================================================================
    # ACPA CORRELATION
    # =========================================================================
    print("\n" + "-" * 80)
    print("CORRELATION WITH ACPA REACTIVITY")
    print("-" * 80)

    acpa_values = [a["acpa_reactivity"] for a in all_analyses]
    correlations = {}

    for metric in basic_metrics:
        metric_values = [a["metrics"][metric] for a in all_analyses]
        r, p = stats.pearsonr(acpa_values, metric_values)
        correlations[metric] = {"r": r, "p": p}
        sig = "*" if p < 0.05 else ""
        print(f"  {metric}: r = {r:.3f}, p = {p:.4f} {sig}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: SIGNIFICANT FINDINGS")
    print("=" * 80)

    significant_findings = []
    for metric, comp in comparisons.items():
        if comp["significant_005"]:
            significant_findings.append(
                {
                    "metric": metric,
                    "p_value": comp["p_value"],
                    "cohens_d": comp["cohens_d"],
                    "direction": ("higher in IMM" if comp["immunodominant_mean"] > comp["silent_mean"] else "lower in IMM"),
                }
            )

    if significant_findings:
        print(f"\n  Found {len(significant_findings)} significant differences (p < 0.05):\n")
        for finding in sorted(significant_findings, key=lambda x: x["p_value"]):
            print(f"  - {finding['metric']}: p={finding['p_value']:.4f}, d={finding['cohens_d']:.2f} ({finding['direction']})")
    else:
        print("\n  No significant differences found at p < 0.05")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output = {
        "metadata": {
            "encoder_version": "3-adic (V5.11.3)",
            "total_epitopes": len(all_analyses),
            "immunodominant_count": len(imm_epitopes),
            "silent_count": len(sil_epitopes),
            "with_arginine": len([a for a in all_analyses if a["has_arginine"]]),
        },
        "statistical_comparisons": comparisons,
        "acpa_correlations": correlations,
        "significant_findings": significant_findings,
        "epitope_analyses": all_analyses,
    }

    output_path = results_dir / "immunogenicity_analysis_augmented.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
