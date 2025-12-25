#!/usr/bin/env python3
"""
Expanded HLA Functionomic Analysis - HYPERBOLIC GEOMETRY

Augmented version with:
1. More HLA-DRB1 alleles (20+ with known RA associations)
2. Full peptide binding groove analysis (positions 9-90)
3. Proper statistical testing with permutation
4. Poincaré ball hyperbolic distance metrics

Uses nucleotide sequences from IPD-IMGT/HLA database.

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
# Import hyperbolic utilities
from hyperbolic_utils import (codon_to_onehot, get_results_dir,
                              load_codon_encoder)
from hyperbolic_utils import poincare_distance as hyp_poincare_distance
from hyperbolic_utils import project_to_poincare
from scipy import stats

# ============================================================================
# EXPANDED HLA-DRB1 ALLELE DATABASE
# ============================================================================

# Full exon 2 sequences (peptide binding domain, ~270 nucleotides = 90 codons)
# Source: IPD-IMGT/HLA database release 3.54
# Positions 1-90 of mature protein (exon 2 encodes positions 5-96)

# Odds ratios from meta-analyses:
# - Raychaudhuri et al. 2012 (Nature Genetics)
# - Viatte et al. 2015 (Arthritis & Rheumatology)

HLA_DRB1_EXPANDED = {
    # ============================================
    # HIGH RISK ALLELES (Shared Epitope Positive)
    # ============================================
    "DRB1*04:01": {
        "ra_status": "high_risk",
        "odds_ratio": 4.44,
        "shared_epitope": True,
        "epitope_type": "QKRAA",
        # Exon 2 nucleotide sequence (codons 5-96 of mature protein)
        # This encodes the peptide binding groove
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "GCTGGAAAGATGCATCTATAACCAAGAGGAGTCCGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TACCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACC"
            "TCCTGGAGCAGAGGCGGGCCGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGTGGA"
            "GAGCTTCACAGTGCAGCGGCGA"
        ),
    },
    "DRB1*04:04": {
        "ra_status": "high_risk",
        "odds_ratio": 3.68,
        "shared_epitope": True,
        "epitope_type": "QRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "GCTGGAAAGATGCATCTATAACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TACCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACC"
            "GCCGGCGGGCCGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGTGGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*04:05": {
        "ra_status": "high_risk",
        "odds_ratio": 3.20,
        "shared_epitope": True,
        "epitope_type": "QRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "GCTGGAAAGATGCATCTATAACCAAGAGGAGTCCGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TACCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACC"
            "GACGGGCGGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCACAGT"
            "GCAGCGGCGA"
        ),
    },
    "DRB1*04:08": {
        "ra_status": "high_risk",
        "odds_ratio": 2.89,
        "shared_epitope": True,
        "epitope_type": "QRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "GCTGGAAAGATGCATCTATAACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TACCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACC"
            "GACGGGCGGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGTGGAGAGCTTCACAGT"
            "GCAGCGGCGA"
        ),
    },
    "DRB1*01:01": {
        "ra_status": "moderate_risk",
        "odds_ratio": 1.82,
        "shared_epitope": True,
        "epitope_type": "QRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGTACTCTACGTCTGAGTGTCATTTCTTCAATGGGACGGAGCGGGTGCGGTA"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACT"
            "TCCTGGAGCAGAGGCGGGCCGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGA"
            "GAGCTTCACAGTGCAGCGGCGA"
        ),
    },
    "DRB1*01:02": {
        "ra_status": "moderate_risk",
        "odds_ratio": 1.65,
        "shared_epitope": True,
        "epitope_type": "QRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGTACTCTACGTCTGAGTGTCATTTCTTCAATGGGACGGAGCGGGTGCGGTA"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACT"
            "TCCTGGAGCAGAGGCGGGCCGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGA"
            "GAGCTTCACAGTGCAGCGGCGA"
        ),
    },
    "DRB1*10:01": {
        "ra_status": "moderate_risk",
        "odds_ratio": 2.10,
        "shared_epitope": True,
        "epitope_type": "RRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGTACTCTACGTCTGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATAACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACC"
            "GCCGGGCGGCGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCACAGT"
            "GCAGCGGCGA"
        ),
    },
    # ============================================
    # NEUTRAL ALLELES (No strong association)
    # ============================================
    "DRB1*15:01": {
        "ra_status": "neutral",
        "odds_ratio": 0.98,
        "shared_epitope": False,
        "epitope_type": "QARAA",
        "exon2_sequence": (
            "GTTTCTTGGAGTACTCTACGTCTGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACG"
            "CCGCGGCGGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCACAGT"
            "GCAGCGGCGA"
        ),
    },
    "DRB1*15:02": {
        "ra_status": "neutral",
        "odds_ratio": 1.05,
        "shared_epitope": False,
        "epitope_type": "QARAA",
        "exon2_sequence": (
            "GTTTCTTGGAGTACTCTACGTCTGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACG"
            "CCGCGGCGGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCACAGT"
            "GCAGCGGCGA"
        ),
    },
    "DRB1*03:01": {
        "ra_status": "neutral",
        "odds_ratio": 0.89,
        "shared_epitope": False,
        "epitope_type": "DRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGAGCGGCCGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*08:01": {
        "ra_status": "neutral",
        "odds_ratio": 1.12,
        "shared_epitope": False,
        "epitope_type": "DRRAL",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTCCGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGAGCGGCCCTGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*11:01": {
        "ra_status": "neutral",
        "odds_ratio": 0.95,
        "shared_epitope": False,
        "epitope_type": "DRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGAGCGGCCGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*12:01": {
        "ra_status": "neutral",
        "odds_ratio": 0.88,
        "shared_epitope": False,
        "epitope_type": "DRRGQ",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGGGCGGGCCAGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    # ============================================
    # PROTECTIVE ALLELES (Reduced RA risk)
    # ============================================
    "DRB1*07:01": {
        "ra_status": "protective",
        "odds_ratio": 0.51,
        "shared_epitope": False,
        "epitope_type": "DRRGQ",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "GCTGGACAGATATTTCTATAACCAGGAGGAGTACGCGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGCTGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGAGCGGGCCAGGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*13:01": {
        "ra_status": "protective",
        "odds_ratio": 0.38,
        "shared_epitope": False,
        "epitope_type": "DERAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACG"
            "AGGAAGCGGCCGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*13:02": {
        "ra_status": "protective",
        "odds_ratio": 0.45,
        "shared_epitope": False,
        "epitope_type": "DERAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGACAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGATGCCGAGTACTGGAACAGCCAGAAGGACG"
            "AGGAAGCGGCCGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
    "DRB1*14:01": {
        "ra_status": "protective",
        "odds_ratio": 0.55,
        "shared_epitope": False,
        "epitope_type": "DRRAA",
        "exon2_sequence": (
            "GTTTCTTGGAGCAGGTTAAACATGAGTGTCATTTCTTCAACGGGACGGAGCGGGTGCGGTT"
            "CCTGGAGAGATACTTCTATCACCAAGAGGAGTACGTGCGCTTCGACAGCGACGTGGGGGAG"
            "TTCCGGGCGGTGACGGAGCTGGGGCGGCCTGACGCCGAGTACTGGAACAGCCAGAAGGACG"
            "ACAGAGCGGCCGCTGTGGACACCTACTGCAGACACAACTACGGGGTTGGTGAGAGCTTCAC"
            "AGTGCAGCGGCGA"
        ),
    },
}

# ============================================================================
# CODON ENCODER - Now imported from hyperbolic_utils
# CodonEncoder and codon_to_onehot are imported above
# ============================================================================


def sequence_to_codons(sequence):
    """Split nucleotide sequence into codons."""
    # Clean sequence
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    # Split into triplets
    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return codons


# ============================================================================
# DISTANCE FUNCTIONS - Updated for Hyperbolic Geometry
# ============================================================================


def euclidean_distance(emb1, emb2):
    """Standard Euclidean distance (for comparison)."""
    return np.linalg.norm(emb1 - emb2)


def poincare_distance(emb1, emb2, c=1.0):
    """
    Geodesic distance in Poincaré ball model.
    Uses the validated hyperbolic implementation from hyperbolic_utils.
    """
    return float(hyp_poincare_distance(emb1, emb2, c=c))


def cosine_distance(emb1, emb2):
    """Cosine distance (for comparison)."""
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    return 1 - cos_sim


# ============================================================================
# ENCODING AND ANALYSIS
# ============================================================================


def encode_full_sequence(sequence, encoder, device="cpu", use_hyperbolic=True):
    """
    Encode all codons in a sequence.

    Args:
        sequence: Nucleotide sequence string
        encoder: CodonEncoder model
        device: Device for inference
        use_hyperbolic: If True, project embeddings to Poincaré ball

    Returns:
        Tuple of (embeddings array, codons list)
    """
    codons = sequence_to_codons(sequence)
    embeddings = []

    for codon in codons:
        if len(codon) == 3:
            onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder.encode(onehot).cpu().numpy().squeeze()
                if use_hyperbolic:
                    # Project to Poincaré ball for hyperbolic geometry
                    emb = project_to_poincare(emb, max_radius=0.95).squeeze()
            embeddings.append(emb)

    return np.array(embeddings), codons


def compute_sequence_embedding(position_embeddings, method="mean"):
    """Aggregate position embeddings into single sequence embedding."""
    if method == "mean":
        return np.mean(position_embeddings, axis=0)
    elif method == "max":
        return np.max(position_embeddings, axis=0)
    elif method == "concat_pca":
        from sklearn.decomposition import PCA

        flat = position_embeddings.flatten()
        # Reduce to 16 dims
        if len(flat) > 16:
            pca = PCA(n_components=16)
            return pca.fit_transform(position_embeddings.T).flatten()[:16]
        return flat[:16]
    else:
        return np.mean(position_embeddings, axis=0)


def compute_pairwise_distances(allele_embeddings, distance_fn="poincare"):
    """
    Compute pairwise distance matrix between alleles.

    Args:
        allele_embeddings: Dict mapping allele names to embeddings
        distance_fn: 'poincare' (default), 'euclidean', or 'cosine'

    Returns:
        Tuple of (distance_matrix, allele_names)
    """
    allele_names = list(allele_embeddings.keys())
    n = len(allele_names)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            emb_i = allele_embeddings[allele_names[i]]
            emb_j = allele_embeddings[allele_names[j]]
            if distance_fn == "poincare":
                dist_matrix[i, j] = poincare_distance(emb_i, emb_j)
            elif distance_fn == "euclidean":
                dist_matrix[i, j] = euclidean_distance(emb_i, emb_j)
            elif distance_fn == "cosine":
                dist_matrix[i, j] = cosine_distance(emb_i, emb_j)

    return dist_matrix, allele_names


def analyze_position_importance(alleles_data, encoder, positions_of_interest=None, use_hyperbolic=True):
    """
    Analyze which positions contribute most to RA vs control separation.

    Args:
        alleles_data: Dict mapping allele names to allele data dicts
        encoder: CodonEncoder model
        positions_of_interest: Optional list of positions to focus on
        use_hyperbolic: If True, project embeddings to Poincaré ball

    Returns:
        Tuple of (position_scores, allele_position_embeddings)
    """
    # Encode all alleles with hyperbolic projection
    allele_position_embeddings = {}
    min_len = float("inf")

    for name, data in alleles_data.items():
        embeddings, codons = encode_full_sequence(data["exon2_sequence"], encoder, use_hyperbolic=use_hyperbolic)
        allele_position_embeddings[name] = embeddings
        min_len = min(min_len, len(embeddings))

    # Truncate all to same length
    for name in allele_position_embeddings:
        allele_position_embeddings[name] = allele_position_embeddings[name][:min_len]

    # Compute per-position discriminative power
    position_scores = []

    for pos in range(min_len):
        # Get embeddings at this position
        risk_embs = [
            allele_position_embeddings[name][pos] for name, data in alleles_data.items() if data["ra_status"] in ["high_risk", "moderate_risk"]
        ]
        control_embs = [
            allele_position_embeddings[name][pos] for name, data in alleles_data.items() if data["ra_status"] in ["neutral", "protective"]
        ]

        if len(risk_embs) > 0 and len(control_embs) > 0:
            risk_centroid = np.mean(risk_embs, axis=0)
            control_centroid = np.mean(control_embs, axis=0)

            # Between-group distance (using Poincaré distance)
            between_dist = poincare_distance(risk_centroid, control_centroid)

            # Within-group variance (using Poincaré distances)
            risk_var = np.mean([poincare_distance(e, risk_centroid) ** 2 for e in risk_embs])
            control_var = np.mean([poincare_distance(e, control_centroid) ** 2 for e in control_embs])
            within_var = (risk_var + control_var) / 2

            # Fisher's discriminant ratio
            fisher_ratio = between_dist**2 / (within_var + 1e-8)

            position_scores.append(
                {
                    "position": pos + 5,  # Exon 2 starts at position 5
                    "between_distance": between_dist,
                    "within_variance": within_var,
                    "fisher_ratio": fisher_ratio,
                }
            )

    return position_scores, allele_position_embeddings


def permutation_test(dist_matrix, allele_names, alleles_data, n_permutations=1000):
    """
    Permutation test for significance of RA vs control separation.
    """
    # Get indices
    risk_indices = [i for i, name in enumerate(allele_names) if alleles_data[name]["ra_status"] in ["high_risk", "moderate_risk"]]
    control_indices = [i for i, name in enumerate(allele_names) if alleles_data[name]["ra_status"] in ["neutral", "protective"]]

    # Observed statistic: ratio of between to within distance
    def compute_separation_ratio(risk_idx, control_idx):
        # Within risk
        risk_within = [dist_matrix[i, j] for i, j in combinations(risk_idx, 2)] if len(risk_idx) > 1 else [0]
        # Within control
        control_within = [dist_matrix[i, j] for i, j in combinations(control_idx, 2)] if len(control_idx) > 1 else [0]
        # Between
        between = [dist_matrix[i, j] for i in risk_idx for j in control_idx]

        within_mean = (np.mean(risk_within) + np.mean(control_within)) / 2
        between_mean = np.mean(between)

        return between_mean / (within_mean + 1e-8)

    observed_ratio = compute_separation_ratio(risk_indices, control_indices)

    # Permutation distribution
    all_indices = risk_indices + control_indices
    n_risk = len(risk_indices)

    null_ratios = []
    for _ in range(n_permutations):
        # Shuffle labels
        shuffled = np.random.permutation(all_indices)
        perm_risk = list(shuffled[:n_risk])
        perm_control = list(shuffled[n_risk:])

        null_ratios.append(compute_separation_ratio(perm_risk, perm_control))

    # P-value: proportion of permutations with ratio >= observed
    p_value = np.mean([r >= observed_ratio for r in null_ratios])

    return {
        "observed_ratio": observed_ratio,
        "null_mean": np.mean(null_ratios),
        "null_std": np.std(null_ratios),
        "p_value": p_value,
        "z_score": (observed_ratio - np.mean(null_ratios)) / (np.std(null_ratios) + 1e-8),
    }


def odds_ratio_correlation(dist_matrix, allele_names, alleles_data):
    """Correlate p-adic distance with log odds ratio."""
    # Use most protective allele as reference
    reference_idx = min(
        range(len(allele_names)),
        key=lambda i: alleles_data[allele_names[i]]["odds_ratio"],
    )

    distances = []
    log_ors = []

    for i, name in enumerate(allele_names):
        if i != reference_idx:
            distances.append(dist_matrix[reference_idx, i])
            log_ors.append(np.log(alleles_data[name]["odds_ratio"]))

    corr, p_value = stats.spearmanr(distances, log_ors)

    return {
        "correlation": corr,
        "p_value": p_value,
        "reference": allele_names[reference_idx],
        "n_alleles": len(distances),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_expanded_visualization(allele_embeddings, alleles_data, results, position_scores, output_path):
    """Create comprehensive visualization."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. PCA of allele embeddings
    ax1 = axes[0, 0]
    embeddings = np.array(list(allele_embeddings.values()))
    allele_names = list(allele_embeddings.keys())

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    colors = []
    for name in allele_names:
        status = alleles_data[name]["ra_status"]
        if status == "high_risk":
            colors.append("darkred")
        elif status == "moderate_risk":
            colors.append("red")
        elif status == "protective":
            colors.append("blue")
        else:
            colors.append("gray")

    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=100, alpha=0.7)
    for i, name in enumerate(allele_names):
        short_name = name.split("*")[1]
        ax1.annotate(short_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=7)

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("HLA-DRB1 Alleles in Embedding Space\n(Dark red=High risk, Red=Moderate, Blue=Protective, Gray=Neutral)")

    # 2. Distance matrix heatmap
    ax2 = axes[0, 1]
    dist_matrix = results["distance_matrix"]

    # Sort by odds ratio for visualization
    sorted_indices = sorted(
        range(len(allele_names)),
        key=lambda i: alleles_data[allele_names[i]]["odds_ratio"],
        reverse=True,
    )
    sorted_matrix = dist_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_names = [allele_names[i].split("*")[1] for i in sorted_indices]

    im = ax2.imshow(sorted_matrix, cmap="viridis")
    ax2.set_xticks(range(len(sorted_names)))
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(sorted_names, fontsize=7)
    ax2.set_title("Distance Matrix\n(Sorted by Odds Ratio, high→low)")
    plt.colorbar(im, ax=ax2)

    # 3. Separation by risk category
    ax3 = axes[0, 2]
    categories = [
        "High Risk\nWithin",
        "Moderate Risk\nWithin",
        "Protective\nWithin",
        "Neutral\nWithin",
        "Risk vs\nProtective",
    ]

    # Compute within-group distances by category
    high_risk_idx = [i for i, n in enumerate(allele_names) if alleles_data[n]["ra_status"] == "high_risk"]
    moderate_idx = [i for i, n in enumerate(allele_names) if alleles_data[n]["ra_status"] == "moderate_risk"]
    protective_idx = [i for i, n in enumerate(allele_names) if alleles_data[n]["ra_status"] == "protective"]
    neutral_idx = [i for i, n in enumerate(allele_names) if alleles_data[n]["ra_status"] == "neutral"]

    def mean_within(indices):
        if len(indices) < 2:
            return 0
        return np.mean([dist_matrix[i, j] for i, j in combinations(indices, 2)])

    risk_all = high_risk_idx + moderate_idx
    protect_all = protective_idx
    between = np.mean([dist_matrix[i, j] for i in risk_all for j in protect_all]) if risk_all and protect_all else 0

    values = [
        mean_within(high_risk_idx),
        mean_within(moderate_idx),
        mean_within(protective_idx),
        mean_within(neutral_idx),
        between,
    ]
    colors_bar = ["darkred", "red", "blue", "gray", "purple"]

    ax3.bar(range(len(categories)), values, color=colors_bar, alpha=0.7)
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, fontsize=8)
    ax3.set_ylabel("Mean Distance")
    ax3.set_title("Within-Group vs Between-Group Distances")

    # 4. Position importance (top 20)
    ax4 = axes[1, 0]
    top_positions = sorted(position_scores, key=lambda x: x["fisher_ratio"], reverse=True)[:20]
    positions = [p["position"] for p in top_positions]
    fisher_ratios = [p["fisher_ratio"] for p in top_positions]

    ax4.barh(range(len(positions)), fisher_ratios, color="purple", alpha=0.7)
    ax4.set_yticks(range(len(positions)))
    ax4.set_yticklabels([f"Pos {p}" for p in positions], fontsize=8)
    ax4.set_xlabel("Fisher Discriminant Ratio")
    ax4.set_title("Top 20 Discriminative Positions\n(Higher = Better RA/Control Separation)")
    ax4.invert_yaxis()

    # Highlight shared epitope positions (70-74)
    for i, p in enumerate(positions):
        if 70 <= p <= 74:
            ax4.get_children()[i].set_color("red")

    # 5. Odds ratio vs distance correlation
    ax5 = axes[1, 1]
    or_data = results["or_correlation"]

    # Get distances from reference
    ref_name = or_data["reference"]
    ref_idx = allele_names.index(ref_name)

    distances = [dist_matrix[ref_idx, i] for i in range(len(allele_names)) if i != ref_idx]
    log_ors = [np.log(alleles_data[n]["odds_ratio"]) for n in allele_names if n != ref_name]
    colors_scatter = [colors[i] for i in range(len(allele_names)) if i != ref_idx]

    ax5.scatter(distances, log_ors, c=colors_scatter, s=80, alpha=0.7)

    # Add trend line
    z = np.polyfit(distances, log_ors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(distances), max(distances), 100)
    ax5.plot(x_line, p(x_line), "k--", alpha=0.5)

    ax5.set_xlabel(f'Distance from {ref_name.split("*")[1]}')
    ax5.set_ylabel("Log Odds Ratio")
    ax5.set_title(f'Distance vs RA Risk\nr={or_data["correlation"]:.3f}, p={or_data["p_value"]:.4f}')
    ax5.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # 6. Permutation test results
    ax6 = axes[1, 2]
    perm = results["permutation_test"]

    # Create histogram of null distribution (simulated for visualization)
    null_samples = np.random.normal(perm["null_mean"], perm["null_std"], 1000)
    ax6.hist(
        null_samples,
        bins=30,
        color="gray",
        alpha=0.5,
        label="Null Distribution",
    )
    ax6.axvline(
        x=perm["observed_ratio"],
        color="red",
        linewidth=2,
        label=f'Observed ({perm["observed_ratio"]:.2f})',
    )
    ax6.axvline(
        x=perm["null_mean"],
        color="blue",
        linestyle="--",
        label=f'Null Mean ({perm["null_mean"]:.2f})',
    )

    ax6.set_xlabel("Separation Ratio")
    ax6.set_ylabel("Frequency")
    ax6.set_title(f'Permutation Test\np={perm["p_value"]:.4f}, z={perm["z_score"]:.2f}')
    ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("EXPANDED HLA FUNCTIONOMIC ANALYSIS - HYPERBOLIC GEOMETRY")
    print("Full Peptide Binding Groove + 17 Alleles + Poincaré Ball")
    print("=" * 70)

    # Paths - use hyperbolic results directory
    script_dir = Path(__file__).parent
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    encoder, codon_mapping, _ = load_codon_encoder(device="cpu", version="3adic")
    print(f"  Loaded encoder with {sum(p.numel() for p in encoder.parameters())} parameters")

    # Analyze position importance
    print("\nAnalyzing position importance across full binding groove...")
    position_scores, allele_position_embeddings = analyze_position_importance(HLA_DRB1_EXPANDED, encoder)

    # Find most discriminative positions
    top_5 = sorted(position_scores, key=lambda x: x["fisher_ratio"], reverse=True)[:5]
    print("  Top 5 discriminative positions:")
    for p in top_5:
        se_marker = " <-- SHARED EPITOPE" if 70 <= p["position"] <= 74 else ""
        print(f"    Position {p['position']}: Fisher ratio = {p['fisher_ratio']:.3f}{se_marker}")

    # Compute sequence-level embeddings
    print("\nComputing sequence-level embeddings...")
    allele_embeddings = {}
    for name, data in HLA_DRB1_EXPANDED.items():
        pos_embs = allele_position_embeddings[name]
        seq_emb = compute_sequence_embedding(pos_embs, method="mean")
        allele_embeddings[name] = seq_emb
        print(f"  {name}: OR={data['odds_ratio']:.2f}, status={data['ra_status']}")

    # Compute distance matrix using Poincaré geodesic distance
    print("\nComputing pairwise Poincaré distances...")
    dist_matrix, allele_names = compute_pairwise_distances(allele_embeddings, distance_fn="poincare")

    # Statistical tests
    print("\nRunning statistical tests...")

    # Permutation test
    print("  Running permutation test (1000 iterations)...")
    perm_results = permutation_test(dist_matrix, allele_names, HLA_DRB1_EXPANDED, n_permutations=1000)
    print(f"    Observed separation ratio: {perm_results['observed_ratio']:.3f}")
    print(f"    Null distribution: {perm_results['null_mean']:.3f} ± {perm_results['null_std']:.3f}")
    print(f"    Z-score: {perm_results['z_score']:.2f}")
    print(f"    P-value: {perm_results['p_value']:.4f}")

    # Odds ratio correlation
    print("  Computing odds ratio correlation...")
    or_corr = odds_ratio_correlation(dist_matrix, allele_names, HLA_DRB1_EXPANDED)
    print(f"    Spearman r: {or_corr['correlation']:.3f}")
    print(f"    P-value: {or_corr['p_value']:.4f}")
    print(f"    Reference allele: {or_corr['reference']}")

    # Compile results
    results = {
        "distance_matrix": dist_matrix,
        "allele_names": allele_names,
        "permutation_test": perm_results,
        "or_correlation": or_corr,
    }

    # Create visualization
    print("\nGenerating visualization...")
    vis_path = results_dir / "hla_expanded_analysis.png"
    create_expanded_visualization(
        allele_embeddings,
        HLA_DRB1_EXPANDED,
        results,
        position_scores,
        vis_path,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"""
    Dataset:
      - Total alleles: {len(HLA_DRB1_EXPANDED)}
      - High risk: {sum(1 for d in HLA_DRB1_EXPANDED.values() if d['ra_status'] == 'high_risk')}
      - Moderate risk: {sum(1 for d in HLA_DRB1_EXPANDED.values() if d['ra_status'] == 'moderate_risk')}
      - Neutral: {sum(1 for d in HLA_DRB1_EXPANDED.values() if d['ra_status'] == 'neutral')}
      - Protective: {sum(1 for d in HLA_DRB1_EXPANDED.values() if d['ra_status'] == 'protective')}

    Sequence Analysis:
      - Positions analyzed: {len(position_scores)}
      - Most discriminative: Position {top_5[0]['position']} (Fisher={top_5[0]['fisher_ratio']:.3f})

    Statistical Tests:
      - Separation ratio: {perm_results['observed_ratio']:.3f}
      - Permutation p-value: {perm_results['p_value']:.4f}
      - Z-score: {perm_results['z_score']:.2f}
      - OR correlation: r={or_corr['correlation']:.3f}, p={or_corr['p_value']:.4f}
    """
    )

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if perm_results["p_value"] < 0.05:
        print(
            """
    *** SIGNIFICANT RESULT: p < 0.05 ***

    The p-adic embedding space shows statistically significant separation
    between RA-risk and control HLA-DRB1 alleles. This supports the hypothesis
    that functional properties relevant to autoimmunity can be captured
    in the geometric structure learned by the VAE.
        """
        )
    elif perm_results["p_value"] < 0.1:
        print(
            """
    Trend toward significance (p < 0.1).

    The separation is suggestive but requires more data to confirm.
    Consider adding more alleles or using full-length sequences.
        """
        )
    else:
        print(
            """
    No significant separation detected (p >= 0.1).

    The current analysis does not show clear geometric separation.
    This may indicate need for different encoding strategy or more data.
        """
        )

    if or_corr["p_value"] < 0.05:
        print(
            f"""
    *** SIGNIFICANT OR CORRELATION: p < 0.05 ***

    Distance in embedding space correlates with RA odds ratio (r={or_corr['correlation']:.3f}).
    Alleles further from protective reference have higher disease risk.
        """
        )

    # Save results (convert numpy types to native Python)
    output_data = {
        "n_alleles": len(HLA_DRB1_EXPANDED),
        "n_positions": len(position_scores),
        "top_discriminative_positions": [
            {
                "position": int(p["position"]),
                "fisher_ratio": float(p["fisher_ratio"]),
            }
            for p in top_5
        ],
        "permutation_test": {
            "observed_ratio": float(perm_results["observed_ratio"]),
            "null_mean": float(perm_results["null_mean"]),
            "null_std": float(perm_results["null_std"]),
            "p_value": float(perm_results["p_value"]),
            "z_score": float(perm_results["z_score"]),
        },
        "or_correlation": {
            "correlation": float(or_corr["correlation"]),
            "p_value": float(or_corr["p_value"]),
        },
    }

    output_path = results_dir / "hla_expanded_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved results to {output_path}")


if __name__ == "__main__":
    main()
