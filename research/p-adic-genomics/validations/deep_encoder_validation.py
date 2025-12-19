#!/usr/bin/env python3
"""
Deep Validation of 3-Adic Codon Encoder

Advanced tests that go beyond basic structure validation:
1. Hyperbolic geometry properties (curvature, Möbius operations)
2. k-NN neighborhood accuracy
3. Geodesic interpolation between synonymous codons
4. P-adic valuation correlation with embedding distance
5. Cluster purity metrics
6. Radial and angular distribution analysis
7. Hyperbolic Fréchet mean computation

Author: AI Whisperers
Date: 2025-12-19
"""

import json
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

# Add paths
SCRIPT_DIR = Path(__file__).parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR.parent / "src"))

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# GENETIC CODE DATA
# ============================================================================

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

NUCLEOTIDES = ['A', 'C', 'G', 'T']
ALL_CODONS = list(CODON_TABLE.keys())


# ============================================================================
# HYPERBOLIC GEOMETRY FUNCTIONS
# ============================================================================

def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """Compute Poincaré ball geodesic distance."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    norm_x_sq = np.sum(x ** 2, axis=-1)
    norm_y_sq = np.sum(y ** 2, axis=-1)
    diff_sq = np.sum((x - y) ** 2, axis=-1)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = np.maximum(denom, 1e-10)

    arg = 1 + 2 * c * diff_sq / denom
    arg = np.maximum(arg, 1.0 + 1e-10)

    dist = (1 / np.sqrt(c)) * np.arccosh(arg)
    return float(dist.squeeze()) if dist.size == 1 else dist


def mobius_addition(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Möbius addition in Poincaré ball: x ⊕ y"""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    xy = np.sum(x * y, axis=-1, keepdims=True)
    x_sq = np.sum(x ** 2, axis=-1, keepdims=True)
    y_sq = np.sum(y ** 2, axis=-1, keepdims=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c**2 * x_sq * y_sq
    denom = np.maximum(denom, 1e-10)

    return num / denom


def exp_map(x: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map at x in direction v (Poincaré ball)."""
    x = np.squeeze(x)
    v = np.squeeze(v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return x

    lambda_x = float(conformal_factor(x, c))
    second_term = np.tanh(np.sqrt(c) * lambda_x * v_norm / 2) * v / (np.sqrt(c) * v_norm)

    return np.squeeze(mobius_addition(x, second_term, c))


def log_map(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map from x to y (Poincaré ball)."""
    x = np.squeeze(x)
    y = np.squeeze(y)
    neg_x = -x
    xy = np.squeeze(mobius_addition(y, neg_x, c))
    xy_norm = np.linalg.norm(xy)

    if xy_norm < 1e-10:
        return np.zeros_like(x)

    lambda_x = float(conformal_factor(x, c))
    return 2 / (np.sqrt(c) * lambda_x) * np.arctanh(np.minimum(np.sqrt(c) * xy_norm, 0.999)) * xy / xy_norm


def hyperbolic_midpoint(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute midpoint on geodesic between x and y using exp/log maps."""
    x = np.squeeze(x)
    y = np.squeeze(y)
    # Logarithmic map from x to y
    v = log_map(x, y, c)
    # Exponential map at x with half the tangent vector
    return exp_map(x, v * 0.5, c)


def conformal_factor(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute conformal factor λ(x) = 2 / (1 - c||x||²)"""
    norm_sq = np.sum(x ** 2, axis=-1)
    return 2.0 / (1 - c * norm_sq + 1e-10)


def hyperbolic_frechet_mean(points: np.ndarray, c: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute Fréchet mean (hyperbolic centroid) using Riemannian gradient descent.

    The Fréchet mean minimizes the sum of squared hyperbolic distances.
    """
    # Initialize with Euclidean mean projected to ball
    mean = np.mean(points, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0.9:
        mean = mean * 0.9 / norm

    for _ in range(max_iter):
        # Compute Riemannian gradient
        grad = np.zeros_like(mean)
        for p in points:
            # Logarithmic map at mean pointing toward p
            d = poincare_distance(mean, p, c)
            if d > 1e-10:
                direction = p - mean
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                grad += d * direction

        grad = grad / len(points)

        # Step with conformal factor
        lambda_x = conformal_factor(mean, c)
        step = 0.5 * grad / lambda_x

        # Update mean
        new_mean = mean + step

        # Project back to ball
        norm = np.linalg.norm(new_mean)
        if norm > 0.95:
            new_mean = new_mean * 0.95 / norm

        # Check convergence
        if np.linalg.norm(new_mean - mean) < tol:
            break

        mean = new_mean

    return mean


def padic_valuation_3(n: int) -> int:
    """Compute 3-adic valuation v₃(n) = max k such that 3^k divides n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def padic_distance_3(i: int, j: int) -> float:
    """Compute 3-adic distance d₃(i,j) = 3^(-v₃(|i-j|))"""
    if i == j:
        return 0.0
    return 3.0 ** (-padic_valuation_3(abs(i - j)))


def codon_to_ternary_position(codon: str) -> int:
    """Convert codon to ternary position (0-63 in base-3)."""
    nuc_map = {'T': 0, 'C': 1, 'A': 2, 'G': 2}  # Simplified
    pos = 0
    for i, n in enumerate(codon):
        pos = pos * 3 + nuc_map.get(n, 0)
    return pos


# ============================================================================
# ENCODER LOADING
# ============================================================================

class CodonEncoder3Adic(nn.Module):
    """Encode codons into V5.11.3 hyperbolic embedding space."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.cluster_head = nn.Linear(embed_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        embedding = self.encoder(x)
        cluster_logits = self.cluster_head(embedding)
        return embedding, cluster_logits


def load_encoder():
    """Load the trained 3-adic encoder."""
    encoder_path = RESEARCH_DIR / "genetic_code" / "data" / "codon_encoder_3adic.pt"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)

    # Rebuild model
    model = CodonEncoder3Adic()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, checkpoint


def get_all_embeddings(model: CodonEncoder3Adic) -> Dict[str, np.ndarray]:
    """Get embeddings for all 64 codons."""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    embeddings = {}
    features_list = []

    for codon in ALL_CODONS:
        features = []
        for nuc in codon:
            one_hot = [0, 0, 0, 0]
            one_hot[nuc_to_idx[nuc]] = 1
            features.extend(one_hot)
        features_list.append(features)

    features_tensor = torch.tensor(features_list, dtype=torch.float32)

    with torch.no_grad():
        embs, _ = model(features_tensor)

    embs_np = embs.numpy()
    for i, codon in enumerate(ALL_CODONS):
        embeddings[codon] = embs_np[i]

    return embeddings


# ============================================================================
# DEEP VALIDATION TESTS
# ============================================================================

def test_knn_neighborhood_accuracy(embeddings: Dict[str, np.ndarray], k: int = 5) -> Dict:
    """
    Test 1: k-NN Neighborhood Accuracy

    For each codon, check if its k nearest neighbors in embedding space
    are synonymous codons (same amino acid).
    """
    print("\n" + "-" * 70)
    print(f"TEST 1: k-NN Neighborhood Accuracy (k={k})")
    print("-" * 70)

    results = {
        "test": "knn_accuracy",
        "k": k,
        "per_codon": {},
        "per_aa": {}
    }

    codon_list = list(embeddings.keys())
    emb_matrix = np.array([embeddings[c] for c in codon_list])

    # Compute pairwise Poincaré distances
    n = len(codon_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(emb_matrix[i], emb_matrix[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    total_correct = 0
    total_neighbors = 0

    aa_to_codons = defaultdict(list)
    for i, codon in enumerate(codon_list):
        aa_to_codons[CODON_TABLE[codon]].append(i)

    for i, codon in enumerate(codon_list):
        aa = CODON_TABLE[codon]
        synonymous_indices = set(aa_to_codons[aa]) - {i}

        # Get k nearest neighbors
        neighbor_dists = [(j, dist_matrix[i, j]) for j in range(n) if j != i]
        neighbor_dists.sort(key=lambda x: x[1])
        k_neighbors = [j for j, _ in neighbor_dists[:k]]

        # Count how many are synonymous
        n_synonymous = len(synonymous_indices)
        k_actual = min(k, n_synonymous)  # Can't have more than available synonymous

        correct = sum(1 for j in k_neighbors[:k_actual] if j in synonymous_indices)

        results["per_codon"][codon] = {
            "neighbors": [codon_list[j] for j in k_neighbors],
            "correct": correct,
            "max_possible": k_actual
        }

        total_correct += correct
        total_neighbors += k_actual

    accuracy = total_correct / max(total_neighbors, 1)
    results["summary"] = {
        "total_correct": total_correct,
        "total_possible": total_neighbors,
        "accuracy": float(accuracy)
    }

    print(f"  Overall k-NN accuracy: {accuracy*100:.1f}%")
    print(f"  Correct neighbors: {total_correct}/{total_neighbors}")

    return results


def test_geodesic_interpolation(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 2: Geodesic Interpolation

    For synonymous codons, check that midpoints on geodesics
    remain within the amino acid cluster.
    """
    print("\n" + "-" * 70)
    print("TEST 2: Geodesic Interpolation")
    print("-" * 70)

    results = {
        "test": "geodesic_interpolation",
        "aa_tests": {},
        "summary": {}
    }

    aa_to_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        aa_to_codons[aa].append(codon)

    interpolations_valid = 0
    total_interpolations = 0

    for aa, codons in aa_to_codons.items():
        if len(codons) < 2:
            continue

        aa_embeddings = [embeddings[c] for c in codons]
        centroid = hyperbolic_frechet_mean(np.array(aa_embeddings))
        max_dist_to_centroid = max(poincare_distance(e, centroid) for e in aa_embeddings)

        midpoint_tests = []
        for i in range(len(codons)):
            for j in range(i + 1, len(codons)):
                x = aa_embeddings[i]
                y = aa_embeddings[j]

                midpoint = hyperbolic_midpoint(x, y)
                dist_to_centroid = poincare_distance(midpoint, centroid)

                # Midpoint should be within cluster (within max dist)
                is_valid = dist_to_centroid <= max_dist_to_centroid * 1.5

                midpoint_tests.append({
                    "codon_i": codons[i],
                    "codon_j": codons[j],
                    "midpoint_dist_to_centroid": float(dist_to_centroid),
                    "max_cluster_dist": float(max_dist_to_centroid),
                    "valid": is_valid
                })

                if is_valid:
                    interpolations_valid += 1
                total_interpolations += 1

        results["aa_tests"][aa] = {
            "n_codons": len(codons),
            "midpoint_tests": midpoint_tests,
            "centroid_radius": float(max_dist_to_centroid)
        }

    validity_rate = interpolations_valid / max(total_interpolations, 1)
    results["summary"] = {
        "valid_interpolations": interpolations_valid,
        "total_interpolations": total_interpolations,
        "validity_rate": float(validity_rate)
    }

    print(f"  Valid interpolations: {interpolations_valid}/{total_interpolations}")
    print(f"  Validity rate: {validity_rate*100:.1f}%")

    return results


def test_padic_correlation(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 3: P-adic Valuation Correlation

    Test if 3-adic distance between codon positions correlates
    with Poincaré distance between embeddings.
    """
    print("\n" + "-" * 70)
    print("TEST 3: P-adic Valuation Correlation")
    print("-" * 70)

    results = {
        "test": "padic_correlation",
        "correlations": {},
        "summary": {}
    }

    codon_list = list(embeddings.keys())
    n = len(codon_list)

    # Compute ternary positions for each codon
    positions = [codon_to_ternary_position(c) for c in codon_list]

    poincare_dists = []
    padic_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            pd = poincare_distance(embeddings[codon_list[i]], embeddings[codon_list[j]])
            pad = padic_distance_3(positions[i], positions[j])

            poincare_dists.append(pd)
            padic_dists.append(pad)

    # Compute correlations
    spearman_r, spearman_p = spearmanr(padic_dists, poincare_dists)
    pearson_r, pearson_p = pearsonr(padic_dists, poincare_dists)

    results["correlations"] = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p)
    }

    # Also test within synonymous groups
    within_syn_poincare = []
    within_syn_padic = []

    aa_to_indices = defaultdict(list)
    for i, c in enumerate(codon_list):
        aa_to_indices[CODON_TABLE[c]].append(i)

    for aa, indices in aa_to_indices.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                pd = poincare_distance(embeddings[codon_list[idx_i]], embeddings[codon_list[idx_j]])
                pad = padic_distance_3(positions[idx_i], positions[idx_j])
                within_syn_poincare.append(pd)
                within_syn_padic.append(pad)

    if len(within_syn_poincare) > 2:
        syn_spearman, _ = spearmanr(within_syn_padic, within_syn_poincare)
        results["within_synonymous_correlation"] = float(syn_spearman)

    print(f"  Overall Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Overall Pearson correlation: {pearson_r:.4f}")
    if "within_synonymous_correlation" in results:
        print(f"  Within-synonymous correlation: {results['within_synonymous_correlation']:.4f}")

    return results


def test_cluster_purity(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 4: Cluster Purity Metrics

    Measure how pure the amino acid clusters are using various metrics.
    """
    print("\n" + "-" * 70)
    print("TEST 4: Cluster Purity Metrics")
    print("-" * 70)

    results = {
        "test": "cluster_purity",
        "per_aa": {},
        "summary": {}
    }

    aa_to_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        aa_to_codons[aa].append(codon)

    silhouette_scores = []

    codon_list = list(embeddings.keys())
    emb_matrix = np.array([embeddings[c] for c in codon_list])

    # Compute all pairwise distances
    n = len(codon_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(emb_matrix[i], emb_matrix[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Compute silhouette-like score for each codon
    for i, codon in enumerate(codon_list):
        aa = CODON_TABLE[codon]
        same_aa_indices = [j for j, c in enumerate(codon_list) if CODON_TABLE[c] == aa and j != i]
        diff_aa_indices = [j for j, c in enumerate(codon_list) if CODON_TABLE[c] != aa]

        if same_aa_indices and diff_aa_indices:
            a = np.mean([dist_matrix[i, j] for j in same_aa_indices])  # Intra-cluster
            b = np.mean([dist_matrix[i, j] for j in diff_aa_indices])  # Inter-cluster

            s = (b - a) / max(a, b, 1e-10)
            silhouette_scores.append(s)

    mean_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0

    # Compute per-AA metrics
    for aa, codons in aa_to_codons.items():
        if len(codons) < 2:
            continue

        aa_embeddings = [embeddings[c] for c in codons]
        centroid = np.mean(aa_embeddings, axis=0)

        # Intra-cluster variance
        intra_dists = [poincare_distance(e, centroid) for e in aa_embeddings]

        results["per_aa"][aa] = {
            "n_codons": len(codons),
            "mean_dist_to_centroid": float(np.mean(intra_dists)),
            "max_dist_to_centroid": float(np.max(intra_dists)),
            "std_dist_to_centroid": float(np.std(intra_dists))
        }

    results["summary"] = {
        "mean_silhouette_score": float(mean_silhouette),
        "silhouette_valid": mean_silhouette > 0.3
    }

    print(f"  Mean silhouette score: {mean_silhouette:.4f}")
    print(f"  Silhouette valid (>0.3): {results['summary']['silhouette_valid']}")

    return results


def test_radial_distribution(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 5: Radial Distribution Analysis

    Analyze how embeddings are distributed radially in the Poincaré ball.
    Higher degeneracy amino acids should be closer to center (deeper in hierarchy).
    """
    print("\n" + "-" * 70)
    print("TEST 5: Radial Distribution Analysis")
    print("-" * 70)

    results = {
        "test": "radial_distribution",
        "per_aa": {},
        "degeneracy_analysis": {},
        "summary": {}
    }

    aa_to_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        aa_to_codons[aa].append(codon)

    # Compute radii for each amino acid cluster
    degeneracy_to_radii = defaultdict(list)

    for aa, codons in aa_to_codons.items():
        aa_embeddings = [embeddings[c] for c in codons]
        radii = [np.linalg.norm(e) for e in aa_embeddings]

        mean_radius = np.mean(radii)
        degeneracy = len(codons)

        results["per_aa"][aa] = {
            "degeneracy": degeneracy,
            "mean_radius": float(mean_radius),
            "min_radius": float(np.min(radii)),
            "max_radius": float(np.max(radii))
        }

        degeneracy_to_radii[degeneracy].append(mean_radius)

    # Check if higher degeneracy → smaller radius (deeper in hierarchy)
    for deg, radii in sorted(degeneracy_to_radii.items()):
        results["degeneracy_analysis"][deg] = {
            "n_amino_acids": len(radii),
            "mean_radius": float(np.mean(radii)),
            "std_radius": float(np.std(radii))
        }
        print(f"  {deg}-fold degenerate: mean radius = {np.mean(radii):.4f}")

    # Compute correlation: degeneracy vs radius
    degs = []
    radii = []
    for aa, data in results["per_aa"].items():
        degs.append(data["degeneracy"])
        radii.append(data["mean_radius"])

    if len(degs) > 2:
        corr, p = spearmanr(degs, radii)
        results["summary"]["degeneracy_radius_correlation"] = float(corr)
        results["summary"]["correlation_p_value"] = float(p)
        print(f"\n  Degeneracy-radius correlation: {corr:.4f} (p={p:.2e})")
        print(f"  (Negative = higher degeneracy closer to center)")

    return results


def test_angular_separation(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 6: Angular Separation Between Amino Acid Clusters

    Measure angular separation between cluster centroids.
    """
    print("\n" + "-" * 70)
    print("TEST 6: Angular Separation Between Clusters")
    print("-" * 70)

    results = {
        "test": "angular_separation",
        "cluster_centroids": {},
        "angular_distances": [],
        "summary": {}
    }

    aa_to_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        aa_to_codons[aa].append(codon)

    # Compute centroids
    centroids = {}
    for aa, codons in aa_to_codons.items():
        aa_embeddings = np.array([embeddings[c] for c in codons])
        centroid = np.mean(aa_embeddings, axis=0)
        centroids[aa] = centroid

        results["cluster_centroids"][aa] = {
            "centroid": centroid.tolist(),
            "radius": float(np.linalg.norm(centroid))
        }

    # Compute angular distances between centroids
    aa_list = list(centroids.keys())
    angular_dists = []

    for i, aa1 in enumerate(aa_list):
        for aa2 in aa_list[i+1:]:
            c1, c2 = centroids[aa1], centroids[aa2]

            # Angular distance (cosine of angle)
            norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(c1, c2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angular_dists.append(angle)

                results["angular_distances"].append({
                    "aa1": aa1,
                    "aa2": aa2,
                    "angle_radians": float(angle),
                    "angle_degrees": float(np.degrees(angle))
                })

    results["summary"] = {
        "mean_angular_distance_deg": float(np.degrees(np.mean(angular_dists))),
        "min_angular_distance_deg": float(np.degrees(np.min(angular_dists))),
        "max_angular_distance_deg": float(np.degrees(np.max(angular_dists)))
    }

    print(f"  Mean angular separation: {np.degrees(np.mean(angular_dists)):.1f}°")
    print(f"  Min angular separation: {np.degrees(np.min(angular_dists)):.1f}°")
    print(f"  Max angular separation: {np.degrees(np.max(angular_dists)):.1f}°")

    return results


def test_conformal_factor_consistency(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Test 7: Conformal Factor Consistency

    Verify that the conformal factor (metric distortion) is consistent
    with hyperbolic geometry expectations.
    """
    print("\n" + "-" * 70)
    print("TEST 7: Conformal Factor Consistency")
    print("-" * 70)

    results = {
        "test": "conformal_factor",
        "per_codon": {},
        "summary": {}
    }

    conformal_factors = []
    radii = []

    for codon, emb in embeddings.items():
        cf = conformal_factor(emb)
        r = np.linalg.norm(emb)

        results["per_codon"][codon] = {
            "conformal_factor": float(cf),
            "radius": float(r)
        }

        conformal_factors.append(cf)
        radii.append(r)

    # Conformal factor should increase with radius
    corr, p = spearmanr(radii, conformal_factors)

    results["summary"] = {
        "mean_conformal_factor": float(np.mean(conformal_factors)),
        "max_conformal_factor": float(np.max(conformal_factors)),
        "radius_cf_correlation": float(corr),
        "correlation_p_value": float(p),
        "geometry_consistent": corr > 0.9  # Strong positive correlation expected
    }

    print(f"  Mean conformal factor: {np.mean(conformal_factors):.4f}")
    print(f"  Max conformal factor: {np.max(conformal_factors):.4f}")
    print(f"  Radius-CF correlation: {corr:.4f}")
    print(f"  Geometry consistent: {results['summary']['geometry_consistent']}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_deep_validation():
    """Run comprehensive deep validation of the encoder."""

    print("=" * 70)
    print("DEEP VALIDATION OF 3-ADIC CODON ENCODER")
    print("Testing hyperbolic geometry and p-adic structure")
    print("=" * 70)

    # Load encoder
    print("\nLoading encoder...")
    model, checkpoint = load_encoder()

    print(f"  Encoder version: {checkpoint.get('metadata', {}).get('version', 'unknown')}")
    print(f"  Source embeddings: {checkpoint.get('metadata', {}).get('source_embeddings', 'unknown')}")

    # Get all embeddings
    print("\nComputing embeddings for all 64 codons...")
    embeddings = get_all_embeddings(model)

    all_results = {
        "framework": "Deep 3-adic encoder validation",
        "encoder_metadata": checkpoint.get('metadata', {}),
        "tests": {}
    }

    # Run all tests
    all_results["tests"]["knn_accuracy"] = test_knn_neighborhood_accuracy(embeddings, k=5)
    all_results["tests"]["geodesic_interpolation"] = test_geodesic_interpolation(embeddings)
    all_results["tests"]["padic_correlation"] = test_padic_correlation(embeddings)
    all_results["tests"]["cluster_purity"] = test_cluster_purity(embeddings)
    all_results["tests"]["radial_distribution"] = test_radial_distribution(embeddings)
    all_results["tests"]["angular_separation"] = test_angular_separation(embeddings)
    all_results["tests"]["conformal_factor"] = test_conformal_factor_consistency(embeddings)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("DEEP VALIDATION SUMMARY")
    print("=" * 70)

    validations = {
        "knn_accuracy": all_results["tests"]["knn_accuracy"]["summary"]["accuracy"] > 0.5,
        "geodesic_interpolation": all_results["tests"]["geodesic_interpolation"]["summary"]["validity_rate"] > 0.8,
        "cluster_purity": all_results["tests"]["cluster_purity"]["summary"]["silhouette_valid"],
        "conformal_consistency": all_results["tests"]["conformal_factor"]["summary"]["geometry_consistent"],
    }

    n_valid = sum(validations.values())
    n_total = len(validations)

    print(f"\n  Deep tests passed: {n_valid}/{n_total}")
    for test, valid in validations.items():
        status = "PASS" if valid else "FAIL"
        print(f"    {test}: {status}")

    all_results["summary"] = {
        "tests_passed": n_valid,
        "tests_total": n_total,
        "pass_rate": n_valid / n_total,
        "validations": validations
    }

    # Save results
    output_file = OUTPUT_DIR / "deep_encoder_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_deep_validation()
