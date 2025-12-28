#!/usr/bin/env python3
"""
Find Natural Codon Positions in V5.11.3 Hyperbolic Space

This script discovers 64 natural positions in the V5.11.3 Poincaré ball that
correspond to the genetic code's 21 amino acid clusters.

Strategy:
1. Use radius (encodes 3-adic valuation) for initial binning
2. Angular clustering within radius bands
3. Select 64 positions matching genetic code degeneracy [1,1,2,2,2,2,2,2,2,2,2,3,3,4,4,4,4,4,6,6,6]
4. Validate that clusters form valid p-adic balls (within < between distances)

Output: research/genetic_code/data/natural_positions_v5_11_3.json
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

# Genetic code degeneracy pattern: 21 clusters with these sizes (sorted)
# Total = 64 codons
DEGENERACY_PATTERN = sorted(
    [
        6,
        6,
        6,  # Leu, Ser, Arg (6 codons each)
        4,
        4,
        4,
        4,
        4,  # Val, Pro, Thr, Ala, Gly (4 codons each)
        3,
        3,  # Ile, stop codons (3 each)
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,  # Phe, Tyr, His, Gln, Asn, Lys, Asp, Glu, Cys (2 each)
        1,
        1,  # Met, Trp (1 each)
    ],
    reverse=True,
)

# For reference: amino acid to degeneracy
AA_DEGENERACY = {
    "L": 6,
    "S": 6,
    "R": 6,
    "V": 4,
    "P": 4,
    "T": 4,
    "A": 4,
    "G": 4,
    "I": 3,
    "*": 3,
    "F": 2,
    "Y": 2,
    "H": 2,
    "Q": 2,
    "N": 2,
    "K": 2,
    "D": 2,
    "E": 2,
    "C": 2,
    "M": 1,
    "W": 1,
}


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincaré ball distance between two points."""
    norm_x_sq = np.sum(x**2)
    norm_y_sq = np.sum(y**2)
    diff_sq = np.sum((x - y) ** 2)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = max(denom, eps)

    arg = 1 + 2 * c * diff_sq / denom
    arg = max(arg, 1.0 + eps)

    return (1 / np.sqrt(c)) * np.arccosh(arg)


def poincare_distance_matrix(embeddings, c=1.0):
    """Compute pairwise Poincaré distances."""
    n = len(embeddings)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(embeddings[i], embeddings[j], c)
            D[i, j] = d
            D[j, i] = d

    return D


def validate_clusters(embeddings, labels):
    """Validate that clusters form valid p-adic balls.

    A valid p-adic ball has: max(within-cluster) < min(between-cluster)

    Returns:
        n_valid: number of valid clusters
        n_total: total clusters
        separation_ratio: mean(between) / mean(within)
    """
    unique_labels = sorted(set(labels))
    D = poincare_distance_matrix(embeddings)

    n_valid = 0
    within_dists = []
    between_dists = []

    for c in unique_labels:
        c_mask = np.array(labels) == c
        c_indices = np.where(c_mask)[0]
        other_indices = np.where(~c_mask)[0]

        if len(c_indices) < 2:
            n_valid += 1
            continue

        # Within-cluster distances
        max_within = 0
        for i in c_indices:
            for j in c_indices:
                if i < j:
                    within_dists.append(D[i, j])
                    max_within = max(max_within, D[i, j])

        # Between-cluster distances
        min_between = float("inf")
        for i in c_indices:
            for j in other_indices:
                between_dists.append(D[i, j])
                min_between = min(min_between, D[i, j])

        if max_within < min_between:
            n_valid += 1

    within_mean = np.mean(within_dists) if within_dists else 0
    between_mean = np.mean(between_dists) if between_dists else 0
    separation_ratio = between_mean / within_mean if within_mean > 0 else float("inf")

    return n_valid, len(unique_labels), separation_ratio


def find_natural_positions(embeddings, target_pattern=DEGENERACY_PATTERN):
    """Find 64 positions matching the genetic code degeneracy pattern.

    Strategy:
    1. Hierarchical clustering into 21 groups
    2. Adjust cluster sizes to match degeneracy pattern
    3. Select representative points from each cluster
    """
    n_clusters = len(target_pattern)
    n_points = sum(target_pattern)

    print(f"\nSearching for {n_points} positions in {n_clusters} clusters...")
    print(f"Target pattern: {target_pattern}")

    # Compute radii
    radii = np.linalg.norm(embeddings, axis=1)

    # Normalize for angular clustering
    z_normalized = embeddings / (radii[:, np.newaxis] + 1e-8)

    # Combined features: angular direction + radius (weighted)
    features = np.hstack([z_normalized, radii[:, np.newaxis] * 0.5])

    # Hierarchical clustering to get 21 initial clusters
    print("  Performing hierarchical clustering...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    initial_labels = clustering.fit_predict(features)

    # Get cluster sizes
    cluster_sizes = Counter(initial_labels)
    print(f"  Initial cluster sizes: {sorted(cluster_sizes.values(), reverse=True)}")

    # For each cluster, select points closest to cluster center
    selected_indices = []
    selected_labels = []

    # Sort clusters by size (largest first) to match degeneracy pattern
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])

    for cluster_idx, (cluster_id, cluster_size) in enumerate(sorted_clusters):
        target_size = target_pattern[cluster_idx] if cluster_idx < len(target_pattern) else 1

        # Get indices in this cluster
        cluster_mask = initial_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Compute cluster center
        cluster_center = embeddings[cluster_indices].mean(axis=0)

        # Distance to center
        dists_to_center = [poincare_distance(embeddings[i], cluster_center) for i in cluster_indices]

        # Select closest to center
        sorted_by_dist = [cluster_indices[i] for i in np.argsort(dists_to_center)]
        n_select = min(target_size, len(sorted_by_dist))

        for idx in sorted_by_dist[:n_select]:
            selected_indices.append(idx)
            selected_labels.append(cluster_idx)

    print(f"  Selected {len(selected_indices)} positions")

    # Validate
    selected_embeddings = embeddings[selected_indices]
    n_valid, n_total, sep_ratio = validate_clusters(selected_embeddings, selected_labels)

    print(f"  Valid p-adic balls: {n_valid}/{n_total}")
    print(f"  Separation ratio: {sep_ratio:.2f}x")

    return selected_indices, selected_labels, sep_ratio


def refine_positions(embeddings, indices, labels, iterations=5):
    """Refine positions by swapping to improve separation ratio."""
    best_indices = indices.copy()
    best_labels = labels.copy()
    best_ratio = validate_clusters(embeddings[best_indices], best_labels)[2]

    print(f"\nRefining positions (initial ratio: {best_ratio:.2f}x)...")

    all_indices = set(range(len(embeddings)))
    selected_set = set(best_indices)

    for iteration in range(iterations):
        improved = False

        for i, idx in enumerate(best_indices):
            cluster = best_labels[i]

            # Find same-cluster points not selected
            cluster_mask = np.array(best_labels) == cluster
            cluster_indices = [best_indices[j] for j in range(len(best_indices)) if best_labels[j] == cluster]

            # Try swapping with nearby unselected points
            radii = np.linalg.norm(embeddings, axis=1)
            idx_radius = radii[idx]

            # Candidates: within 10% radius
            candidates = [j for j in all_indices - selected_set if abs(radii[j] - idx_radius) < 0.1 * idx_radius]

            for candidate in candidates[:10]:  # Limit candidates
                # Try swap
                test_indices = best_indices.copy()
                test_indices[i] = candidate

                _, _, new_ratio = validate_clusters(embeddings[test_indices], best_labels)

                if new_ratio > best_ratio:
                    best_indices = test_indices
                    best_ratio = new_ratio
                    selected_set.discard(idx)
                    selected_set.add(candidate)
                    improved = True
                    break

            if improved:
                break

        print(f"  Iteration {iteration + 1}: ratio = {best_ratio:.2f}x")

        if not improved:
            break

    return best_indices, best_labels, best_ratio


def main():
    print("=" * 70)
    print("FIND NATURAL POSITIONS IN V5.11.3 HYPERBOLIC SPACE")
    print("=" * 70)

    # Load embeddings
    data_dir = Path(__file__).parent.parent / "data"
    embeddings_path = data_dir / "v5_11_3_embeddings.pt"

    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        print("Run 07_extract_v5_11_3_embeddings.py first.")
        return 1

    print(f"\nLoading embeddings from: {embeddings_path}")
    data = torch.load(embeddings_path, weights_only=False)

    # Use VAE-B embeddings (the one with stronger hierarchy)
    z_B = data["z_B_hyp"].numpy()
    valuations = data["valuations"].numpy()

    print(f"  Loaded {len(z_B)} embeddings, dim={z_B.shape[1]}")
    print(f"  Hierarchy correlation: {data['metadata']['hierarchy_correlation']:.4f}")

    # Find natural positions
    indices, labels, sep_ratio = find_natural_positions(z_B)

    # Refine
    indices, labels, sep_ratio = refine_positions(z_B, indices, labels)

    # Organize by cluster
    clusters = defaultdict(list)
    for idx, label in zip(indices, labels):
        clusters[label].append(int(idx))

    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    # Create output structure
    output = {
        "metadata": {
            "source": "V5.11.3 embeddings",
            "n_positions": len(indices),
            "n_clusters": len(clusters),
            "separation_ratio": float(sep_ratio),
            "timestamp": datetime.now().isoformat(),
        },
        "positions": [int(i) for i in indices],
        "labels": [int(l) for l in labels],
        "clusters": {str(label): positions for label, positions in sorted_clusters},
        "cluster_sizes": [len(positions) for _, positions in sorted_clusters],
        "degeneracy_pattern": DEGENERACY_PATTERN,
    }

    # Validate final result
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    selected_embeddings = z_B[indices]
    n_valid, n_total, final_ratio = validate_clusters(selected_embeddings, labels)

    print(f"\n  Positions found: {len(indices)}")
    print(f"  Clusters: {n_total}")
    print(f"  Valid p-adic balls: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"  Separation ratio: {final_ratio:.2f}x")

    # Cluster size distribution
    print(f"\n  Cluster sizes: {output['cluster_sizes']}")
    print(f"  Target pattern: {DEGENERACY_PATTERN}")

    # Check radii distribution
    radii = np.linalg.norm(z_B[indices], axis=1)
    print(f"\n  Radius range: [{radii.min():.4f}, {radii.max():.4f}]")

    # Save
    output_path = data_dir / "natural_positions_v5_11_3.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {output_path}")

    # Also save as Python-importable format
    positions_py = f'''"""Natural positions discovered in V5.11.3 hyperbolic space.

Generated: {datetime.now().isoformat()}
Separation ratio: {final_ratio:.2f}x
Valid p-adic balls: {n_valid}/{n_total}
"""

NATURAL_POSITIONS_V5_11_3 = {indices}

CLUSTER_LABELS_V5_11_3 = {labels}

CLUSTER_SIZES_V5_11_3 = {output['cluster_sizes']}
'''

    py_path = data_dir / "natural_positions_v5_11_3.py"
    with open(py_path, "w") as f:
        f.write(positions_py)

    print(f"  Saved Python module: {py_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
