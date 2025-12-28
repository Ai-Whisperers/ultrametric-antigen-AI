"""
07b_reverse_padic_search.py - Reverse approach: Find 64 indices that form 20 p-adic balls

Instead of mapping codons to ternary space, we ask:
"Which 64 points in the 19,683-point embedding space naturally form 20 clusters
with p-adic ball properties (within-distance < between-distance)?"

If we find such points, we can then:
1. Characterize their properties (valuation, radius, angular structure)
2. See if they match the genetic code degeneracy pattern (1,1,2,2,2,2,2,2,2,3,3,4,4,4,4,4,6,6,6)
3. Discover the "natural" codon→ternary mapping

Usage:
    python 07b_reverse_padic_search.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# Genetic code degeneracy pattern: how many codons per amino acid
# Sorted: [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6]
# That's: 2×1, 9×2, 2×3, 5×4, 3×6 = 2+18+6+20+18 = 64 codons, 21 "amino acids" (including stop)
GENETIC_CODE_DEGENERACY = [
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    6,
    6,
    6,
]
N_AMINO_ACIDS = 21  # Including stop codon


def poincare_distance(x, y, c=1.0, eps=1e-7):
    """Compute Poincare geodesic distance."""
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


def compute_ball_quality(indices, embeddings, cluster_labels):
    """Compute p-adic ball quality for a set of points with cluster assignments.

    Returns:
        quality: mean(between_dist) / mean(within_dist) - higher is better
        n_valid_balls: number of clusters where eps_within < eps_between
    """
    n_points = len(indices)
    unique_clusters = set(cluster_labels)

    if len(unique_clusters) < 2:
        return 0.0, 0

    # Get embeddings for these indices
    emb = embeddings[indices]

    # Compute pairwise distances
    within_distances = []
    between_distances = []

    for c in unique_clusters:
        c_mask = np.array(cluster_labels) == c
        c_indices = np.where(c_mask)[0]

        if len(c_indices) < 2:
            continue

        # Within-cluster distances
        for i, j in combinations(c_indices, 2):
            d = poincare_distance(emb[i], emb[j])[0]
            within_distances.append(d)

        # Between-cluster distances
        other_indices = np.where(~c_mask)[0]
        for ci in c_indices:
            for oi in other_indices:
                d = poincare_distance(emb[ci], emb[oi])[0]
                between_distances.append(d)

    if not within_distances or not between_distances:
        return 0.0, 0

    mean_within = np.mean(within_distances)
    mean_between = np.mean(between_distances)

    quality = mean_between / mean_within if mean_within > 0 else 0

    # Count valid balls (eps_within < min_eps_between for each cluster)
    n_valid = 0
    for c in unique_clusters:
        c_mask = np.array(cluster_labels) == c
        c_indices = np.where(c_mask)[0]

        if len(c_indices) < 2:
            n_valid += 1  # Single-point clusters are trivially valid
            continue

        # Max within-cluster distance
        eps_within = 0
        for i, j in combinations(c_indices, 2):
            d = poincare_distance(emb[i], emb[j])[0]
            eps_within = max(eps_within, d)

        # Min between-cluster distance
        other_indices = np.where(~c_mask)[0]
        eps_between = float("inf")
        for ci in c_indices:
            for oi in other_indices:
                d = poincare_distance(emb[ci], emb[oi])[0]
                eps_between = min(eps_between, d)

        if eps_within < eps_between:
            n_valid += 1

    return quality, n_valid


def find_natural_clusters(embeddings, n_points=64, n_clusters=21, n_samples=1000):
    """Find n_points that naturally form n_clusters p-adic balls.

    Strategy: Sample random subsets and evaluate their clustering quality.
    """
    print(f"\n  Searching for {n_points} points forming {n_clusters} clusters...")
    print(f"  Sampling {n_samples} random configurations...")

    n_total = len(embeddings)
    best_quality = 0
    best_indices = None
    best_labels = None

    np.random.seed(42)

    for i in range(n_samples):
        # Sample random indices
        indices = np.random.choice(n_total, n_points, replace=False)

        # Get embeddings
        emb = embeddings[indices]

        # Cluster using hierarchical clustering on geodesic distances
        # (Computing full pairwise is expensive, use Euclidean as proxy for speed)
        distances = pdist(emb, metric="euclidean")
        Z = linkage(distances, method="ward")
        labels = fcluster(Z, n_clusters, criterion="maxclust")

        # Evaluate quality
        quality, n_valid = compute_ball_quality(indices, embeddings, labels)

        if quality > best_quality:
            best_quality = quality
            best_indices = indices
            best_labels = labels

            if i % 100 == 0:
                print(f"    Iter {i}: quality={quality:.3f}, valid_balls={n_valid}/{n_clusters}")

    print(f"\n  Best configuration: quality={best_quality:.3f}")

    return best_indices, best_labels, best_quality


def find_by_valuation_structure(embeddings, valuations, n_clusters=21):
    """Find 64 points using valuation-based selection.

    Idea: The genetic code has a specific degeneracy pattern.
    We look for points at different valuation levels that cluster naturally.
    """
    print("\n  Searching using valuation structure...")

    # Target degeneracy pattern (sorted)
    target_pattern = sorted(GENETIC_CODE_DEGENERACY)

    # Group indices by valuation
    val_to_indices = {}
    for i, v in enumerate(valuations):
        v_int = int(v)
        if v_int not in val_to_indices:
            val_to_indices[v_int] = []
        val_to_indices[v_int].append(i)

    print(f"    Valuation distribution: {[(v, len(idx)) for v, idx in sorted(val_to_indices.items())]}")

    # Strategy: Select from high-valuation levels first (they're rarer and more structured)
    # Then fill with lower-valuation points

    best_quality = 0
    best_indices = None
    best_labels = None

    np.random.seed(42)

    for trial in range(500):
        selected_indices = []
        selected_labels = []
        cluster_id = 0

        # For each target cluster size, pick points from similar valuation
        remaining_pattern = list(target_pattern)
        np.random.shuffle(remaining_pattern)

        for cluster_size in remaining_pattern:
            # Prefer higher valuation points (more structured)
            # Weight by inverse of count (rare = better)
            weights = []
            valid_vals = []
            for v, idx_list in val_to_indices.items():
                available = [i for i in idx_list if i not in selected_indices]
                if len(available) >= cluster_size:
                    weights.append(1.0 / (len(idx_list) + 1))
                    valid_vals.append(v)

            if not valid_vals:
                break

            # Sample a valuation level
            weights = np.array(weights) / sum(weights)
            chosen_val = np.random.choice(valid_vals, p=weights)

            # Pick cluster_size points from this valuation
            available = [i for i in val_to_indices[chosen_val] if i not in selected_indices]
            chosen_points = np.random.choice(available, cluster_size, replace=False)

            selected_indices.extend(chosen_points)
            selected_labels.extend([cluster_id] * cluster_size)
            cluster_id += 1

        if len(selected_indices) != 64:
            continue

        # Evaluate quality
        quality, n_valid = compute_ball_quality(np.array(selected_indices), embeddings, selected_labels)

        if quality > best_quality:
            best_quality = quality
            best_indices = np.array(selected_indices)
            best_labels = selected_labels

    print(f"    Best valuation-based configuration: quality={best_quality:.3f}")

    return best_indices, best_labels, best_quality


def find_by_radius_bands(embeddings, n_clusters=21):
    """Find 64 points using radius-based bands.

    Idea: Points at similar radius (= similar valuation) should cluster together.
    """
    print("\n  Searching using radius bands...")

    radii = np.linalg.norm(embeddings, axis=1)

    # Sort by radius
    sorted_indices = np.argsort(radii)

    # Create radius bands
    n_total = len(embeddings)

    best_quality = 0
    best_indices = None
    best_labels = None

    target_pattern = sorted(GENETIC_CODE_DEGENERACY)

    np.random.seed(42)

    for trial in range(500):
        # Sample starting positions for each cluster within radius-sorted order
        selected_indices = []
        selected_labels = []

        # Divide into 21 bands, pick cluster_size points from each
        band_size = n_total // 21

        for cluster_id, cluster_size in enumerate(target_pattern):
            band_start = cluster_id * band_size
            band_end = min((cluster_id + 1) * band_size, n_total)
            band_indices = sorted_indices[band_start:band_end]

            # Randomly pick from this band
            if len(band_indices) >= cluster_size:
                chosen = np.random.choice(band_indices, cluster_size, replace=False)
                selected_indices.extend(chosen)
                selected_labels.extend([cluster_id] * cluster_size)

        if len(selected_indices) != 64:
            continue

        quality, n_valid = compute_ball_quality(np.array(selected_indices), embeddings, selected_labels)

        if quality > best_quality:
            best_quality = quality
            best_indices = np.array(selected_indices)
            best_labels = selected_labels

    print(f"    Best radius-band configuration: quality={best_quality:.3f}")

    return best_indices, best_labels, best_quality


def find_optimal_64_greedy(embeddings, valuations, n_clusters=21):
    """Greedy search for optimal 64 points.

    Start with highest-valuation points (most structured), add points
    that maximize cluster separation.
    """
    print("\n  Greedy search for optimal 64 points...")

    target_pattern = sorted(GENETIC_CODE_DEGENERACY, reverse=True)  # Start with largest clusters

    # Sort all points by valuation (descending) then radius
    radii = np.linalg.norm(embeddings, axis=1)
    sort_key = list(zip(-valuations, radii, range(len(embeddings))))
    sort_key.sort()
    sorted_indices = [x[2] for x in sort_key]

    best_quality = 0
    best_indices = None
    best_labels = None

    np.random.seed(42)

    for trial in range(200):
        selected_indices = []
        selected_labels = []
        used = set()

        # Start with the highest valuation point
        start_idx = 0
        for i, idx in enumerate(sorted_indices):
            if idx not in used:
                start_idx = i
                break

        cluster_id = 0
        for cluster_size in target_pattern:
            # Find cluster_size points that are close to each other but far from existing
            cluster_points = []

            # Seed with a random high-valuation unused point
            candidates = [sorted_indices[i] for i in range(min(1000, len(sorted_indices))) if sorted_indices[i] not in used]

            if not candidates:
                candidates = [i for i in range(len(embeddings)) if i not in used]

            if len(candidates) < cluster_size:
                break

            # Pick seed randomly from top candidates
            seed = np.random.choice(candidates[: max(1, len(candidates) // 10 + 1)])
            cluster_points.append(seed)
            used.add(seed)

            # Greedily add nearest unused points
            while len(cluster_points) < cluster_size:
                best_next = None
                best_dist = float("inf")

                for cand in candidates:
                    if cand in used:
                        continue
                    # Distance to cluster centroid
                    centroid = embeddings[cluster_points].mean(axis=0)
                    d = np.linalg.norm(embeddings[cand] - centroid)
                    if d < best_dist:
                        best_dist = d
                        best_next = cand

                if best_next is None:
                    break

                cluster_points.append(best_next)
                used.add(best_next)

            selected_indices.extend(cluster_points)
            selected_labels.extend([cluster_id] * len(cluster_points))
            cluster_id += 1

        if len(selected_indices) != 64:
            continue

        quality, n_valid = compute_ball_quality(np.array(selected_indices), embeddings, selected_labels)

        if quality > best_quality:
            best_quality = quality
            best_indices = np.array(selected_indices)
            best_labels = selected_labels
            print(f"    Trial {trial}: quality={quality:.3f}, valid={n_valid}/21")

    print(f"    Best greedy configuration: quality={best_quality:.3f}")

    return best_indices, best_labels, best_quality


def analyze_best_configuration(indices, labels, embeddings, valuations):
    """Analyze the properties of the best configuration."""
    print("\n" + "=" * 70)
    print("ANALYSIS OF BEST CONFIGURATION")
    print("=" * 70)

    # Get properties of selected points
    selected_valuations = valuations[indices]
    selected_radii = np.linalg.norm(embeddings[indices], axis=1)

    # Cluster statistics
    unique_labels = sorted(set(labels))

    print(f"\n  Selected 64 points form {len(unique_labels)} clusters")
    print(f"\n  Cluster size distribution: {sorted(Counter(labels).values())}")
    print(f"  Target (genetic code):     {sorted(GENETIC_CODE_DEGENERACY)}")

    # Compare distributions
    actual_sizes = sorted(Counter(labels).values())
    target_sizes = sorted(GENETIC_CODE_DEGENERACY)

    if actual_sizes == target_sizes:
        print("\n  *** PERFECT MATCH to genetic code degeneracy pattern! ***")

    # Valuation distribution
    print("\n  Valuation distribution of selected points:")
    val_counts = Counter(selected_valuations.astype(int))
    for v in sorted(val_counts.keys()):
        print(f"    v={v}: {val_counts[v]} points")

    # Radius by cluster
    print("\n  Mean radius by cluster:")
    for c in unique_labels[:5]:  # Show first 5
        c_mask = np.array(labels) == c
        c_radii = selected_radii[c_mask]
        print(f"    Cluster {c} (n={c_mask.sum()}): radius = {c_radii.mean():.4f} +/- {c_radii.std():.4f}")

    # Valuation by cluster
    print("\n  Mean valuation by cluster:")
    for c in unique_labels[:5]:
        c_mask = np.array(labels) == c
        c_vals = selected_valuations[c_mask]
        print(f"    Cluster {c} (n={c_mask.sum()}): valuation = {c_vals.mean():.2f} +/- {c_vals.std():.2f}")

    return {
        "n_clusters": len(unique_labels),
        "cluster_sizes": sorted(Counter(labels).values()),
        "valuation_distribution": dict(val_counts),
        "mean_radius": float(selected_radii.mean()),
        "mean_valuation": float(selected_valuations.mean()),
    }


def visualize_configuration(indices, labels, embeddings, valuations, output_dir):
    """Visualize the best configuration."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    selected_emb = embeddings[indices]
    selected_vals = valuations[indices]
    selected_radii = np.linalg.norm(selected_emb, axis=1)

    # 1. PCA projection colored by cluster
    ax1 = axes[0, 0]
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(selected_emb)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, c in enumerate(unique_labels):
        mask = np.array(labels) == c
        ax1.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[i]],
            s=80,
            alpha=0.7,
            label=f"C{c}",
            edgecolors="black",
        )

    ax1.set_title("PCA of Selected 64 Points (by cluster)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # 2. Radius vs Valuation
    ax2 = axes[0, 1]
    for i, c in enumerate(unique_labels):
        mask = np.array(labels) == c
        ax2.scatter(
            selected_vals[mask],
            selected_radii[mask],
            c=[colors[i]],
            s=80,
            alpha=0.7,
        )

    ax2.set_xlabel("3-adic Valuation")
    ax2.set_ylabel("Hyperbolic Radius")
    ax2.set_title("Radius vs Valuation (by cluster)")

    # 3. Cluster size histogram
    ax3 = axes[1, 0]
    actual_sizes = sorted(Counter(labels).values())
    target_sizes = sorted(GENETIC_CODE_DEGENERACY)

    x = np.arange(len(actual_sizes))
    width = 0.35
    ax3.bar(x - width / 2, actual_sizes, width, label="Found", alpha=0.7)
    ax3.bar(x + width / 2, target_sizes, width, label="Genetic Code", alpha=0.7)
    ax3.set_xlabel("Cluster Index (sorted by size)")
    ax3.set_ylabel("Cluster Size")
    ax3.set_title("Cluster Size Distribution")
    ax3.legend()

    # 4. Valuation histogram of selected vs all
    ax4 = axes[1, 1]
    ax4.hist(valuations, bins=10, alpha=0.5, label="All 19683", density=True)
    ax4.hist(selected_vals, bins=10, alpha=0.7, label="Selected 64", density=True)
    ax4.set_xlabel("3-adic Valuation")
    ax4.set_ylabel("Density")
    ax4.set_title("Valuation Distribution")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "reverse_padic_search.png", dpi=150)
    plt.close()

    print(f"\n  Saved visualization to {output_dir}/reverse_padic_search.png")


def main():
    # Use local data directory instead of deprecated riemann_hypothesis_sandbox
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REVERSE P-ADIC SEARCH")
    print("Find 64 points that naturally form 21 p-adic balls")
    print("=" * 70)

    # Load embeddings (from 3-adic hyperbolic extraction)
    print("\nLoading hyperbolic embeddings...")
    embeddings_path = data_dir / "v5_11_3_embeddings.pt"
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        print("Run 07_extract_v5_11_3_embeddings.py first")
        return
    data = torch.load(embeddings_path, weights_only=False)

    z_B = data.get("z_B_hyp", data.get("z_hyperbolic"))
    if torch.is_tensor(z_B):
        z_B = z_B.numpy()

    print(f"Loaded embeddings: shape = {z_B.shape}")

    # Compute valuations
    print("Computing 3-adic valuations...")
    indices = np.arange(len(z_B))

    def valuation_3(n):
        if n == 0:
            return 9  # Max valuation for 0
        v = 0
        while n % 3 == 0:
            v += 1
            n //= 3
        return v

    valuations = np.array([valuation_3(i) for i in indices])
    print(f"Valuation range: {valuations.min()} - {valuations.max()}")

    # Results container
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "reverse_padic_search",
        "target_n_points": 64,
        "target_n_clusters": 21,
        "methods": {},
    }

    # Method 1: Random sampling with clustering
    print("\n" + "-" * 50)
    print("METHOD 1: Random Sampling + Hierarchical Clustering")
    print("-" * 50)
    idx1, labels1, quality1 = find_natural_clusters(z_B, n_points=64, n_clusters=21, n_samples=500)
    results["methods"]["random_clustering"] = {"quality": quality1}

    # Method 2: Valuation-based selection
    print("\n" + "-" * 50)
    print("METHOD 2: Valuation-Based Selection")
    print("-" * 50)
    idx2, labels2, quality2 = find_by_valuation_structure(z_B, valuations, n_clusters=21)
    results["methods"]["valuation_based"] = {"quality": quality2}

    # Method 3: Radius bands
    print("\n" + "-" * 50)
    print("METHOD 3: Radius Band Selection")
    print("-" * 50)
    idx3, labels3, quality3 = find_by_radius_bands(z_B, n_clusters=21)
    results["methods"]["radius_bands"] = {"quality": quality3}

    # Method 4: Greedy optimization
    print("\n" + "-" * 50)
    print("METHOD 4: Greedy Optimization")
    print("-" * 50)
    idx4, labels4, quality4 = find_optimal_64_greedy(z_B, valuations, n_clusters=21)
    results["methods"]["greedy"] = {"quality": quality4}

    # Find best method
    methods = [
        ("random_clustering", idx1, labels1, quality1),
        ("valuation_based", idx2, labels2, quality2),
        ("radius_bands", idx3, labels3, quality3),
        ("greedy", idx4, labels4, quality4),
    ]

    best_method = max(methods, key=lambda x: x[3] if x[3] else 0)

    print("\n" + "=" * 70)
    print("BEST METHOD: " + best_method[0])
    print(f"Quality score: {best_method[3]:.4f}")
    print("=" * 70)

    if best_method[1] is not None:
        # Analyze best configuration
        analysis = analyze_best_configuration(best_method[1], best_method[2], z_B, valuations)
        results["best_method"] = best_method[0]
        results["best_quality"] = best_method[3]
        results["best_analysis"] = analysis

        # Visualize
        visualize_configuration(best_method[1], best_method[2], z_B, valuations, output_dir)

        # Key insight: What makes these 64 points special?
        print("\n" + "=" * 70)
        print("KEY INSIGHT: What makes these 64 points special?")
        print("=" * 70)

        selected_vals = valuations[best_method[1]]
        all_vals_mean = valuations.mean()
        selected_vals_mean = selected_vals.mean()

        print(f"\n  Mean valuation (all 19683): {all_vals_mean:.3f}")
        print(f"  Mean valuation (selected 64): {selected_vals_mean:.3f}")

        if selected_vals_mean > all_vals_mean + 0.5:
            print("\n  *** Selected points have HIGHER valuation (more divisible by 3) ***")
            print("  This suggests the genetic code maps to 'deeper' algebraic operations!")

        # Check if high-valuation points cluster better
        high_val_mask = selected_vals >= 2
        if high_val_mask.sum() > 10:
            print(f"\n  Points with valuation >= 2: {high_val_mask.sum()}/64")

    # Save results
    results_file = output_dir / "reverse_padic_search.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    main()
