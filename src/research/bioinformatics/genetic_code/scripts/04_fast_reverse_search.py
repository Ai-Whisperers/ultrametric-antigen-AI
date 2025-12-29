"""
07c_fast_reverse_search.py - Fast reverse search using model's embedding structure

OPTIMIZATION: Use radius (encodes valuation) for O(n) binning, angular clustering
within bands, then only compute O(64²) geodesics for final verification.

Usage:
    python 07c_fast_reverse_search.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

# Genetic code degeneracy: 21 groups with sizes [1,1,2,2,2,2,2,2,2,2,2,3,3,4,4,4,4,4,6,6,6]
DEGENERACY_PATTERN = [
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


def poincare_distance_matrix(embeddings, c=1.0, eps=1e-7):
    """Compute pairwise Poincare distances for small set of points."""
    n = len(embeddings)
    D = np.zeros((n, n))

    norms_sq = np.sum(embeddings**2, axis=1)

    for i in range(n):
        for j in range(i + 1, n):
            diff_sq = np.sum((embeddings[i] - embeddings[j]) ** 2)
            denom = (1 - c * norms_sq[i]) * (1 - c * norms_sq[j])
            denom = max(denom, eps)
            arg = 1 + 2 * c * diff_sq / denom
            arg = max(arg, 1.0 + eps)
            d = (1 / np.sqrt(c)) * np.arccosh(arg)
            D[i, j] = d
            D[j, i] = d

    return D


def compute_ball_quality(D, labels):
    """Check if clusters form valid p-adic balls."""
    unique_labels = sorted(set(labels))
    n_valid = 0
    margins = []

    for c in unique_labels:
        c_mask = np.array(labels) == c
        c_indices = np.where(c_mask)[0]

        if len(c_indices) < 2:
            n_valid += 1
            continue

        # Max within-cluster distance
        eps_within = 0
        for i in c_indices:
            for j in c_indices:
                if i < j:
                    eps_within = max(eps_within, D[i, j])

        # Min between-cluster distance
        other_indices = np.where(~c_mask)[0]
        eps_between = float("inf")
        for i in c_indices:
            for j in other_indices:
                eps_between = min(eps_between, D[i, j])

        margin = eps_between - eps_within
        margins.append(margin)

        if eps_within < eps_between:
            n_valid += 1

    return n_valid, len(unique_labels), margins


def main():
    # Use local data directory instead of deprecated riemann_hypothesis_sandbox
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAST REVERSE P-ADIC SEARCH")
    print("Using model's radius structure for O(n) binning")
    print("=" * 70)

    # Load embeddings (from 3-adic hyperbolic extraction)
    print("\nLoading embeddings...")
    embeddings_path = data_dir / "v5_11_3_embeddings.pt"
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        print("Run 07_extract_v5_11_3_embeddings.py first")
        return
    data = torch.load(embeddings_path, weights_only=False)

    z_B = data.get("z_B_hyp", data.get("z_hyperbolic"))
    if torch.is_tensor(z_B):
        z_B = z_B.numpy()

    n_total = len(z_B)
    print(f"Loaded {n_total} embeddings, dim={z_B.shape[1]}")

    # Step 1: Compute radius for all points (O(n))
    print("\nStep 1: Computing radii (O(n))...")
    radii = np.linalg.norm(z_B, axis=1)

    # Step 2: Bin by radius into 10 bands (matching valuation levels 0-9)
    print("Step 2: Binning by radius into valuation bands...")

    # Use percentiles to create balanced bands
    percentiles = np.percentile(radii, np.linspace(0, 100, 11))
    band_assignments = np.digitize(radii, percentiles[1:-1])  # 0-9

    band_counts = Counter(band_assignments)
    print(f"  Band sizes: {[band_counts[i] for i in range(10)]}")

    # Step 3: Angular clustering within bands
    print("Step 3: Angular clustering within bands...")

    # Normalize to unit sphere for angular clustering
    z_normalized = z_B / (radii[:, np.newaxis] + 1e-8)

    # For each band, find natural angular clusters
    band_clusters = {}  # band -> list of (center_idx, cluster_indices)

    for band in range(10):
        band_mask = band_assignments == band
        band_indices = np.where(band_mask)[0]

        if len(band_indices) < 5:
            # Too few points, treat as one cluster
            band_clusters[band] = [(band_indices[0], band_indices)]
            continue

        # K-means on normalized vectors (angular clustering)
        n_clusters = min(10, len(band_indices) // 5)
        if n_clusters < 2:
            band_clusters[band] = [(band_indices[0], band_indices)]
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(z_normalized[band_indices])

        clusters = []
        for c in range(n_clusters):
            c_mask = cluster_labels == c
            c_indices = band_indices[c_mask]
            if len(c_indices) > 0:
                # Find center (point closest to centroid)
                centroid = z_B[c_indices].mean(axis=0)
                dists = np.linalg.norm(z_B[c_indices] - centroid, axis=1)
                center_local = np.argmin(dists)
                center_idx = c_indices[center_local]
                clusters.append((center_idx, c_indices))

        band_clusters[band] = clusters

    total_clusters = sum(len(v) for v in band_clusters.values())
    print(f"  Found {total_clusters} angular clusters across all bands")

    # Step 4: Select 64 points matching degeneracy pattern
    print("\nStep 4: Selecting 64 points matching genetic code pattern...")

    # Strategy: Pick from higher bands (higher valuation = more structured)
    # Match the degeneracy pattern [1,1,2,2,...,6,6,6]

    selected_indices = []
    selected_labels = []
    used_clusters = set()

    # Sort pattern by size (pick larger clusters first from structured bands)
    pattern_with_id = list(enumerate(DEGENERACY_PATTERN))

    # Prioritize bands 3-7 (mid-valuation, good structure)
    priority_bands = [5, 4, 6, 3, 7, 2, 8, 1, 9, 0]

    cluster_id = 0
    for target_size in sorted(DEGENERACY_PATTERN, reverse=True):
        found = False

        for band in priority_bands:
            if found:
                break
            for center_idx, cluster_indices in band_clusters.get(band, []):
                if center_idx in used_clusters:
                    continue
                if len(cluster_indices) >= target_size:
                    # Pick target_size points from this cluster
                    chosen = cluster_indices[:target_size]
                    selected_indices.extend(chosen)
                    selected_labels.extend([cluster_id] * target_size)
                    used_clusters.add(center_idx)
                    cluster_id += 1
                    found = True
                    break

        if not found:
            # Fallback: pick any unused points
            for band in range(10):
                if found:
                    break
                for center_idx, cluster_indices in band_clusters.get(band, []):
                    if center_idx in used_clusters:
                        continue
                    available = [i for i in cluster_indices if i not in selected_indices]
                    if len(available) >= target_size:
                        chosen = available[:target_size]
                        selected_indices.extend(chosen)
                        selected_labels.extend([cluster_id] * target_size)
                        used_clusters.add(center_idx)
                        cluster_id += 1
                        found = True
                        break

    print(f"  Selected {len(selected_indices)} points in {cluster_id} clusters")
    print(f"  Cluster sizes: {sorted(Counter(selected_labels).values())}")
    print(f"  Target sizes:  {sorted(DEGENERACY_PATTERN)}")

    if len(selected_indices) != 64:
        print(f"  WARNING: Only found {len(selected_indices)}/64 points")
        # Pad with random points if needed
        remaining = 64 - len(selected_indices)
        unused = [i for i in range(n_total) if i not in selected_indices]
        extra = np.random.choice(unused, remaining, replace=False)
        selected_indices.extend(extra)
        selected_labels.extend([cluster_id] * remaining)

    selected_indices = np.array(selected_indices[:64])
    selected_labels = selected_labels[:64]

    # Step 5: Verify with geodesic distances (O(64²) = 4096 operations)
    print("\nStep 5: Computing geodesic distances for 64 points...")

    selected_emb = z_B[selected_indices]
    D = poincare_distance_matrix(selected_emb)

    print(f"  Geodesic distance range: [{D[D>0].min():.4f}, {D.max():.4f}]")

    # Check ball quality
    n_valid, n_total_clusters, margins = compute_ball_quality(D, selected_labels)

    print("\n  P-ADIC BALL VERIFICATION:")
    print(f"    Valid balls: {n_valid}/{n_total_clusters}")
    print(f"    Mean margin: {np.mean(margins):.4f}")
    print(f"    Positive margins: {sum(1 for m in margins if m > 0)}/{len(margins)}")

    # Step 6: Analyze the selected indices
    print("\n" + "=" * 70)
    print("ANALYSIS OF SELECTED 64 INDICES")
    print("=" * 70)

    selected_radii = radii[selected_indices]
    selected_bands = band_assignments[selected_indices]

    print(f"\n  Radius range: [{selected_radii.min():.4f}, {selected_radii.max():.4f}]")
    print(f"  Mean radius: {selected_radii.mean():.4f} (all points: {radii.mean():.4f})")
    print(f"  Band distribution: {dict(Counter(selected_bands))}")

    # The key output: these 64 ternary indices
    print("\n  SELECTED TERNARY INDICES (these are the 'natural codon positions'):")
    print(f"  {sorted(selected_indices)[:20]}... (showing first 20)")

    # Step 7: Visualization
    print("\nStep 7: Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Radius histogram: selected vs all
    ax1 = axes[0, 0]
    ax1.hist(radii, bins=50, alpha=0.5, label="All 19683", density=True)
    ax1.hist(selected_radii, bins=20, alpha=0.7, label="Selected 64", density=True)
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("Density")
    ax1.set_title("Radius Distribution")
    ax1.legend()

    # 2. Geodesic distance matrix
    ax2 = axes[0, 1]
    # Sort by cluster label
    sort_idx = np.argsort(selected_labels)
    D_sorted = D[np.ix_(sort_idx, sort_idx)]
    im = ax2.imshow(D_sorted, cmap="viridis")
    ax2.set_title("Geodesic Distance Matrix (sorted by cluster)")
    plt.colorbar(im, ax=ax2)

    # 3. Within vs between distances
    ax3 = axes[1, 0]
    within_dists = []
    between_dists = []
    for i in range(64):
        for j in range(i + 1, 64):
            if selected_labels[i] == selected_labels[j]:
                within_dists.append(D[i, j])
            else:
                between_dists.append(D[i, j])

    ax3.hist(
        within_dists,
        bins=20,
        alpha=0.7,
        label=f"Within (n={len(within_dists)})",
        density=True,
    )
    ax3.hist(
        between_dists,
        bins=20,
        alpha=0.7,
        label=f"Between (n={len(between_dists)})",
        density=True,
    )
    ax3.axvline(np.mean(within_dists), color="blue", linestyle="--")
    ax3.axvline(np.mean(between_dists), color="orange", linestyle="--")
    ax3.set_xlabel("Geodesic Distance")
    ax3.set_title("Within vs Between Cluster Distances")
    ax3.legend()

    # 4. Cluster sizes comparison
    ax4 = axes[1, 1]
    actual = sorted(Counter(selected_labels).values())
    target = sorted(DEGENERACY_PATTERN)
    x = np.arange(len(actual))
    ax4.bar(x - 0.2, actual, 0.4, label="Found")
    ax4.bar(x + 0.2, target, 0.4, label="Genetic Code")
    ax4.set_xlabel("Cluster (sorted by size)")
    ax4.set_ylabel("Size")
    ax4.set_title("Cluster Size Distribution")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "fast_reverse_search.png", dpi=150)
    plt.close()
    print(f"  Saved to {output_dir}/fast_reverse_search.png")

    # Save results
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_valid_balls": int(n_valid),
        "n_total_clusters": int(n_total_clusters),
        "mean_margin": float(np.mean(margins)),
        "positive_margins": int(sum(1 for m in margins if m > 0)),
        "selected_indices": [int(i) for i in selected_indices],
        "cluster_labels": [int(l) for l in selected_labels],
        "cluster_sizes": sorted(Counter(selected_labels).values()),
        "mean_within_dist": float(np.mean(within_dists)),
        "mean_between_dist": float(np.mean(between_dists)),
        "separation_ratio": float(np.mean(between_dists) / np.mean(within_dists)),
    }

    with open(output_dir / "fast_reverse_search.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved results to {output_dir}/fast_reverse_search.json")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Valid p-adic balls: {n_valid}/{n_total_clusters} ({100*n_valid/n_total_clusters:.1f}%)
    Mean within-cluster distance: {np.mean(within_dists):.4f}
    Mean between-cluster distance: {np.mean(between_dists):.4f}
    Separation ratio: {np.mean(between_dists)/np.mean(within_dists):.2f}x

    {'*** GOOD SEPARATION: Clusters are well-defined! ***' if np.mean(between_dists) > np.mean(within_dists) * 1.5 else 'Weak separation between clusters'}
    """
    )

    return results


if __name__ == "__main__":
    main()
