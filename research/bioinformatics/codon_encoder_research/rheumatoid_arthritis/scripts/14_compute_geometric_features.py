#!/usr/bin/env python3
"""
Compute Geometric Features for Arginine Sites

For each arginine site extracted in Script 13:
- Encode the window sequence using 3-adic codon encoder
- Compute hyperbolic (Poincaré ball) features
- Simulate citrullination and compute shift metrics

Output directory: results/proteome_wide/14_geometric_features/

Version: 1.0
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
# Local imports
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, load_codon_encoder,
                              poincare_distance)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Processing
BATCH_SIZE = 1000  # Process in batches for progress reporting
CHECKPOINT_INTERVAL = 10000  # Save checkpoint every N sites

# Output configuration
SCRIPT_NUM = "14"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_geometric_features"
INPUT_SUBDIR = "13_arginine_contexts"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "proteome_wide" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_input_dir() -> Path:
    """Get input directory (from previous script)."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "proteome_wide" / INPUT_SUBDIR


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================


def encode_window(window: str, encoder, device: str = "cpu") -> tuple:
    """
    Encode a window sequence to hyperbolic embeddings.

    Returns:
        (embeddings, cluster_probs) or (None, None) if encoding fails
    """
    embeddings = []
    cluster_probs = []

    for aa in window:
        if aa == "X":  # Padding
            continue

        codon = AA_TO_CODON.get(aa)
        if codon is None:
            continue

        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs, emb = encoder.get_cluster_probs(onehot)
            embeddings.append(emb.cpu().numpy().squeeze())
            cluster_probs.append(probs.cpu().numpy().squeeze())

    if len(embeddings) == 0:
        return None, None

    return np.array(embeddings), np.array(cluster_probs)


def compute_site_features(site: Dict, encoder, device: str = "cpu") -> Optional[Dict]:
    """
    Compute all geometric features for a single arginine site.

    Returns feature dict or None if computation fails.
    """
    window = site["window_sequence"]
    r_pos = site["r_pos_in_window"]

    # Encode window
    embeddings, cluster_probs = encode_window(window, encoder, device)

    if embeddings is None or len(embeddings) < 2:
        return None

    # Basic embedding metrics
    norms = np.linalg.norm(embeddings, axis=1)

    # Cluster homogeneity
    cluster_ids = np.argmax(cluster_probs, axis=1)
    if len(cluster_ids) > 0:
        majority_cluster = np.argmax(np.bincount(cluster_ids))
        homogeneity = np.mean(cluster_ids == majority_cluster)
    else:
        homogeneity = 0.0

    # Neighbor distances (Poincaré geodesic)
    neighbor_dists = []
    for i in range(len(embeddings) - 1):
        d = poincare_distance(
            torch.tensor(embeddings[i]).float(),
            torch.tensor(embeddings[i + 1]).float(),
        ).item()
        neighbor_dists.append(d)
    mean_neighbor_dist = np.mean(neighbor_dists) if neighbor_dists else 0.0

    # Boundary potential
    boundary_potentials = []
    for i, emb in enumerate(embeddings):
        my_cluster = cluster_ids[i]
        other_mask = cluster_ids != my_cluster
        if np.any(other_mask):
            other_embs = embeddings[other_mask]
            dists = [poincare_distance(torch.tensor(emb).float(), torch.tensor(other).float()).item() for other in other_embs]
            if dists:
                boundary_potentials.append(min(dists))
    mean_boundary = np.mean(boundary_potentials) if boundary_potentials else 0.0

    features = {
        "embedding_norm": float(np.mean(norms)),
        "embedding_norm_std": float(np.std(norms)),
        "cluster_homogeneity": float(homogeneity),
        "mean_neighbor_distance": float(mean_neighbor_dist),
        "boundary_potential": float(mean_boundary),
    }

    # Find R position in the encoded sequence (accounting for X padding)
    valid_positions = [i for i, aa in enumerate(window) if aa != "X"]

    # Citrullination simulation
    # Find which embedding index corresponds to R
    r_idx = None
    encoded_idx = 0
    for i, aa in enumerate(window):
        if aa == "X":
            continue
        if i == r_pos:
            r_idx = encoded_idx
            break
        encoded_idx += 1

    if r_idx is not None and len(embeddings) > 1:
        # Original centroid and distribution
        original_centroid = np.mean(embeddings, axis=0)
        original_probs = np.mean(cluster_probs, axis=0)
        original_entropy = -np.sum(original_probs * np.log(original_probs + 1e-10))

        # Citrullinated (R removed)
        cit_embeddings = np.delete(embeddings, r_idx, axis=0)
        cit_probs = np.delete(cluster_probs, r_idx, axis=0)

        if len(cit_embeddings) > 0:
            cit_centroid = np.mean(cit_embeddings, axis=0)
            cit_probs_mean = np.mean(cit_probs, axis=0)
            cit_entropy = -np.sum(cit_probs_mean * np.log(cit_probs_mean + 1e-10))

            # Centroid shift (Poincaré distance)
            centroid_shift = poincare_distance(
                torch.tensor(original_centroid).float(),
                torch.tensor(cit_centroid).float(),
            ).item()

            # JS divergence
            m = 0.5 * (original_probs + cit_probs_mean)
            js_div = 0.5 * (
                np.sum(original_probs * np.log((original_probs + 1e-10) / (m + 1e-10)))
                + np.sum(cit_probs_mean * np.log((cit_probs_mean + 1e-10) / (m + 1e-10)))
            )

            # Entropy change
            entropy_change = cit_entropy - original_entropy

            features["centroid_shift"] = float(centroid_shift)
            features["js_divergence"] = float(js_div)
            features["entropy_change"] = float(entropy_change)
        else:
            features["centroid_shift"] = 0.0
            features["js_divergence"] = 0.0
            features["entropy_change"] = 0.0
    else:
        features["centroid_shift"] = 0.0
        features["js_divergence"] = 0.0
        features["entropy_change"] = 0.0

    # Derived features
    n_arginines = site.get("n_arginines_in_window", window.count("R"))
    seq_len = len([aa for aa in window if aa != "X"])

    features["sequence_length"] = seq_len
    features["n_arginines"] = n_arginines
    features["r_density"] = n_arginines / max(seq_len, 1)
    features["entropy_per_r"] = features["entropy_change"] / max(n_arginines, 1)

    return features


def process_all_sites(sites: List[Dict], encoder, device: str = "cpu", output_dir: Path = None) -> List[Dict]:
    """
    Process all arginine sites and compute features.

    Includes checkpointing for long runs.
    """
    print(f"\n[1] Processing {len(sites):,} arginine sites...")

    results = []
    failed = 0
    start_time = time.time()

    # Check for existing checkpoint
    checkpoint_path = output_dir / "checkpoint.json" if output_dir else None
    start_idx = 0

    if checkpoint_path and checkpoint_path.exists():
        print("  Found checkpoint, resuming...")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        results = checkpoint.get("results", [])
        start_idx = checkpoint.get("processed", 0)
        print(f"  Resuming from site {start_idx:,}")

    for i, site in enumerate(sites[start_idx:], start=start_idx):
        # Compute features
        features = compute_site_features(site, encoder, device)

        if features is None:
            failed += 1
            continue

        # Combine site metadata with computed features
        result = {
            "protein_id": site["protein_id"],
            "gene_name": site["gene_name"],
            "r_position": site["r_position"],
            "window_sequence": site["window_sequence"],
            **features,
            # Keep some metadata for enrichment
            "go_cc": site.get("go_cc", ""),
            "go_bp": site.get("go_bp", ""),
            "subcellular_location": site.get("subcellular_location", ""),
            "has_disease_annotation": site.get("has_disease_annotation", False),
        }
        results.append(result)

        # Progress reporting
        processed = i + 1
        if processed % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (len(sites) - processed) / rate if rate > 0 else 0
            print(
                f"  Processed {processed:,} / {len(sites):,} " f"({100*processed/len(sites):.1f}%) - " f"{rate:.0f} sites/sec - ETA: {eta/60:.1f} min"
            )

        # Checkpointing
        if checkpoint_path and processed % CHECKPOINT_INTERVAL == 0:
            checkpoint = {"processed": processed, "results": results}
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Successful: {len(results):,}")
    print(f"  Failed: {failed:,}")

    # Remove checkpoint on success
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


# ============================================================================
# STATISTICS
# ============================================================================


def compute_feature_statistics(results: List[Dict], output_dir: Path) -> Dict:
    """Compute statistics on computed features."""
    print("\n[2] Computing feature statistics...")

    feature_names = [
        "embedding_norm",
        "embedding_norm_std",
        "cluster_homogeneity",
        "mean_neighbor_distance",
        "boundary_potential",
        "centroid_shift",
        "js_divergence",
        "entropy_change",
        "r_density",
        "entropy_per_r",
    ]

    stats = {"total_sites": len(results), "features": {}}

    for feat in feature_names:
        values = [r[feat] for r in results if feat in r]
        if values:
            stats["features"][feat] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    # Save
    stats_path = output_dir / "feature_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Print key stats
    print("\n  Feature summary:")
    for feat in ["centroid_shift", "entropy_change", "js_divergence"]:
        if feat in stats["features"]:
            s = stats["features"][feat]
            print(f"    {feat}: mean={s['mean']:.4f}, std={s['std']:.4f}")

    return stats


# ============================================================================
# SAVE OUTPUTS
# ============================================================================


def save_results(results: List[Dict], output_dir: Path):
    """Save computed features in multiple formats."""
    print("\n[3] Saving results...")

    # Save as JSON
    json_path = output_dir / "geometric_features.json"
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved: {json_path} ({json_path.stat().st_size:,} bytes)")

    # Save as Parquet (efficient)
    try:
        df = pd.DataFrame(results)
        parquet_path = output_dir / "geometric_features.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"  Warning: Could not save Parquet ({e})")

    # Save as CSV (readable)
    csv_columns = [
        "protein_id",
        "gene_name",
        "r_position",
        "window_sequence",
        "embedding_norm",
        "cluster_homogeneity",
        "centroid_shift",
        "js_divergence",
        "entropy_change",
        "r_density",
    ]
    df_csv = pd.DataFrame([{k: r.get(k) for k in csv_columns} for r in results])
    csv_path = output_dir / "geometric_features_summary.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({csv_path.stat().st_size:,} bytes)")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("COMPUTE GEOMETRIC FEATURES")
    print("Hyperbolic (Poincaré ball) features for arginine sites")
    print("=" * 80)

    # Setup directories
    input_dir = get_input_dir()
    output_dir = get_output_dir()
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load arginine sites
    print("\n[0] Loading arginine sites...")
    sites_path = input_dir / "arginine_sites.json"

    if not sites_path.exists():
        print(f"  ERROR: Sites file not found: {sites_path}")
        print("  Please run script 13_extract_arginine_contexts.py first")
        return

    with open(sites_path, "r") as f:
        sites = json.load(f)
    print(f"  Loaded {len(sites):,} arginine sites")

    # Load encoder
    print("\n  Loading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, _, _ = load_codon_encoder(device=device, version="3adic")
    print("  Encoder loaded")

    # Process sites
    results = process_all_sites(sites, encoder, device, output_dir)

    # Compute statistics
    stats = compute_feature_statistics(results, output_dir)

    # Save results
    save_results(results, output_dir)

    print("\n" + "=" * 80)
    print("FEATURE COMPUTATION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    print(f"\nReady for Script 15: {len(results):,} sites with features")

    return results


if __name__ == "__main__":
    main()
