#!/usr/bin/env python3
"""
Predict Immunogenicity for All Arginine Sites

Apply the trained immunogenicity predictor (from Script 11) to all
arginine sites in the human proteome.

Output directory: results/proteome_wide/15_predictions/

Version: 1.0
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature columns (must match Script 11)
FEATURE_COLUMNS = [
    "embedding_norm",
    "embedding_norm_std",
    "cluster_homogeneity",
    "mean_neighbor_distance",
    "boundary_potential",
    "sequence_length",
    "n_arginines",
    "centroid_shift",
    "js_divergence",
    "entropy_change",
    "r_density",
    "entropy_per_r",
]

# Risk categories
RISK_THRESHOLDS = {
    "very_high": 0.90,
    "high": 0.75,
    "moderate": 0.50,
    "low": 0.25,
}

# Output configuration
SCRIPT_NUM = "15"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_predictions"
INPUT_SUBDIR = "14_geometric_features"


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


def get_training_data_dir() -> Path:
    """Get directory with training data for model fitting."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "hyperbolic"


# ============================================================================
# MODEL TRAINING (on known epitopes)
# ============================================================================


def load_training_data() -> tuple:
    """
    Load training data from the known RA epitopes analysis.

    Returns (X, y) for model training.
    """
    print("\n[1] Loading training data from known epitopes...")

    # Import the epitope database
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "augmented_db",
        Path(__file__).parent / "08_augmented_epitope_database.py",
    )
    augmented_db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augmented_db)
    RA_AUTOANTIGENS = augmented_db.RA_AUTOANTIGENS_AUGMENTED

    # Import feature extraction from Script 11
    import torch
    from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot,
                                  load_codon_encoder, poincare_distance)

    device = "cpu"
    encoder, _, _ = load_codon_encoder(device=device, version="3adic")

    X_list = []
    y_list = []

    for protein_id, protein in RA_AUTOANTIGENS.items():
        for epitope in protein["epitopes"]:
            sequence = epitope["sequence"]

            # Encode sequence
            embeddings = []
            cluster_probs_list = []

            for aa in sequence:
                codon = AA_TO_CODON.get(aa, "NNN")
                if codon == "NNN":
                    continue
                onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    probs, emb = encoder.get_cluster_probs(onehot)
                    embeddings.append(emb.numpy().squeeze())
                    cluster_probs_list.append(probs.numpy().squeeze())

            if len(embeddings) < 2:
                continue

            embeddings = np.array(embeddings)
            cluster_probs = np.array(cluster_probs_list)

            # Compute features (matching Script 11)
            norms = np.linalg.norm(embeddings, axis=1)
            cluster_ids = np.argmax(cluster_probs, axis=1)
            majority = np.argmax(np.bincount(cluster_ids))
            homogeneity = np.mean(cluster_ids == majority)

            neighbor_dists = [
                poincare_distance(
                    torch.tensor(embeddings[i]).float(),
                    torch.tensor(embeddings[i + 1]).float(),
                ).item()
                for i in range(len(embeddings) - 1)
            ]

            boundary_pots = []
            for i, emb in enumerate(embeddings):
                other_mask = cluster_ids != cluster_ids[i]
                if np.any(other_mask):
                    dists = [
                        poincare_distance(
                            torch.tensor(emb).float(),
                            torch.tensor(embeddings[j]).float(),
                        ).item()
                        for j in np.where(other_mask)[0]
                    ]
                    if dists:
                        boundary_pots.append(min(dists))

            # Citrullination features
            arg_positions = [i for i, aa in enumerate(sequence) if aa == "R"]
            if arg_positions:
                orig_centroid = np.mean(embeddings, axis=0)
                orig_probs = np.mean(cluster_probs, axis=0)
                orig_entropy = -np.sum(orig_probs * np.log(orig_probs + 1e-10))

                shifts, js_divs, ent_changes = [], [], []
                for r_pos in arg_positions:
                    cit_emb = np.delete(embeddings, r_pos, axis=0)
                    cit_probs = np.delete(cluster_probs, r_pos, axis=0)
                    if len(cit_emb) > 0:
                        cit_cent = np.mean(cit_emb, axis=0)
                        cit_p = np.mean(cit_probs, axis=0)
                        shifts.append(
                            poincare_distance(
                                torch.tensor(orig_centroid).float(),
                                torch.tensor(cit_cent).float(),
                            ).item()
                        )
                        m = 0.5 * (orig_probs + cit_p)
                        js_divs.append(
                            0.5
                            * (
                                np.sum(orig_probs * np.log((orig_probs + 1e-10) / (m + 1e-10)))
                                + np.sum(cit_p * np.log((cit_p + 1e-10) / (m + 1e-10)))
                            )
                        )
                        ent_changes.append(-np.sum(cit_p * np.log(cit_p + 1e-10)) - orig_entropy)

                centroid_shift = np.mean(shifts) if shifts else 0.0
                js_div = np.mean(js_divs) if js_divs else 0.0
                entropy_change = np.mean(ent_changes) if ent_changes else 0.0
            else:
                centroid_shift = js_div = entropy_change = 0.0

            n_arginines = len(arg_positions)
            seq_len = len(sequence)

            features = [
                np.mean(norms),  # embedding_norm
                np.std(norms),  # embedding_norm_std
                homogeneity,  # cluster_homogeneity
                (np.mean(neighbor_dists) if neighbor_dists else 0.0),  # mean_neighbor_distance
                (np.mean(boundary_pots) if boundary_pots else 0.0),  # boundary_potential
                seq_len,  # sequence_length
                n_arginines,  # n_arginines
                centroid_shift,  # centroid_shift
                js_div,  # js_divergence
                entropy_change,  # entropy_change
                n_arginines / seq_len,  # r_density
                entropy_change / max(n_arginines, 1),  # entropy_per_r
            ]

            X_list.append(features)
            y_list.append(1 if epitope["immunodominant"] else 0)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"  Loaded {len(y)} training samples")
    print(f"  Positive: {sum(y)} ({100*sum(y)/len(y):.1f}%)")

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train logistic regression model on known epitopes.

    Returns (model, scaler).
    """
    print("\n[2] Training prediction model...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # Report training performance
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    from sklearn.metrics import accuracy_score, roc_auc_score

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"  Training accuracy: {acc:.3f}")
    print(f"  Training AUC: {auc:.3f}")

    return model, scaler


# ============================================================================
# PREDICTION
# ============================================================================


def predict_immunogenicity(sites: List[Dict], model, scaler) -> List[Dict]:
    """
    Apply trained model to predict immunogenicity for all sites.
    """
    print(f"\n[3] Predicting immunogenicity for {len(sites):,} sites...")

    # Build feature matrix
    X_list = []
    valid_indices = []

    for i, site in enumerate(sites):
        try:
            features = [site.get(col, 0.0) for col in FEATURE_COLUMNS]
            if any(f is None or (isinstance(f, float) and np.isnan(f)) for f in features):
                continue
            X_list.append(features)
            valid_indices.append(i)
        except Exception:
            continue

    X = np.array(X_list)
    print(f"  Valid sites: {len(X):,} / {len(sites):,}")

    # Scale and predict
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Add predictions to sites
    results = []
    prob_idx = 0

    for i, site in enumerate(sites):
        result = site.copy()

        if i in valid_indices:
            prob = probabilities[prob_idx]
            prob_idx += 1

            result["immunogenic_probability"] = float(prob)

            # Assign risk category
            if prob >= RISK_THRESHOLDS["very_high"]:
                result["risk_category"] = "very_high"
            elif prob >= RISK_THRESHOLDS["high"]:
                result["risk_category"] = "high"
            elif prob >= RISK_THRESHOLDS["moderate"]:
                result["risk_category"] = "moderate"
            elif prob >= RISK_THRESHOLDS["low"]:
                result["risk_category"] = "low"
            else:
                result["risk_category"] = "very_low"
        else:
            result["immunogenic_probability"] = None
            result["risk_category"] = "unknown"

        results.append(result)

    # Statistics
    valid_probs = [r["immunogenic_probability"] for r in results if r["immunogenic_probability"] is not None]
    risk_counts = Counter(r["risk_category"] for r in results)

    print("\n  Prediction statistics:")
    print(f"    Mean probability: {np.mean(valid_probs):.3f}")
    print(f"    Median probability: {np.median(valid_probs):.3f}")
    print("\n  Risk distribution:")
    for category in [
        "very_high",
        "high",
        "moderate",
        "low",
        "very_low",
        "unknown",
    ]:
        count = risk_counts.get(category, 0)
        pct = 100 * count / len(results) if results else 0
        print(f"    {category}: {count:,} ({pct:.1f}%)")

    return results


# ============================================================================
# OUTPUT
# ============================================================================


def save_predictions(results: List[Dict], output_dir: Path):
    """Save predictions in multiple formats."""
    print("\n[4] Saving predictions...")

    # Full results as JSON
    json_path = output_dir / "predictions_full.json"
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved: {json_path} ({json_path.stat().st_size:,} bytes)")

    # Parquet for efficient storage
    try:
        df = pd.DataFrame(results)
        parquet_path = output_dir / "predictions.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"  Warning: Could not save Parquet ({e})")

    # CSV summary
    csv_cols = [
        "protein_id",
        "gene_name",
        "r_position",
        "window_sequence",
        "immunogenic_probability",
        "risk_category",
        "centroid_shift",
        "entropy_change",
    ]
    df_csv = pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in results])
    csv_path = output_dir / "predictions_summary.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Top candidates (high risk)
    high_risk = [r for r in results if r.get("risk_category") in ["very_high", "high"]]
    high_risk_sorted = sorted(
        high_risk,
        key=lambda x: x.get("immunogenic_probability", 0),
        reverse=True,
    )

    top_path = output_dir / "high_risk_candidates.csv"
    df_top = pd.DataFrame(high_risk_sorted[:1000])  # Top 1000
    df_top.to_csv(top_path, index=False)
    print(f"  Saved: {top_path} ({len(high_risk_sorted):,} high-risk sites)")

    # Top by protein (aggregate)
    protein_risk = {}
    for r in results:
        pid = r.get("protein_id")
        prob = r.get("immunogenic_probability")
        if pid and prob is not None:
            if pid not in protein_risk:
                protein_risk[pid] = {
                    "protein_id": pid,
                    "gene_name": r.get("gene_name"),
                    "probs": [],
                    "high_risk_count": 0,
                }
            protein_risk[pid]["probs"].append(prob)
            if r.get("risk_category") in ["very_high", "high"]:
                protein_risk[pid]["high_risk_count"] += 1

    # Calculate protein-level metrics
    for pid, data in protein_risk.items():
        data["mean_probability"] = np.mean(data["probs"])
        data["max_probability"] = np.max(data["probs"])
        data["n_arginines"] = len(data["probs"])
        del data["probs"]

    protein_df = pd.DataFrame(list(protein_risk.values()))
    protein_df = protein_df.sort_values("max_probability", ascending=False)
    protein_path = output_dir / "protein_risk_summary.csv"
    protein_df.to_csv(protein_path, index=False)
    print(f"  Saved: {protein_path} ({len(protein_df):,} proteins)")


def save_statistics(results: List[Dict], output_dir: Path):
    """Save prediction statistics."""
    print("\n[5] Saving statistics...")

    valid_results = [r for r in results if r.get("immunogenic_probability") is not None]

    stats = {
        "total_sites": len(results),
        "valid_predictions": len(valid_results),
        "risk_distribution": dict(Counter(r["risk_category"] for r in results)),
        "probability_stats": {
            "mean": float(np.mean([r["immunogenic_probability"] for r in valid_results])),
            "std": float(np.std([r["immunogenic_probability"] for r in valid_results])),
            "median": float(np.median([r["immunogenic_probability"] for r in valid_results])),
            "min": float(np.min([r["immunogenic_probability"] for r in valid_results])),
            "max": float(np.max([r["immunogenic_probability"] for r in valid_results])),
        },
        "high_risk_proteins": len(set(r["protein_id"] for r in valid_results if r.get("risk_category") in ["very_high", "high"])),
    }

    stats_path = output_dir / "prediction_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    return stats


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("PREDICT IMMUNOGENICITY")
    print("Applying trained model to all human proteome arginine sites")
    print("=" * 80)

    # Setup directories
    input_dir = get_input_dir()
    output_dir = get_output_dir()
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load geometric features
    print("\n[0] Loading geometric features...")
    features_path = input_dir / "geometric_features.json"

    if not features_path.exists():
        print(f"  ERROR: Features file not found: {features_path}")
        print("  Please run script 14_compute_geometric_features.py first")
        return

    with open(features_path, "r") as f:
        sites = json.load(f)
    print(f"  Loaded {len(sites):,} sites with features")

    # Train model on known epitopes
    X_train, y_train = load_training_data()
    model, scaler = train_model(X_train, y_train)

    # Predict
    results = predict_immunogenicity(sites, model, scaler)

    # Save outputs
    save_predictions(results, output_dir)
    stats = save_statistics(results, output_dir)

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")

    print("\nKey results:")
    print(f"  High-risk sites: {stats['risk_distribution'].get('very_high', 0) + stats['risk_distribution'].get('high', 0):,}")
    print(f"  High-risk proteins: {stats['high_risk_proteins']:,}")

    return results


if __name__ == "__main__":
    main()
