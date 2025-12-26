# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
HIV Tropism Switching Analysis

Analyze coreceptor tropism (CCR5 vs CXCR4) using V3 loop sequences
and gp120 alignments with p-adic hyperbolic codon geometry.

Analyses performed:
1. CCR5 vs CXCR4 hyperbolic separation
2. Tropism-switching trajectory mapping
3. Glycan shield correlation with tropism
4. Tropism classifier using hyperbolic features

Input:
- HIV_V3_coreceptor (2,935 V3 sequences with tropism labels)
- cview_gp120 (712 aligned gp120 sequences)

Output: Tropism analysis results and visualizations
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from codon_extraction import encode_amino_acid_sequence, find_glycan_sites
from hyperbolic_utils import load_hyperbolic_encoder
from unified_data_loader import (
    load_gp120_alignments,
    load_gp120_tropism_labels,
    load_v3_coreceptor,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "tropism"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# V3 loop key positions (11/25 rule positions for tropism)
KEY_V3_POSITIONS = {
    11: "Position 11 (basic charge)",
    25: "Position 25 (basic charge)",
    13: "Position 13",
    24: "Position 24",
    32: "Position 32",
}


# ============================================================================
# DATA LOADING
# ============================================================================


def load_tropism_data() -> tuple:
    """Load V3 coreceptor and gp120 tropism data."""
    print("Loading tropism datasets...")

    v3_df = None
    gp120_seqs = None
    gp120_labels = None

    # Load V3 coreceptor data
    try:
        v3_df = load_v3_coreceptor()
        print(f"  V3 coreceptor: {len(v3_df)} sequences")

        # Identify tropism column
        tropism_col = None
        for col in ["tropism", "Tropism", "label", "Label", "target"]:
            if col in v3_df.columns:
                tropism_col = col
                break

        if tropism_col:
            v3_df["tropism_label"] = v3_df[tropism_col].astype(str).str.upper()
            print(f"    CCR5 (R5): {(v3_df['tropism_label'].str.contains('R5|CCR5')).sum()}")
            print(f"    CXCR4 (X4): {(v3_df['tropism_label'].str.contains('X4|CXCR4')).sum()}")
    except Exception as e:
        print(f"  V3 coreceptor: Failed to load - {e}")

    # Load gp120 alignments
    try:
        gp120_seqs = load_gp120_alignments()
        gp120_labels = load_gp120_tropism_labels()
        print(f"  gp120 alignments: {len(gp120_seqs)} sequences")
        if gp120_labels:
            print(f"    With tropism labels: {len(gp120_labels)}")
    except Exception as e:
        print(f"  gp120 alignments: Failed to load - {e}")

    return v3_df, gp120_seqs, gp120_labels


def prepare_v3_dataset(v3_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare V3 dataset for analysis."""
    print("\nPreparing V3 dataset...")

    if v3_df is None or v3_df.empty:
        return pd.DataFrame()

    # Find sequence column
    seq_col = None
    for col in ["sequence", "Sequence", "seq", "V3", "v3_sequence"]:
        if col in v3_df.columns:
            seq_col = col
            break

    if seq_col is None:
        # Try to find a column with sequence-like content
        for col in v3_df.columns:
            sample = v3_df[col].iloc[0] if len(v3_df) > 0 else ""
            if isinstance(sample, str) and len(sample) > 20 and all(c in "ACDEFGHIKLMNPQRSTVWY-" for c in sample.upper()):
                seq_col = col
                break

    if seq_col is None:
        print("  Could not identify sequence column")
        return pd.DataFrame()

    v3_df["v3_sequence"] = v3_df[seq_col].astype(str).str.upper()

    # Clean sequences
    v3_df = v3_df[v3_df["v3_sequence"].str.len() >= 20].copy()

    # Standardize tropism labels - handle both boolean columns and label columns
    if "CXCR4" in v3_df.columns and "CCR5" in v3_df.columns:
        # Boolean columns directly
        v3_df["is_x4"] = v3_df["CXCR4"].astype(bool)
        v3_df["is_r5"] = v3_df["CCR5"].astype(bool)
    elif "tropism_label" in v3_df.columns:
        v3_df["is_x4"] = v3_df["tropism_label"].str.contains("X4|CXCR4", case=False, na=False)
        v3_df["is_r5"] = v3_df["tropism_label"].str.contains("R5|CCR5", case=False, na=False)
    else:
        # Try to find tropism column
        for col in ["tropism", "Tropism", "label", "Label", "target"]:
            if col in v3_df.columns:
                v3_df["is_x4"] = v3_df[col].astype(str).str.upper().str.contains("X4|CXCR4")
                v3_df["is_r5"] = v3_df[col].astype(str).str.upper().str.contains("R5|CCR5")
                break
        else:
            print("  Could not identify tropism column")
            return pd.DataFrame()

    # Calculate sequence properties
    v3_df["seq_length"] = v3_df["v3_sequence"].str.len()
    v3_df["n_basic"] = v3_df["v3_sequence"].apply(lambda x: sum(c in "RK" for c in x))
    v3_df["n_acidic"] = v3_df["v3_sequence"].apply(lambda x: sum(c in "DE" for c in x))
    v3_df["net_charge"] = v3_df["n_basic"] - v3_df["n_acidic"]

    print(f"  Prepared {len(v3_df)} sequences")

    return v3_df


# ============================================================================
# HYPERBOLIC ANALYSIS
# ============================================================================


def encode_v3_sequences(v3_df: pd.DataFrame, encoder) -> pd.DataFrame:
    """Encode V3 sequences to hyperbolic embeddings."""
    print("\nEncoding V3 sequences to hyperbolic space...")

    embeddings_list = []
    mean_embeddings = []
    radii = []

    for _, row in v3_df.iterrows():
        seq = row["v3_sequence"]

        try:
            emb = encode_amino_acid_sequence(seq, encoder)
            if len(emb) > 0:
                embeddings_list.append(emb)
                mean_embeddings.append(emb.mean(axis=0))
                radii.append(np.mean(np.linalg.norm(emb, axis=1)))
            else:
                embeddings_list.append(None)
                mean_embeddings.append(None)
                radii.append(None)
        except Exception:
            embeddings_list.append(None)
            mean_embeddings.append(None)
            radii.append(None)

    v3_df["embeddings"] = embeddings_list
    v3_df["mean_embedding"] = mean_embeddings
    v3_df["mean_radius"] = radii

    valid_count = v3_df["embeddings"].notna().sum()
    print(f"  Encoded {valid_count} sequences")

    return v3_df


def analyze_tropism_separation(v3_df: pd.DataFrame) -> dict:
    """Analyze geometric separation between CCR5 and CXCR4 sequences."""
    print("\nAnalyzing tropism separation...")

    results = {}

    # Filter valid embeddings
    valid_df = v3_df[v3_df["mean_embedding"].notna()].copy()

    if len(valid_df) < 10:
        print("  Insufficient data for analysis")
        return results

    # Get CCR5 and CXCR4 groups
    r5_df = valid_df[valid_df["is_r5"]]
    x4_df = valid_df[valid_df["is_x4"]]

    print(f"  CCR5 sequences: {len(r5_df)}")
    print(f"  CXCR4 sequences: {len(x4_df)}")

    if len(r5_df) < 5 or len(x4_df) < 5:
        print("  Insufficient data for comparison")
        return results

    # Calculate centroids
    r5_centroids = np.array([e for e in r5_df["mean_embedding"] if e is not None])
    x4_centroids = np.array([e for e in x4_df["mean_embedding"] if e is not None])

    r5_centroid = r5_centroids.mean(axis=0)
    x4_centroid = x4_centroids.mean(axis=0)

    # Distance between centroids
    centroid_distance = np.linalg.norm(r5_centroid - x4_centroid)

    # Radial positions
    r5_radii = r5_df["mean_radius"].dropna().values
    x4_radii = x4_df["mean_radius"].dropna().values

    # Statistical comparison
    mw_stat, mw_pval = stats.mannwhitneyu(r5_radii, x4_radii)

    results = {
        "n_r5": len(r5_df),
        "n_x4": len(x4_df),
        "r5_mean_radius": r5_radii.mean(),
        "r5_std_radius": r5_radii.std(),
        "x4_mean_radius": x4_radii.mean(),
        "x4_std_radius": x4_radii.std(),
        "centroid_distance": centroid_distance,
        "mann_whitney_pval": mw_pval,
        "r5_centroid": r5_centroid,
        "x4_centroid": x4_centroid,
    }

    print(f"  Centroid distance: {centroid_distance:.4f}")
    print(f"  R5 mean radius: {r5_radii.mean():.4f}")
    print(f"  X4 mean radius: {x4_radii.mean():.4f}")
    print(f"  Mann-Whitney p-value: {mw_pval:.6f}")

    return results


def analyze_position_importance(v3_df: pd.DataFrame, encoder) -> pd.DataFrame:
    """Analyze importance of each V3 position for tropism."""
    print("\nAnalyzing position importance for tropism...")

    results = []

    valid_df = v3_df[(v3_df["is_r5"] | v3_df["is_x4"]) & v3_df["embeddings"].notna()].copy()

    if len(valid_df) < 20:
        print("  Insufficient data")
        return pd.DataFrame()

    # Get minimum sequence length
    min_len = min(len(e) for e in valid_df["embeddings"] if e is not None)

    for pos in range(min_len):
        # Extract embeddings at this position
        r5_emb = []
        x4_emb = []

        for _, row in valid_df.iterrows():
            emb = row["embeddings"]
            if emb is not None and len(emb) > pos:
                if row["is_r5"]:
                    r5_emb.append(emb[pos])
                elif row["is_x4"]:
                    x4_emb.append(emb[pos])

        if len(r5_emb) < 5 or len(x4_emb) < 5:
            continue

        r5_emb = np.array(r5_emb)
        x4_emb = np.array(x4_emb)

        # Calculate separation
        r5_mean = r5_emb.mean(axis=0)
        x4_mean = x4_emb.mean(axis=0)
        separation = np.linalg.norm(r5_mean - x4_mean)

        # Radial comparison
        r5_radii = np.linalg.norm(r5_emb, axis=1)
        x4_radii = np.linalg.norm(x4_emb, axis=1)
        _, pval = stats.mannwhitneyu(r5_radii, x4_radii)

        results.append(
            {
                "position": pos + 1,  # 1-indexed
                "separation": separation,
                "r5_mean_radius": r5_radii.mean(),
                "x4_mean_radius": x4_radii.mean(),
                "radius_pval": pval,
                "is_key_position": (pos + 1) in KEY_V3_POSITIONS,
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# CLASSIFICATION
# ============================================================================


def build_tropism_classifier(v3_df: pd.DataFrame) -> dict:
    """Build ML classifier for tropism prediction."""
    print("\nBuilding tropism classifier...")

    # Prepare features
    valid_df = v3_df[
        (v3_df["is_r5"] | v3_df["is_x4"]) & v3_df["mean_embedding"].notna()
    ].copy()

    if len(valid_df) < 50:
        print("  Insufficient data for classification")
        return {}

    # Feature matrix from embeddings
    X = np.array([e for e in valid_df["mean_embedding"]])
    y = valid_df["is_x4"].astype(int).values

    # Add sequence-based features
    seq_features = valid_df[["net_charge", "n_basic", "seq_length"]].values
    X_full = np.hstack([X, seq_features])

    print(f"  Features: {X_full.shape[1]} dimensions")
    print(f"  Samples: {len(X_full)} (R5: {(y == 0).sum()}, X4: {(y == 1).sum()})")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifiers
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)

    print(f"  Logistic Regression: Accuracy={lr_acc:.3f}, AUC={lr_auc:.3f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_prob = rf.predict_proba(X_test_scaled)[:, 1]

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)

    print(f"  Random Forest: Accuracy={rf_acc:.3f}, AUC={rf_auc:.3f}")

    # Cross-validation
    lr_cv = cross_val_score(lr, X_full, y, cv=5, scoring="accuracy")
    rf_cv = cross_val_score(rf, X_full, y, cv=5, scoring="accuracy")

    results = {
        "lr_accuracy": lr_acc,
        "lr_auc": lr_auc,
        "lr_cv_mean": lr_cv.mean(),
        "lr_cv_std": lr_cv.std(),
        "rf_accuracy": rf_acc,
        "rf_auc": rf_auc,
        "rf_cv_mean": rf_cv.mean(),
        "rf_cv_std": rf_cv.std(),
        "y_test": y_test,
        "lr_prob": lr_prob,
        "rf_prob": rf_prob,
        "feature_importance": rf.feature_importances_,
    }

    return results


# ============================================================================
# GLYCAN ANALYSIS
# ============================================================================


def analyze_glycan_tropism_correlation(gp120_seqs: dict, gp120_labels: dict) -> pd.DataFrame:
    """Analyze correlation between glycan patterns and tropism."""
    print("\nAnalyzing glycan-tropism correlation...")

    if not gp120_seqs or not gp120_labels:
        print("  Missing gp120 data")
        return pd.DataFrame()

    results = []

    for seq_id, sequence in gp120_seqs.items():
        tropism = gp120_labels.get(seq_id)
        if tropism is None:
            continue

        # Find glycan sites
        glycan_sites = find_glycan_sites(sequence)

        results.append(
            {
                "seq_id": seq_id,
                "tropism": tropism,
                "is_x4": tropism.upper() == "CXCR4",
                "n_glycan_sites": len(glycan_sites),
                "glycan_density": len(glycan_sites) / len(sequence) if sequence else 0,
                "seq_length": len(sequence),
            }
        )

    df = pd.DataFrame(results)

    if len(df) > 10:
        r5_glycans = df[~df["is_x4"]]["n_glycan_sites"]
        x4_glycans = df[df["is_x4"]]["n_glycan_sites"]

        if len(r5_glycans) > 5 and len(x4_glycans) > 5:
            stat, pval = stats.mannwhitneyu(r5_glycans, x4_glycans)
            print(f"  R5 mean glycan sites: {r5_glycans.mean():.1f}")
            print(f"  X4 mean glycan sites: {x4_glycans.mean():.1f}")
            print(f"  Mann-Whitney p-value: {pval:.6f}")

    return df


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_tropism_separation(v3_df: pd.DataFrame, separation_results: dict):
    """Plot CCR5 vs CXCR4 geometric separation."""
    if not separation_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid_df = v3_df[v3_df["mean_embedding"].notna()].copy()
    r5_df = valid_df[valid_df["is_r5"]]
    x4_df = valid_df[valid_df["is_x4"]]

    # Plot 1: Radial distribution
    ax1 = axes[0]
    ax1.hist(r5_df["mean_radius"], bins=30, alpha=0.6, label="CCR5", color="blue")
    ax1.hist(x4_df["mean_radius"], bins=30, alpha=0.6, label="CXCR4", color="red")
    ax1.axvline(
        separation_results["r5_mean_radius"],
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    ax1.axvline(
        separation_results["x4_mean_radius"],
        color="red",
        linestyle="--",
        linewidth=2,
    )
    ax1.set_xlabel("Mean Radial Position")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Radial Distribution by Tropism")
    ax1.legend()

    # Plot 2: Net charge distribution
    ax2 = axes[1]
    ax2.hist(r5_df["net_charge"], bins=15, alpha=0.6, label="CCR5", color="blue")
    ax2.hist(x4_df["net_charge"], bins=15, alpha=0.6, label="CXCR4", color="red")
    ax2.set_xlabel("Net Charge")
    ax2.set_ylabel("Frequency")
    ax2.set_title("V3 Net Charge by Tropism")
    ax2.legend()

    # Plot 3: PCA of embeddings
    ax3 = axes[2]
    embeddings = np.array([e for e in valid_df["mean_embedding"] if e is not None])
    labels = valid_df[valid_df["mean_embedding"].notna()]["is_x4"].values

    if len(embeddings) > 10:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)

        ax3.scatter(
            pca_result[~labels, 0],
            pca_result[~labels, 1],
            alpha=0.5,
            label="CCR5",
            color="blue",
            s=10,
        )
        ax3.scatter(
            pca_result[labels, 0],
            pca_result[labels, 1],
            alpha=0.5,
            label="CXCR4",
            color="red",
            s=10,
        )
        ax3.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
        ax3.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
        ax3.set_title("PCA of V3 Embeddings")
        ax3.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tropism_separation.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'tropism_separation.png'}")


def plot_position_importance(position_df: pd.DataFrame):
    """Plot position importance for tropism."""
    if position_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Separation by position
    ax1 = axes[0]
    colors = ["red" if p else "steelblue" for p in position_df["is_key_position"]]
    ax1.bar(position_df["position"], position_df["separation"], color=colors, alpha=0.7)
    ax1.set_xlabel("V3 Position")
    ax1.set_ylabel("Tropism Separation")
    ax1.set_title("Per-Position Tropism Separation\n(Red = Known Key Positions)")

    # Plot 2: P-value by position
    ax2 = axes[1]
    pvals = -np.log10(position_df["radius_pval"] + 1e-10)
    ax2.bar(position_df["position"], pvals, color=colors, alpha=0.7)
    ax2.axhline(-np.log10(0.05), color="gray", linestyle="--", label="p=0.05")
    ax2.axhline(-np.log10(0.01), color="red", linestyle="--", label="p=0.01")
    ax2.set_xlabel("V3 Position")
    ax2.set_ylabel("-log10(p-value)")
    ax2.set_title("Statistical Significance of Tropism Difference")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_importance.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'position_importance.png'}")


def plot_classifier_results(classifier_results: dict):
    """Plot classifier performance."""
    if not classifier_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curves
    ax1 = axes[0]
    y_test = classifier_results["y_test"]

    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, classifier_results["lr_prob"])
    ax1.plot(fpr_lr, tpr_lr, label=f"LR (AUC={classifier_results['lr_auc']:.3f})")

    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, classifier_results["rf_prob"])
    ax1.plot(fpr_rf, tpr_rf, label=f"RF (AUC={classifier_results['rf_auc']:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Tropism Classifier ROC Curves")
    ax1.legend()

    # Cross-validation comparison
    ax2 = axes[1]
    models = ["Logistic Regression", "Random Forest"]
    means = [classifier_results["lr_cv_mean"], classifier_results["rf_cv_mean"]]
    stds = [classifier_results["lr_cv_std"], classifier_results["rf_cv_std"]]

    ax2.bar(models, means, yerr=stds, capsize=10, color=["steelblue", "coral"], alpha=0.7)
    ax2.set_ylabel("Cross-Validation Accuracy")
    ax2.set_title("5-Fold CV Performance")
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classifier_performance.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'classifier_performance.png'}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_tropism_report(
    v3_df: pd.DataFrame,
    separation_results: dict,
    position_df: pd.DataFrame,
    classifier_results: dict,
    glycan_df: pd.DataFrame,
):
    """Generate comprehensive tropism analysis report."""
    report_path = OUTPUT_DIR / "TROPISM_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# HIV Tropism Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary Statistics\n\n")

        if not v3_df.empty:
            f.write(f"- Total V3 sequences: {len(v3_df)}\n")
            f.write(f"- CCR5 (R5): {v3_df['is_r5'].sum()}\n")
            f.write(f"- CXCR4 (X4): {v3_df['is_x4'].sum()}\n")
            f.write(f"- Sequences with embeddings: {v3_df['mean_embedding'].notna().sum()}\n\n")

        # Separation analysis
        if separation_results:
            f.write("## Tropism Separation Analysis\n\n")
            f.write("| Metric | CCR5 | CXCR4 |\n")
            f.write("|--------|------|-------|\n")
            f.write(
                f"| Mean Radius | {separation_results['r5_mean_radius']:.4f} | "
                f"{separation_results['x4_mean_radius']:.4f} |\n"
            )
            f.write(
                f"| Std Radius | {separation_results['r5_std_radius']:.4f} | "
                f"{separation_results['x4_std_radius']:.4f} |\n"
            )
            f.write(f"\n- Centroid Distance: {separation_results['centroid_distance']:.4f}\n")
            f.write(f"- Mann-Whitney p-value: {separation_results['mann_whitney_pval']:.6f}\n\n")

        # Position importance
        if not position_df.empty:
            f.write("## Key Positions for Tropism\n\n")
            f.write("Top 10 most discriminative positions:\n\n")
            f.write("| Position | Separation | p-value | Key Position? |\n")
            f.write("|----------|------------|---------|---------------|\n")

            top_positions = position_df.nlargest(10, "separation")
            for _, row in top_positions.iterrows():
                key = "Yes" if row["is_key_position"] else ""
                f.write(
                    f"| {int(row['position'])} | {row['separation']:.4f} | "
                    f"{row['radius_pval']:.6f} | {key} |\n"
                )
            f.write("\n")

        # Classifier results
        if classifier_results:
            f.write("## Tropism Classifier Performance\n\n")
            f.write("| Classifier | Accuracy | AUC | CV Mean | CV Std |\n")
            f.write("|------------|----------|-----|---------|--------|\n")
            f.write(
                f"| Logistic Regression | {classifier_results['lr_accuracy']:.3f} | "
                f"{classifier_results['lr_auc']:.3f} | {classifier_results['lr_cv_mean']:.3f} | "
                f"{classifier_results['lr_cv_std']:.3f} |\n"
            )
            f.write(
                f"| Random Forest | {classifier_results['rf_accuracy']:.3f} | "
                f"{classifier_results['rf_auc']:.3f} | {classifier_results['rf_cv_mean']:.3f} | "
                f"{classifier_results['rf_cv_std']:.3f} |\n"
            )
            f.write("\n")

        # Glycan analysis
        if not glycan_df.empty:
            f.write("## Glycan-Tropism Correlation\n\n")
            r5_glyc = glycan_df[~glycan_df["is_x4"]]["n_glycan_sites"]
            x4_glyc = glycan_df[glycan_df["is_x4"]]["n_glycan_sites"]

            if len(r5_glyc) > 0 and len(x4_glyc) > 0:
                f.write(f"- CCR5 mean glycan sites: {r5_glyc.mean():.1f}\n")
                f.write(f"- CXCR4 mean glycan sites: {x4_glyc.mean():.1f}\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        if separation_results:
            sig = "*" if separation_results["mann_whitney_pval"] < 0.05 else ""
            f.write(
                f"1. **Geometric Separation**: CCR5 and CXCR4 sequences show "
                f"{separation_results['centroid_distance']:.4f} distance in embedding space{sig}\n"
            )

        if classifier_results:
            best_auc = max(classifier_results["lr_auc"], classifier_results["rf_auc"])
            f.write(f"2. **Tropism Prediction**: Best classifier achieves AUC of {best_auc:.3f}\n")

        if not position_df.empty:
            top_pos = position_df.nlargest(1, "separation").iloc[0]
            f.write(f"3. **Most Discriminative Position**: Position {int(top_pos['position'])}\n")

        f.write("\n## Generated Files\n\n")
        f.write("- `tropism_separation.png` - CCR5 vs CXCR4 visualization\n")
        f.write("- `position_importance.png` - Per-position discrimination\n")
        f.write("- `classifier_performance.png` - ML classifier results\n")
        f.write("- `v3_data.csv` - V3 sequence data with embeddings info\n")
        f.write("- `position_importance.csv` - Position importance data\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete tropism analysis."""
    print("=" * 70)
    print("HIV Tropism Switching Analysis")
    print("=" * 70)

    # Load encoder
    print("\nLoading hyperbolic codon encoder...")
    try:
        encoder, _ = load_hyperbolic_encoder()
        print("  Encoder loaded successfully")
    except FileNotFoundError as e:
        print(f"  Error loading encoder: {e}")
        encoder = None

    # Load data
    v3_df, gp120_seqs, gp120_labels = load_tropism_data()

    # Prepare V3 data
    if v3_df is not None and not v3_df.empty:
        v3_df = prepare_v3_dataset(v3_df)

        # Encode sequences
        if encoder and not v3_df.empty:
            v3_df = encode_v3_sequences(v3_df, encoder)
    else:
        v3_df = pd.DataFrame()

    # Run analyses
    separation_results = {}
    position_df = pd.DataFrame()
    classifier_results = {}
    glycan_df = pd.DataFrame()

    if not v3_df.empty and encoder:
        separation_results = analyze_tropism_separation(v3_df)
        position_df = analyze_position_importance(v3_df, encoder)
        classifier_results = build_tropism_classifier(v3_df)

    if gp120_seqs and gp120_labels:
        glycan_df = analyze_glycan_tropism_correlation(gp120_seqs, gp120_labels)

    # Generate visualizations
    print("\nGenerating visualizations...")

    if not v3_df.empty:
        plot_tropism_separation(v3_df, separation_results)
        plot_position_importance(position_df)
        plot_classifier_results(classifier_results)

    # Save data
    print("\nSaving results...")

    if not v3_df.empty:
        save_cols = [c for c in v3_df.columns if c not in ["embeddings", "mean_embedding"]]
        v3_df[save_cols].to_csv(OUTPUT_DIR / "v3_data.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'v3_data.csv'}")

    if not position_df.empty:
        position_df.to_csv(OUTPUT_DIR / "position_importance.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'position_importance.csv'}")

    if not glycan_df.empty:
        glycan_df.to_csv(OUTPUT_DIR / "glycan_tropism.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'glycan_tropism.csv'}")

    # Generate report
    print("\nGenerating report...")
    generate_tropism_report(v3_df, separation_results, position_df, classifier_results, glycan_df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
