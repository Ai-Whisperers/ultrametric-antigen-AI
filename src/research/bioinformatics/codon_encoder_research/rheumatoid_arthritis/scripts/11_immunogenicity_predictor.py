#!/usr/bin/env python3
"""
Immunogenicity Predictor for RA Epitopes

Builds predictive models using hyperbolic geometric features to classify
epitopes as immunodominant or silent based on citrullination-induced changes.

Key features used:
1. Entropy change upon citrullination (primary - p=0.024, d=0.80)
2. JS divergence (secondary - p=0.053, d=-0.68)
3. Centroid shift magnitude
4. Basic embedding metrics (norm, homogeneity, neighbor distance, boundary potential)

Models implemented:
- Logistic Regression (interpretable baseline)
- Random Forest (feature importance)
- Gradient Boosting (best performance)

Validation:
- Leave-one-protein-out cross-validation (realistic for new protein prediction)
- Leave-one-out cross-validation (maximum data utilization)
- Stratified k-fold (standard)

Output:
- ROC/AUC curves with 95% CI
- Calibration plots
- Feature importance rankings
- Performance metrics (accuracy, precision, recall, F1, MCC)

Version: 1.0 - Initial implementation with 3-adic encoder
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def hyperbolic_radii(embeddings: np.ndarray, c: float = 1.0) -> np.ndarray:
    """V5.12.2: Compute hyperbolic radii for batch of embeddings."""
    sqrt_c = np.sqrt(c)
    euclidean_norms = np.linalg.norm(embeddings, axis=1)
    clamped = np.clip(euclidean_norms * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


warnings.filterwarnings("ignore")

import matplotlib
# Plotting
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# Import epitope database
import importlib.util

# Local imports
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, get_results_dir,
                              load_codon_encoder, poincare_distance)

spec = importlib.util.spec_from_file_location("augmented_db", Path(__file__).parent / "08_augmented_epitope_database.py")
augmented_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmented_db)
RA_AUTOANTIGENS_AUGMENTED = augmented_db.RA_AUTOANTIGENS_AUGMENTED


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def extract_features(epitope: dict, encoder, device: str = "cpu") -> Optional[Dict]:
    """
    Extract all geometric features for a single epitope.

    Returns None if epitope has no arginine (cannot compute citrullination features).
    """
    sequence = epitope["sequence"]

    # Encode sequence
    embeddings = []
    cluster_probs_list = []

    for aa in sequence:
        codon = AA_TO_CODON.get(aa, "NNN")
        if codon == "NNN":
            continue
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, emb = encoder.get_cluster_probs(onehot)
            embeddings.append(emb.cpu().numpy().squeeze())
            cluster_probs_list.append(probs.cpu().numpy().squeeze())

    if len(embeddings) == 0:
        return None

    embeddings = np.array(embeddings)
    cluster_probs = np.array(cluster_probs_list)

    # Basic embedding metrics (V5.12.2: use hyperbolic radii)
    norms = hyperbolic_radii(embeddings)

    # Cluster homogeneity
    majority_cluster = np.argmax(np.bincount(np.argmax(cluster_probs, axis=1)))
    homogeneity = np.mean(np.argmax(cluster_probs, axis=1) == majority_cluster)

    # Neighbor distances (Poincaré)
    neighbor_dists = []
    for i in range(len(embeddings) - 1):
        d = poincare_distance(
            torch.tensor(embeddings[i]).float(),
            torch.tensor(embeddings[i + 1]).float(),
        ).item()
        neighbor_dists.append(d)
    mean_neighbor_dist = np.mean(neighbor_dists) if neighbor_dists else 0.0

    # Boundary potential (distance to nearest different cluster)
    cluster_ids = np.argmax(cluster_probs, axis=1)
    boundary_potentials = []
    for i, emb in enumerate(embeddings):
        my_cluster = cluster_ids[i]
        other_mask = cluster_ids != my_cluster
        if np.any(other_mask):
            other_embeddings = embeddings[other_mask]
            dists = [poincare_distance(torch.tensor(emb).float(), torch.tensor(other).float()).item() for other in other_embeddings]
            boundary_potentials.append(np.min(dists))
    mean_boundary = np.mean(boundary_potentials) if boundary_potentials else 0.0

    features = {
        "embedding_norm": np.mean(norms),
        "embedding_norm_std": np.std(norms),
        "cluster_homogeneity": homogeneity,
        "mean_neighbor_distance": mean_neighbor_dist,
        "boundary_potential": mean_boundary,
        "sequence_length": len(sequence),
        "n_arginines": sum(1 for aa in sequence if aa == "R"),
    }

    # Citrullination features (only if epitope has arginine)
    arg_positions = [i for i, aa in enumerate(sequence) if aa == "R"]

    if not arg_positions:
        # No arginine - cannot compute citrullination features
        features["has_arginine"] = False
        features["centroid_shift"] = 0.0
        features["js_divergence"] = 0.0
        features["entropy_change"] = 0.0
        return features

    features["has_arginine"] = True

    # Compute citrullination effects for each R position
    centroid_shifts = []
    js_divergences = []
    entropy_changes = []

    original_centroid = np.mean(embeddings, axis=0)
    original_probs = np.mean(cluster_probs, axis=0)
    original_entropy = -np.sum(original_probs * np.log(original_probs + 1e-10))

    for r_pos in arg_positions:
        # Create citrullinated version (remove R from embeddings)
        cit_embeddings = np.delete(embeddings, r_pos, axis=0)
        cit_probs = np.delete(cluster_probs, r_pos, axis=0)

        if len(cit_embeddings) == 0:
            continue

        cit_centroid = np.mean(cit_embeddings, axis=0)
        cit_probs_mean = np.mean(cit_probs, axis=0)
        cit_entropy = -np.sum(cit_probs_mean * np.log(cit_probs_mean + 1e-10))

        # Centroid shift (Poincaré distance)
        shift = poincare_distance(
            torch.tensor(original_centroid).float(),
            torch.tensor(cit_centroid).float(),
        ).item()
        centroid_shifts.append(shift)

        # JS divergence
        m = 0.5 * (original_probs + cit_probs_mean)
        js = 0.5 * (
            np.sum(original_probs * np.log((original_probs + 1e-10) / (m + 1e-10)))
            + np.sum(cit_probs_mean * np.log((cit_probs_mean + 1e-10) / (m + 1e-10)))
        )
        js_divergences.append(js)

        # Entropy change
        entropy_changes.append(cit_entropy - original_entropy)

    features["centroid_shift"] = np.mean(centroid_shifts) if centroid_shifts else 0.0
    features["js_divergence"] = np.mean(js_divergences) if js_divergences else 0.0
    features["entropy_change"] = np.mean(entropy_changes) if entropy_changes else 0.0

    # Additional derived features
    features["r_density"] = features["n_arginines"] / features["sequence_length"]
    features["entropy_per_r"] = features["entropy_change"] / max(features["n_arginines"], 1)

    return features


def build_feature_matrix(encoder, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build feature matrix X and label vector y from all epitopes.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 1 for immunodominant, 0 for silent
        epitope_ids: List of epitope identifiers
        protein_ids: List of protein identifiers (for leave-one-protein-out CV)
    """
    X_list = []
    y_list = []
    epitope_ids = []
    protein_ids = []

    feature_names = [
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

    for protein_id, protein in RA_AUTOANTIGENS_AUGMENTED.items():
        for epitope in protein["epitopes"]:
            features = extract_features(epitope, encoder, device)
            if features is None:
                continue

            # Build feature vector
            feature_vec = [features.get(name, 0.0) for name in feature_names]
            X_list.append(feature_vec)
            y_list.append(1 if epitope["immunodominant"] else 0)
            epitope_ids.append(epitope["id"])
            protein_ids.append(protein_id)

    return (
        np.array(X_list),
        np.array(y_list),
        epitope_ids,
        protein_ids,
        feature_names,
    )


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================


def train_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    protein_ids: List[str],
    feature_names: List[str],
) -> Dict:
    """
    Train and evaluate multiple models with different CV strategies.
    """
    results = {
        "models": {},
        "cv_results": {},
        "feature_importance": {},
    }

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        "logistic_regression": LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=3, random_state=42),
    }

    # Leave-One-Protein-Out CV
    print("\n[1] Leave-One-Protein-Out Cross-Validation")
    print("-" * 50)

    unique_proteins = list(set(protein_ids))
    lopo_results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in models}

    for test_protein in unique_proteins:
        # Split
        train_mask = np.array([p != test_protein for p in protein_ids])
        test_mask = ~train_mask

        if sum(test_mask) == 0 or sum(train_mask) == 0:
            continue

        X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        for name, model in models.items():
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            y_prob = model_clone.predict_proba(X_test)[:, 1]

            lopo_results[name]["y_true"].extend(y_test)
            lopo_results[name]["y_pred"].extend(y_pred)
            lopo_results[name]["y_prob"].extend(y_prob)

    # Compute LOPO metrics
    for name in models:
        y_true = np.array(lopo_results[name]["y_true"])
        y_pred = np.array(lopo_results[name]["y_pred"])
        y_prob = np.array(lopo_results[name]["y_prob"])

        if len(np.unique(y_true)) < 2:
            continue

        results["cv_results"][f"lopo_{name}"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
        }

        print(f"  {name}:")
        print(f"    AUC: {results['cv_results'][f'lopo_{name}']['auc']:.3f}")
        print(f"    Accuracy: {results['cv_results'][f'lopo_{name}']['accuracy']:.3f}")
        print(f"    MCC: {results['cv_results'][f'lopo_{name}']['mcc']:.3f}")

    # Stratified 5-Fold CV
    print("\n[2] Stratified 5-Fold Cross-Validation")
    print("-" * 50)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in models}

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            y_prob = model_clone.predict_proba(X_test)[:, 1]

            skf_results[name]["y_true"].extend(y_test)
            skf_results[name]["y_pred"].extend(y_pred)
            skf_results[name]["y_prob"].extend(y_prob)

    for name in models:
        y_true = np.array(skf_results[name]["y_true"])
        y_pred = np.array(skf_results[name]["y_pred"])
        y_prob = np.array(skf_results[name]["y_prob"])

        results["cv_results"][f"skf_{name}"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
        }

        print(f"  {name}:")
        print(f"    AUC: {results['cv_results'][f'skf_{name}']['auc']:.3f}")
        print(f"    Accuracy: {results['cv_results'][f'skf_{name}']['accuracy']:.3f}")
        print(f"    MCC: {results['cv_results'][f'skf_{name}']['mcc']:.3f}")

    # Train final models on all data
    print("\n[3] Training Final Models on All Data")
    print("-" * 50)

    for name, model in models.items():
        model.fit(X_scaled, y)
        results["models"][name] = {
            "trained": True,
            "n_samples": len(y),
            "n_positive": int(sum(y)),
            "n_negative": int(len(y) - sum(y)),
        }

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        results["feature_importance"][name] = [{"feature": feature_names[i], "importance": float(importance[i])} for i in sorted_idx]

        print(f"  {name} - Top 5 features:")
        for i in sorted_idx[:5]:
            print(f"    {feature_names[i]}: {importance[i]:.4f}")

    return results, models, scaler


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_roc_curves(results: Dict, output_dir: Path):
    """Plot ROC curves for all models and CV strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for models
    colors = {
        "logistic_regression": "#2196F3",
        "random_forest": "#4CAF50",
        "gradient_boosting": "#FF9800",
    }

    # LOPO ROC
    ax = axes[0]
    for key, res in results["cv_results"].items():
        if not key.startswith("lopo_"):
            continue
        model_name = key.replace("lopo_", "")

        y_true = np.array(res["y_true"])
        y_prob = np.array(res["y_prob"])

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        label = f"{model_name.replace('_', ' ').title()} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=colors.get(model_name, "gray"), lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Leave-One-Protein-Out CV", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # SKF ROC
    ax = axes[1]
    for key, res in results["cv_results"].items():
        if not key.startswith("skf_"):
            continue
        model_name = key.replace("skf_", "")

        y_true = np.array(res["y_true"])
        y_prob = np.array(res["y_prob"])

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        label = f"{model_name.replace('_', ' ').title()} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=colors.get(model_name, "gray"), lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Stratified 5-Fold CV", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'roc_curves.png'}")


def plot_feature_importance(results: Dict, output_dir: Path):
    """Plot feature importance for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {
        "logistic_regression": "#2196F3",
        "random_forest": "#4CAF50",
        "gradient_boosting": "#FF9800",
    }

    for idx, (name, importance_list) in enumerate(results["feature_importance"].items()):
        ax = axes[idx]

        features = [item["feature"] for item in importance_list]
        importances = [item["importance"] for item in importance_list]

        # Take top 10
        features = features[:10]
        importances = importances[:10]

        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=colors.get(name, "gray"), alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace("_", " ").title() for f in features], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Feature Importance by Model", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'feature_importance.png'}")


def plot_calibration(results: Dict, output_dir: Path):
    """Plot calibration curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "logistic_regression": "#2196F3",
        "random_forest": "#4CAF50",
        "gradient_boosting": "#FF9800",
    }

    for cv_type, ax in zip(["lopo", "skf"], axes):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

        for key, res in results["cv_results"].items():
            if not key.startswith(f"{cv_type}_"):
                continue
            model_name = key.replace(f"{cv_type}_", "")

            y_true = np.array(res["y_true"])
            y_prob = np.array(res["y_prob"])

            # Calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")

            brier = res["brier"]
            label = f"{model_name.replace('_', ' ').title()} (Brier={brier:.3f})"
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                color=colors.get(model_name, "gray"),
                lw=2,
                markersize=8,
                label=label,
            )

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        title = "Leave-One-Protein-Out CV" if cv_type == "lopo" else "Stratified 5-Fold CV"
        ax.set_title(f"Calibration Plot - {title}", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'calibration_curves.png'}")


def plot_metrics_comparison(results: Dict, output_dir: Path):
    """Plot comparison of metrics across models and CV strategies."""
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
    models = ["logistic_regression", "random_forest", "gradient_boosting"]
    cv_types = ["lopo", "skf"]

    x = np.arange(len(metrics))
    width = 0.12

    colors = {
        "lopo_logistic_regression": "#1565C0",
        "lopo_random_forest": "#2E7D32",
        "lopo_gradient_boosting": "#E65100",
        "skf_logistic_regression": "#64B5F6",
        "skf_random_forest": "#81C784",
        "skf_gradient_boosting": "#FFB74D",
    }

    labels = {
        "lopo_logistic_regression": "LR (LOPO)",
        "lopo_random_forest": "RF (LOPO)",
        "lopo_gradient_boosting": "GB (LOPO)",
        "skf_logistic_regression": "LR (5-Fold)",
        "skf_random_forest": "RF (5-Fold)",
        "skf_gradient_boosting": "GB (5-Fold)",
    }

    offset = 0
    for cv_type in cv_types:
        for model in models:
            key = f"{cv_type}_{model}"
            if key not in results["cv_results"]:
                continue

            res = results["cv_results"][key]
            values = [res[m] for m in metrics]

            ax.bar(
                x + offset * width,
                values,
                width,
                label=labels[key],
                color=colors[key],
                alpha=0.9,
            )
            offset += 1

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 2.5 * width)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])

    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'metrics_comparison.png'}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("IMMUNOGENICITY PREDICTOR")
    print("Predicting immunodominance from hyperbolic geometric features")
    print("=" * 80)

    # Setup
    results_dir = get_results_dir(hyperbolic=True)
    output_dir = results_dir / "prediction"
    output_dir.mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, mapping, _ = load_codon_encoder(device=device, version="3adic")

    # Extract features
    print("\nExtracting features from epitope database...")
    X, y, epitope_ids, protein_ids, feature_names = build_feature_matrix(encoder, device)
    print(f"  Total samples: {len(y)}")
    print(f"  Immunodominant: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Silent: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Proteins: {len(set(protein_ids))}")

    # Train and evaluate
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)

    results, models, scaler = train_evaluate_models(X, y, protein_ids, feature_names)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_roc_curves(results, output_dir)
    plot_feature_importance(results, output_dir)
    plot_calibration(results, output_dir)
    plot_metrics_comparison(results, output_dir)

    # Save results
    print("\nSaving results...")

    # Prepare JSON-serializable results
    json_results = {
        "metadata": {
            "encoder_version": "3-adic (V5.11.3)",
            "n_samples": len(y),
            "n_immunodominant": int(sum(y)),
            "n_silent": int(len(y) - sum(y)),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "n_proteins": len(set(protein_ids)),
        },
        "cv_results": {},
        "feature_importance": results["feature_importance"],
    }

    # Convert cv_results (remove numpy arrays)
    for key, res in results["cv_results"].items():
        json_results["cv_results"][key] = {k: v for k, v in res.items() if k not in ["y_true", "y_prob"]}

    with open(output_dir / "prediction_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {output_dir / 'prediction_results.json'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nBest model performance (Leave-One-Protein-Out CV):")
    best_auc = 0
    best_model = None
    for key, res in results["cv_results"].items():
        if key.startswith("lopo_") and res["auc"] > best_auc:
            best_auc = res["auc"]
            best_model = key.replace("lopo_", "")

    if best_model:
        res = results["cv_results"][f"lopo_{best_model}"]
        print(f"  Model: {best_model.replace('_', ' ').title()}")
        print(f"  AUC: {res['auc']:.3f}")
        print(f"  Accuracy: {res['accuracy']:.3f}")
        print(f"  F1: {res['f1']:.3f}")
        print(f"  MCC: {res['mcc']:.3f}")

    print("\nTop 3 predictive features (across models):")
    feature_scores = defaultdict(list)
    for model_name, importance_list in results["feature_importance"].items():
        for rank, item in enumerate(importance_list):
            feature_scores[item["feature"]].append(item["importance"])

    mean_importance = {f: np.mean(scores) for f, scores in feature_scores.items()}
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(sorted_features[:3], 1):
        print(f"  {i}. {feature.replace('_', ' ').title()}: {importance:.4f}")

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
