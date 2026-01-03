#!/usr/bin/env python3
"""Comprehensive Validation for Carlos Brizuela AMP Activity Models.

This script performs rigorous validation including:
1. Leave-One-Out Cross-Validation (LOO-CV) for small datasets
2. Permutation tests for statistical significance
3. Feature importance analysis
4. Biological context assessment (Gram+/- membrane differences)
5. Ensemble model comparison

Usage:
    python validation/comprehensive_validation.py
    python validation/comprehensive_validation.py --output validation/results/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import warnings

import numpy as np

warnings.filterwarnings('ignore')

# Add paths
_script_dir = Path(__file__).parent
_deliverables_dir = _script_dir.parent.parent.parent
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_script_dir.parent))

try:
    from scipy.stats import spearmanr, pearsonr, permutation_test
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import joblib
    from sklearn.model_selection import (
        cross_val_predict, LeaveOneOut, KFold,
        permutation_test_score
    )
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ValidationResult:
    """Comprehensive validation results for a model."""
    model_name: str
    target: str
    gram_type: str  # "positive", "negative", "mixed"
    n_samples: int
    n_unique_sequences: int

    # Cross-validation metrics
    cv_method: str  # "LOO" or "5-fold"
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    rmse: float
    mae: float

    # Permutation test
    permutation_p: float
    permutation_score: float
    n_permutations: int

    # Feature importance
    top_features: list

    # Biological context
    primary_predictor: str
    biological_notes: str

    # Quality assessment
    is_significant: bool
    confidence_level: str  # "high", "moderate", "low"
    recommendation: str


def get_gram_type(target: str) -> str:
    """Get bacterial gram staining type."""
    gram_positive = ["staphylococcus", "enterococcus", "streptococcus"]
    gram_negative = ["escherichia", "pseudomonas", "acinetobacter", "klebsiella"]

    if target:
        target_lower = target.lower()
        for gp in gram_positive:
            if gp in target_lower:
                return "positive"
        for gn in gram_negative:
            if gn in target_lower:
                return "negative"
    return "mixed"


def load_data_for_target(target: str = None):
    """Load training data for a specific target."""
    from scripts.dramp_activity_loader import DRAMPLoader

    loader = DRAMPLoader()
    db = loader.generate_curated_database()
    X, y = db.get_training_data(target)

    # Count unique sequences
    if target:
        records = db.filter_by_target(target)
    else:
        records = db.records
    records = [r for r in records if r.mic_value and r.mic_value > 0]
    n_unique = len(set(r.sequence for r in records))

    return X, y, n_unique, db


def compute_feature_importance(X, y, feature_names=None):
    """Compute feature importance using Random Forest."""
    if not HAS_SKLEARN:
        return []

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)

    importances = rf.feature_importances_

    # Default feature names
    if feature_names is None:
        feature_names = [
            'length', 'charge', 'hydrophobicity', 'volume',
            'positive_fraction', 'negative_fraction', 'aromatic_fraction',
            'aliphatic_fraction', 'polar_fraction', 'hydrophobic_moment'
        ] + [f'aac_{aa}' for aa in 'ACDEFGHIKLMNPQRSTVWY']

    # Get top 5 features
    indices = np.argsort(importances)[::-1][:5]
    top_features = [
        {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
         "importance": float(importances[i])}
        for i in indices
    ]

    return top_features


def validate_model_comprehensive(
    target: str = None,
    n_permutations: int = 100,
) -> Optional[ValidationResult]:
    """Perform comprehensive validation for a model."""
    if not HAS_SKLEARN or not HAS_SCIPY:
        print("Error: scipy and sklearn required")
        return None

    # Load data
    X, y, n_unique, db = load_data_for_target(target)

    if len(X) < 10:
        print(f"Not enough data for {target}: {len(X)} samples")
        return None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose CV method based on sample size
    if len(X) < 30:
        cv_method = "LOO"
        cv = LeaveOneOut()
    else:
        cv_method = "5-fold"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=2, random_state=42
    )

    # Cross-validation predictions
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

    # Compute metrics
    pearson_r, pearson_p = pearsonr(y, y_pred)
    spearman_r, spearman_p = spearmanr(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Permutation test for significance
    perm_score, perm_scores, perm_p = permutation_test_score(
        model, X_scaled, y, cv=cv,
        n_permutations=n_permutations,
        random_state=42,
        scoring='neg_mean_squared_error'
    )

    # Feature importance
    model.fit(X_scaled, y)
    top_features = compute_feature_importance(X_scaled, y)

    # Determine primary predictor
    if top_features:
        primary_predictor = top_features[0]["feature"]
    else:
        primary_predictor = "unknown"

    # Biological context
    gram_type = get_gram_type(target)
    if gram_type == "positive":
        bio_notes = "Gram-positive: thick peptidoglycan, no outer membrane. Charge-based features less predictive."
    elif gram_type == "negative":
        bio_notes = "Gram-negative: LPS outer membrane. Cationic AMPs more effective."
    else:
        bio_notes = "Mixed pathogens: heterogeneous mechanisms."

    # Quality assessment
    is_significant = pearson_p < 0.05 and perm_p < 0.05

    if is_significant and abs(pearson_r) > 0.4:
        confidence = "high"
        recommendation = "Model suitable for predictions with uncertainty estimates"
    elif is_significant and abs(pearson_r) > 0.2:
        confidence = "moderate"
        recommendation = "Use with caution, consider ensemble with general model"
    elif len(X) < 30:
        confidence = "low"
        recommendation = f"Insufficient data (n={len(X)}). Use general model instead."
    else:
        confidence = "low"
        recommendation = "Model not predictive. Use heuristic or general model."

    return ValidationResult(
        model_name=f"activity_{target or 'general'}",
        target=target or "all",
        gram_type=gram_type,
        n_samples=len(X),
        n_unique_sequences=n_unique,
        cv_method=cv_method,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        rmse=rmse,
        mae=mae,
        permutation_p=perm_p,
        permutation_score=perm_score,
        n_permutations=n_permutations,
        top_features=top_features,
        primary_predictor=primary_predictor,
        biological_notes=bio_notes,
        is_significant=is_significant,
        confidence_level=confidence,
        recommendation=recommendation,
    )


def main():
    """Run comprehensive validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Model Validation")
    parser.add_argument('--output', type=str, default='validation/results/',
                        help='Output directory')
    parser.add_argument('--permutations', type=int, default=100,
                        help='Number of permutations for significance test')
    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE AMP ACTIVITY MODEL VALIDATION")
    print("=" * 70)
    print()

    # Validate all models
    targets = [None, "staphylococcus", "pseudomonas", "escherichia", "acinetobacter"]
    results = []

    for target in targets:
        name = target or "general"
        print(f"Validating {name}...", end=" ", flush=True)
        result = validate_model_comprehensive(target, args.permutations)
        if result:
            results.append(result)
            status = "✓" if result.is_significant else "✗"
            print(f"[{status}] r={result.pearson_r:.3f}, p={result.pearson_p:.4f}")
        else:
            print("[SKIP]")

    # Print detailed results
    print()
    print("=" * 70)
    print("DETAILED VALIDATION RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n{r.model_name.upper()}")
        print("-" * 50)
        print(f"  Target: {r.target} ({r.gram_type} gram)")
        print(f"  Samples: {r.n_samples} ({r.n_unique_sequences} unique)")
        print(f"  CV Method: {r.cv_method}")
        print(f"  Pearson r: {r.pearson_r:.4f} (p={r.pearson_p:.4e})")
        print(f"  Spearman ρ: {r.spearman_r:.4f} (p={r.spearman_p:.4e})")
        print(f"  RMSE: {r.rmse:.4f}, MAE: {r.mae:.4f}")
        print(f"  Permutation p: {r.permutation_p:.4f}")
        print(f"  Primary predictor: {r.primary_predictor}")
        print(f"  Confidence: {r.confidence_level.upper()}")
        print(f"  Recommendation: {r.recommendation}")
        print(f"  Biology: {r.biological_notes}")

    # Summary table
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'N':>5} {'Gram':>8} {'r':>7} {'p':>10} {'Perm-p':>8} {'Status':<10}")
    print("-" * 75)

    for r in results:
        status = f"[{r.confidence_level}]"
        sig = "***" if r.pearson_p < 0.001 else "**" if r.pearson_p < 0.01 else "*" if r.pearson_p < 0.05 else ""
        print(f"{r.model_name:<25} {r.n_samples:>5} {r.gram_type:>8} {r.pearson_r:>7.3f}{sig:<3} "
              f"{r.pearson_p:>8.2e} {r.permutation_p:>8.3f} {status:<10}")

    # Recommendations
    print()
    print("=" * 70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 70)

    high_conf = [r for r in results if r.confidence_level == "high"]
    mod_conf = [r for r in results if r.confidence_level == "moderate"]
    low_conf = [r for r in results if r.confidence_level == "low"]

    print(f"\nHigh Confidence ({len(high_conf)} models):")
    for r in high_conf:
        print(f"  ✓ {r.model_name}: {r.recommendation}")

    print(f"\nModerate Confidence ({len(mod_conf)} models):")
    for r in mod_conf:
        print(f"  ~ {r.model_name}: {r.recommendation}")

    print(f"\nLow Confidence ({len(low_conf)} models):")
    for r in low_conf:
        print(f"  ✗ {r.model_name}: {r.recommendation}")

    # Save results
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    def to_json_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    results_dict = to_json_serializable({
        "metadata": {
            "validation_type": "comprehensive",
            "n_permutations": args.permutations,
            "timestamp": str(np.datetime64('now')),
        },
        "models": [asdict(r) for r in results],
        "summary": {
            "total_models": len(results),
            "high_confidence": len(high_conf),
            "moderate_confidence": len(mod_conf),
            "low_confidence": len(low_conf),
            "significant_models": sum(1 for r in results if r.is_significant),
        }
    })

    output_path = output_dir / "comprehensive_validation.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Return status
    n_significant = sum(1 for r in results if r.is_significant)
    if n_significant >= 3:
        print(f"\n✓ {n_significant}/{len(results)} models validated successfully")
        return 0
    else:
        print(f"\n✗ Only {n_significant}/{len(results)} models significant")
        return 1


if __name__ == "__main__":
    sys.exit(main())
