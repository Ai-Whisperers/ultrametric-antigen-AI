#!/usr/bin/env python3
"""Comprehensive Analysis of P-adic DDG Predictions on Full S669 Dataset

This script:
1. Runs V2 predictor on all 669 mutations
2. Analyzes prediction errors by mutation type
3. Performs proper cross-validation
4. Compares with other tools
5. Determines if model is trivial or meaningful

Usage:
    python analyze_padic_ddg_full.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.biology.codons import AMINO_ACID_TO_CODONS, CODON_TO_INDEX
from src.encoders.codon_encoder import (
    compute_padic_distance_between_codons,
    AA_PROPERTIES,
)


def compute_aa_padic_distance(aa1: str, aa2: str) -> float:
    """Compute minimum p-adic distance between amino acids."""
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    min_dist = 1.0
    for c1 in codons1:
        idx1 = CODON_TO_INDEX[c1]
        for c2 in codons2:
            idx2 = CODON_TO_INDEX[c2]
            dist = compute_padic_distance_between_codons(idx1, idx2)
            min_dist = min(min_dist, dist)

    return min_dist


def compute_aa_mean_padic_distance(aa1: str, aa2: str) -> float:
    """Compute mean p-adic distance over all codon pairs."""
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    total_dist = 0.0
    count = 0
    for c1 in codons1:
        idx1 = CODON_TO_INDEX[c1]
        for c2 in codons2:
            idx2 = CODON_TO_INDEX[c2]
            total_dist += compute_padic_distance_between_codons(idx1, idx2)
            count += 1

    return total_dist / count if count > 0 else 1.0


def extract_features(wt_aa: str, mut_aa: str) -> np.ndarray:
    """Extract features for a mutation."""
    features = []

    # P-adic features (proper codon distances)
    padic_min = compute_aa_padic_distance(wt_aa, mut_aa)
    padic_mean = compute_aa_mean_padic_distance(wt_aa, mut_aa)
    features.extend([padic_min, padic_mean])

    # Physicochemical features
    wt_props = AA_PROPERTIES.get(wt_aa, (0, 0, 0, 0))
    mut_props = AA_PROPERTIES.get(mut_aa, (0, 0, 0, 0))

    delta_hydro = mut_props[0] - wt_props[0]
    delta_charge = mut_props[1] - wt_props[1]
    delta_size = mut_props[2] - wt_props[2]
    delta_polarity = mut_props[3] - wt_props[3]

    features.extend([delta_size, delta_hydro, abs(delta_charge), delta_polarity])

    # Degeneracy ratio
    wt_deg = len(AMINO_ACID_TO_CODONS.get(wt_aa, []))
    mut_deg = len(AMINO_ACID_TO_CODONS.get(mut_aa, []))
    deg_ratio = mut_deg / max(wt_deg, 1)
    features.append(deg_ratio)

    return np.array(features)


def parse_mutation(mut_str: str) -> tuple:
    """Parse mutation string like 'S11A' -> ('S', 11, 'A')"""
    match = re.match(r'([A-Z])(\d+)([A-Z])', mut_str)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None, None, None


def load_s669_full(filepath: Path) -> list:
    """Load full S669 dataset with all tool predictions."""
    mutations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')

    # Find column indices
    seq_mut_idx = header.index('Seq_Mut')
    ddg_idx = header.index('Experimental_DDG_dir')

    # Find tool prediction columns
    tool_columns = {}
    for i, col in enumerate(header):
        if col.endswith('_dir') and col != 'Experimental_DDG_dir':
            tool_name = col.replace('_dir', '')
            tool_columns[tool_name] = i

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < max(seq_mut_idx, ddg_idx) + 1:
            continue

        try:
            mut_str = parts[seq_mut_idx]
            wt_aa, pos, mut_aa = parse_mutation(mut_str)
            if wt_aa is None:
                continue

            ddg_exp = float(parts[ddg_idx])

            # Get tool predictions
            tool_preds = {}
            for tool, idx in tool_columns.items():
                try:
                    tool_preds[tool] = float(parts[idx])
                except (ValueError, IndexError):
                    pass

            mutations.append({
                'mutation': mut_str,
                'wild_type': wt_aa,
                'mutant': mut_aa,
                'position': pos,
                'ddg_experimental': ddg_exp,
                'tool_predictions': tool_preds
            })
        except (ValueError, IndexError):
            continue

    return mutations


def analyze_errors(mutations: list, predictions: np.ndarray, y_true: np.ndarray):
    """Analyze prediction errors by various categories."""
    errors = predictions - y_true
    abs_errors = np.abs(errors)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS BY CATEGORY")
    print("=" * 70)

    # By mutation type
    mut_type_errors = defaultdict(list)
    for i, mut in enumerate(mutations):
        key = f"{mut['wild_type']}→{mut['mutant']}"
        mut_type_errors[key].append(abs_errors[i])

    # Sort by mean error
    sorted_types = sorted(mut_type_errors.items(), key=lambda x: np.mean(x[1]))

    print("\n## Best Predicted Mutations (lowest error):")
    print("| Mutation | Count | MAE | Std |")
    print("|----------|-------|-----|-----|")
    for mut_type, errs in sorted_types[:10]:
        if len(errs) >= 2:
            print(f"| {mut_type} | {len(errs)} | {np.mean(errs):.3f} | {np.std(errs):.3f} |")

    print("\n## Worst Predicted Mutations (highest error):")
    print("| Mutation | Count | MAE | Std |")
    print("|----------|-------|-----|-----|")
    for mut_type, errs in sorted_types[-10:][::-1]:
        if len(errs) >= 2:
            print(f"| {mut_type} | {len(errs)} | {np.mean(errs):.3f} | {np.std(errs):.3f} |")

    # By DDG magnitude
    print("\n## Error by DDG Magnitude:")
    print("| DDG Range | Count | MAE | Spearman |")
    print("|-----------|-------|-----|----------|")

    ddg_bins = [(-10, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 10)]
    for low, high in ddg_bins:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 5:
            r = spearmanr(predictions[mask], y_true[mask])[0] if HAS_SCIPY else 0
            print(f"| [{low}, {high}) | {mask.sum()} | {np.mean(abs_errors[mask]):.3f} | {r:.3f} |")

    # By p-adic distance
    print("\n## Error by P-adic Distance:")
    print("| P-adic Dist | Count | MAE | Note |")
    print("|-------------|-------|-----|------|")

    padic_dists = defaultdict(list)
    for i, mut in enumerate(mutations):
        d = compute_aa_padic_distance(mut['wild_type'], mut['mutant'])
        padic_dists[f"{d:.2f}"].append(abs_errors[i])

    for dist in sorted(padic_dists.keys()):
        errs = padic_dists[dist]
        print(f"| {dist} | {len(errs)} | {np.mean(errs):.3f} | |")

    return sorted_types


def cross_validate_properly(mutations: list, feature_names: list):
    """Perform rigorous cross-validation."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 70)

    X = []
    y = []
    valid_mutations = []

    for mut in mutations:
        wt_aa = mut['wild_type']
        mut_aa = mut['mutant']

        if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
            features = extract_features(wt_aa, mut_aa)
            X.append(features)
            y.append(mut['ddg_experimental'])
            valid_mutations.append(mut)

    X = np.array(X)
    y = np.array(y)

    print(f"\nValid mutations: {len(y)}")
    print(f"Feature matrix shape: {X.shape}")

    # 5-fold CV
    model = Ridge(alpha=1.0)
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv5_scores = cross_val_score(model, X, y, cv=cv5, scoring='r2')

    print(f"\n5-Fold CV R²: {cv5_scores.mean():.4f} ± {cv5_scores.std():.4f}")
    print(f"  Folds: {[f'{s:.3f}' for s in cv5_scores]}")

    # 10-fold CV
    cv10 = KFold(n_splits=10, shuffle=True, random_state=42)
    cv10_scores = cross_val_score(model, X, y, cv=cv10, scoring='r2')

    print(f"\n10-Fold CV R²: {cv10_scores.mean():.4f} ± {cv10_scores.std():.4f}")

    # Spearman in CV
    from sklearn.model_selection import cross_val_predict
    y_pred_cv = cross_val_predict(model, X, y, cv=cv5)
    cv_spearman = spearmanr(y_pred_cv, y)[0] if HAS_SCIPY else 0
    cv_pearson = pearsonr(y_pred_cv, y)[0] if HAS_SCIPY else np.corrcoef(y_pred_cv, y)[0, 1]

    print(f"\nCV Spearman r: {cv_spearman:.4f}")
    print(f"CV Pearson r: {cv_pearson:.4f}")
    print(f"CV MAE: {mean_absolute_error(y, y_pred_cv):.4f}")
    print(f"CV RMSE: {np.sqrt(mean_squared_error(y, y_pred_cv)):.4f}")

    # Feature importance via ablation
    print("\n## Feature Ablation Study:")
    print("| Removed Feature | CV R² | Δ R² |")
    print("|-----------------|-------|------|")

    base_score = cv5_scores.mean()
    for i, fname in enumerate(feature_names):
        X_ablated = np.delete(X, i, axis=1)
        ablated_scores = cross_val_score(model, X_ablated, y, cv=cv5, scoring='r2')
        delta = base_score - ablated_scores.mean()
        print(f"| {fname} | {ablated_scores.mean():.4f} | {delta:+.4f} |")

    # Test only p-adic features
    print("\n## P-adic Features Only:")
    X_padic = X[:, :2]  # Just padic_min and padic_mean
    padic_scores = cross_val_score(model, X_padic, y, cv=cv5, scoring='r2')
    print(f"CV R² (p-adic only): {padic_scores.mean():.4f} ± {padic_scores.std():.4f}")

    # Test only physicochemical features
    print("\n## Physicochemical Features Only:")
    X_phys = X[:, 2:]  # Everything except p-adic
    phys_scores = cross_val_score(model, X_phys, y, cv=cv5, scoring='r2')
    print(f"CV R² (physico only): {phys_scores.mean():.4f} ± {phys_scores.std():.4f}")

    return {
        'cv5_r2': cv5_scores.mean(),
        'cv5_std': cv5_scores.std(),
        'cv10_r2': cv10_scores.mean(),
        'cv_spearman': cv_spearman,
        'cv_pearson': cv_pearson,
        'padic_only_r2': padic_scores.mean(),
        'phys_only_r2': phys_scores.mean(),
        'y_pred_cv': y_pred_cv,
        'y_true': y,
        'valid_mutations': valid_mutations
    }


def compare_with_tools(mutations: list, our_predictions: np.ndarray, y_true: np.ndarray):
    """Compare our predictions with published tools."""
    print("\n" + "=" * 70)
    print("COMPARISON WITH PUBLISHED TOOLS")
    print("=" * 70)

    # Collect tool predictions
    tool_results = {}

    for tool in ['FoldX', 'DUET', 'mCSM', 'DDGun', 'DDGun3D', 'MAESTRO',
                 'PremPS', 'ThermoNet', 'ACDC-NN', 'PopMusic']:
        preds = []
        exp = []
        for i, mut in enumerate(mutations):
            if tool in mut['tool_predictions']:
                preds.append(mut['tool_predictions'][tool])
                exp.append(y_true[i])

        if len(preds) > 10:
            preds = np.array(preds)
            exp = np.array(exp)
            r_spearman = spearmanr(preds, exp)[0] if HAS_SCIPY else 0
            r_pearson = pearsonr(preds, exp)[0] if HAS_SCIPY else 0
            mae = np.mean(np.abs(preds - exp))
            tool_results[tool] = {
                'n': len(preds),
                'spearman': r_spearman,
                'pearson': r_pearson,
                'mae': mae
            }

    # Our results
    r_spearman = spearmanr(our_predictions, y_true)[0] if HAS_SCIPY else 0
    r_pearson = pearsonr(our_predictions, y_true)[0] if HAS_SCIPY else 0
    mae = np.mean(np.abs(our_predictions - y_true))

    print("\n| Tool | N | Spearman r | Pearson r | MAE |")
    print("|------|---|------------|-----------|-----|")

    # Sort by Spearman
    sorted_tools = sorted(tool_results.items(), key=lambda x: x[1]['spearman'], reverse=True)
    for tool, res in sorted_tools:
        print(f"| {tool} | {res['n']} | {res['spearman']:.3f} | {res['pearson']:.3f} | {res['mae']:.3f} |")

    print(f"| **P-adic V2** | {len(y_true)} | **{r_spearman:.3f}** | **{r_pearson:.3f}** | **{mae:.3f}** |")

    return tool_results


def assess_triviality(cv_results: dict):
    """Determine if the model is trivial or meaningful."""
    print("\n" + "=" * 70)
    print("TRIVIALITY ASSESSMENT")
    print("=" * 70)

    cv_r2 = cv_results['cv5_r2']
    padic_r2 = cv_results['padic_only_r2']
    phys_r2 = cv_results['phys_only_r2']
    cv_spearman = cv_results['cv_spearman']

    print("\n## Key Metrics:")
    print(f"  - 5-Fold CV R²: {cv_r2:.4f}")
    print(f"  - P-adic only R²: {padic_r2:.4f}")
    print(f"  - Physicochemical only R²: {phys_r2:.4f}")
    print(f"  - CV Spearman: {cv_spearman:.4f}")

    print("\n## Triviality Criteria:")

    # Criterion 1: CV R² positive = learns something
    c1 = cv_r2 > 0
    print(f"  1. CV R² > 0: {c1} ({cv_r2:.4f})")

    # Criterion 2: P-adic features contribute
    padic_contribution = cv_r2 - phys_r2
    c2 = padic_contribution > 0.01
    print(f"  2. P-adic adds to physico: {c2} (Δ = {padic_contribution:.4f})")

    # Criterion 3: Not just memorizing (CV ≈ train)
    c3 = cv_r2 > 0.05
    print(f"  3. CV R² > 0.05 (generalizes): {c3}")

    # Criterion 4: Spearman meaningful
    c4 = cv_spearman > 0.3
    print(f"  4. CV Spearman > 0.3: {c4} ({cv_spearman:.4f})")

    print("\n## Assessment:")
    if c1 and c2 and c3 and c4:
        print("  ✓ MODEL IS NON-TRIVIAL")
        print("    - P-adic features provide unique predictive signal")
        print("    - Model generalizes beyond training data")
    elif c1 and c4:
        print("  ~ MODEL IS PARTIALLY MEANINGFUL")
        print("    - Shows correlation but may be dominated by physicochemical features")
    else:
        print("  ✗ MODEL MAY BE TRIVIAL")
        print("    - Limited generalization or p-adic contribution")

    # Identify improvement paths
    print("\n## Potential Improvements:")

    if padic_r2 < 0.02:
        print("  1. P-adic features alone have weak signal")
        print("     → Consider codon context (neighboring AAs)")
        print("     → Add structural context if available")

    if cv_r2 < phys_r2:
        print("  2. Physicochemical features dominate")
        print("     → P-adic captures redundant information")
        print("     → Need better p-adic feature engineering")

    if cv_spearman < 0.4:
        print("  3. Ranking accuracy is moderate")
        print("     → Focus on relative rankings, not absolute values")
        print("     → Consider ranking-based loss functions")

    return {
        'is_nontrivial': c1 and c2 and c3 and c4,
        'cv_r2': cv_r2,
        'padic_contribution': padic_contribution,
        'criteria': [c1, c2, c3, c4]
    }


def main():
    script_dir = Path(__file__).parent

    # Try full dataset first, fall back to original
    full_path = script_dir / "data" / "S669" / "S669.csv"
    fallback_path = script_dir / "data" / "s669.csv"

    if full_path.exists():
        data_path = full_path
        print("Using FULL S669 dataset (669 mutations)")
    else:
        data_path = fallback_path
        print("Using fallback dataset")

    print("=" * 70)
    print("COMPREHENSIVE P-ADIC DDG ANALYSIS")
    print("=" * 70)
    print(f"\nData: {data_path}")

    # Load data
    mutations = load_s669_full(data_path)
    print(f"Loaded {len(mutations)} mutations")

    # Extract features and train
    feature_names = ['padic_min', 'padic_mean', 'delta_volume',
                     'delta_hydro', 'delta_charge', 'delta_polarity',
                     'degeneracy_ratio']

    X = []
    y = []
    valid_mutations = []

    for mut in mutations:
        wt_aa = mut['wild_type']
        mut_aa = mut['mutant']

        if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
            features = extract_features(wt_aa, mut_aa)
            X.append(features)
            y.append(mut['ddg_experimental'])
            valid_mutations.append(mut)

    X = np.array(X)
    y = np.array(y)

    print(f"Valid mutations: {len(y)}")

    # Train on full data for error analysis
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Training metrics
    r_spearman = spearmanr(y_pred, y)[0] if HAS_SCIPY else 0
    r_pearson = pearsonr(y_pred, y)[0] if HAS_SCIPY else 0
    mae = np.mean(np.abs(y_pred - y))
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))

    print("\n## Training Results (Full Data):")
    print(f"  Spearman r: {r_spearman:.4f}")
    print(f"  Pearson r: {r_pearson:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    print("\n## Learned Weights:")
    for i, fname in enumerate(feature_names):
        print(f"  {fname}: {model.coef_[i]:.4f}")
    print(f"  bias: {model.intercept_:.4f}")

    # Error analysis
    analyze_errors(valid_mutations, y_pred, y)

    # Cross-validation
    cv_results = cross_validate_properly(mutations, feature_names)

    # Compare with tools
    compare_with_tools(valid_mutations, y_pred, y)

    # Triviality assessment
    triviality = assess_triviality(cv_results)

    # Save results
    results_path = script_dir / "results" / "full_analysis_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': str(data_path),
        'n_mutations': len(y),
        'training': {
            'spearman': float(r_spearman),
            'pearson': float(r_pearson),
            'mae': float(mae),
            'rmse': float(rmse)
        },
        'cross_validation': {
            'cv5_r2': float(cv_results['cv5_r2']),
            'cv5_std': float(cv_results['cv5_std']),
            'cv10_r2': float(cv_results['cv10_r2']),
            'cv_spearman': float(cv_results['cv_spearman']),
            'cv_pearson': float(cv_results['cv_pearson']),
            'padic_only_r2': float(cv_results['padic_only_r2']),
            'phys_only_r2': float(cv_results['phys_only_r2'])
        },
        'triviality': {
            'is_nontrivial': bool(triviality['is_nontrivial']),
            'padic_contribution': float(triviality['padic_contribution'])
        },
        'weights': {fname: float(model.coef_[i]) for i, fname in enumerate(feature_names)}
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
