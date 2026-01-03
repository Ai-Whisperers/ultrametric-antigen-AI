"""
Arrow Flip Clustering: Defining Soft Boundaries for Prediction Regimes

This module expands the arrow flip experiments with:
1. Cross-validation to ensure robust findings
2. Soft boundary detection using logistic regression
3. Clustering of AA pairs by prediction regime
4. Decision rules for when to use hybrid vs simple approaches

The goal is to precisely define WHERE the arrow flips from
"sequence is sufficient" to "p-adic/hybrid structure adds value".
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import from parent
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from functional_profiles import AMINO_ACID_PROFILES, compute_functional_similarity_matrix


# =============================================================================
# AMINO ACID PROPERTIES (same as arrow_flip_experiments.py)
# =============================================================================

AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 'polar', 'aromatic': False},
    'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 'charged', 'aromatic': False},
    'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 'charged', 'aromatic': False},
    'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': True},
    'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0, 'polarity': 'charged', 'aromatic': True},
    'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 'charged', 'aromatic': False},
    'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 'polar', 'aromatic': False},
    'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 'polar', 'aromatic': False},
    'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 'charged', 'aromatic': False},
    'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 'polar', 'aromatic': False},
    'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 'polar', 'aromatic': False},
    'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': False},
    'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 'nonpolar', 'aromatic': True},
    'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 'polar', 'aromatic': True},
}


# =============================================================================
# FEATURE EXTRACTION FOR AA PAIRS
# =============================================================================

@dataclass
class AAPairFeatures:
    """Features describing an amino acid pair for classification."""
    aa1: str
    aa2: str

    # Absolute differences
    hydro_diff: float
    volume_diff: float
    charge_diff: float

    # Categorical
    same_polarity: bool
    same_charge: bool
    both_aromatic: bool
    one_aromatic: bool

    # Derived
    size_category: str  # 'small', 'medium', 'large'
    charge_category: str  # 'same', 'neutral_to_charged', 'opposite'

    # Ground truth
    func_similarity: float
    hybrid_cost: float
    simple_dist: float
    hybrid_wins: bool  # Does hybrid outperform simple?

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical feature vector for ML."""
        return np.array([
            self.hydro_diff,
            self.volume_diff / 100.0,  # Normalize
            abs(self.charge_diff),
            float(self.same_polarity),
            float(self.same_charge),
            float(self.both_aromatic),
            float(self.one_aromatic),
            1.0 if self.size_category == 'small' else (0.5 if self.size_category == 'medium' else 0.0),
            1.0 if self.charge_category == 'same' else (0.5 if self.charge_category == 'neutral_to_charged' else 0.0),
        ])

    @property
    def feature_names(self) -> List[str]:
        return [
            'hydro_diff', 'volume_diff', 'charge_diff',
            'same_polarity', 'same_charge', 'both_aromatic', 'one_aromatic',
            'size_category', 'charge_category'
        ]


def compute_hybrid_cost(aa1: str, aa2: str) -> float:
    """Compute hybrid cost with charge/size penalties."""
    if aa1 not in AA_PROPERTIES or aa2 not in AA_PROPERTIES:
        return float('inf')

    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]

    hydro_dist = abs(p1['hydrophobicity'] - p2['hydrophobicity'])
    vol_dist = abs(p1['volume'] - p2['volume']) / 50.0
    base_cost = np.sqrt(hydro_dist**2 + vol_dist**2)

    if p1['charge'] != p2['charge']:
        if p1['charge'] * p2['charge'] < 0:
            base_cost += 5.0
        else:
            base_cost += 2.0

    if abs(p1['volume'] - p2['volume']) > 60:
        base_cost += 3.0

    return base_cost


def compute_simple_distance(aa1: str, aa2: str) -> float:
    """Compute simple Euclidean distance."""
    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]
    return np.sqrt(
        (p1['hydrophobicity'] - p2['hydrophobicity'])**2 +
        (p1['charge'] - p2['charge'])**2 +
        ((p1['volume'] - p2['volume']) / 100)**2
    )


def extract_pair_features(aa1: str, aa2: str, similarity_matrix: np.ndarray,
                          aa_codes: List[str]) -> AAPairFeatures:
    """Extract all features for an AA pair."""
    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]

    # Compute costs
    hybrid_cost = compute_hybrid_cost(aa1, aa2)
    simple_dist = compute_simple_distance(aa1, aa2)

    # Get functional similarity
    i = aa_codes.index(aa1)
    j = aa_codes.index(aa2)
    func_sim = similarity_matrix[i, j]

    # Determine which wins (normalized comparison)
    # Higher similarity = better; lower cost = better
    # Normalize to compare: -cost scaled to match similarity range
    max_cost = 15.0  # Approximate max
    hybrid_pred = -hybrid_cost / max_cost  # Scale to ~[-1, 0]
    simple_pred = -simple_dist / 10.0  # Scale similarly

    # Error comparison
    hybrid_error = abs(hybrid_pred - func_sim)
    simple_error = abs(simple_pred - func_sim)
    hybrid_wins = hybrid_error < simple_error

    # Categorical features
    vol_diff = abs(p1['volume'] - p2['volume'])
    if vol_diff < 30:
        size_cat = 'small'
    elif vol_diff < 80:
        size_cat = 'medium'
    else:
        size_cat = 'large'

    if p1['charge'] == p2['charge']:
        charge_cat = 'same'
    elif p1['charge'] * p2['charge'] < 0:
        charge_cat = 'opposite'
    else:
        charge_cat = 'neutral_to_charged'

    return AAPairFeatures(
        aa1=aa1,
        aa2=aa2,
        hydro_diff=abs(p1['hydrophobicity'] - p2['hydrophobicity']),
        volume_diff=vol_diff,
        charge_diff=p1['charge'] - p2['charge'],
        same_polarity=p1['polarity'] == p2['polarity'],
        same_charge=p1['charge'] == p2['charge'],
        both_aromatic=p1['aromatic'] and p2['aromatic'],
        one_aromatic=p1['aromatic'] != p2['aromatic'],
        size_category=size_cat,
        charge_category=charge_cat,
        func_similarity=func_sim,
        hybrid_cost=hybrid_cost,
        simple_dist=simple_dist,
        hybrid_wins=hybrid_wins,
    )


def build_pair_dataset() -> Tuple[List[AAPairFeatures], np.ndarray, List[str]]:
    """Build complete dataset of AA pair features."""
    similarity_matrix, aa_codes = compute_functional_similarity_matrix()

    pairs = []
    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue
            pair = extract_pair_features(aa1, aa2, similarity_matrix, aa_codes)
            pairs.append(pair)

    return pairs, similarity_matrix, aa_codes


# =============================================================================
# EXPERIMENT 1: CROSS-VALIDATION
# =============================================================================

def experiment_1_cross_validation(pairs: List[AAPairFeatures]) -> Dict:
    """
    Cross-validate the hybrid vs simple classification.

    Uses multiple CV strategies:
    1. K-fold (k=5, 10)
    2. Leave-one-out
    3. Bootstrap
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: CROSS-VALIDATION OF PREDICTION REGIMES")
    print("="*70)

    # Prepare data
    X = np.array([p.to_feature_vector() for p in pairs])
    y = np.array([1 if p.hybrid_wins else 0 for p in pairs])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ==========================================================================
    # 1. K-Fold Cross-Validation
    # ==========================================================================
    print("\n--- K-Fold Cross-Validation ---")

    for k in [5, 10]:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr_scores = cross_val_score(lr, X_scaled, y, cv=kf, scoring='accuracy')

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_scores = cross_val_score(dt, X_scaled, y, cv=kf, scoring='accuracy')

        print(f"\n{k}-Fold CV:")
        print(f"  Logistic Regression: {lr_scores.mean():.3f} (+/- {lr_scores.std()*2:.3f})")
        print(f"  Decision Tree: {dt_scores.mean():.3f} (+/- {dt_scores.std()*2:.3f})")

        results[f'{k}_fold'] = {
            'logistic_regression': {
                'mean': float(lr_scores.mean()),
                'std': float(lr_scores.std()),
                'scores': lr_scores.tolist()
            },
            'decision_tree': {
                'mean': float(dt_scores.mean()),
                'std': float(dt_scores.std()),
                'scores': dt_scores.tolist()
            }
        }

    # ==========================================================================
    # 2. Leave-One-Out Cross-Validation
    # ==========================================================================
    print("\n--- Leave-One-Out Cross-Validation ---")

    loo = LeaveOneOut()

    # Use simpler model for LOO (faster)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    loo_scores = cross_val_score(lr, X_scaled, y, cv=loo, scoring='accuracy')

    print(f"  LOO Accuracy: {loo_scores.mean():.3f} ({sum(loo_scores)}/{len(loo_scores)} correct)")

    results['loo'] = {
        'accuracy': float(loo_scores.mean()),
        'n_correct': int(sum(loo_scores)),
        'n_total': len(loo_scores)
    }

    # ==========================================================================
    # 3. Bootstrap Validation
    # ==========================================================================
    print("\n--- Bootstrap Validation (100 iterations) ---")

    n_bootstrap = 100
    bootstrap_accuracies = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X_scaled[indices]
        y_boot = y[indices]

        # Out-of-bag samples
        oob_indices = list(set(range(len(X))) - set(indices))
        if len(oob_indices) < 5:
            continue
        X_oob = X_scaled[oob_indices]
        y_oob = y[oob_indices]

        # Train and evaluate
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_boot, y_boot)
        acc = lr.score(X_oob, y_oob)
        bootstrap_accuracies.append(acc)

    boot_mean = np.mean(bootstrap_accuracies)
    boot_std = np.std(bootstrap_accuracies)
    boot_ci = (np.percentile(bootstrap_accuracies, 2.5), np.percentile(bootstrap_accuracies, 97.5))

    print(f"  Bootstrap Accuracy: {boot_mean:.3f} (+/- {boot_std*2:.3f})")
    print(f"  95% CI: [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]")

    results['bootstrap'] = {
        'mean': float(boot_mean),
        'std': float(boot_std),
        'ci_95': [float(boot_ci[0]), float(boot_ci[1])],
        'n_iterations': n_bootstrap
    }

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n--- Cross-Validation Summary ---")
    print(f"  Overall baseline (majority class): {max(y.mean(), 1-y.mean()):.3f}")
    print(f"  Best CV accuracy: {max(results['5_fold']['logistic_regression']['mean'], results['10_fold']['logistic_regression']['mean']):.3f}")
    print(f"  Improvement over baseline: {max(results['5_fold']['logistic_regression']['mean'], results['10_fold']['logistic_regression']['mean']) - max(y.mean(), 1-y.mean()):.3f}")

    return results


# =============================================================================
# EXPERIMENT 2: SOFT BOUNDARY DETECTION
# =============================================================================

def experiment_2_soft_boundaries(pairs: List[AAPairFeatures]) -> Dict:
    """
    Detect soft boundaries using logistic regression probabilities.

    Identifies:
    1. Hard hybrid zone (p > 0.8)
    2. Soft hybrid zone (0.5 < p <= 0.8)
    3. Uncertain zone (0.4 < p <= 0.6)
    4. Soft simple zone (0.2 < p <= 0.5)
    5. Hard simple zone (p <= 0.2)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: SOFT BOUNDARY DETECTION")
    print("="*70)

    # Prepare data
    X = np.array([p.to_feature_vector() for p in pairs])
    y = np.array([1 if p.hybrid_wins else 0 for p in pairs])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_scaled, y)

    # Get probabilities
    probs = lr.predict_proba(X_scaled)[:, 1]  # Probability of hybrid_wins

    # Define zones
    zones = {
        'hard_hybrid': [],      # p > 0.8
        'soft_hybrid': [],      # 0.6 < p <= 0.8
        'uncertain': [],        # 0.4 <= p <= 0.6
        'soft_simple': [],      # 0.2 <= p < 0.4
        'hard_simple': [],      # p < 0.2
    }

    for pair, prob in zip(pairs, probs):
        pair_info = {
            'pair': f"{pair.aa1}-{pair.aa2}",
            'prob_hybrid': float(prob),
            'actual': 'hybrid' if pair.hybrid_wins else 'simple',
            'hydro_diff': pair.hydro_diff,
            'volume_diff': pair.volume_diff,
            'charge_category': pair.charge_category,
        }

        if prob > 0.8:
            zones['hard_hybrid'].append(pair_info)
        elif prob > 0.6:
            zones['soft_hybrid'].append(pair_info)
        elif prob >= 0.4:
            zones['uncertain'].append(pair_info)
        elif prob >= 0.2:
            zones['soft_simple'].append(pair_info)
        else:
            zones['hard_simple'].append(pair_info)

    # Print results
    print("\n--- Zone Distribution ---")
    for zone, members in zones.items():
        correct = sum(1 for m in members if
                     (m['actual'] == 'hybrid' and 'hybrid' in zone) or
                     (m['actual'] == 'simple' and 'simple' in zone))
        print(f"\n{zone.upper()} (n={len(members)}):")
        print(f"  Accuracy: {correct}/{len(members)} ({100*correct/len(members) if members else 0:.1f}%)")

        if members:
            # Show characteristics
            avg_hydro = np.mean([m['hydro_diff'] for m in members])
            avg_vol = np.mean([m['volume_diff'] for m in members])
            charge_cats = [m['charge_category'] for m in members]

            print(f"  Avg hydro_diff: {avg_hydro:.2f}")
            print(f"  Avg volume_diff: {avg_vol:.1f}")
            print(f"  Charge categories: {dict((c, charge_cats.count(c)) for c in set(charge_cats))}")

    # ==========================================================================
    # Identify boundary characteristics
    # ==========================================================================
    print("\n--- Boundary Characteristics ---")

    # Feature importance from logistic regression
    feature_names = pairs[0].feature_names
    importance = np.abs(lr.coef_[0])
    sorted_idx = np.argsort(importance)[::-1]

    print("\nFeature Importance (for predicting hybrid zone):")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]}: {importance[idx]:.3f}")

    # Decision rules
    print("\n--- Soft Boundary Rules ---")

    # Analyze uncertain zone
    uncertain_pairs = zones['uncertain']
    if uncertain_pairs:
        print("\nUNCERTAIN ZONE characteristics (where prediction is hardest):")
        for p in uncertain_pairs[:5]:
            print(f"  {p['pair']}: prob={p['prob_hybrid']:.2f}, actual={p['actual']}, "
                  f"hydro_diff={p['hydro_diff']:.1f}, vol_diff={p['volume_diff']:.0f}, "
                  f"charge={p['charge_category']}")

    results = {
        'zones': {name: len(members) for name, members in zones.items()},
        'zone_details': zones,
        'feature_importance': {
            feature_names[i]: float(importance[i])
            for i in sorted_idx
        },
        'model_accuracy': float(lr.score(X_scaled, y)),
    }

    return results


# =============================================================================
# EXPERIMENT 3: CLUSTERING AA PAIRS BY REGIME
# =============================================================================

def experiment_3_regime_clustering(pairs: List[AAPairFeatures]) -> Dict:
    """
    Cluster AA pairs into distinct prediction regimes.

    Uses:
    1. K-means clustering on feature space
    2. Hierarchical clustering for interpretability
    3. Regime characterization
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: REGIME CLUSTERING")
    print("="*70)

    # Prepare data
    X = np.array([p.to_feature_vector() for p in pairs])
    y = np.array([1 if p.hybrid_wins else 0 for p in pairs])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================================================
    # Find optimal number of clusters
    # ==========================================================================
    print("\n--- Finding Optimal Cluster Count ---")

    silhouette_scores = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append((k, score))
        print(f"  k={k}: silhouette={score:.3f}")

    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"\nOptimal k: {optimal_k}")

    # ==========================================================================
    # Final clustering
    # ==========================================================================
    print(f"\n--- Clustering with k={optimal_k} ---")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Analyze each cluster
    clusters = {}
    for k in range(optimal_k):
        cluster_mask = cluster_labels == k
        cluster_pairs = [p for p, m in zip(pairs, cluster_mask) if m]

        # Statistics
        n_hybrid_wins = sum(1 for p in cluster_pairs if p.hybrid_wins)
        hybrid_rate = n_hybrid_wins / len(cluster_pairs) if cluster_pairs else 0

        avg_hydro = np.mean([p.hydro_diff for p in cluster_pairs])
        avg_vol = np.mean([p.volume_diff for p in cluster_pairs])

        charge_same = sum(1 for p in cluster_pairs if p.same_charge)
        polarity_same = sum(1 for p in cluster_pairs if p.same_polarity)

        # Determine cluster type
        if hybrid_rate > 0.7:
            cluster_type = "HYBRID_ZONE"
        elif hybrid_rate < 0.3:
            cluster_type = "SIMPLE_ZONE"
        else:
            cluster_type = "MIXED_ZONE"

        clusters[k] = {
            'type': cluster_type,
            'n_pairs': len(cluster_pairs),
            'hybrid_rate': hybrid_rate,
            'avg_hydro_diff': avg_hydro,
            'avg_volume_diff': avg_vol,
            'pct_same_charge': charge_same / len(cluster_pairs) if cluster_pairs else 0,
            'pct_same_polarity': polarity_same / len(cluster_pairs) if cluster_pairs else 0,
            'pairs': [f"{p.aa1}-{p.aa2}" for p in cluster_pairs],
        }

        print(f"\nCluster {k} ({cluster_type}):")
        print(f"  N pairs: {len(cluster_pairs)}")
        print(f"  Hybrid win rate: {hybrid_rate:.1%}")
        print(f"  Avg hydro_diff: {avg_hydro:.2f}")
        print(f"  Avg volume_diff: {avg_vol:.1f}")
        print(f"  Same charge: {100*charge_same/len(cluster_pairs):.0f}%")
        print(f"  Same polarity: {100*polarity_same/len(cluster_pairs):.0f}%")
        print(f"  Example pairs: {', '.join([f'{p.aa1}-{p.aa2}' for p in cluster_pairs[:5]])}")

    # ==========================================================================
    # Create interpretable decision rules
    # ==========================================================================
    print("\n" + "="*70)
    print("DECISION RULES FOR REGIME SELECTION")
    print("="*70)

    # Train decision tree for interpretability
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    feature_names = pairs[0].feature_names

    print("\nDecision Tree Rules:")

    def extract_rules(tree, feature_names, class_names=['Simple', 'Hybrid']):
        """Extract rules from decision tree."""
        from sklearn.tree import _tree

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        rules = []

        def recurse(node, depth, rule):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Left child
                left_rule = rule + [f"{name} <= {threshold:.2f}"]
                recurse(tree_.children_left[node], depth + 1, left_rule)

                # Right child
                right_rule = rule + [f"{name} > {threshold:.2f}"]
                recurse(tree_.children_right[node], depth + 1, right_rule)
            else:
                # Leaf node
                value = tree_.value[node][0]
                class_idx = np.argmax(value)
                confidence = value[class_idx] / value.sum()
                rules.append({
                    'conditions': rule,
                    'prediction': class_names[class_idx],
                    'confidence': confidence,
                    'samples': int(value.sum())
                })

        recurse(0, 0, [])
        return rules

    rules = extract_rules(dt, feature_names)

    for i, rule in enumerate(rules):
        print(f"\nRule {i+1}: {rule['prediction']} (confidence={rule['confidence']:.1%}, n={rule['samples']})")
        for condition in rule['conditions']:
            print(f"  IF {condition}")

    results = {
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'clusters': clusters,
        'decision_rules': rules,
        'decision_tree_accuracy': float(dt.score(X, y)),
    }

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_experiments() -> Dict:
    """Run all clustering and boundary detection experiments."""

    print("="*70)
    print("ARROW FLIP CLUSTERING: SOFT BOUNDARY DETECTION")
    print("="*70)

    # Build dataset
    print("\nBuilding AA pair dataset...")
    pairs, similarity_matrix, aa_codes = build_pair_dataset()
    print(f"Total pairs: {len(pairs)}")

    n_hybrid = sum(1 for p in pairs if p.hybrid_wins)
    print(f"Hybrid wins: {n_hybrid} ({100*n_hybrid/len(pairs):.1f}%)")
    print(f"Simple wins: {len(pairs)-n_hybrid} ({100*(len(pairs)-n_hybrid)/len(pairs):.1f}%)")

    results = {}

    # Experiment 1: Cross-validation
    results['cross_validation'] = experiment_1_cross_validation(pairs)

    # Experiment 2: Soft boundaries
    results['soft_boundaries'] = experiment_2_soft_boundaries(pairs)

    # Experiment 3: Regime clustering
    results['regime_clustering'] = experiment_3_regime_clustering(pairs)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'arrow_flip_clustering_results.json')

    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY: SOFT BOUNDARY CHARACTERIZATION")
    print("="*70)

    print("\n1. CROSS-VALIDATION:")
    cv_acc = results['cross_validation']['10_fold']['logistic_regression']['mean']
    print(f"   10-fold CV accuracy: {cv_acc:.1%}")
    print(f"   Prediction of regime is {'RELIABLE' if cv_acc > 0.6 else 'MODERATE'}")

    print("\n2. SOFT BOUNDARIES:")
    zones = results['soft_boundaries']['zones']
    print(f"   Hard hybrid zone: {zones['hard_hybrid']} pairs")
    print(f"   Soft hybrid zone: {zones['soft_hybrid']} pairs")
    print(f"   Uncertain zone: {zones['uncertain']} pairs")
    print(f"   Soft simple zone: {zones['soft_simple']} pairs")
    print(f"   Hard simple zone: {zones['hard_simple']} pairs")

    print("\n3. REGIME CLUSTERS:")
    clusters = results['regime_clustering']['clusters']
    for k, info in clusters.items():
        print(f"   Cluster {k} ({info['type']}): {info['n_pairs']} pairs, "
              f"hybrid_rate={info['hybrid_rate']:.0%}")

    print("\n4. KEY DECISION RULES:")
    rules = results['regime_clustering']['decision_rules']
    for rule in rules[:3]:
        conditions = ' AND '.join(rule['conditions'][:2])
        print(f"   IF {conditions} → {rule['prediction']} ({rule['confidence']:.0%})")

    return results


if __name__ == "__main__":
    results = run_all_experiments()
