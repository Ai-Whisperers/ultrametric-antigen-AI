#!/usr/bin/env python3
"""Adaptive Feature Selection for P-adic Features.

Key insight from rigorous_validation.py:
- P-adic features help MOST for: neutral→charged, small DDG, size changes
- P-adic features HURT for: charge reversals, large DDG effects

Strategy: Train mutation-type-specific models or use adaptive weighting.

Usage:
    python adaptive_feature_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance, exp_map_zero, project_to_poincare


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Amino acid classifications
AA_CHARGED_POS = set('KRH')
AA_CHARGED_NEG = set('DE')


def get_hydrophobicity_ordering() -> List[str]:
    """Get amino acids ordered by hydrophobicity."""
    props = {aa: AA_PROPERTIES.get(aa, (0, 0, 0, 0)) for aa in AMINO_ACIDS}
    return sorted(AMINO_ACIDS, key=lambda aa: -props[aa][0])


def compute_valuation(n: int, p: int) -> int:
    """P-adic valuation."""
    if n == 0:
        return p
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


class OrderedPadicEncoder(nn.Module):
    """P-adic encoder with hydrophobicity ordering."""

    def __init__(self, prime: int = 5, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()

        self.ordering = get_hydrophobicity_ordering()
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.ordering)}
        self.prime = prime
        self.latent_dim = latent_dim
        self.curvature = 1.0

        self.aa_embedding = nn.Parameter(torch.randn(20, latent_dim) * 0.01)

        self.property_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        hierarchy = [min(compute_valuation(i, prime), 4) for i in range(20)]
        self.register_buffer('hierarchy_levels', torch.tensor(hierarchy, dtype=torch.long))
        max_level = max(hierarchy) if max(hierarchy) > 0 else 1
        target_radii = [0.85 - (h / max_level) * 0.6 for h in hierarchy]
        self.register_buffer('target_radii', torch.tensor(target_radii, dtype=torch.float32))

        props = [list(AA_PROPERTIES.get(aa, (0, 0, 0, 0))) for aa in self.ordering]
        self.register_buffer('aa_properties', torch.tensor(props, dtype=torch.float32))

    def forward(self, aa_indices: torch.Tensor):
        z_base = self.aa_embedding[aa_indices]
        props = self.aa_properties[aa_indices]
        z_prop = self.property_encoder(props)
        z_fused = self.fusion(torch.cat([z_base, z_prop], dim=-1))
        z_hyp = exp_map_zero(z_fused, c=self.curvature)
        z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)
        return {'z_hyp': z_hyp, 'radii': radii}

    def get_aa_embeddings(self):
        indices = torch.arange(20, device=self.aa_embedding.device)
        out = self.forward(indices)
        return {aa: out['z_hyp'][i] for i, aa in enumerate(self.ordering)}

    def get_aa_valuations(self):
        return {aa: int(self.hierarchy_levels[i].item()) for i, aa in enumerate(self.ordering)}


def train_encoder(n_epochs: int = 200) -> OrderedPadicEncoder:
    """Train the encoder."""
    encoder = OrderedPadicEncoder(prime=5, latent_dim=16)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        encoder.train()
        optimizer.zero_grad()

        indices = torch.arange(20)
        out = encoder(indices)

        hierarchy_loss = torch.mean((out['radii'] - encoder.target_radii) ** 2)

        z = out['z_hyp']
        prop_dist = torch.cdist(encoder.aa_properties, encoder.aa_properties, p=2)
        prop_dist = prop_dist / (prop_dist.max() + 1e-8)

        emb_dist = torch.zeros(20, 20)
        for i in range(20):
            for j in range(i+1, 20):
                d = poincare_distance(z[i:i+1], z[j:j+1], c=1.0).squeeze().clamp(0, 10)
                emb_dist[i, j] = emb_dist[j, i] = d
        emb_dist = emb_dist / (emb_dist.max() + 1e-8)

        mask = torch.triu(torch.ones(20, 20), diagonal=1).bool()
        property_loss = torch.mean((emb_dist[mask] - prop_dist[mask]) ** 2)

        loss = hierarchy_loss * 2.0 + property_loss * 1.0

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

    return encoder


def load_s669(filepath: Path) -> List[Dict]:
    """Load S669 dataset."""
    mutations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    ddg_idx = header.index('Experimental_DDG_dir') if 'Experimental_DDG_dir' in header else 11

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) <= ddg_idx:
            continue
        try:
            seq_mut = parts[2] if len(parts) > 2 else parts[1]
            if len(seq_mut) >= 3:
                wt = seq_mut[0].upper()
                mt = seq_mut[-1].upper()
                if wt in AMINO_ACIDS and mt in AMINO_ACIDS and wt != mt:
                    ddg = float(parts[ddg_idx])
                    mutations.append({'wild_type': wt, 'mutant': mt, 'ddg_exp': ddg})
        except:
            continue
    return mutations


def classify_charge_change(wt: str, mt: str) -> str:
    """Classify charge change type."""
    wt_pos = wt in AA_CHARGED_POS
    wt_neg = wt in AA_CHARGED_NEG
    mt_pos = mt in AA_CHARGED_POS
    mt_neg = mt in AA_CHARGED_NEG

    if (wt_pos or wt_neg) and not (mt_pos or mt_neg):
        return 'charged_to_neutral'
    elif not (wt_pos or wt_neg) and (mt_pos or mt_neg):
        return 'neutral_to_charged'
    elif (wt_pos and mt_neg) or (wt_neg and mt_pos):
        return 'charge_reversal'
    else:
        return 'same_charge'


def extract_extended_features(mutations: List[Dict], encoder: OrderedPadicEncoder) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """Extract ALL possible features for selection experiments."""

    encoder.eval()
    embeddings = encoder.get_aa_embeddings()
    valuations = encoder.get_aa_valuations()

    X, y, feature_names, metadata = [], [], [], []

    for mut in mutations:
        wt, mt = mut['wild_type'], mut['mutant']
        if wt not in embeddings or mt not in embeddings:
            continue

        wt_emb = embeddings[wt].detach().cpu()
        mt_emb = embeddings[mt].detach().cpu()
        wt_props = np.array(AA_PROPERTIES.get(wt, (0, 0, 0, 0)))
        mt_props = np.array(AA_PROPERTIES.get(mt, (0, 0, 0, 0)))

        features = []

        # 1. Basic physicochemical deltas (4)
        delta_props = mt_props - wt_props
        features.extend(delta_props.tolist())

        # 2. Derived physicochemical (6)
        features.append(abs(delta_props[0]))  # abs hydro change
        features.append(abs(delta_props[1]))  # abs charge change
        features.append(abs(delta_props[2]))  # abs volume change
        features.append(abs(delta_props[3]))  # abs polarity change
        features.append(np.linalg.norm(delta_props))  # L2 norm
        features.append(np.linalg.norm(delta_props[:2]))  # Hydro+charge L2

        # 3. P-adic hyperbolic features (6)
        hyp_dist = poincare_distance(wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=1.0).item()
        origin = torch.zeros(1, 16)
        wt_r = poincare_distance(wt_emb.unsqueeze(0), origin, c=1.0).item()
        mt_r = poincare_distance(mt_emb.unsqueeze(0), origin, c=1.0).item()

        features.append(hyp_dist)           # Hyperbolic distance
        features.append(mt_r - wt_r)        # Delta radius
        features.append(wt_r)               # WT radius
        features.append(mt_r)               # MT radius
        features.append(wt_r + mt_r)        # Sum radius
        features.append(abs(mt_r - wt_r))   # Abs delta radius

        # 4. P-adic valuation features (6)
        wt_val = valuations.get(wt, 0)
        mt_val = valuations.get(mt, 0)
        features.append(wt_val)              # WT valuation
        features.append(mt_val)              # MT valuation
        features.append(mt_val - wt_val)     # Delta valuation
        features.append(abs(mt_val - wt_val))  # Abs delta val
        features.append(wt_val * mt_val)     # Product (interaction)
        features.append(max(wt_val, mt_val)) # Max valuation

        # 5. Embedding-based features (6)
        wt_vec = wt_emb.numpy()
        mt_vec = mt_emb.numpy()
        diff_vec = mt_vec - wt_vec

        cos_sim = np.dot(wt_vec, mt_vec) / (np.linalg.norm(wt_vec) * np.linalg.norm(mt_vec) + 1e-8)
        features.append(cos_sim)                    # Cosine similarity
        features.append(np.var(diff_vec))           # Direction variance
        features.append(np.max(np.abs(diff_vec)))   # Max dim change
        features.append(np.mean(np.abs(diff_vec)))  # Mean dim change
        features.append(np.linalg.norm(diff_vec))   # Euclidean emb dist
        features.append(np.std(wt_vec) + np.std(mt_vec))  # Sum stds

        # 6. Charge-aware indicator features (4)
        charge_type = classify_charge_change(wt, mt)
        features.append(1.0 if charge_type == 'charge_reversal' else 0.0)
        features.append(1.0 if charge_type == 'neutral_to_charged' else 0.0)
        features.append(1.0 if charge_type == 'charged_to_neutral' else 0.0)
        features.append(1.0 if charge_type == 'same_charge' else 0.0)

        # 7. DDG magnitude indicator (for testing, not using DDG itself)
        # This is for meta-analysis only - we predict magnitude from other features
        features.append(np.sign(delta_props[0]))  # Hydro direction
        features.append(np.sign(delta_props[1]))  # Charge direction

        X.append(features)
        y.append(mut['ddg_exp'])
        metadata.append({**mut, 'charge_type': charge_type})

    # Feature names
    feature_names = [
        'delta_hydro', 'delta_charge', 'delta_volume', 'delta_polarity',
        'abs_hydro', 'abs_charge', 'abs_volume', 'abs_polarity', 'l2_all', 'l2_hc',
        'hyp_dist', 'delta_r', 'wt_r', 'mt_r', 'sum_r', 'abs_delta_r',
        'wt_val', 'mt_val', 'delta_val', 'abs_delta_val', 'val_product', 'max_val',
        'cos_sim', 'dir_var', 'max_dim', 'mean_dim', 'euc_dist', 'sum_std',
        'is_charge_rev', 'is_n2c', 'is_c2n', 'is_same_charge',
        'sign_hydro', 'sign_charge',
    ]

    return np.array(X), np.array(y), feature_names, metadata


def evaluate_feature_set(X: np.ndarray, y: np.ndarray,
                         feature_indices: List[int], n_repeats: int = 5) -> Dict:
    """Evaluate a feature set with cross-validation."""
    X_subset = X[:, feature_indices]

    scores = []
    for seed in range(n_repeats):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
        r, _ = spearmanr(y, y_pred)
        scores.append(r)

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores,
    }


def adaptive_model_selection(X: np.ndarray, y: np.ndarray, metadata: List[Dict],
                             feature_names: List[str]) -> Dict:
    """Build adaptive model that uses different features for different mutation types."""

    # Define feature groups
    physico_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # First 10 features
    padic_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # P-adic features
    embedding_idx = [22, 23, 24, 25, 26, 27]  # Embedding features
    indicator_idx = [28, 29, 30, 31]  # Charge indicators

    # Evaluate different strategies
    strategies = {
        'physico_only': physico_idx,
        'physico+padic': physico_idx + padic_idx,
        'all_features': list(range(len(feature_names) - 2)),  # Exclude sign features
        'physico+embedding': physico_idx + embedding_idx,
        'physico+indicators': physico_idx + indicator_idx,
        'physico+padic+indicators': physico_idx + padic_idx + indicator_idx,
    }

    results = {}
    for name, indices in strategies.items():
        result = evaluate_feature_set(X, y, indices)
        results[name] = result
        print(f"{name}: {result['mean']:.4f} +/- {result['std']:.4f}")

    return results


def feature_importance_analysis(X: np.ndarray, y: np.ndarray,
                                feature_names: List[str]) -> Dict:
    """Analyze feature importance with multiple methods."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, :len(feature_names)-2])  # Exclude sign features

    # 1. Ridge coefficients
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_importance = np.abs(ridge.coef_)

    # 2. Mutual information
    mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

    # 3. Random forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_scaled, y)
    rf_importance = rf.feature_importances_

    # 4. Gradient boosting importance
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
    gb.fit(X_scaled, y)
    gb_importance = gb.feature_importances_

    # Aggregate
    importance_df = []
    for i, name in enumerate(feature_names[:-2]):
        importance_df.append({
            'feature': name,
            'ridge': ridge_importance[i],
            'mi': mi_scores[i],
            'rf': rf_importance[i],
            'gb': gb_importance[i],
            'avg_rank': np.mean([
                np.argsort(-ridge_importance).tolist().index(i),
                np.argsort(-mi_scores).tolist().index(i),
                np.argsort(-rf_importance).tolist().index(i),
                np.argsort(-gb_importance).tolist().index(i),
            ])
        })

    importance_df.sort(key=lambda x: x['avg_rank'])
    return importance_df


def greedy_feature_selection(X: np.ndarray, y: np.ndarray,
                             feature_names: List[str], max_features: int = 20) -> Dict:
    """Greedy forward feature selection to find optimal subset."""

    n_features = len(feature_names) - 2  # Exclude sign features
    selected = []
    remaining = list(range(n_features))
    history = []

    best_score = 0.0

    while len(selected) < max_features and remaining:
        best_addition = None
        best_new_score = best_score

        for feat in remaining:
            candidate = selected + [feat]
            result = evaluate_feature_set(X, y, candidate, n_repeats=3)

            if result['mean'] > best_new_score:
                best_new_score = result['mean']
                best_addition = feat

        if best_addition is not None and best_new_score > best_score + 0.001:
            selected.append(best_addition)
            remaining.remove(best_addition)
            best_score = best_new_score
            history.append({
                'n_features': len(selected),
                'added': feature_names[best_addition],
                'score': best_score,
                'features': [feature_names[i] for i in selected],
            })
            print(f"  Added {feature_names[best_addition]}: score = {best_score:.4f}")
        else:
            break

    return {
        'selected_features': [feature_names[i] for i in selected],
        'selected_indices': selected,
        'final_score': best_score,
        'history': history,
    }


def evaluate_nonlinear_models(X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Dict:
    """Compare linear vs nonlinear models."""

    # Feature subsets
    physico_idx = list(range(10))
    padic_idx = list(range(10, 22))
    best_idx = physico_idx + padic_idx

    X_subset = X[:, best_idx]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    results = {}

    models = {
        'Ridge': Ridge(alpha=1.0),
        'RidgeCV': RidgeCV(alphas=[0.1, 1.0, 10.0]),
        'RF_shallow': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'RF_deep': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GB_shallow': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'GB_deep': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
        r, _ = spearmanr(y, y_pred)
        mae = np.mean(np.abs(y - y_pred))
        results[name] = {'spearman': r, 'mae': mae}
        print(f"{name}: Spearman = {r:.4f}, MAE = {mae:.3f}")

    return results


def main():
    # Load data
    data_dir = PROJECT_ROOT / 'deliverables' / 'partners' / 'jose_colbes' / 'reproducibility' / 'data'
    data_path = data_dir / 's669_full.csv'
    if not data_path.exists():
        data_path = data_dir / 'S669' / 'S669.csv'

    mutations = load_s669(data_path)
    print(f"Loaded {len(mutations)} mutations")

    # Train encoder
    print("\nTraining ordered p-adic encoder...")
    encoder = train_encoder(n_epochs=200)

    # Extract all features
    print("\nExtracting extended feature set...")
    X, y, feature_names, metadata = extract_extended_features(mutations, encoder)
    print(f"Total features: {len(feature_names)}")

    # ================================================================
    # 1. Strategy Comparison
    # ================================================================
    print("\n" + "="*70)
    print("1. FEATURE SET STRATEGY COMPARISON")
    print("="*70)

    strategy_results = adaptive_model_selection(X, y, metadata, feature_names)

    # ================================================================
    # 2. Feature Importance Analysis
    # ================================================================
    print("\n" + "="*70)
    print("2. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    importance = feature_importance_analysis(X, y, feature_names)
    print("\nTop 15 features by average rank:")
    for i, feat in enumerate(importance[:15]):
        print(f"  {i+1:2d}. {feat['feature']:<20} (avg_rank: {feat['avg_rank']:.1f})")

    # ================================================================
    # 3. Greedy Feature Selection
    # ================================================================
    print("\n" + "="*70)
    print("3. GREEDY FORWARD FEATURE SELECTION")
    print("="*70)

    greedy_result = greedy_feature_selection(X, y, feature_names, max_features=15)
    print(f"\nOptimal features ({len(greedy_result['selected_features'])} total):")
    for feat in greedy_result['selected_features']:
        print(f"  - {feat}")
    print(f"Final score: {greedy_result['final_score']:.4f}")

    # ================================================================
    # 4. Nonlinear Model Comparison
    # ================================================================
    print("\n" + "="*70)
    print("4. NONLINEAR MODEL COMPARISON")
    print("="*70)

    nonlinear_results = evaluate_nonlinear_models(X, y, feature_names)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY: OPTIMIZATION RESULTS")
    print("="*70)

    # Get best strategy
    best_strategy = max(strategy_results.items(), key=lambda x: x[1]['mean'])
    print(f"\nBest strategy: {best_strategy[0]} (Spearman = {best_strategy[1]['mean']:.4f})")

    # Get best nonlinear
    best_nonlinear = max(nonlinear_results.items(), key=lambda x: x[1]['spearman'])
    print(f"Best model: {best_nonlinear[0]} (Spearman = {best_nonlinear[1]['spearman']:.4f})")

    print(f"\nGreedy selection score: {greedy_result['final_score']:.4f}")

    # Compare to baseline
    baseline_result = evaluate_feature_set(X, y, list(range(10)))  # Physico only
    print(f"\nBaseline (physico only): {baseline_result['mean']:.4f}")

    improvement = (greedy_result['final_score'] - baseline_result['mean']) / baseline_result['mean'] * 100
    print(f"Improvement with optimized features: {improvement:+.1f}%")

    # Save results
    output_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'feature_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump({
            'strategies': {k: {'mean': v['mean'], 'std': v['std']} for k, v in strategy_results.items()},
            'importance': importance[:20],
            'greedy_selection': {
                'features': greedy_result['selected_features'],
                'score': greedy_result['final_score'],
            },
            'nonlinear_models': nonlinear_results,
            'baseline': baseline_result['mean'],
            'improvement_percent': improvement,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
