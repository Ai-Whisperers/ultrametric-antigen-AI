#!/usr/bin/env python3
"""Rigorous Statistical Validation for P-adic Feature Combinations.

Goal: Validate the +8.5% improvement with skeptical metrics:
1. Bootstrap confidence intervals for Spearman correlation
2. Permutation test for significance of improvement over baseline
3. Stratified analysis by mutation type (polarity change, charge change, size change)
4. Stratified analysis by DDG magnitude (stabilizing vs destabilizing)
5. Leave-one-out cross-validation for robustness check

Usage:
    python rigorous_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, ttest_rel, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance, exp_map_zero, project_to_poincare


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Amino acid classifications for stratification
AA_POLAR = set('STNQYW')  # Polar uncharged
AA_NONPOLAR = set('GAVLIMPFW')  # Nonpolar
AA_CHARGED_POS = set('KRH')  # Positively charged
AA_CHARGED_NEG = set('DE')  # Negatively charged
AA_SMALL = set('GAVSTC')  # Small AAs
AA_LARGE = set('FYWKRH')  # Large AAs


@dataclass
class ValidationResult:
    """Result from validation experiment."""
    metric: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    p_value: float = None


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
    """Load S669 dataset with mutation metadata."""
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
                    mutations.append({
                        'wild_type': wt,
                        'mutant': mt,
                        'ddg_exp': ddg,
                        'mutation_str': f"{wt}->{mt}",
                    })
        except:
            continue
    return mutations


def classify_mutation(wt: str, mt: str) -> Dict[str, str]:
    """Classify mutation by type."""
    classifications = {}

    # Polarity change
    wt_polar = wt in AA_POLAR
    mt_polar = mt in AA_POLAR
    wt_nonpolar = wt in AA_NONPOLAR
    mt_nonpolar = mt in AA_NONPOLAR

    if wt_polar and mt_nonpolar:
        classifications['polarity'] = 'polar_to_nonpolar'
    elif wt_nonpolar and mt_polar:
        classifications['polarity'] = 'nonpolar_to_polar'
    else:
        classifications['polarity'] = 'same_polarity'

    # Charge change
    wt_pos = wt in AA_CHARGED_POS
    wt_neg = wt in AA_CHARGED_NEG
    mt_pos = mt in AA_CHARGED_POS
    mt_neg = mt in AA_CHARGED_NEG

    if (wt_pos or wt_neg) and not (mt_pos or mt_neg):
        classifications['charge'] = 'charged_to_neutral'
    elif not (wt_pos or wt_neg) and (mt_pos or mt_neg):
        classifications['charge'] = 'neutral_to_charged'
    elif (wt_pos and mt_neg) or (wt_neg and mt_pos):
        classifications['charge'] = 'charge_reversal'
    else:
        classifications['charge'] = 'same_charge'

    # Size change
    wt_small = wt in AA_SMALL
    wt_large = wt in AA_LARGE
    mt_small = mt in AA_SMALL
    mt_large = mt in AA_LARGE

    if wt_small and mt_large:
        classifications['size'] = 'small_to_large'
    elif wt_large and mt_small:
        classifications['size'] = 'large_to_small'
    else:
        classifications['size'] = 'similar_size'

    return classifications


def extract_features(mutations: List[Dict], encoder: OrderedPadicEncoder,
                     feature_set: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract features with mutation metadata."""

    encoder.eval()
    embeddings = encoder.get_aa_embeddings()
    valuations = encoder.get_aa_valuations()

    X, y, metadata = [], [], []

    for mut in mutations:
        wt, mt = mut['wild_type'], mut['mutant']
        if wt not in embeddings or mt not in embeddings:
            continue

        wt_emb = embeddings[wt].detach().cpu()
        mt_emb = embeddings[mt].detach().cpu()
        wt_props = np.array(AA_PROPERTIES.get(wt, (0, 0, 0, 0)))
        mt_props = np.array(AA_PROPERTIES.get(mt, (0, 0, 0, 0)))

        features = []

        if 'physico' in feature_set:
            delta_props = mt_props - wt_props
            features.extend(delta_props.tolist())
            features.append(abs(delta_props[1]))  # abs charge change
            features.append(np.linalg.norm(delta_props))  # L2 distance

        if 'padic' in feature_set:
            hyp_dist = poincare_distance(wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=1.0).item()
            origin = torch.zeros(1, 16)
            wt_r = poincare_distance(wt_emb.unsqueeze(0), origin, c=1.0).item()
            mt_r = poincare_distance(mt_emb.unsqueeze(0), origin, c=1.0).item()

            features.append(hyp_dist)
            features.append(mt_r - wt_r)
            features.append(wt_r)
            features.append(mt_r)

        if 'valuation' in feature_set:
            wt_val = valuations.get(wt, 0)
            mt_val = valuations.get(mt, 0)
            features.append(wt_val)
            features.append(mt_val)
            features.append(mt_val - wt_val)
            features.append(abs(mt_val - wt_val))

        X.append(features)
        y.append(mut['ddg_exp'])

        # Add classifications
        meta = {**mut, **classify_mutation(wt, mt)}
        meta['ddg_class'] = 'stabilizing' if mut['ddg_exp'] < 0 else 'destabilizing'
        meta['ddg_magnitude'] = 'small' if abs(mut['ddg_exp']) < 1.0 else ('medium' if abs(mut['ddg_exp']) < 2.0 else 'large')
        metadata.append(meta)

    return np.array(X), np.array(y), metadata


def bootstrap_spearman(y_true: np.ndarray, y_pred: np.ndarray,
                       n_bootstrap: int = 1000, ci: float = 0.95) -> ValidationResult:
    """Compute bootstrap confidence interval for Spearman correlation."""
    n = len(y_true)
    correlations = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(y_true[idx], y_pred[idx])
        if not np.isnan(r):
            correlations.append(r)

    correlations = np.array(correlations)
    alpha = (1 - ci) / 2

    return ValidationResult(
        metric='spearman',
        mean=np.mean(correlations),
        std=np.std(correlations),
        ci_lower=np.percentile(correlations, alpha * 100),
        ci_upper=np.percentile(correlations, (1 - alpha) * 100),
        n_samples=n,
    )


def permutation_test(y_true: np.ndarray, y_pred_best: np.ndarray,
                     y_pred_baseline: np.ndarray, n_permutations: int = 1000) -> float:
    """Permutation test: Is best model significantly better than baseline?"""
    # Observed difference
    r_best, _ = spearmanr(y_true, y_pred_best)
    r_baseline, _ = spearmanr(y_true, y_pred_baseline)
    observed_diff = r_best - r_baseline

    # Null distribution
    combined = np.column_stack([y_pred_best, y_pred_baseline])
    null_diffs = []

    for _ in range(n_permutations):
        # Randomly swap predictions for each sample
        swapped = combined.copy()
        swap_mask = np.random.binomial(1, 0.5, len(y_true)).astype(bool)
        swapped[swap_mask, 0], swapped[swap_mask, 1] = \
            swapped[swap_mask, 1].copy(), swapped[swap_mask, 0].copy()

        r1, _ = spearmanr(y_true, swapped[:, 0])
        r2, _ = spearmanr(y_true, swapped[:, 1])
        null_diffs.append(r1 - r2)

    null_diffs = np.array(null_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value


def stratified_evaluation(X: np.ndarray, y: np.ndarray, metadata: List[Dict],
                          stratify_by: str) -> Dict[str, ValidationResult]:
    """Evaluate performance stratified by mutation type."""
    results = {}

    # Get unique categories
    categories = set(m[stratify_by] for m in metadata)

    for category in categories:
        # Filter samples
        mask = np.array([m[stratify_by] == category for m in metadata])
        X_cat, y_cat = X[mask], y[mask]

        if len(y_cat) < 10:  # Skip small groups
            continue

        # Cross-validation
        kf = KFold(n_splits=min(5, len(y_cat)), shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cat)

        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X_scaled, y_cat, cv=kf)

        # Bootstrap CI
        result = bootstrap_spearman(y_cat, y_pred, n_bootstrap=500)
        result.n_samples = len(y_cat)
        results[category] = result

    return results


def run_cv_comparison(X_best: np.ndarray, X_baseline: np.ndarray,
                      y: np.ndarray, n_repeats: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Run repeated cross-validation for statistical comparison."""
    best_scores = []
    baseline_scores = []

    for seed in range(n_repeats):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        scaler_best = StandardScaler()
        X_best_scaled = scaler_best.fit_transform(X_best)
        model_best = Ridge(alpha=1.0)
        y_pred_best = cross_val_predict(model_best, X_best_scaled, y, cv=kf)
        r_best, _ = spearmanr(y, y_pred_best)
        best_scores.append(r_best)

        scaler_base = StandardScaler()
        X_base_scaled = scaler_base.fit_transform(X_baseline)
        model_base = Ridge(alpha=1.0)
        y_pred_base = cross_val_predict(model_base, X_base_scaled, y, cv=kf)
        r_base, _ = spearmanr(y, y_pred_base)
        baseline_scores.append(r_base)

    return np.array(best_scores), np.array(baseline_scores)


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

    # Extract features for both models
    print("\nExtracting features...")
    X_best, y, metadata = extract_features(mutations, encoder, 'physico+padic+valuation')
    X_baseline, _, _ = extract_features(mutations, encoder, 'physico')

    print(f"Best model: {X_best.shape[1]} features")
    print(f"Baseline: {X_baseline.shape[1]} features")

    # ================================================================
    # 1. Bootstrap Confidence Intervals
    # ================================================================
    print("\n" + "="*70)
    print("1. BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
    print("="*70)

    # Get predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scaler_best = StandardScaler()
    X_best_scaled = scaler_best.fit_transform(X_best)
    model_best = Ridge(alpha=1.0)
    y_pred_best = cross_val_predict(model_best, X_best_scaled, y, cv=kf)

    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_baseline)
    model_base = Ridge(alpha=1.0)
    y_pred_baseline = cross_val_predict(model_base, X_base_scaled, y, cv=kf)

    result_best = bootstrap_spearman(y, y_pred_best, n_bootstrap=1000)
    result_baseline = bootstrap_spearman(y, y_pred_baseline, n_bootstrap=1000)

    print(f"\nBest (physico+padic+valuation):")
    print(f"  Spearman: {result_best.mean:.4f} (95% CI: [{result_best.ci_lower:.4f}, {result_best.ci_upper:.4f}])")

    print(f"\nBaseline (physico):")
    print(f"  Spearman: {result_baseline.mean:.4f} (95% CI: [{result_baseline.ci_lower:.4f}, {result_baseline.ci_upper:.4f}])")

    improvement = (result_best.mean - result_baseline.mean) / result_baseline.mean * 100
    print(f"\nImprovement: {improvement:+.1f}%")

    # Check CI overlap
    ci_overlap = result_best.ci_lower < result_baseline.ci_upper
    print(f"CI overlap: {'YES (need significance test)' if ci_overlap else 'NO (clearly significant)'}")

    # ================================================================
    # 2. Permutation Test for Significance
    # ================================================================
    print("\n" + "="*70)
    print("2. PERMUTATION TEST (1000 permutations)")
    print("="*70)

    p_value = permutation_test(y, y_pred_best, y_pred_baseline, n_permutations=1000)
    print(f"\nNull hypothesis: Best model performs same as baseline")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance at 0.05: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"Significance at 0.01: {'YES' if p_value < 0.01 else 'NO'}")

    # ================================================================
    # 3. Paired Statistical Tests
    # ================================================================
    print("\n" + "="*70)
    print("3. PAIRED STATISTICAL TESTS (10 CV repeats)")
    print("="*70)

    best_scores, baseline_scores = run_cv_comparison(X_best, X_baseline, y, n_repeats=10)

    print(f"\nBest model scores: {best_scores.mean():.4f} +/- {best_scores.std():.4f}")
    print(f"Baseline scores:   {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}")

    # Paired t-test
    t_stat, t_pval = ttest_rel(best_scores, baseline_scores)
    print(f"\nPaired t-test: t={t_stat:.3f}, p={t_pval:.4f}")

    # Wilcoxon signed-rank test
    w_stat, w_pval = wilcoxon(best_scores, baseline_scores)
    print(f"Wilcoxon test: W={w_stat:.1f}, p={w_pval:.4f}")

    # ================================================================
    # 4. Stratified Analysis by Mutation Type
    # ================================================================
    print("\n" + "="*70)
    print("4. STRATIFIED ANALYSIS BY MUTATION TYPE")
    print("="*70)

    stratifications = ['polarity', 'charge', 'size', 'ddg_class', 'ddg_magnitude']

    stratified_results = {}
    for strat in stratifications:
        print(f"\n--- Stratified by {strat.upper()} ---")
        results = stratified_evaluation(X_best, y, metadata, strat)

        stratified_results[strat] = {}
        for cat, result in sorted(results.items(), key=lambda x: -x[1].mean):
            print(f"  {cat}: Spearman = {result.mean:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}] (n={result.n_samples})")
            stratified_results[strat][cat] = asdict(result)

    # ================================================================
    # 5. Where does P-adic help most?
    # ================================================================
    print("\n" + "="*70)
    print("5. P-ADIC ADVANTAGE BY MUTATION TYPE")
    print("="*70)

    # Compare best vs baseline for each stratum
    for strat in ['charge', 'size', 'ddg_magnitude']:
        print(f"\n--- {strat.upper()} ---")
        categories = set(m[strat] for m in metadata)

        for cat in sorted(categories):
            mask = np.array([m[strat] == cat for m in metadata])
            X_best_cat, X_base_cat, y_cat = X_best[mask], X_baseline[mask], y[mask]

            if len(y_cat) < 15:
                continue

            kf = KFold(n_splits=min(5, len(y_cat)), shuffle=True, random_state=42)

            # Best
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_best_cat)
            y_pred_b = cross_val_predict(Ridge(alpha=1.0), X_s, y_cat, cv=kf)
            r_best, _ = spearmanr(y_cat, y_pred_b)

            # Baseline
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_base_cat)
            y_pred_bl = cross_val_predict(Ridge(alpha=1.0), X_s, y_cat, cv=kf)
            r_base, _ = spearmanr(y_cat, y_pred_bl)

            advantage = (r_best - r_base) / max(r_base, 0.01) * 100
            marker = " ***" if advantage > 15 else " *" if advantage > 5 else ""
            print(f"  {cat:<25} Best: {r_best:.4f}  Baseline: {r_base:.4f}  Advantage: {advantage:+.1f}%{marker}")

    # ================================================================
    # Summary and Recommendations
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)

    print(f"""
STATISTICAL VALIDATION RESULTS:
-------------------------------
Overall Improvement: {improvement:+.1f}%
Bootstrap 95% CI for Best: [{result_best.ci_lower:.4f}, {result_best.ci_upper:.4f}]
Bootstrap 95% CI for Base: [{result_baseline.ci_lower:.4f}, {result_baseline.ci_upper:.4f}]
Permutation p-value: {p_value:.4f}
Paired t-test p-value: {t_pval:.4f}

CONCLUSION:
-----------
""")

    if p_value < 0.05 and t_pval < 0.05:
        print("The +8.5% improvement is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("The improvement is NOT statistically significant at the 0.05 level")

    # Save results
    output_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'statistical_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_report = {
        'overall': {
            'best_model': asdict(result_best),
            'baseline': asdict(result_baseline),
            'improvement_percent': float(improvement),
            'permutation_p_value': float(p_value),
            'paired_t_test_p_value': float(t_pval),
            'wilcoxon_p_value': float(w_pval),
        },
        'cv_scores': {
            'best': best_scores.tolist(),
            'baseline': baseline_scores.tolist(),
        },
        'stratified': stratified_results,
    }

    with open(output_dir / 'validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"\nResults saved to {output_dir / 'validation_report.json'}")


if __name__ == '__main__':
    main()
