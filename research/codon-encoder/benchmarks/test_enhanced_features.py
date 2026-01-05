#!/usr/bin/env python3
"""Test Enhanced Feature Combinations.

Goal: Surpass the physicochemical baseline (Spearman 0.366) by combining:
1. Ordered p-adic encoder features (from hydrophobicity ordering)
2. Raw physicochemical features
3. Explicit p-adic valuation features
4. Embedding angle/direction features

Usage:
    python test_enhanced_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance, exp_map_zero, project_to_poincare


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


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


def extract_features(mutations: List[Dict], encoder: OrderedPadicEncoder,
                     feature_set: str) -> tuple:
    """Extract features based on feature set name."""

    encoder.eval()
    embeddings = encoder.get_aa_embeddings()
    valuations = encoder.get_aa_valuations()

    X, y = [], []

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
            # Raw physicochemical features
            delta_props = mt_props - wt_props
            features.extend(delta_props.tolist())
            features.append(abs(delta_props[1]))  # abs charge change
            features.append(np.linalg.norm(delta_props))  # L2 distance

        if 'padic' in feature_set:
            # P-adic encoder features
            hyp_dist = poincare_distance(wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=1.0).item()
            origin = torch.zeros(1, 16)
            wt_r = poincare_distance(wt_emb.unsqueeze(0), origin, c=1.0).item()
            mt_r = poincare_distance(mt_emb.unsqueeze(0), origin, c=1.0).item()

            features.append(hyp_dist)
            features.append(mt_r - wt_r)
            features.append(wt_r)
            features.append(mt_r)

        if 'valuation' in feature_set:
            # Explicit p-adic valuation features
            wt_val = valuations.get(wt, 0)
            mt_val = valuations.get(mt, 0)
            features.append(wt_val)
            features.append(mt_val)
            features.append(mt_val - wt_val)  # Delta valuation
            features.append(abs(mt_val - wt_val))  # Abs delta

        if 'direction' in feature_set:
            # Embedding direction features
            wt_vec = wt_emb.numpy()
            mt_vec = mt_emb.numpy()
            diff_vec = mt_vec - wt_vec

            # Cosine similarity
            cos_sim = np.dot(wt_vec, mt_vec) / (np.linalg.norm(wt_vec) * np.linalg.norm(mt_vec) + 1e-8)
            features.append(cos_sim)

            # Direction entropy (variance of diff vector)
            features.append(np.var(diff_vec))

            # Max dimension change
            features.append(np.max(np.abs(diff_vec)))

        X.append(features)
        y.append(mut['ddg_exp'])

    return np.array(X), np.array(y)


def evaluate(X: np.ndarray, y: np.ndarray, regressor='ridge') -> Dict:
    """Evaluate with cross-validation."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if regressor == 'ridge':
        model = Ridge(alpha=1.0)
    else:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)

    spearman_r, _ = spearmanr(y, y_pred)
    pearson_r, _ = pearsonr(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    return {
        'spearman': float(spearman_r),
        'pearson': float(pearson_r),
        'mae': float(mae),
        'n_features': X.shape[1],
    }


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

    # Feature sets to test
    feature_sets = [
        ('physico', 'Physicochemical only (baseline)'),
        ('padic', 'P-adic encoder only'),
        ('valuation', 'Valuation only'),
        ('physico+padic', 'Physico + P-adic'),
        ('physico+valuation', 'Physico + Valuation'),
        ('padic+valuation', 'P-adic + Valuation'),
        ('physico+padic+valuation', 'All three'),
        ('physico+padic+valuation+direction', 'All + Direction'),
    ]

    print("\n" + "="*70)
    print("Testing Feature Combinations")
    print("="*70)

    results = []
    for feature_set, description in feature_sets:
        X, y = extract_features(mutations, encoder, feature_set)
        metrics = evaluate(X, y, regressor='ridge')
        metrics['feature_set'] = feature_set
        metrics['description'] = description
        results.append(metrics)
        print(f"\n{description}")
        print(f"  Features: {metrics['n_features']}")
        print(f"  Spearman: {metrics['spearman']:.4f}")

    # Also test with GradientBoosting for best feature set
    print("\n" + "-"*70)
    print("Testing with GradientBoosting Regressor")
    print("-"*70)

    for feature_set, description in [('physico+padic+valuation+direction', 'All features')]:
        X, y = extract_features(mutations, encoder, feature_set)
        metrics_gb = evaluate(X, y, regressor='gb')
        print(f"\n{description} (GradientBoosting)")
        print(f"  Spearman: {metrics_gb['spearman']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Feature Combination Results")
    print("="*70)
    print(f"{'Feature Set':<40} {'Spearman':>10} {'#Feats':>8}")
    print("-"*60)

    results.sort(key=lambda x: -x['spearman'])
    for r in results:
        marker = " <-- BEST" if r == results[0] else ""
        print(f"{r['description']:<40} {r['spearman']:>10.4f} {r['n_features']:>8}{marker}")

    baseline = next(r for r in results if r['feature_set'] == 'physico')
    best = results[0]

    print("-"*60)
    print(f"\nBaseline (physico): {baseline['spearman']:.4f}")
    print(f"Best combination:   {best['spearman']:.4f} ({best['description']})")
    print(f"Improvement:        {(best['spearman'] - baseline['spearman']) / baseline['spearman'] * 100:+.1f}%")

    # Save results
    output_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'enhanced_features'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'feature_combinations.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
