#!/usr/bin/env python3
"""Test Ordered Amino Acid Indices for P-adic Encoding.

Hypothesis: If we order amino acids by a biological property (e.g., hydrophobicity),
then p-adic valuation creates meaningful hierarchies where:
- Indices divisible by p are "central" in that property dimension
- This could improve DDG prediction for hydrophobicity-driven mutations

Orderings to test:
1. Alphabetical (baseline, arbitrary)
2. Hydrophobicity (most hydrophobic = index 0)
3. Molecular weight (lightest = index 0)
4. Charge (most negative = index 0, then neutral, then positive)
5. Volume (smallest = index 0)

Usage:
    python test_ordered_indices.py
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
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance, exp_map_zero, project_to_poincare


# Amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Property data (hydrophobicity, charge, volume, polarity)
# From AA_PROPERTIES: hydrophobicity, charge, volume, polarity


def get_orderings() -> Dict[str, List[str]]:
    """Get different amino acid orderings."""

    # Get properties
    props = {aa: AA_PROPERTIES.get(aa, (0, 0, 0, 0)) for aa in AMINO_ACIDS}

    orderings = {}

    # 1. Alphabetical (baseline)
    orderings['alphabetical'] = sorted(AMINO_ACIDS)

    # 2. Hydrophobicity (most hydrophobic first)
    orderings['hydrophobicity'] = sorted(AMINO_ACIDS, key=lambda aa: -props[aa][0])

    # 3. Molecular weight / Volume (smallest first)
    orderings['volume'] = sorted(AMINO_ACIDS, key=lambda aa: props[aa][2])

    # 4. Charge (negative → neutral → positive)
    orderings['charge'] = sorted(AMINO_ACIDS, key=lambda aa: props[aa][1])

    # 5. Polarity (least polar first)
    orderings['polarity'] = sorted(AMINO_ACIDS, key=lambda aa: props[aa][3])

    # 6. Combined score (hydrophobic + small + neutral)
    def combined_score(aa):
        h, c, v, p = props[aa]
        return -h + v/100 + abs(c)  # Prefer hydrophobic, small, neutral
    orderings['combined'] = sorted(AMINO_ACIDS, key=combined_score)

    return orderings


class OrderedPadicEncoder(nn.Module):
    """P-adic encoder with custom amino acid ordering."""

    def __init__(self, ordering: List[str], prime: int = 5, latent_dim: int = 16,
                 hidden_dim: int = 64, curvature: float = 1.0):
        super().__init__()

        self.ordering = ordering
        self.aa_to_idx = {aa: i for i, aa in enumerate(ordering)}
        self.prime = prime
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.n_aa = 20

        # Learnable embeddings
        self.aa_embedding = nn.Parameter(torch.randn(self.n_aa, latent_dim) * 0.01)

        # Property encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Setup hierarchy based on ordering
        self._setup_hierarchy()
        self._setup_properties()

    def _setup_hierarchy(self):
        """Compute hierarchy based on p-adic valuation of ordered indices."""
        def valuation(n, p):
            if n == 0:
                return p
            v = 0
            while n % p == 0:
                n //= p
                v += 1
            return v

        hierarchy = [min(valuation(i, self.prime), 4) for i in range(self.n_aa)]
        self.register_buffer('hierarchy_levels', torch.tensor(hierarchy, dtype=torch.long))

        max_level = max(hierarchy) if max(hierarchy) > 0 else 1
        target_radii = [0.85 - (h / max_level) * 0.6 for h in hierarchy]
        self.register_buffer('target_radii', torch.tensor(target_radii, dtype=torch.float32))

    def _setup_properties(self):
        """Setup properties in ordering order."""
        props = [list(AA_PROPERTIES.get(aa, (0, 0, 0, 0))) for aa in self.ordering]
        self.register_buffer('aa_properties', torch.tensor(props, dtype=torch.float32))

    def forward(self, aa_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_base = self.aa_embedding[aa_indices]
        props = self.aa_properties[aa_indices]
        z_prop = self.property_encoder(props)
        z_fused = self.fusion(torch.cat([z_base, z_prop], dim=-1))

        z_hyp = exp_map_zero(z_fused, c=self.curvature)
        z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)

        return {'z_hyp': z_hyp, 'radii': radii}

    def get_aa_embeddings(self) -> Dict[str, torch.Tensor]:
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)
        return {aa: out['z_hyp'][i] for i, aa in enumerate(self.ordering)}


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
        except (ValueError, IndexError):
            continue
    return mutations


def train_and_evaluate(ordering_name: str, ordering: List[str], mutations: List[Dict],
                       prime: int = 5, n_epochs: int = 150) -> Dict:
    """Train encoder and evaluate on DDG prediction."""

    device = 'cpu'
    encoder = OrderedPadicEncoder(ordering, prime=prime, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training
    for epoch in range(n_epochs):
        encoder.train()
        optimizer.zero_grad()

        indices = torch.arange(20, device=device)
        out = encoder(indices)

        # Hierarchy loss
        hierarchy_loss = torch.mean((out['radii'] - encoder.target_radii) ** 2)

        # Property alignment
        z = out['z_hyp']
        prop_dist = torch.cdist(encoder.aa_properties, encoder.aa_properties, p=2)
        prop_dist = prop_dist / (prop_dist.max() + 1e-8)

        emb_dist = torch.zeros(20, 20, device=device)
        for i in range(20):
            for j in range(i+1, 20):
                d = poincare_distance(z[i:i+1], z[j:j+1], c=encoder.curvature).squeeze().clamp(0, 10)
                emb_dist[i, j] = emb_dist[j, i] = d
        emb_dist = emb_dist / (emb_dist.max() + 1e-8)

        mask = torch.triu(torch.ones(20, 20, device=device), diagonal=1).bool()
        property_loss = torch.mean((emb_dist[mask] - prop_dist[mask]) ** 2)

        loss = hierarchy_loss * 2.0 + property_loss * 1.0

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluation
    encoder.eval()
    embeddings = encoder.get_aa_embeddings()

    X, y = [], []
    for mut in mutations:
        wt, mt = mut['wild_type'], mut['mutant']
        if wt not in embeddings or mt not in embeddings:
            continue

        wt_emb = embeddings[wt].detach().cpu()
        mt_emb = embeddings[mt].detach().cpu()

        hyp_dist = poincare_distance(wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=encoder.curvature).item()
        origin = torch.zeros(1, encoder.latent_dim)
        wt_r = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        mt_r = poincare_distance(mt_emb.unsqueeze(0), origin, c=encoder.curvature).item()

        wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
        mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))

        X.append([hyp_dist, mt_r - wt_r, mt_props[0] - wt_props[0],
                  abs(mt_props[1] - wt_props[1]), mt_props[2] - wt_props[2]])
        y.append(mut['ddg_exp'])

    X, y = np.array(X), np.array(y)

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)

    spearman_r, _ = spearmanr(y, y_pred)
    pearson_r, _ = pearsonr(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    # Check if ordering creates meaningful hierarchy
    # For hydrophobicity ordering, check if hydrophobic AAs are p-adically central
    hierarchy_check = {}
    for i, aa in enumerate(ordering[:5]):  # Top 5 in ordering
        val = int(encoder.hierarchy_levels[i].item())
        hierarchy_check[aa] = val

    return {
        'ordering': ordering_name,
        'spearman': float(spearman_r),
        'pearson': float(pearson_r),
        'mae': float(mae),
        'n_samples': len(y),
        'top5_hierarchy': hierarchy_check,
        'ordering_list': ordering,
    }


def main():
    # Load data
    data_dir = PROJECT_ROOT / 'deliverables' / 'partners' / 'jose_colbes' / 'reproducibility' / 'data'
    data_path = data_dir / 's669_full.csv'
    if not data_path.exists():
        data_path = data_dir / 'S669' / 'S669.csv'

    mutations = load_s669(data_path)
    print(f"Loaded {len(mutations)} mutations")

    # Get orderings
    orderings = get_orderings()

    print("\n" + "="*70)
    print("Testing Amino Acid Orderings for P-adic Encoding")
    print("="*70)

    results = []
    for name, ordering in orderings.items():
        print(f"\n[{name.upper()}] Order: {' '.join(ordering[:5])}... → {' '.join(ordering[-3:])}")
        result = train_and_evaluate(name, ordering, mutations, prime=5, n_epochs=150)
        results.append(result)
        print(f"  Spearman: {result['spearman']:.4f}")
        print(f"  Top-5 p-adic levels: {result['top5_hierarchy']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Ordered Index Results")
    print("="*70)
    print(f"{'Ordering':<20} {'Spearman':>12} {'Pearson':>12} {'MAE':>10}")
    print("-"*60)

    results.sort(key=lambda x: -x['spearman'])
    for r in results:
        print(f"{r['ordering']:<20} {r['spearman']:>12.4f} {r['pearson']:>12.4f} {r['mae']:>10.4f}")

    print("-"*60)
    print(f"\nBest ordering: {results[0]['ordering']} (Spearman = {results[0]['spearman']:.4f})")

    # Save results
    output_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'ordered_indices'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'ordering_comparison.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'ordering_comparison.json'}")


if __name__ == '__main__':
    main()
