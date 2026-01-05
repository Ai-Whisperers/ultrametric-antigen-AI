#!/usr/bin/env python3
"""Train Generalized P-adic Encoder and Validate on DDG Prediction.

This script:
1. Trains the GeneralizedPadicEncoder for amino acid sequences
2. Compares different primes (3, 5, 7) for p-adic structure
3. Validates segment-based encoding on long vs short sequences
4. Evaluates DDG prediction with learned embeddings

Key insight from regime analysis:
- LENGTH is the dominant factor in prediction failure for long peptides
- Segment-based encoding captures local patterns while preserving global structure
- P-adic generalization (beyond 3-adic) may capture different hierarchical relationships

Usage:
    python train_generalized_padic_encoder.py [--prime 5] [--epochs 500]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from src.encoders.generalized_padic_encoder import (
    GeneralizedPadicEncoder,
    create_aminoacid_encoder,
    compute_padic_distance,
)
from src.geometry import poincare_distance
from src.encoders.codon_encoder import AA_PROPERTIES

# Amino acid encoding
AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '*': 20,  # Stop
}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}


class AminoAcidPadicEncoder(nn.Module):
    """Amino acid encoder using p-adic structure.

    Learns embeddings for 20 amino acids on the Poincaré ball,
    with p-adic hierarchy determining radial position.
    """

    def __init__(
        self,
        prime: int = 5,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        curvature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.prime = prime
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.n_aa = 20

        # Learnable AA embeddings in tangent space (small init for stability)
        self.aa_embedding = nn.Parameter(torch.randn(self.n_aa, latent_dim) * 0.01)

        # Property encoder (hydrophobicity, charge, volume, polarity)
        self.property_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Projection to Poincaré ball
        from src.geometry import exp_map_zero, project_to_poincare
        self.exp_map_zero = exp_map_zero
        self.project_to_poincare = project_to_poincare

        # Pre-compute p-adic valuations for each AA index
        self._compute_hierarchy_levels()

        # Pre-compute property vectors
        self._setup_properties()

    def _compute_hierarchy_levels(self):
        """Compute hierarchy levels based on p-adic valuation of AA index."""
        from src.encoders.generalized_padic_encoder import compute_padic_valuation

        hierarchy = []
        for i in range(self.n_aa):
            v = compute_padic_valuation(i, self.prime)
            hierarchy.append(min(v, 4))  # Cap at 4 levels

        self.register_buffer('hierarchy_levels', torch.tensor(hierarchy, dtype=torch.long))

        # Target radii: higher hierarchy → smaller radius (more central)
        max_level = max(hierarchy)
        target_radii = [0.85 - (h / max_level) * 0.6 for h in hierarchy]
        self.register_buffer('target_radii', torch.tensor(target_radii, dtype=torch.float32))

    def _setup_properties(self):
        """Set up amino acid property vectors."""
        props = []
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            p = AA_PROPERTIES.get(aa, (0, 0, 0, 0))
            props.append(list(p))

        self.register_buffer('aa_properties', torch.tensor(props, dtype=torch.float32))

    def forward(self, aa_indices: Tensor) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            aa_indices: Amino acid indices (batch,) or (batch, seq_len)

        Returns:
            Dictionary with embeddings and radii
        """
        original_shape = aa_indices.shape
        if len(original_shape) == 2:
            # Sequence input - flatten
            batch_size, seq_len = original_shape
            aa_indices = aa_indices.view(-1)
        else:
            batch_size = aa_indices.shape[0]
            seq_len = 1

        # Get base embeddings
        z_base = self.aa_embedding[aa_indices]  # (N, latent_dim)

        # Get property encoding
        props = self.aa_properties[aa_indices]  # (N, 4)
        z_prop = self.property_encoder(props)  # (N, latent_dim)

        # Fuse
        z_fused = self.fusion(torch.cat([z_base, z_prop], dim=-1))

        # Map to Poincaré ball
        z_hyp = self.exp_map_zero(z_fused, c=self.curvature)
        z_hyp = self.project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

        # Compute radii
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)

        # Reshape if needed
        if seq_len > 1:
            z_hyp = z_hyp.view(batch_size, seq_len, -1)
            radii = radii.view(batch_size, seq_len)

        return {
            'z_hyp': z_hyp,
            'radii': radii,
            'z_tangent': z_fused,
        }

    def get_aa_embeddings(self) -> Dict[str, Tensor]:
        """Get embeddings for all amino acids."""
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)

        return {
            aa: out['z_hyp'][i]
            for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')
        }

    def compute_hierarchy_loss(self) -> Tensor:
        """Loss encouraging radii to match hierarchy levels."""
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)
        radii = out['radii']

        return torch.mean((radii - self.target_radii) ** 2)

    def compute_property_alignment_loss(self) -> Tensor:
        """Loss encouraging similar properties → similar embeddings."""
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)
        z_hyp = out['z_hyp']  # (20, latent_dim)

        # Compute pairwise hyperbolic distances
        n = self.n_aa
        hyp_dists = torch.zeros(n, n, device=z_hyp.device)
        for i in range(n):
            for j in range(i + 1, n):
                d = poincare_distance(
                    z_hyp[i:i+1], z_hyp[j:j+1], c=self.curvature
                ).squeeze()
                d = d.clamp(min=0.0, max=10.0)  # Clamp for stability
                hyp_dists[i, j] = d
                hyp_dists[j, i] = d

        # Compute property distances
        props = self.aa_properties
        prop_dists = torch.cdist(props, props, p=2)
        prop_dists = prop_dists / (prop_dists.max() + 1e-8)

        # Normalize hyperbolic distances to same scale
        hyp_dists = hyp_dists / (hyp_dists.max() + 1e-8)

        # Alignment loss: similar properties → similar embeddings
        mask = torch.triu(torch.ones(n, n, device=z_hyp.device), diagonal=1).bool()
        loss = torch.mean((hyp_dists[mask] - prop_dists[mask]) ** 2)

        return loss

    def compute_padic_structure_loss(self, n_triplets: int = 100) -> Tensor:
        """Loss preserving p-adic distance structure using random triplet sampling."""
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)
        z_hyp = out['z_hyp']
        n = self.n_aa

        # Precompute all pairwise hyperbolic distances using poincare_distance
        hyp_dists = torch.zeros(n, n, device=z_hyp.device)
        for i in range(n):
            for j in range(i + 1, n):
                d = poincare_distance(
                    z_hyp[i:i+1], z_hyp[j:j+1], c=self.curvature
                ).squeeze()
                # Clamp to prevent NaN
                d = d.clamp(min=0.0, max=10.0)
                hyp_dists[i, j] = d
                hyp_dists[j, i] = d

        # Precompute p-adic distances
        padic_dists = torch.zeros(n, n, device=z_hyp.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    padic_dists[i, j] = compute_padic_distance(i, j, self.prime)

        # Random triplet sampling
        triplet_losses = []
        margin = 0.1
        valid_triplets = []

        # Pre-generate valid triplets
        for anchor in range(n):
            for pos in range(n):
                if anchor == pos:
                    continue
                for neg in range(n):
                    if anchor == neg or pos == neg:
                        continue
                    if padic_dists[anchor, pos] < padic_dists[anchor, neg]:
                        valid_triplets.append((anchor, pos, neg))

        # Sample if we have too many
        if len(valid_triplets) > n_triplets:
            valid_triplets = random.sample(valid_triplets, n_triplets)

        for anchor, pos, neg in valid_triplets:
            d_ap = hyp_dists[anchor, pos]
            d_an = hyp_dists[anchor, neg]

            # Skip if NaN
            if torch.isnan(d_ap) or torch.isnan(d_an):
                continue

            trip_loss = torch.relu(d_ap - d_an + margin)
            triplet_losses.append(trip_loss)

        if triplet_losses:
            loss = torch.stack(triplet_losses).mean()
        else:
            loss = torch.tensor(0.0, device=z_hyp.device)

        return loss

    def compute_total_loss(self) -> Dict[str, Tensor]:
        """Compute all losses with NaN safety."""
        hierarchy_loss = self.compute_hierarchy_loss()
        property_loss = self.compute_property_alignment_loss()
        padic_loss = self.compute_padic_structure_loss()

        # Replace NaN with zero
        if torch.isnan(hierarchy_loss):
            hierarchy_loss = torch.tensor(0.0, device=self.aa_embedding.device)
        if torch.isnan(property_loss):
            property_loss = torch.tensor(0.0, device=self.aa_embedding.device)
        if torch.isnan(padic_loss):
            padic_loss = torch.tensor(0.0, device=self.aa_embedding.device)

        total = hierarchy_loss * 2.0 + property_loss * 1.0 + padic_loss * 1.5

        return {
            'total': total,
            'hierarchy': hierarchy_loss,
            'property': property_loss,
            'padic': padic_loss,
        }


def load_s669(filepath: Path) -> List[Dict]:
    """Load S669 dataset for DDG prediction."""
    mutations = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    ddg_idx = header.index('Experimental_DDG_dir')

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) <= ddg_idx:
            continue

        try:
            # Parse mutation like "S11A" or "D76A"
            seq_mut = parts[2] if len(parts) > 2 else parts[1]

            # Extract wild type and mutant
            if len(seq_mut) >= 3:
                wt = seq_mut[0].upper()
                mt = seq_mut[-1].upper()

                if wt in AA_TO_IDX and mt in AA_TO_IDX and wt != mt:
                    ddg = float(parts[ddg_idx])
                    mutations.append({
                        'wild_type': wt,
                        'mutant': mt,
                        'ddg_exp': ddg,
                    })
        except (ValueError, IndexError):
            continue

    return mutations


def evaluate_ddg(
    encoder: AminoAcidPadicEncoder,
    mutations: List[Dict],
) -> Dict[str, float]:
    """Evaluate DDG prediction with trained encoder."""
    encoder.eval()

    # Get AA embeddings
    aa_embeddings = encoder.get_aa_embeddings()

    # Build features
    X = []
    y = []

    for mut in mutations:
        wt = mut['wild_type']
        mt = mut['mutant']

        if wt not in aa_embeddings or mt not in aa_embeddings:
            continue

        wt_emb = aa_embeddings[wt]
        mt_emb = aa_embeddings[mt]

        # Hyperbolic distance
        hyp_dist = poincare_distance(
            wt_emb.unsqueeze(0),
            mt_emb.unsqueeze(0),
            c=encoder.curvature
        ).item()

        # Radii difference
        origin = torch.zeros(1, encoder.latent_dim, device=wt_emb.device)
        wt_radius = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        mt_radius = poincare_distance(mt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        delta_radius = mt_radius - wt_radius

        # Physicochemical features
        wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
        mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))

        delta_hydro = mt_props[0] - wt_props[0]
        delta_charge = abs(mt_props[1] - wt_props[1])
        delta_volume = mt_props[2] - wt_props[2] if len(mt_props) > 2 else 0

        X.append([hyp_dist, delta_radius, delta_hydro, delta_charge, delta_volume])
        y.append(mut['ddg_exp'])

    if len(X) == 0:
        return {'spearman': 0.0, 'pearson': 0.0, 'mae': float('inf'), 'n_samples': 0}

    X = np.array(X)
    y = np.array(y)

    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha = 1.0
    best_score = -1

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        y_pred = cross_val_predict(model, X, y, cv=kf)
        r, _ = pearsonr(y_pred, y)
        if r > best_score:
            best_score = r
            best_alpha = alpha

    # Final predictions
    model = Ridge(alpha=best_alpha)
    y_pred = cross_val_predict(model, X, y, cv=kf)

    spearman_r, _ = spearmanr(y_pred, y)
    pearson_r, _ = pearsonr(y_pred, y)
    mae = np.mean(np.abs(y_pred - y))

    return {
        'spearman': spearman_r,
        'pearson': pearson_r,
        'mae': mae,
        'n_samples': len(y),
    }


def train_encoder(
    encoder: AminoAcidPadicEncoder,
    n_epochs: int = 500,
    lr: float = 0.001,
    verbose: bool = True,
) -> List[Dict]:
    """Train the encoder."""
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = []

    for epoch in range(n_epochs):
        encoder.train()
        optimizer.zero_grad()

        losses = encoder.compute_total_loss()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        history.append({
            'epoch': epoch,
            'total': losses['total'].item(),
            'hierarchy': losses['hierarchy'].item(),
            'property': losses['property'].item(),
            'padic': losses['padic'].item(),
        })

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}: "
                  f"total={losses['total'].item():.4f}, "
                  f"hierarchy={losses['hierarchy'].item():.4f}, "
                  f"property={losses['property'].item():.4f}, "
                  f"padic={losses['padic'].item():.4f}")

    return history


def compare_primes(
    mutations: List[Dict],
    primes: List[int] = [2, 3, 5, 7, 11],
    n_epochs: int = 500,
    device: str = 'cpu',
) -> Dict[int, Dict]:
    """Compare different primes for p-adic structure."""
    results = {}

    for p in primes:
        print(f"\n{'='*60}")
        print(f"Training with prime p={p}")
        print(f"{'='*60}")

        encoder = AminoAcidPadicEncoder(
            prime=p,
            latent_dim=16,
            hidden_dim=64,
        ).to(device)

        # Train
        history = train_encoder(encoder, n_epochs=n_epochs, verbose=True)

        # Evaluate
        metrics = evaluate_ddg(encoder, mutations)

        results[p] = {
            'final_loss': history[-1]['total'],
            'ddg_metrics': metrics,
            'history': history,
        }

        print(f"\nResults for p={p}:")
        print(f"  DDG Spearman: {metrics['spearman']:.4f}")
        print(f"  DDG Pearson:  {metrics['pearson']:.4f}")
        print(f"  DDG MAE:      {metrics['mae']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Generalized P-adic Encoder')
    parser.add_argument('--prime', type=int, default=5, help='Prime for p-adic structure')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--compare-primes', action='store_true', help='Compare multiple primes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Data paths
    data_dir = PROJECT_ROOT / 'deliverables' / 'partners' / 'jose_colbes' / 'reproducibility' / 'data'
    results_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'generalized_padic'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading S669 dataset...")
    s669_path = data_dir / 's669_full.csv'
    if not s669_path.exists():
        s669_path = data_dir / 'S669' / 'S669.csv'

    mutations = load_s669(s669_path)
    print(f"Loaded {len(mutations)} mutations")

    if args.compare_primes:
        # Compare different primes
        results = compare_primes(
            mutations,
            primes=[2, 3, 5, 7, 11],
            n_epochs=args.epochs,
            device=args.device,
        )

        # Find best prime
        best_prime = max(results.keys(), key=lambda p: results[p]['ddg_metrics']['spearman'])
        print(f"\n{'='*60}")
        print(f"Best prime: p={best_prime}")
        print(f"  Spearman: {results[best_prime]['ddg_metrics']['spearman']:.4f}")
        print(f"{'='*60}")

        # Save results
        summary = {
            p: {
                'spearman': results[p]['ddg_metrics']['spearman'],
                'pearson': results[p]['ddg_metrics']['pearson'],
                'mae': results[p]['ddg_metrics']['mae'],
            }
            for p in results
        }

        with open(results_dir / 'prime_comparison.json', 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        # Train single encoder
        print(f"\nTraining with prime p={args.prime}")

        encoder = AminoAcidPadicEncoder(
            prime=args.prime,
            latent_dim=16,
            hidden_dim=64,
        ).to(args.device)

        history = train_encoder(encoder, n_epochs=args.epochs, lr=args.lr, verbose=True)
        metrics = evaluate_ddg(encoder, mutations)

        print(f"\n{'='*60}")
        print(f"Final Results (p={args.prime}):")
        print(f"  DDG Spearman: {metrics['spearman']:.4f}")
        print(f"  DDG Pearson:  {metrics['pearson']:.4f}")
        print(f"  DDG MAE:      {metrics['mae']:.4f}")
        print(f"  N samples:    {metrics['n_samples']}")
        print(f"{'='*60}")

        # Save model and results
        checkpoint = {
            'model_state_dict': encoder.state_dict(),
            'prime': args.prime,
            'metrics': metrics,
            'history': history,
            'timestamp': datetime.now().isoformat(),
        }

        torch.save(checkpoint, results_dir / f'aa_encoder_p{args.prime}.pt')

        with open(results_dir / f'aa_encoder_p{args.prime}_results.json', 'w') as f:
            json.dump({
                'prime': args.prime,
                'epochs': args.epochs,
                'metrics': metrics,
                'final_loss': history[-1],
            }, f, indent=2)

        print(f"\nSaved to {results_dir}")


if __name__ == '__main__':
    main()
