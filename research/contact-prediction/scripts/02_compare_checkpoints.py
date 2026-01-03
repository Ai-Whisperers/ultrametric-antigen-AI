#!/usr/bin/env python3
"""
Compare contact prediction signal across different checkpoints.

Tests whether high-richness embeddings provide better contact discrimination
than ceiling-hierarchy (collapsed) embeddings.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY

# Same insulin data
INSULIN_B_CHAIN = {
    'sequence': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
    'codons': [
        'TTT', 'GTG', 'AAC', 'CAG', 'CAC', 'CTG', 'TGC', 'GGC', 'AGC', 'CAC',
        'CTG', 'GTG', 'GAG', 'GCC', 'CTG', 'TAC', 'CTG', 'GTG', 'TGC', 'GGC',
        'GAG', 'CGC', 'GGC', 'TTC', 'TTC', 'TAC', 'ACC', 'CCC', 'AAG', 'ACC',
    ],
    'ca_coords': np.array([
        [12.86, 3.91, 5.44], [10.41, 1.04, 5.87], [11.06, -2.54, 4.74],
        [8.15, -4.90, 4.73], [8.09, -6.97, 1.63], [5.30, -9.45, 1.13],
        [5.93, -11.16, -2.17], [3.77, -14.24, -2.23], [5.11, -17.64, -1.28],
        [3.62, -18.50, 1.97], [4.18, -15.02, 3.36], [2.57, -13.82, 6.53],
        [3.97, -10.33, 7.18], [1.66, -8.64, 9.76], [2.21, -4.92, 10.21],
        [-0.25, -3.05, 12.22], [-0.45, 0.76, 11.53], [-2.61, 2.40, 14.04],
        [-2.01, 6.14, 14.10], [-4.33, 8.63, 12.24], [-3.21, 11.88, 10.71],
        [-5.07, 13.41, 7.87], [-3.04, 16.59, 7.33], [-4.27, 18.21, 4.16],
        [-1.82, 21.00, 3.59], [-2.52, 22.65, 0.24], [0.67, 24.59, -0.68],
        [1.15, 26.10, -4.07], [4.59, 27.54, -4.82], [5.08, 29.17, -8.07],
    ])
}

CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def extract_embeddings_from_checkpoint(checkpoint_path):
    """Extract embeddings from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Infer dimensions
    encoder_weight = state_dict.get('encoder_A.net.0.weight')
    hidden_dim = encoder_weight.shape[0] if encoder_weight is not None else 64

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16, hidden_dim=hidden_dim, max_radius=0.99,
        curvature=1.0, use_controller=False, use_dual_projection=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Convert indices to ternary representation
    all_indices = torch.arange(TERNARY.N_OPERATIONS)
    all_ternary = TERNARY.to_ternary(all_indices)  # (N, 9) with values {-1, 0, 1}

    with torch.no_grad():
        out = model(all_ternary.float(), compute_control=False)

    return out['z_B_hyp']


def compute_contact_map(coords, threshold=8.0):
    """Compute binary contact map."""
    dist_matrix = squareform(pdist(coords))
    contact_map = (dist_matrix < threshold).astype(float)
    n = len(coords)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                contact_map[i, j] = 0
    return contact_map


def load_codon_mapping():
    """Load the actual codon mapping."""
    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'
    with open(map_path) as f:
        mapping = json.load(f)
    return mapping['codon_to_position']


def evaluate_checkpoint(z_hyp, codons, coords, name):
    """Evaluate contact prediction for a checkpoint."""
    # Use the actual codon mapping
    codon_to_idx = load_codon_mapping()

    n = len(codons)
    contact_map = compute_contact_map(coords)

    # Compute hyperbolic distances
    hyp_dists = []
    contacts = []

    for i in range(n):
        for j in range(i + 4, n):
            idx_i = codon_to_idx[codons[i]]
            idx_j = codon_to_idx[codons[j]]

            d = poincare_distance(
                z_hyp[idx_i:idx_i+1],
                z_hyp[idx_j:idx_j+1],
                c=1.0
            ).item()

            hyp_dists.append(d)
            contacts.append(contact_map[i, j])

    hyp_dists = np.array(hyp_dists)
    contacts = np.array(contacts)

    # Metrics
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(contacts, -hyp_dists)

    contact_dists = hyp_dists[contacts == 1]
    noncontact_dists = hyp_dists[contacts == 0]

    pooled_std = np.sqrt((contact_dists.var() + noncontact_dists.var()) / 2)
    cohens_d = (contact_dists.mean() - noncontact_dists.mean()) / pooled_std

    return {
        'name': name,
        'auc': auc,
        'cohens_d': cohens_d,
        'mean_contact': contact_dists.mean(),
        'mean_noncontact': noncontact_dists.mean(),
    }


def main():
    print("=" * 70)
    print("CHECKPOINT COMPARISON: Contact Prediction Signal")
    print("=" * 70)
    print()

    checkpoints = [
        ('final_rich_lr5e5_best.pt', 'High Richness (0.00858)'),
        ('homeostatic_rich_best.pt', 'Balanced (0.00662)'),
        ('v5_11_structural_best.pt', 'Ceiling Hierarchy'),
    ]

    ckpt_dir = Path(__file__).parent.parent / 'checkpoints'
    protein = INSULIN_B_CHAIN

    results = []

    for ckpt_file, description in checkpoints:
        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"  Skipping {ckpt_file} (not found)")
            continue

        print(f"Testing: {description}")
        print(f"  Loading {ckpt_file}...")

        try:
            z_hyp = extract_embeddings_from_checkpoint(ckpt_path)
            result = evaluate_checkpoint(
                z_hyp, protein['codons'], protein['ca_coords'], description
            )
            results.append(result)
            print(f"  AUC = {result['auc']:.4f}, Cohen's d = {result['cohens_d']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Checkpoint':<30} {'AUC':>8} {'Cohen d':>10} {'Δ mean':>10}")
    print("-" * 60)

    for r in results:
        delta = r['mean_contact'] - r['mean_noncontact']
        print(f"{r['name']:<30} {r['auc']:>8.4f} {r['cohens_d']:>+10.4f} {delta:>+10.4f}")

    # Best checkpoint
    if results:
        best = max(results, key=lambda x: x['auc'])
        print()
        print(f"BEST: {best['name']} (AUC = {best['auc']:.4f})")

        if best['auc'] > 0.6:
            print()
            print(">>> Strong signal! High-richness embeddings should be used.")
        elif best['auc'] > 0.55:
            print()
            print(">>> Moderate signal. Worth pursuing with more data.")

    return results


if __name__ == '__main__':
    results = main()
