#!/usr/bin/env python3
"""
RA Handshake Interface Analysis

Analyzes PTM sites in context of immunological handshake interfaces:
1. HLA-DRB1 Peptide Groove - citrullinated peptide presentation
2. TCR-pMHC Interface - T-cell recognition of presented peptides
3. PAD Enzyme-Substrate - arginine deimination sites

Computes geometric convergence between:
- Modified epitope embeddings and HLA binding pockets
- WT vs modified TCR contact residues
- PAD recognition motifs

Part of Phase 1: RA Extensions (PRIORITY)
See: research/genetic_code/PTM_EXTENSION_PLAN.md

Input: research/bioinformatics/rheumatoid_arthritis/data/ra_ptm_sweep_results.json
Output: research/bioinformatics/rheumatoid_arthritis/data/ra_handshake_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch

# Add path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from hyperbolic_utils import (
    load_hyperbolic_encoder,
    encode_codon_hyperbolic,
    hyperbolic_centroid,
    poincare_distance,
    AA_TO_CODON,
)

# =============================================================================
# HLA-DRB1 BINDING POCKET DEFINITIONS
# =============================================================================

# P1, P4, P6, P7, P9 pockets are key for peptide binding
# Shared epitope (SE) is at positions 70-74 (QKRAA, QRRAA, or RRRAA)
HLA_POCKETS = {
    'P1': {'positions': [85, 86], 'function': 'anchor', 'preference': 'hydrophobic'},
    'P4': {'positions': [13, 70, 71, 74], 'function': 'shared_epitope_contact', 'preference': 'charged'},
    'P6': {'positions': [11, 13], 'function': 'anchor', 'preference': 'small'},
    'P7': {'positions': [28, 47], 'function': 'TCR_contact', 'preference': 'exposed'},
    'P9': {'positions': [9, 37, 57], 'function': 'anchor', 'preference': 'basic/neutral'},
}

# Risk-associated HLA alleles with known RA associations
HLA_RISK_ALLELES = {
    'DRB1*04:01': {'odds_ratio': 4.44, 'shared_epitope': 'QKRAA', 'risk': 'high'},
    'DRB1*04:04': {'odds_ratio': 3.68, 'shared_epitope': 'QRRAA', 'risk': 'high'},
    'DRB1*04:05': {'odds_ratio': 3.20, 'shared_epitope': 'QRRAA', 'risk': 'high'},
    'DRB1*01:01': {'odds_ratio': 2.50, 'shared_epitope': 'QRRAA', 'risk': 'moderate'},
    'DRB1*10:01': {'odds_ratio': 2.30, 'shared_epitope': 'RRRAA', 'risk': 'moderate'},
    'DRB1*04:02': {'odds_ratio': 0.46, 'shared_epitope': None, 'risk': 'protective'},
    'DRB1*13:02': {'odds_ratio': 0.34, 'shared_epitope': None, 'risk': 'protective'},
}

# Shared epitope sequences for embedding comparison
SHARED_EPITOPE_SEQS = {
    'QKRAA': 'QKRAA',  # DR4 (04:01)
    'QRRAA': 'QRRAA',  # DR4 (04:04, 04:05), DR1
    'RRRAA': 'RRRAA',  # DR10
    'DERAA': 'DERAA',  # DR4:02 - protective
}

# PAD4 recognition motif patterns
PAD4_MOTIFS = {
    'canonical': {'pattern': 'G-R-G', 'description': 'Glycine-flanked arginine'},
    'extended': {'pattern': 'X-R-S', 'description': 'Serine C-terminal'},
    'histone': {'pattern': 'R-K-X', 'description': 'Adjacent to lysine'},
}


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def encode_context(encoder, context: str) -> np.ndarray:
    """Encode amino acid context to hyperbolic embeddings."""
    embeddings = []
    for aa in context.upper():
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            emb = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(emb)
        elif aa == '-' or aa == 'X':
            # Padding - use glycine as neutral
            emb = encode_codon_hyperbolic('GGC', encoder)
            embeddings.append(emb)
    return np.array(embeddings) if embeddings else np.array([])


def compute_context_centroid(encoder, context: str) -> np.ndarray:
    """Compute hyperbolic centroid of a context."""
    embeddings = encode_context(encoder, context)
    if len(embeddings) == 0:
        return np.zeros(16)
    return hyperbolic_centroid(embeddings)


def compute_convergence(encoder, seq1: str, seq2: str) -> dict:
    """Compute geometric convergence between two sequences."""
    emb1 = encode_context(encoder, seq1)
    emb2 = encode_context(encoder, seq2)

    if len(emb1) == 0 or len(emb2) == 0:
        return {'distance': 1.0, 'convergent': False}

    cent1 = hyperbolic_centroid(emb1)
    cent2 = hyperbolic_centroid(emb2)

    cent1_t = torch.from_numpy(cent1).float().unsqueeze(0)
    cent2_t = torch.from_numpy(cent2).float().unsqueeze(0)

    dist = poincare_distance(cent1_t, cent2_t).item()

    # Normalize by diameter (~2)
    normalized_dist = min(dist / 2.0, 1.0)

    return {
        'distance': normalized_dist,
        'convergent': normalized_dist < 0.20,  # Convergent if < 20% of max
        'centroid_1': cent1.tolist(),
        'centroid_2': cent2.tolist(),
    }


# =============================================================================
# HLA-PEPTIDE INTERFACE ANALYSIS
# =============================================================================

def analyze_hla_peptide_interface(encoder, ptm_sample: dict) -> dict:
    """
    Analyze PTM site compatibility with HLA binding groove.

    Key insight: Citrullination (R->Q) changes the charge at P4 pocket,
    potentially creating neoepitopes recognized by autoreactive T-cells.
    """
    wt_context = ptm_sample['wt_context']
    mod_context = ptm_sample['mod_context']

    # Encode wild-type and modified contexts
    wt_centroid = compute_context_centroid(encoder, wt_context)
    mod_centroid = compute_context_centroid(encoder, mod_context)

    # Compare with shared epitope sequences
    se_matches = {}
    for se_name, se_seq in SHARED_EPITOPE_SEQS.items():
        se_centroid = compute_context_centroid(encoder, se_seq)

        # Distance from modified peptide to shared epitope
        mod_t = torch.from_numpy(mod_centroid).float().unsqueeze(0)
        se_t = torch.from_numpy(se_centroid).float().unsqueeze(0)
        dist_mod = poincare_distance(mod_t, se_t).item()

        # Distance from WT peptide to shared epitope
        wt_t = torch.from_numpy(wt_centroid).float().unsqueeze(0)
        dist_wt = poincare_distance(wt_t, se_t).item()

        se_matches[se_name] = {
            'wt_distance': round(dist_wt / 2.0, 6),
            'mod_distance': round(dist_mod / 2.0, 6),
            'shift_toward_se': round((dist_wt - dist_mod) / 2.0, 6),
            'converges': dist_mod < dist_wt,  # PTM moves toward SE
        }

    # Find best SE match for modified peptide
    best_se = min(se_matches.items(), key=lambda x: x[1]['mod_distance'])

    return {
        'shared_epitope_analysis': se_matches,
        'best_se_match': best_se[0],
        'best_se_distance': best_se[1]['mod_distance'],
        'converges_to_risk_se': best_se[0] in ['QKRAA', 'QRRAA', 'RRRAA'],
        'moves_toward_se': best_se[1]['shift_toward_se'] > 0,
    }


# =============================================================================
# PAD4 SUBSTRATE RECOGNITION ANALYSIS
# =============================================================================

def analyze_pad_substrate(ptm_sample: dict) -> dict:
    """
    Analyze PAD4 recognition motif compatibility.

    PAD4 preferentially citrullinates:
    - Arginines flanked by glycine (G-R-G)
    - Arginines in unstructured regions
    - Histones (especially H2B R29, H4 R3)
    """
    if ptm_sample['ptm_type'] != 'R->Q':
        return {'applicable': False, 'reason': 'Not citrullination'}

    context = ptm_sample['wt_context']
    position = len(context) // 2  # Center position

    # Check flanking residues
    left_flank = context[position-1] if position > 0 else '-'
    right_flank = context[position+1] if position < len(context)-1 else '-'

    motif_matches = {
        'glycine_flanked': left_flank == 'G' or right_flank == 'G',
        'serine_adjacent': right_flank == 'S',
        'lysine_adjacent': 'K' in context[max(0,position-2):position+3],
        'proline_adjacent': 'P' in context[max(0,position-2):position+3],
    }

    # Score based on known PAD4 preferences
    pad_score = 0.0
    if motif_matches['glycine_flanked']:
        pad_score += 0.3
    if motif_matches['serine_adjacent']:
        pad_score += 0.2
    if motif_matches['lysine_adjacent']:
        pad_score += 0.2
    if not motif_matches['proline_adjacent']:  # Proline inhibits PAD
        pad_score += 0.1

    return {
        'applicable': True,
        'motif_matches': motif_matches,
        'pad_accessibility_score': round(pad_score, 3),
        'predicted_substrate': pad_score >= 0.4,
    }


# =============================================================================
# TCR-pMHC INTERFACE ANALYSIS
# =============================================================================

def analyze_tcr_interface(encoder, ptm_sample: dict) -> dict:
    """
    Analyze TCR contact potential of modified peptides.

    Key positions for TCR contact are P5-P8 (solvent exposed).
    Modifications at these positions have highest immunogenic potential.
    """
    wt_context = ptm_sample['wt_context']
    mod_context = ptm_sample['mod_context']

    # TCR typically contacts central 3-4 residues
    center = len(wt_context) // 2
    tcr_window = 2  # ±2 residues from modification

    wt_tcr_region = wt_context[max(0, center-tcr_window):center+tcr_window+1]
    mod_tcr_region = mod_context[max(0, center-tcr_window):center+tcr_window+1]

    # Compute change in TCR contact surface
    convergence = compute_convergence(encoder, wt_tcr_region, mod_tcr_region)

    # Estimate immunogenic potential based on geometric shift
    shift = convergence['distance']

    if shift < 0.10:
        immunogenic_class = 'LOW'
        description = 'Minimal change, likely tolerized'
    elif shift < 0.20:
        immunogenic_class = 'MODERATE'
        description = 'Detectable change, may break tolerance'
    elif shift < 0.30:
        immunogenic_class = 'HIGH'
        description = 'Goldilocks zone - strong neoepitope potential'
    else:
        immunogenic_class = 'VERY_HIGH'
        description = 'Large shift - may be too foreign'

    return {
        'tcr_contact_region_wt': wt_tcr_region,
        'tcr_contact_region_mod': mod_tcr_region,
        'tcr_interface_shift': round(shift, 6),
        'immunogenic_class': immunogenic_class,
        'description': description,
        'goldilocks_zone': 0.15 <= shift <= 0.30,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_handshake_analysis(sweep_results: dict, encoder) -> dict:
    """Run comprehensive handshake analysis on PTM sweep results."""

    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'encoder': '3-adic (V5.11.3)',
            'analysis_type': 'RA Handshake Interface Mapping',
            'interfaces': ['HLA-peptide', 'TCR-pMHC', 'PAD-substrate'],
        },
        'samples': [],
        'summary': {
            'total_analyzed': 0,
            'by_interface': {
                'hla_converges_to_se': 0,
                'pad_predicted_substrate': 0,
                'tcr_goldilocks': 0,
            },
            'by_ptm_type': defaultdict(lambda: {
                'total': 0, 'hla_converges': 0, 'pad_substrate': 0, 'tcr_goldilocks': 0
            }),
            'high_priority_targets': [],
        }
    }

    total = len(sweep_results['samples'])
    print(f"\nAnalyzing {total} PTM samples...")

    for i, sample in enumerate(sweep_results['samples']):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total}...")

        # Run interface analyses
        hla_analysis = analyze_hla_peptide_interface(encoder, sample)
        tcr_analysis = analyze_tcr_interface(encoder, sample)
        pad_analysis = analyze_pad_substrate(sample)

        # Combine results
        analyzed_sample = {
            'protein': sample['protein'],
            'position': sample['position'],
            'ptm_type': sample['ptm_type'],
            'wt_context': sample['wt_context'],
            'mod_context': sample['mod_context'],
            'centroid_shift': sample['centroid_shift'],
            'goldilocks_zone': sample['goldilocks_zone'],
            'is_known_acpa': sample.get('is_known_acpa', False),
            'hla_interface': hla_analysis,
            'tcr_interface': tcr_analysis,
            'pad_substrate': pad_analysis,
        }

        results['samples'].append(analyzed_sample)
        results['summary']['total_analyzed'] += 1

        # Update statistics
        ptm = sample['ptm_type']
        results['summary']['by_ptm_type'][ptm]['total'] += 1

        if hla_analysis['converges_to_risk_se']:
            results['summary']['by_interface']['hla_converges_to_se'] += 1
            results['summary']['by_ptm_type'][ptm]['hla_converges'] += 1

        if pad_analysis.get('predicted_substrate', False):
            results['summary']['by_interface']['pad_predicted_substrate'] += 1
            results['summary']['by_ptm_type'][ptm]['pad_substrate'] += 1

        if tcr_analysis['goldilocks_zone']:
            results['summary']['by_interface']['tcr_goldilocks'] += 1
            results['summary']['by_ptm_type'][ptm]['tcr_goldilocks'] += 1

        # Identify high-priority targets (multiple interface hits)
        priority_score = sum([
            hla_analysis['converges_to_risk_se'],
            pad_analysis.get('predicted_substrate', False),
            tcr_analysis['goldilocks_zone'],
        ])

        if priority_score >= 2:
            results['summary']['high_priority_targets'].append({
                'protein': sample['protein'],
                'position': sample['position'],
                'ptm_type': sample['ptm_type'],
                'priority_score': priority_score,
                'hla_converges': hla_analysis['converges_to_risk_se'],
                'pad_substrate': pad_analysis.get('predicted_substrate', False),
                'tcr_goldilocks': tcr_analysis['goldilocks_zone'],
                'is_known_acpa': sample.get('is_known_acpa', False),
            })

    # Convert defaultdict
    results['summary']['by_ptm_type'] = dict(results['summary']['by_ptm_type'])

    # Sort high-priority targets
    results['summary']['high_priority_targets'].sort(
        key=lambda x: (x['priority_score'], x['is_known_acpa']),
        reverse=True
    )

    return results


def main():
    print("=" * 70)
    print("RA HANDSHAKE INTERFACE ANALYSIS")
    print("Phase 1: RA Extensions (PRIORITY)")
    print("=" * 70)

    # Load PTM sweep results
    data_dir = SCRIPT_DIR.parent / 'data'
    sweep_path = data_dir / 'ra_ptm_sweep_results.json'

    if not sweep_path.exists():
        print(f"ERROR: PTM sweep results not found. Run 19_comprehensive_ra_ptm_sweep.py first.")
        return 1

    print(f"\nLoading PTM sweep results from: {sweep_path}")
    with open(sweep_path) as f:
        sweep_results = json.load(f)

    print(f"  Total samples: {sweep_results['statistics']['total_samples']}")

    # Load encoder
    print("\nLoading 3-adic hyperbolic encoder...")
    try:
        encoder, mapping = load_hyperbolic_encoder(device='cpu', version='3adic')
        print("  Encoder loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load encoder: {e}")
        return 1

    # Run analysis
    print("\nRunning handshake interface analysis...")
    results = run_handshake_analysis(sweep_results, encoder)

    # Summary
    print("\n" + "=" * 70)
    print("HANDSHAKE ANALYSIS SUMMARY")
    print("=" * 70)

    summary = results['summary']
    total = summary['total_analyzed']

    print(f"\n  Total samples analyzed: {total}")

    print(f"\n  Interface Hits:")
    for interface, count in summary['by_interface'].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"    {interface}: {count} ({pct:.1f}%)")

    print(f"\n  By PTM Type:")
    for ptm, stats in summary['by_ptm_type'].items():
        tcr_rate = stats['tcr_goldilocks'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {ptm}: {stats['total']} total, {stats['tcr_goldilocks']} TCR goldilocks ({tcr_rate:.1f}%)")

    print(f"\n  High-Priority Targets (2+ interface hits): {len(summary['high_priority_targets'])}")
    if summary['high_priority_targets'][:5]:
        print("    Top 5:")
        for target in summary['high_priority_targets'][:5]:
            known = " [KNOWN ACPA]" if target['is_known_acpa'] else ""
            print(f"      {target['protein']} {target['ptm_type']} @ {target['position']} "
                  f"(score={target['priority_score']}){known}")

    # Save results
    output_path = data_dir / 'ra_handshake_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Save high-priority targets separately
    targets_path = data_dir / 'ra_high_priority_targets.json'
    with open(targets_path, 'w') as f:
        json.dump(summary['high_priority_targets'], f, indent=2)
    print(f"  Saved: {targets_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run 21_ra_alphafold_jobs.py to generate structural validation jobs")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
