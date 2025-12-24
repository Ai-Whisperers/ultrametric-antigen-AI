#!/usr/bin/env python3
"""
Tau Phosphorylation Sweep Analysis

Applies the p-adic geometric framework to analyze how phosphorylation
at each of tau's ~50 characterized sites affects protein geometry.

Hypothesis: Phosphorylation-induced geometric shifts correlate with:
  - Microtubule binding disruption
  - Aggregation propensity
  - Pathological stage (early vs late AD)

Uses the 3-adic codon encoder to map sequences to hyperbolic space
and compute geometric perturbations from S/T/Y → D (phosphomimic).
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add paths
SCRIPT_DIR = Path(__file__).parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "data"))
sys.path.insert(0, str(RESEARCH_DIR / "bioinformatics" / "rheumatoid_arthritis" / "scripts"))

from tau_phospho_database import (
    TAU_2N4R_SEQUENCE,
    TAU_PHOSPHO_SITES,
    TAU_EPITOPES,
    TAU_DOMAINS,
    TAU_TUBULIN_CONTACTS,
    KXGS_MOTIFS
)

from hyperbolic_utils import (
    load_hyperbolic_encoder,
    encode_codon_hyperbolic,
    hyperbolic_centroid,
    poincare_distance,
    AA_TO_CODON
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dysfunction zone thresholds (different from Goldilocks for neurodegeneration)
ZONE_THRESHOLDS = {
    'tolerated': 0.15,       # <15%: Normal phosphorylation, tolerated
    'transition': 0.35,      # 15-35%: Transition zone, early dysfunction
    # >35%: Severe dysfunction, aggregation-prone
}

# Context window size (residues on each side)
CONTEXT_WINDOW = 7  # 15-mer total


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_sequence(sequence: str, encoder) -> np.ndarray:
    """Encode amino acid sequence to hyperbolic embeddings."""
    embeddings = []
    for aa in sequence:
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            embedding = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


def extract_context(sequence: str, position: int, window: int = CONTEXT_WINDOW) -> Tuple[str, int]:
    """
    Extract sequence context around a position.

    Returns:
        Tuple of (context_string, position_in_context)
    """
    # Convert to 0-indexed
    pos_idx = position - 1

    start = max(0, pos_idx - window)
    end = min(len(sequence), pos_idx + window + 1)

    context = sequence[start:end]
    pos_in_context = pos_idx - start

    return context, pos_in_context


def apply_phosphomimic(sequence: str, position: int) -> str:
    """
    Apply phosphomimetic mutation (S/T/Y → D) at specified position.
    Position is 1-indexed.
    """
    seq_list = list(sequence)
    pos_idx = position - 1

    if seq_list[pos_idx] in ['S', 'T', 'Y']:
        seq_list[pos_idx] = 'D'  # Aspartate mimics phosphate charge

    return ''.join(seq_list)


def classify_zone(shift: float) -> str:
    """Classify geometric shift into dysfunction zones."""
    if shift < ZONE_THRESHOLDS['tolerated']:
        return 'tolerated'
    elif shift < ZONE_THRESHOLDS['transition']:
        return 'transition'
    else:
        return 'severe'


def compute_dysfunction_score(shift: float, zone: str) -> float:
    """
    Compute dysfunction score (0-1) based on geometric shift.
    Higher score = more likely to cause dysfunction.
    """
    if zone == 'tolerated':
        return shift / ZONE_THRESHOLDS['tolerated'] * 0.3
    elif zone == 'transition':
        range_size = ZONE_THRESHOLDS['transition'] - ZONE_THRESHOLDS['tolerated']
        normalized = (shift - ZONE_THRESHOLDS['tolerated']) / range_size
        return 0.3 + normalized * 0.4
    else:
        # Severe zone
        excess = shift - ZONE_THRESHOLDS['transition']
        return min(1.0, 0.7 + excess * 2)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_single_site(
    position: int,
    site_data: Dict,
    sequence: str,
    encoder
) -> Dict:
    """Analyze phosphorylation at a single site."""

    # Extract context
    context_wt, pos_in_ctx = extract_context(sequence, position)

    # Apply phosphomimic
    context_mut = list(context_wt)
    context_mut[pos_in_ctx] = 'D'
    context_mut = ''.join(context_mut)

    # Encode both
    emb_wt = encode_sequence(context_wt, encoder)
    emb_mut = encode_sequence(context_mut, encoder)

    if len(emb_wt) == 0 or len(emb_mut) == 0:
        return None

    # Compute centroids and distance
    centroid_wt = hyperbolic_centroid(emb_wt)
    centroid_mut = hyperbolic_centroid(emb_mut)
    shift = float(poincare_distance(centroid_wt, centroid_mut))

    # Classify
    zone = classify_zone(shift)
    dysfunction_score = compute_dysfunction_score(shift, zone)

    return {
        'position': position,
        'aa': site_data['aa'],
        'domain': site_data['domain'],
        'epitope': site_data.get('epitope'),
        'stage': site_data['stage'],
        'kinases': site_data.get('kinases', []),
        'context_wt': context_wt,
        'context_mut': context_mut,
        'centroid_shift': shift,
        'zone': zone,
        'dysfunction_score': dysfunction_score,
        'is_mtbr': site_data['domain'] in ['R1', 'R2', 'R3', 'R4', 'MTBR'],
        'is_kxgs': position in [262, 293, 324, 356],  # KXGS motif serines
        'near_tubulin_contact': position in TAU_TUBULIN_CONTACTS or
                                any(abs(position - tc) <= 3 for tc in TAU_TUBULIN_CONTACTS)
    }


def analyze_epitope_combination(
    epitope_name: str,
    epitope_data: Dict,
    sequence: str,
    encoder
) -> Dict:
    """Analyze combined phosphorylation at an epitope."""

    sites = epitope_data['sites']
    if not sites:
        return None

    # Get context around first site (or middle if multiple)
    center_pos = sites[len(sites) // 2]

    # Create multi-phosphorylated context
    context_wt, _ = extract_context(sequence, center_pos, window=10)

    # Apply all phosphomimics
    context_mut = list(context_wt)
    start_pos = center_pos - 10

    for site in sites:
        idx_in_context = site - start_pos - 1
        if 0 <= idx_in_context < len(context_mut):
            context_mut[idx_in_context] = 'D'

    context_mut = ''.join(context_mut)

    # Encode
    emb_wt = encode_sequence(context_wt, encoder)
    emb_mut = encode_sequence(context_mut, encoder)

    if len(emb_wt) == 0 or len(emb_mut) == 0:
        return None

    centroid_wt = hyperbolic_centroid(emb_wt)
    centroid_mut = hyperbolic_centroid(emb_mut)
    shift = float(poincare_distance(centroid_wt, centroid_mut))

    zone = classify_zone(shift)

    return {
        'epitope': epitope_name,
        'sites': sites,
        'description': epitope_data['description'],
        'stage': epitope_data['stage'],
        'context_wt': context_wt,
        'context_mut': context_mut,
        'combined_shift': shift,
        'zone': zone,
        'num_phospho': len(sites)
    }


def main():
    print("=" * 70)
    print("TAU PHOSPHORYLATION SWEEP ANALYSIS")
    print("P-adic Geometric Framework for Alzheimer's Disease")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")

    print(f"\nTau sequence length: {len(TAU_2N4R_SEQUENCE)} amino acids")
    print(f"Phospho-sites to analyze: {len(TAU_PHOSPHO_SITES)}")

    results = {
        'metadata': {
            'protein': 'Tau (MAPT)',
            'isoform': '2N4R (441 aa)',
            'encoder': '3-adic (V5.11.3)',
            'analysis': 'Single-site phosphorylation sweep',
            'zone_thresholds': ZONE_THRESHOLDS
        },
        'single_site_results': [],
        'epitope_results': [],
        'summary': {}
    }

    # ========================================================================
    # Single-site analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Single-Site Phosphorylation Analysis")
    print("-" * 70)

    site_results = []

    for position, site_data in sorted(TAU_PHOSPHO_SITES.items()):
        result = analyze_single_site(position, site_data, TAU_2N4R_SEQUENCE, encoder)
        if result:
            site_results.append(result)

            # Print progress
            zone_marker = {'tolerated': '.', 'transition': '*', 'severe': '!'}[result['zone']]
            print(f"  {result['aa']}{position:3d} ({result['domain']:12s}): "
                  f"shift={result['centroid_shift']*100:5.1f}% [{result['zone']:10s}] {zone_marker}")

    results['single_site_results'] = site_results

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Summary Statistics")
    print("-" * 70)

    # By zone
    zone_counts = defaultdict(int)
    for r in site_results:
        zone_counts[r['zone']] += 1

    print("\nSites by dysfunction zone:")
    for zone in ['tolerated', 'transition', 'severe']:
        count = zone_counts[zone]
        pct = count / len(site_results) * 100
        print(f"  {zone:12s}: {count:3d} ({pct:5.1f}%)")

    # By domain
    print("\nMean shift by domain:")
    domain_shifts = defaultdict(list)
    for r in site_results:
        domain_shifts[r['domain']].append(r['centroid_shift'])

    for domain, shifts in sorted(domain_shifts.items(), key=lambda x: -np.mean(x[1])):
        mean_shift = np.mean(shifts) * 100
        print(f"  {domain:12s}: {mean_shift:5.1f}% (n={len(shifts)})")

    # By pathological stage
    print("\nMean shift by pathological stage:")
    stage_shifts = defaultdict(list)
    for r in site_results:
        stage_shifts[r['stage']].append(r['centroid_shift'])

    for stage, shifts in sorted(stage_shifts.items()):
        mean_shift = np.mean(shifts) * 100
        print(f"  {stage:12s}: {mean_shift:5.1f}% (n={len(shifts)})")

    # ========================================================================
    # Top dysfunction sites
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Top Dysfunction Sites (by geometric shift)")
    print("-" * 70)

    sorted_sites = sorted(site_results, key=lambda x: x['centroid_shift'], reverse=True)

    print("\n--- Top 15 Highest Geometric Perturbation ---")
    for i, r in enumerate(sorted_sites[:15]):
        epitope_str = f"[{r['epitope']}]" if r['epitope'] else ""
        mtbr_str = "[MTBR]" if r['is_mtbr'] else ""
        kxgs_str = "[KXGS]" if r['is_kxgs'] else ""

        print(f"  {i+1:2d}. {r['aa']}{r['position']:3d} ({r['domain']:8s}): "
              f"shift={r['centroid_shift']*100:5.1f}% [{r['zone']}] "
              f"{epitope_str} {mtbr_str} {kxgs_str}")

    # ========================================================================
    # MTBR-specific analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. MTBR (Microtubule Binding Region) Analysis")
    print("-" * 70)

    mtbr_sites = [r for r in site_results if r['is_mtbr']]
    print(f"\nMTBR phospho-sites: {len(mtbr_sites)}")

    if mtbr_sites:
        mtbr_sorted = sorted(mtbr_sites, key=lambda x: x['centroid_shift'], reverse=True)
        print("\nMTBR sites ranked by dysfunction potential:")
        for r in mtbr_sorted:
            kxgs_str = "[KXGS]" if r['is_kxgs'] else ""
            tubulin_str = "[TUB]" if r['near_tubulin_contact'] else ""
            print(f"  {r['aa']}{r['position']:3d} ({r['domain']:3s}): "
                  f"shift={r['centroid_shift']*100:5.1f}% [{r['zone']}] "
                  f"{kxgs_str} {tubulin_str}")

    # ========================================================================
    # Epitope combination analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. Pathological Epitope Analysis (Combined Sites)")
    print("-" * 70)

    epitope_results = []

    for epitope_name, epitope_data in TAU_EPITOPES.items():
        result = analyze_epitope_combination(
            epitope_name, epitope_data, TAU_2N4R_SEQUENCE, encoder
        )
        if result:
            epitope_results.append(result)
            print(f"\n  {epitope_name}:")
            print(f"    Sites: {result['sites']}")
            print(f"    Combined shift: {result['combined_shift']*100:.1f}%")
            print(f"    Zone: {result['zone']}")
            print(f"    Stage: {result['stage']}")

    results['epitope_results'] = epitope_results

    # ========================================================================
    # Synergy analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("6. Synergy Analysis (Additive vs Combinatorial)")
    print("-" * 70)

    print("\nComparing individual vs combined phosphorylation:")

    for epi in epitope_results:
        if epi['num_phospho'] >= 2:
            # Get individual shifts
            individual_shifts = []
            for site in epi['sites']:
                for r in site_results:
                    if r['position'] == site:
                        individual_shifts.append(r['centroid_shift'])
                        break

            if len(individual_shifts) == len(epi['sites']):
                additive_expected = sum(individual_shifts)
                actual = epi['combined_shift']

                if additive_expected > 0:
                    synergy_ratio = actual / additive_expected
                    synergy_type = "SYNERGISTIC" if synergy_ratio > 1.2 else \
                                   "ANTAGONISTIC" if synergy_ratio < 0.8 else "ADDITIVE"

                    print(f"\n  {epi['epitope']}:")
                    print(f"    Individual shifts: {[f'{s*100:.1f}%' for s in individual_shifts]}")
                    print(f"    Expected (additive): {additive_expected*100:.1f}%")
                    print(f"    Actual (combined): {actual*100:.1f}%")
                    print(f"    Synergy ratio: {synergy_ratio:.2f} [{synergy_type}]")

    # ========================================================================
    # Therapeutic targets
    # ========================================================================
    print("\n" + "-" * 70)
    print("7. Therapeutic Target Prioritization")
    print("-" * 70)

    # High-value targets: transition/severe zone + MTBR or early stage
    priority_targets = []
    for r in site_results:
        if r['zone'] in ['transition', 'severe']:
            priority = 0
            reasons = []

            if r['is_mtbr']:
                priority += 3
                reasons.append("MTBR location")
            if r['is_kxgs']:
                priority += 2
                reasons.append("KXGS motif")
            if r['stage'] == 'early':
                priority += 2
                reasons.append("Early pathology marker")
            if r['epitope']:
                priority += 1
                reasons.append(f"Epitope: {r['epitope']}")

            if priority > 0:
                priority_targets.append((r, priority, reasons))

    priority_targets.sort(key=lambda x: (-x[1], -x[0]['centroid_shift']))

    print("\nPriority dephosphorylation targets (for therapeutic intervention):")
    for r, priority, reasons in priority_targets[:10]:
        print(f"\n  {r['aa']}{r['position']} (priority score: {priority})")
        print(f"    Shift: {r['centroid_shift']*100:.1f}% [{r['zone']}]")
        print(f"    Reasons: {', '.join(reasons)}")
        print(f"    Target kinases: {', '.join(r['kinases'])}")

    # ========================================================================
    # Save results
    # ========================================================================
    results['summary'] = {
        'total_sites': len(site_results),
        'zone_distribution': dict(zone_counts),
        'mean_shift': float(np.mean([r['centroid_shift'] for r in site_results])),
        'top_sites': [r['position'] for r in sorted_sites[:10]],
        'mtbr_sites_in_transition_or_severe': len([r for r in mtbr_sites
                                                    if r['zone'] in ['transition', 'severe']])
    }

    output_path = SCRIPT_DIR / "results" / "tau_phospho_sweep_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # ========================================================================
    # Key findings
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    severe_sites = [r for r in site_results if r['zone'] == 'severe']
    transition_sites = [r for r in site_results if r['zone'] == 'transition']

    print(f"""
1. DYSFUNCTION DISTRIBUTION
   - Severe zone (>35% shift): {len(severe_sites)} sites
   - Transition zone (15-35%): {len(transition_sites)} sites
   - Tolerated (<15%): {zone_counts['tolerated']} sites

2. HIGHEST RISK SITES (for aggregation/dysfunction)
   Top 3: {', '.join([f"{r['aa']}{r['position']}" for r in sorted_sites[:3]])}

3. MTBR VULNERABILITY
   {len([r for r in mtbr_sites if r['zone'] != 'tolerated'])} of {len(mtbr_sites)} MTBR sites
   cause significant geometric perturbation

4. EARLY INTERVENTION TARGETS
   Sites with early-stage pathology AND high geometric shift:
   {', '.join([f"{r['aa']}{r['position']}" for r in sorted_sites
              if r['stage'] == 'early' and r['zone'] != 'tolerated'][:5])}

5. KINASE TARGETS FOR DRUG DEVELOPMENT
   Based on high-shift sites, prioritize inhibitors for:
   {', '.join(set([k for r in sorted_sites[:10] for k in r['kinases']]))}
""")


if __name__ == "__main__":
    main()
