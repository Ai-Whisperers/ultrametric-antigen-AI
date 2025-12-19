#!/usr/bin/env python3
"""
Deep Handshake Sweep: Comprehensive Interface Analysis

Expands the handshake analysis with:
1. All RBD-ACE2 contact residues with extended context windows
2. Expanded modification library (12 modification types)
3. Lower convergence thresholds to find more pairs
4. Full asymmetric perturbation scan across all positions
5. Cross-interface hotspot detection
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import product

# Add path to hyperbolic utils
RESEARCH_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(RESEARCH_DIR / "bioinformatics" / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (
    load_hyperbolic_encoder,
    encode_codon_hyperbolic,
    hyperbolic_centroid,
    poincare_distance,
    AA_TO_CODON
)


# ============================================================================
# EXPANDED MODIFICATION LIBRARY
# ============================================================================

MODIFICATIONS = {
    # Deglycosylation
    'N_to_Q': {'N': 'Q'},  # Asparagine deglycosylation mimic

    # Phosphorylation mimics
    'S_to_D': {'S': 'D'},  # Phosphoserine mimic
    'T_to_D': {'T': 'D'},  # Phosphothreonine mimic
    'Y_to_D': {'Y': 'D'},  # Phosphotyrosine mimic

    # Citrullination
    'R_to_Q': {'R': 'Q'},  # Citrulline (PAD enzyme product)

    # Acetylation
    'K_to_Q': {'K': 'Q'},  # Acetyl-lysine mimic

    # Methylation proxies
    'K_to_R': {'K': 'R'},  # Methyl-lysine (charge retention)
    'R_to_K': {'R': 'K'},  # Demethylation proxy

    # Oxidation
    'M_to_Q': {'M': 'Q'},  # Methionine sulfoxide proxy
    'C_to_S': {'C': 'S'},  # Cysteine oxidation
    'W_to_F': {'W': 'F'},  # Tryptophan oxidation proxy

    # Hydroxylation
    'P_to_A': {'P': 'A'},  # Hydroxyproline disruption
    'F_to_Y': {'F': 'Y'},  # Phenylalanine hydroxylation

    # Charge modifications
    'D_to_N': {'D': 'N'},  # Deamidation reverse
    'E_to_Q': {'E': 'Q'},  # Glutamate neutralization

    # Size modifications
    'G_to_A': {'G': 'A'},  # Glycine expansion
    'A_to_G': {'A': 'G'},  # Alanine contraction
    'V_to_I': {'V': 'I'},  # Valine expansion
    'L_to_V': {'L': 'V'},  # Leucine contraction
}


# ============================================================================
# COMPLETE SEQUENCES
# ============================================================================

# Full SARS-CoV-2 Spike RBD (319-541)
SPIKE_RBD = """RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK
CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS
NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ
PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"""

# Human ACE2 ectodomain (19-615)
ACE2_ECTODOMAIN = """STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQST
LAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNP
QECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYED
YGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISP
IGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSV
GLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGH
IQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLL
KQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYC
DPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNML
RLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSI
KVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPR
ISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPP
VS"""

# Extended contact maps based on crystal structures
RBD_CONTACTS_EXTENDED = {
    # Direct contacts (from PDB 6M0J)
    417: ('K', 'direct', ['D30']),
    446: ('G', 'direct', ['Q42']),
    449: ('Y', 'direct', ['D38', 'Q42']),
    453: ('Y', 'direct', ['H34']),
    455: ('L', 'direct', ['H34', 'D30']),
    456: ('F', 'direct', ['D30', 'K31']),
    475: ('A', 'direct', ['S19', 'Q24']),
    476: ('G', 'direct', ['S19', 'Q24']),
    477: ('S', 'direct', ['S19']),
    486: ('F', 'direct', ['M82', 'L79', 'Q24']),
    487: ('N', 'direct', ['Q24', 'Y83']),
    489: ('Y', 'direct', ['K31', 'Y83', 'F28']),
    493: ('Q', 'direct', ['K31', 'H34', 'E35']),
    496: ('G', 'direct', ['K353', 'D38']),
    498: ('Q', 'direct', ['D38', 'Y41', 'Q42', 'K353']),
    500: ('T', 'direct', ['Y41', 'D355']),
    501: ('N', 'direct', ['Y41', 'K353']),
    502: ('G', 'direct', ['K353', 'G354']),
    505: ('Y', 'direct', ['E37', 'R393']),
    # Proximal residues (within 6Å)
    403: ('R', 'proximal', ['K353']),
    408: ('R', 'proximal', ['D38']),
    439: ('N', 'proximal', ['Q42']),
    440: ('N', 'proximal', ['Q42']),
    444: ('K', 'proximal', ['D38']),
    445: ('V', 'proximal', ['Q42']),
    484: ('E', 'proximal', ['K31']),
    485: ('G', 'proximal', ['Q24']),
    490: ('F', 'proximal', ['K31']),
    492: ('L', 'proximal', ['K31']),
    494: ('S', 'proximal', ['E35']),
    495: ('Y', 'proximal', ['D38']),
    499: ('P', 'proximal', ['K353']),
    503: ('V', 'proximal', ['K353']),
    504: ('G', 'proximal', ['E37']),
}

ACE2_CONTACTS_EXTENDED = {
    # Direct contacts
    19: ('S', 'direct', ['A475', 'G476']),
    24: ('Q', 'direct', ['A475', 'G476', 'N487', 'F486']),
    27: ('T', 'direct', ['F456']),
    28: ('F', 'direct', ['Y489']),
    30: ('D', 'direct', ['K417', 'L455', 'F456']),
    31: ('K', 'direct', ['F456', 'Y489', 'Q493', 'L492']),
    34: ('H', 'direct', ['Y453', 'L455', 'Q493']),
    35: ('E', 'direct', ['Q493']),
    37: ('E', 'direct', ['Y505']),
    38: ('D', 'direct', ['Y449', 'G496', 'Q498']),
    41: ('Y', 'direct', ['Q498', 'T500', 'N501']),
    42: ('Q', 'direct', ['G446', 'Y449', 'Q498']),
    45: ('L', 'direct', ['Q498']),
    79: ('L', 'direct', ['F486']),
    82: ('M', 'direct', ['F486']),
    83: ('Y', 'direct', ['N487', 'Y489']),
    353: ('K', 'direct', ['G496', 'Q498', 'N501', 'G502']),
    354: ('G', 'direct', ['G502']),
    355: ('D', 'direct', ['T500']),
    357: ('R', 'proximal', ['T500']),
    393: ('R', 'direct', ['Y505']),
    # Proximal
    21: ('E', 'proximal', ['A475']),
    23: ('K', 'proximal', ['A475']),
    25: ('T', 'proximal', ['G476']),
    80: ('M', 'proximal', ['F486']),
    81: ('Y', 'proximal', ['F486']),
    329: ('N', 'proximal', ['T500']),
    330: ('N', 'proximal', ['T500']),
    351: ('L', 'proximal', ['N501']),
    352: ('N', 'proximal', ['N501']),
    356: ('F', 'proximal', ['T500']),
}


def clean_sequence(seq: str) -> str:
    """Remove whitespace and newlines from sequence."""
    return ''.join(seq.split())


def encode_sequence(sequence: str, encoder) -> np.ndarray:
    """Encode a sequence and return embeddings for each position."""
    embeddings = []
    for aa in sequence:
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            embedding = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


def extract_context(sequence: str, position: int, window: int = 7) -> str:
    """Extract sequence context around a position with larger window."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)
    return sequence[start:end]


def encode_all_interfaces(
    sequence: str,
    contacts: Dict,
    seq_offset: int,
    encoder,
    window: int = 7
) -> Dict[int, Dict]:
    """Encode all contact residues with extended metadata."""
    results = {}
    seq = clean_sequence(sequence)

    for pos, (aa, contact_type, partners) in contacts.items():
        seq_pos = pos - seq_offset
        if 0 <= seq_pos < len(seq):
            context = extract_context(seq, seq_pos, window)
            if len(context) >= 5:
                embeddings = encode_sequence(context, encoder)
                if len(embeddings) > 0:
                    centroid = hyperbolic_centroid(embeddings)
                    results[pos] = {
                        'aa': aa,
                        'context': context,
                        'centroid': centroid,
                        'contact_type': contact_type,
                        'partners': partners,
                        'seq_pos': seq_pos
                    }
    return results


def compute_all_distances(viral: Dict, host: Dict) -> List[Dict]:
    """Compute all pairwise distances with full metadata."""
    distances = []

    for v_pos, v_data in viral.items():
        for h_pos, h_data in host.items():
            dist = float(poincare_distance(v_data['centroid'], h_data['centroid']))
            distances.append({
                'viral_pos': v_pos,
                'viral_aa': v_data['aa'],
                'viral_context': v_data['context'],
                'viral_type': v_data['contact_type'],
                'host_pos': h_pos,
                'host_aa': h_data['aa'],
                'host_context': h_data['context'],
                'host_type': h_data['contact_type'],
                'distance': dist,
                'is_known_pair': h_pos in [int(p.replace('D', '').replace('K', '').replace('E', '').replace('Y', '').replace('Q', '').replace('H', '').replace('S', '').replace('T', '').replace('F', '').replace('M', '').replace('L', '').replace('R', '').replace('G', '').replace('N', '').replace('A', '').replace('V', '').replace('W', '').replace('P', '').replace('C', '').replace('I', ''))
                             for p in v_data['partners'] if any(c.isdigit() for c in p)]
            })

    return sorted(distances, key=lambda x: x['distance'])


def deep_asymmetric_scan(
    viral_data: Dict,
    host_data: Dict,
    encoder,
    modifications: Dict = MODIFICATIONS
) -> List[Dict]:
    """
    Comprehensive asymmetric perturbation scan across all positions
    and all modification types.
    """
    results = []

    v_ctx = viral_data['context']
    h_ctx = host_data['context']

    # Encode originals
    v_emb = encode_sequence(v_ctx, encoder)
    h_emb = encode_sequence(h_ctx, encoder)

    if len(v_emb) == 0 or len(h_emb) == 0:
        return []

    v_orig = hyperbolic_centroid(v_emb)
    h_orig = hyperbolic_centroid(h_emb)

    # Test each position in viral context
    for v_pos in range(len(v_ctx)):
        v_aa = v_ctx[v_pos]

        for mod_name, mod_map in modifications.items():
            if v_aa in mod_map:
                # Apply modification to viral
                v_new = list(v_ctx)
                v_new[v_pos] = mod_map[v_aa]
                v_new_emb = encode_sequence(''.join(v_new), encoder)

                if len(v_new_emb) > 0:
                    v_shift = float(poincare_distance(v_orig, hyperbolic_centroid(v_new_emb)))

                    # Check corresponding host position if exists
                    h_shift = 0.0
                    if v_pos < len(h_ctx):
                        h_aa = h_ctx[v_pos]
                        if h_aa in mod_map:
                            h_new = list(h_ctx)
                            h_new[v_pos] = mod_map[h_aa]
                            h_new_emb = encode_sequence(''.join(h_new), encoder)
                            if len(h_new_emb) > 0:
                                h_shift = float(poincare_distance(h_orig, hyperbolic_centroid(h_new_emb)))

                    asymmetry = v_shift - h_shift

                    # Classify therapeutic potential
                    if v_shift > 0.20 and h_shift < 0.05:
                        potential = 'EXCELLENT'
                    elif v_shift > 0.15 and h_shift < 0.10:
                        potential = 'HIGH'
                    elif v_shift > 0.10 and h_shift < 0.15:
                        potential = 'MEDIUM'
                    elif v_shift > 0.05:
                        potential = 'LOW'
                    else:
                        potential = 'MINIMAL'

                    results.append({
                        'viral_interface_pos': viral_data['aa'] + str(viral_data.get('pos', '')),
                        'host_interface_pos': host_data['aa'] + str(host_data.get('pos', '')),
                        'context_position': v_pos,
                        'viral_aa': v_aa,
                        'modification': mod_name,
                        'new_aa': mod_map[v_aa],
                        'viral_shift': v_shift,
                        'host_shift': h_shift,
                        'asymmetry': asymmetry,
                        'therapeutic_potential': potential,
                        'viral_context': v_ctx,
                        'host_context': h_ctx
                    })

    return results


def find_hotspots(asymmetric_results: List[Dict]) -> Dict:
    """Identify modification hotspots across all interfaces."""
    # Group by modification type
    by_modification = defaultdict(list)
    for r in asymmetric_results:
        by_modification[r['modification']].append(r)

    # Group by viral amino acid
    by_viral_aa = defaultdict(list)
    for r in asymmetric_results:
        by_viral_aa[r['viral_aa']].append(r)

    # Find top performers
    excellent = [r for r in asymmetric_results if r['therapeutic_potential'] == 'EXCELLENT']
    high = [r for r in asymmetric_results if r['therapeutic_potential'] == 'HIGH']

    return {
        'by_modification': {k: len(v) for k, v in by_modification.items()},
        'by_viral_aa': {k: len(v) for k, v in by_viral_aa.items()},
        'excellent_count': len(excellent),
        'high_count': len(high),
        'top_modifications': sorted(
            [(k, np.mean([r['asymmetry'] for r in v])) for k, v in by_modification.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
    }


def main():
    print("=" * 70)
    print("DEEP HANDSHAKE SWEEP")
    print("Comprehensive SARS-CoV-2 / Human Interface Analysis")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")
    print(f"Testing {len(MODIFICATIONS)} modification types")

    results = {
        'metadata': {
            'encoder': '3-adic (V5.11.3)',
            'analysis': 'Deep Handshake Sweep',
            'modifications_tested': list(MODIFICATIONS.keys()),
            'window_size': 7
        }
    }

    # ========================================================================
    # 1. Encode All Interfaces
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Encoding Extended Interface Maps")
    print("-" * 70)

    rbd_seq = clean_sequence(SPIKE_RBD)
    ace2_seq = clean_sequence(ACE2_ECTODOMAIN)

    print(f"RBD: {len(rbd_seq)} residues, {len(RBD_CONTACTS_EXTENDED)} contact sites")
    print(f"ACE2: {len(ace2_seq)} residues, {len(ACE2_CONTACTS_EXTENDED)} contact sites")

    rbd_interfaces = encode_all_interfaces(SPIKE_RBD, RBD_CONTACTS_EXTENDED, 319, encoder)
    ace2_interfaces = encode_all_interfaces(ACE2_ECTODOMAIN, ACE2_CONTACTS_EXTENDED, 19, encoder)

    print(f"\nEncoded {len(rbd_interfaces)} RBD interfaces")
    print(f"Encoded {len(ace2_interfaces)} ACE2 interfaces")

    # ========================================================================
    # 2. Compute All Distances
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Computing Pairwise Distances")
    print("-" * 70)

    all_distances = compute_all_distances(rbd_interfaces, ace2_interfaces)
    print(f"Computed {len(all_distances)} pairwise distances")

    # Multiple convergence thresholds
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    for thresh in thresholds:
        count = len([d for d in all_distances if d['distance'] < thresh])
        print(f"  Distance < {thresh}: {count} pairs")

    # Top convergences
    print("\n--- Top 15 Convergent Handshakes ---")
    for i, d in enumerate(all_distances[:15]):
        print(f"  {i+1:2d}. RBD-{d['viral_pos']} ({d['viral_aa']}) ↔ "
              f"ACE2-{d['host_pos']} ({d['host_aa']}): dist={d['distance']:.4f}")
        print(f"      V: {d['viral_context']}")
        print(f"      H: {d['host_context']}")

    results['convergences'] = all_distances[:50]

    # ========================================================================
    # 3. Deep Asymmetric Scan
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Deep Asymmetric Perturbation Scan")
    print("-" * 70)
    print(f"Scanning top 20 convergent pairs across {len(MODIFICATIONS)} modifications...")

    all_asymmetric = []
    for d in all_distances[:20]:
        v_pos = d['viral_pos']
        h_pos = d['host_pos']

        if v_pos in rbd_interfaces and h_pos in ace2_interfaces:
            v_data = rbd_interfaces[v_pos].copy()
            v_data['pos'] = v_pos
            h_data = ace2_interfaces[h_pos].copy()
            h_data['pos'] = h_pos

            asym_results = deep_asymmetric_scan(v_data, h_data, encoder)
            for r in asym_results:
                r['viral_interface'] = v_pos
                r['host_interface'] = h_pos
                r['interface_distance'] = d['distance']
            all_asymmetric.extend(asym_results)

    print(f"\nTotal modifications tested: {len(all_asymmetric)}")

    # Count by potential
    excellent = [r for r in all_asymmetric if r['therapeutic_potential'] == 'EXCELLENT']
    high = [r for r in all_asymmetric if r['therapeutic_potential'] == 'HIGH']
    medium = [r for r in all_asymmetric if r['therapeutic_potential'] == 'MEDIUM']

    print(f"  EXCELLENT potential: {len(excellent)}")
    print(f"  HIGH potential: {len(high)}")
    print(f"  MEDIUM potential: {len(medium)}")

    # Sort by asymmetry
    sorted_asym = sorted(all_asymmetric, key=lambda x: x['asymmetry'], reverse=True)

    print("\n--- Top 20 Asymmetric Targets ---")
    for i, a in enumerate(sorted_asym[:20]):
        print(f"  {i+1:2d}. RBD-{a['viral_interface']} → {a['modification']}")
        print(f"      {a['viral_aa']}→{a['new_aa']} at context pos {a['context_position']}")
        print(f"      Viral: {a['viral_shift']:.3f} | Host: {a['host_shift']:.3f} | "
              f"Asym: {a['asymmetry']:.3f} [{a['therapeutic_potential']}]")

    results['asymmetric_targets'] = sorted_asym[:100]

    # ========================================================================
    # 4. Hotspot Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Hotspot Analysis")
    print("-" * 70)

    hotspots = find_hotspots(all_asymmetric)

    print("\n--- Top Modifications by Mean Asymmetry ---")
    for mod, mean_asym in hotspots['top_modifications']:
        count = hotspots['by_modification'].get(mod, 0)
        print(f"  {mod}: mean_asymmetry={mean_asym:.4f} (n={count})")

    print("\n--- Modifications by Target Amino Acid ---")
    for aa, count in sorted(hotspots['by_viral_aa'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {aa}: {count} modifications tested")

    results['hotspots'] = {
        'by_modification': hotspots['by_modification'],
        'top_modifications': hotspots['top_modifications'],
        'excellent_count': hotspots['excellent_count'],
        'high_count': hotspots['high_count']
    }

    # ========================================================================
    # 5. Actionable Therapeutic Candidates
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. ACTIONABLE THERAPEUTIC CANDIDATES")
    print("-" * 70)

    # Group excellent/high by modification type
    actionable = defaultdict(list)
    for r in excellent + high:
        actionable[r['modification']].append(r)

    print("\n--- Grouped by Modification Type ---")
    for mod, targets in sorted(actionable.items(), key=lambda x: len(x[1]), reverse=True):
        if len(targets) >= 2:
            best = max(targets, key=lambda x: x['asymmetry'])
            print(f"\n  {mod} ({len(targets)} targets)")
            print(f"    Best: RBD-{best['viral_interface']} context pos {best['context_position']}")
            print(f"    {best['viral_aa']}→{best['new_aa']}: asymmetry={best['asymmetry']:.3f}")
            print(f"    Context: {best['viral_context']}")

    # ========================================================================
    # Save Results
    # ========================================================================
    output_path = Path(__file__).parent / "deep_sweep_results.json"

    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: THERAPEUTIC CANDIDATES FOR FURTHER INVESTIGATION")
    print("=" * 70)

    print("\n1. PHOSPHORYLATION MIMICS (S→D, T→D, Y→D)")
    phospho = [r for r in sorted_asym if 'to_D' in r['modification']][:5]
    for p in phospho:
        print(f"   - {p['viral_aa']}→D at RBD-{p['viral_interface']}: "
              f"viral shift {p['viral_shift']:.1%}, host shift {p['host_shift']:.1%}")

    print("\n2. CITRULLINATION (R→Q)")
    cit = [r for r in sorted_asym if r['modification'] == 'R_to_Q'][:5]
    for c in cit:
        print(f"   - R→Q at RBD-{c['viral_interface']}: "
              f"viral shift {c['viral_shift']:.1%}, host shift {c['host_shift']:.1%}")

    print("\n3. ACETYLATION (K→Q)")
    acet = [r for r in sorted_asym if r['modification'] == 'K_to_Q'][:5]
    for a in acet:
        print(f"   - K→Q at RBD-{a['viral_interface']}: "
              f"viral shift {a['viral_shift']:.1%}, host shift {a['host_shift']:.1%}")


if __name__ == "__main__":
    main()
