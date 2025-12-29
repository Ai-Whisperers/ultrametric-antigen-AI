#!/usr/bin/env python3
"""
SARS-CoV-2 / Human Receptor Handshake Interface Analysis

Identifies the geometric "handshake signatures" where viral and host proteins
must converge in p-adic space for successful infection.

Targets:
1. Spike RBD ↔ ACE2 binding interface
2. Spike S1/S2 ↔ Furin cleavage site
3. Spike S2' ↔ TMPRSS2 cleavage site
4. Spike C-terminus ↔ NRP1 binding

The goal is to find:
- Convergence zones: Where viral and host geometry must align
- Forbidden zones: Regions accessible to host but not viral evolution
- Disruption points: Modifications that break viral docking but preserve host function
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add path to hyperbolic utils
# Path: .../codon_encoder_research/sars_cov_2/glycan_shield/this_script.py
# parent = glycan_shield, parent.parent = sars_cov_2, parent.parent.parent = codon_encoder_research
CODON_RESEARCH_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CODON_RESEARCH_DIR / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)


@dataclass
class InterfaceResidue:
    """A residue at a protein-protein interface."""

    position: int
    amino_acid: str
    protein: str
    role: str  # 'contact', 'proximal', 'structural'
    partner_contacts: List[int]  # positions in partner protein


# ============================================================================
# SEQUENCE DATA
# ============================================================================

# SARS-CoV-2 Spike RBD (residues 319-541)
SPIKE_RBD = """RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK
CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS
NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ
PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"""

# Human ACE2 (residues 19-615, extracellular domain)
# Key binding residues: 24, 27, 28, 30, 31, 34, 35, 37, 38, 41, 42, 45, 79, 82, 83, 353, 354, 355, 357
ACE2_ECTODOMAIN = """STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQST
LAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNP
QECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYED
YGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISP
IGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSV
GLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGH
IQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEinf
LLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYC
DPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNML
RLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSI
KVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPR
ISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPP
VS"""

# TMPRSS2 catalytic domain (key residues around active site)
# Active site: H296, D345, S441 (catalytic triad)
TMPRSS2_CATALYTIC = """IVGGTSSAEGVSPWQVSLQDKTGFHFCGGSLINENWVVTAAHCVQNLGKEWVLTAAHCMR
CKSNVVTYNCTKLPSDAQNTKCQYSFSLSFSWNPCYQVSITVNKKYWIVKNSWGLEWCDI
TQYTQIGGIHLKLQAAAVHSPVAQVSHGQVALKGWKLGHSAQCQSKGSIQVRLGEDNINV
VEGNEQFISATRSYRCRTAGFLTGFIKLAQPGDKITVPGSDGGPLVCKKGAWWLASGVKD
QSVSPTLDILSRLQPIGITQVGSFSVNDGGPQCTRAPGQFTVLLPSASCTANHSFFYRSG
VYYEPEVDNVN"""

# Furin cleavage site context in Spike (S1/S2 junction)
# PRRAR↓SV (positions 681-687 in spike)
SPIKE_FURIN_SITE = "TNSPRRARSVASQS"  # Extended context around PRRAR

# Spike S2' cleavage site (TMPRSS2 target)
# KR↓SF (positions 814-817)
SPIKE_S2PRIME_SITE = "PSKRSFIEDLLFNK"  # Extended context

# NRP1 C-end rule binding motif
# C-terminus after furin cleavage: ...RRAR
SPIKE_NRP1_MOTIF = "TNSPR RRAR"  # Space indicates cleavage


# ============================================================================
# INTERFACE DEFINITIONS
# ============================================================================

# RBD residues that contact ACE2 (from crystal structure PDB 6M0J)
RBD_ACE2_CONTACTS = {
    417: "K",  # contacts ACE2 D30
    446: "G",  # contacts ACE2 Q42
    449: "Y",  # contacts ACE2 D38, Q42
    453: "Y",  # contacts ACE2 H34
    455: "L",  # contacts ACE2 H34, D30
    456: "F",  # contacts ACE2 D30, K31
    475: "A",  # contacts ACE2 S19, Q24
    476: "G",  # contacts ACE2 S19, Q24
    477: "S",  # contacts ACE2 S19
    486: "F",  # contacts ACE2 M82, L79, Q24
    487: "N",  # contacts ACE2 Q24, Y83
    489: "Y",  # contacts ACE2 K31, Y83, F28
    493: "Q",  # contacts ACE2 K31, H34, E35
    496: "G",  # contacts ACE2 K353, D38
    498: "Q",  # contacts ACE2 D38, Y41, Q42, K353
    500: "T",  # contacts ACE2 Y41, D355
    501: "N",  # contacts ACE2 Y41, K353
    502: "G",  # contacts ACE2 K353, G354
    505: "Y",  # contacts ACE2 E37, R393
}

# ACE2 residues that contact RBD
ACE2_RBD_CONTACTS = {
    19: "S",
    24: "Q",
    27: "T",
    28: "F",
    30: "D",
    31: "K",
    34: "H",
    35: "E",
    37: "E",
    38: "D",
    41: "Y",
    42: "Q",
    45: "L",
    79: "L",
    82: "M",
    83: "Y",
    353: "K",
    354: "G",
    355: "D",
    357: "R",
    393: "R",
}


def clean_sequence(seq: str) -> str:
    """Remove whitespace and newlines from sequence."""
    return "".join(seq.split())


def extract_context(sequence: str, position: int, window: int = 5) -> str:
    """Extract sequence context around a position."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)
    return sequence[start:end]


def encode_sequence(sequence: str, encoder) -> np.ndarray:
    """Encode a sequence and return embeddings for each position."""
    embeddings = []
    for aa in sequence:
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            embedding = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


def encode_interface_contexts(
    sequence: str,
    contact_positions: Dict[int, str],
    seq_offset: int,
    encoder,
    window: int = 5,
) -> Dict[int, Tuple[str, np.ndarray]]:
    """Encode sequence contexts around interface residues."""
    results = {}
    seq = clean_sequence(sequence)

    for pos, expected_aa in contact_positions.items():
        # Adjust for sequence offset
        seq_pos = pos - seq_offset
        if 0 <= seq_pos < len(seq):
            context = extract_context(seq, seq_pos, window)
            if len(context) >= 3:  # Need at least some context
                embeddings = encode_sequence(context, encoder)
                if len(embeddings) > 0:
                    centroid = hyperbolic_centroid(embeddings)
                    results[pos] = (context, centroid)

    return results


def compute_interface_distances(
    viral_interfaces: Dict[int, Tuple[str, np.ndarray]],
    host_interfaces: Dict[int, Tuple[str, np.ndarray]],
) -> List[Dict]:
    """Compute pairwise distances between viral and host interface points."""
    distances = []

    for v_pos, (v_ctx, v_emb) in viral_interfaces.items():
        for h_pos, (h_ctx, h_emb) in host_interfaces.items():
            dist = poincare_distance(v_emb, h_emb)
            distances.append(
                {
                    "viral_pos": v_pos,
                    "viral_context": v_ctx,
                    "host_pos": h_pos,
                    "host_context": h_ctx,
                    "distance": float(dist),
                    "viral_embedding": v_emb.tolist(),
                    "host_embedding": h_emb.tolist(),
                }
            )

    return distances


def find_convergence_zones(distances: List[Dict], threshold: float = 0.3) -> List[Dict]:
    """Find interface pairs with high geometric convergence (low distance)."""
    convergent = [d for d in distances if d["distance"] < threshold]
    return sorted(convergent, key=lambda x: x["distance"])


def simulate_disruption(
    context: str,
    position_in_context: int,
    encoder,
    modifications: Dict[str, str] = None,
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Simulate modifications and compute resulting geometric shifts."""
    if modifications is None:
        # Default modifications to test
        modifications = {
            "phospho_mimic": {"S": "D", "T": "D", "Y": "D"},  # Phospho-mimetic
            "methylation": {"K": "M", "R": "M"},  # Rough methylation proxy
            "acetylation": {"K": "Q"},  # Acetyl-lysine mimic
            "citrullination": {"R": "Q"},  # Citrulline
        }

    # Encode original
    orig_embeddings = encode_sequence(context, encoder)
    if len(orig_embeddings) == 0:
        return {}
    orig_centroid = hyperbolic_centroid(orig_embeddings)

    results = {}
    target_aa = context[position_in_context] if position_in_context < len(context) else None

    for mod_name, mod_map in modifications.items():
        if target_aa and target_aa in mod_map:
            # Apply modification
            new_context = list(context)
            new_context[position_in_context] = mod_map[target_aa]
            new_context = "".join(new_context)

            # Encode modified
            new_embeddings = encode_sequence(new_context, encoder)
            if len(new_embeddings) > 0:
                new_centroid = hyperbolic_centroid(new_embeddings)
                shift = poincare_distance(orig_centroid, new_centroid)
                results[mod_name] = (new_centroid, float(shift))

    return results


def analyze_asymmetric_perturbation(viral_context: str, host_context: str, encoder) -> Dict:
    """
    Find modifications that maximally disrupt viral geometry while
    minimally affecting host geometry - the key to selective therapeutics.
    """
    modifications = {
        "N_to_Q": {"N": "Q"},  # Deglycosylation mimic
        "R_to_Cit": {"R": "Q"},  # Citrullination
        "K_to_Ac": {"K": "Q"},  # Acetylation mimic
        "S_to_pS": {"S": "D"},  # Phosphorylation mimic
        "T_to_pT": {"T": "D"},  # Phosphorylation mimic
        "Y_to_pY": {"Y": "D"},  # Phosphorylation mimic
        "F_to_Y": {"F": "Y"},  # Hydroxylation
        "P_to_Hyp": {"P": "O"},  # Hydroxyproline (using O as proxy)
    }

    # Encode originals
    v_emb = encode_sequence(viral_context, encoder)
    h_emb = encode_sequence(host_context, encoder)

    if len(v_emb) == 0 or len(h_emb) == 0:
        return []

    v_orig = hyperbolic_centroid(v_emb)
    h_orig = hyperbolic_centroid(h_emb)

    results = []

    # Test each position in both contexts
    for pos in range(min(len(viral_context), len(host_context))):
        v_aa = viral_context[pos]
        h_aa = host_context[pos]

        for mod_name, mod_map in modifications.items():
            v_shift = 0.0
            h_shift = 0.0

            # Test on viral sequence
            if v_aa in mod_map:
                v_new = list(viral_context)
                v_new[pos] = mod_map[v_aa]
                v_new_emb = encode_sequence("".join(v_new), encoder)
                if len(v_new_emb) > 0:
                    v_shift = float(poincare_distance(v_orig, hyperbolic_centroid(v_new_emb)))

            # Test on host sequence
            if h_aa in mod_map:
                h_new = list(host_context)
                h_new[pos] = mod_map[h_aa]
                h_new_emb = encode_sequence("".join(h_new), encoder)
                if len(h_new_emb) > 0:
                    h_shift = float(poincare_distance(h_orig, hyperbolic_centroid(h_new_emb)))

            if v_shift > 0 or h_shift > 0:
                # Asymmetry ratio: positive = disrupts viral more than host
                asymmetry = v_shift - h_shift if h_shift > 0 else v_shift
                results.append(
                    {
                        "position": pos,
                        "viral_aa": v_aa,
                        "host_aa": h_aa,
                        "modification": mod_name,
                        "viral_shift": v_shift,
                        "host_shift": h_shift,
                        "asymmetry": asymmetry,
                        "therapeutic_potential": (
                            "HIGH" if v_shift > 0.15 and h_shift < 0.10 else ("MEDIUM" if v_shift > 0.10 and h_shift < 0.15 else "LOW")
                        ),
                    }
                )

    return sorted(results, key=lambda x: x["asymmetry"], reverse=True)


def main():
    print("=" * 70)
    print("SARS-CoV-2 / Human Receptor Handshake Interface Analysis")
    print("P-adic Geometric Framework")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")

    results = {
        "metadata": {
            "encoder": "3-adic (V5.11.3)",
            "analysis": "Handshake Interface Mapping",
            "interfaces": ["RBD-ACE2", "Spike-Furin", "Spike-TMPRSS2"],
        },
        "interfaces": {},
        "convergence_zones": [],
        "asymmetric_targets": [],
    }

    # ========================================================================
    # 1. RBD-ACE2 Interface Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. RBD-ACE2 Binding Interface")
    print("-" * 70)

    # Encode RBD contact residues
    rbd_seq = clean_sequence(SPIKE_RBD)
    print(f"RBD sequence length: {len(rbd_seq)} residues")
    print(f"Contact residues to analyze: {len(RBD_ACE2_CONTACTS)}")

    rbd_interfaces = encode_interface_contexts(SPIKE_RBD, RBD_ACE2_CONTACTS, 319, encoder)

    # Encode ACE2 contact residues
    ace2_seq = clean_sequence(ACE2_ECTODOMAIN)
    print(f"ACE2 sequence length: {len(ace2_seq)} residues")
    print(f"Contact residues to analyze: {len(ACE2_RBD_CONTACTS)}")

    ace2_interfaces = encode_interface_contexts(ACE2_ECTODOMAIN, ACE2_RBD_CONTACTS, 19, encoder)

    print(f"\nEncoded {len(rbd_interfaces)} RBD interface contexts")
    print(f"Encoded {len(ace2_interfaces)} ACE2 interface contexts")

    # Compute interface distances
    interface_distances = compute_interface_distances(rbd_interfaces, ace2_interfaces)
    print(f"Computed {len(interface_distances)} pairwise distances")

    # Find convergence zones
    convergence = find_convergence_zones(interface_distances, threshold=0.25)
    print(f"\nFound {len(convergence)} convergent interface pairs (distance < 0.25)")

    print("\n--- Top 10 Convergent Handshakes ---")
    for i, c in enumerate(convergence[:10]):
        print(f"  {i+1}. RBD-{c['viral_pos']} ↔ ACE2-{c['host_pos']}: " f"dist={c['distance']:.4f}")
        print(f"      Viral: {c['viral_context']}")
        print(f"      Host:  {c['host_context']}")

    results["interfaces"]["RBD_ACE2"] = {
        "viral_contacts": len(rbd_interfaces),
        "host_contacts": len(ace2_interfaces),
        "total_pairs": len(interface_distances),
        "convergent_pairs": len(convergence),
        "top_convergences": convergence[:20],
    }

    # ========================================================================
    # 2. Asymmetric Perturbation Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Asymmetric Perturbation Analysis")
    print("-" * 70)
    print("Finding modifications that disrupt viral >> host geometry")

    # Analyze top convergent pairs for asymmetric perturbation
    asymmetric_results = []
    for conv in convergence[:5]:  # Top 5 convergent pairs
        v_ctx = conv["viral_context"]
        h_ctx = conv["host_context"]

        asym = analyze_asymmetric_perturbation(v_ctx, h_ctx, encoder)

        for a in asym:
            a["viral_position"] = conv["viral_pos"]
            a["host_position"] = conv["host_pos"]
            asymmetric_results.append(a)

    # Filter for high therapeutic potential
    high_potential = [a for a in asymmetric_results if a["therapeutic_potential"] == "HIGH"]
    medium_potential = [a for a in asymmetric_results if a["therapeutic_potential"] == "MEDIUM"]

    print(f"\nHigh therapeutic potential modifications: {len(high_potential)}")
    print(f"Medium therapeutic potential modifications: {len(medium_potential)}")

    print("\n--- Top Asymmetric Targets (Viral >> Host disruption) ---")
    for i, a in enumerate(sorted(asymmetric_results, key=lambda x: x["asymmetry"], reverse=True)[:10]):
        print(f"  {i+1}. RBD-{a['viral_position']} pos-{a['position']}: " f"{a['viral_aa']}→{a['modification']}")
        print(
            f"      Viral shift: {a['viral_shift']:.3f} | Host shift: {a['host_shift']:.3f} | "
            f"Asymmetry: {a['asymmetry']:.3f} [{a['therapeutic_potential']}]"
        )

    results["asymmetric_targets"] = asymmetric_results

    # ========================================================================
    # 3. Cleavage Site Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Cleavage Site Interface Analysis")
    print("-" * 70)

    # Furin site
    furin_emb = encode_sequence(SPIKE_FURIN_SITE, encoder)
    if len(furin_emb) > 0:
        furin_centroid = hyperbolic_centroid(furin_emb)
        print(f"Furin site (PRRAR): encoded, centroid norm = {np.linalg.norm(furin_centroid):.4f}")

        # Test disruption
        print("\nFurin site modification analysis:")
        for pos, aa in enumerate(SPIKE_FURIN_SITE):
            if aa == "R":
                # Test R→Q (citrullination-like)
                modified = list(SPIKE_FURIN_SITE)
                modified[pos] = "Q"
                mod_emb = encode_sequence("".join(modified), encoder)
                if len(mod_emb) > 0:
                    mod_centroid = hyperbolic_centroid(mod_emb)
                    shift = poincare_distance(furin_centroid, mod_centroid)
                    print(f"  Position {pos} (R→Q): shift = {shift:.4f}")

    # TMPRSS2 site
    tmprss2_emb = encode_sequence(SPIKE_S2PRIME_SITE, encoder)
    if len(tmprss2_emb) > 0:
        tmprss2_centroid = hyperbolic_centroid(tmprss2_emb)
        print(f"\nTMPRSS2 site (S2'): encoded, centroid norm = {np.linalg.norm(tmprss2_centroid):.4f}")

    results["interfaces"]["cleavage_sites"] = {
        "furin_site": SPIKE_FURIN_SITE,
        "tmprss2_site": SPIKE_S2PRIME_SITE,
    }

    # ========================================================================
    # 4. Geometric Summary
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Geometric Summary")
    print("-" * 70)

    # Compute overall interface geometry
    all_viral_emb = []
    all_host_emb = []

    for pos, (ctx, emb) in rbd_interfaces.items():
        all_viral_emb.append(emb)
    for pos, (ctx, emb) in ace2_interfaces.items():
        all_host_emb.append(emb)

    if all_viral_emb and all_host_emb:
        viral_centroid = hyperbolic_centroid(np.array(all_viral_emb))
        host_centroid = hyperbolic_centroid(np.array(all_host_emb))
        interface_distance = poincare_distance(viral_centroid, host_centroid)

        print(f"Overall RBD interface centroid norm: {np.linalg.norm(viral_centroid):.4f}")
        print(f"Overall ACE2 interface centroid norm: {np.linalg.norm(host_centroid):.4f}")
        print(f"Interface geometric distance: {interface_distance:.4f}")

        results["geometry"] = {
            "viral_centroid": viral_centroid.tolist(),
            "host_centroid": host_centroid.tolist(),
            "interface_distance": float(interface_distance),
        }

    # ========================================================================
    # Save Results
    # ========================================================================
    output_path = Path(__file__).parent / "handshake_analysis_results.json"

    # Convert numpy arrays for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    results = convert_for_json(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # ========================================================================
    # Key Findings
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("\n1. CONVERGENCE ZONES (Handshake Signatures)")
    print("   These are positions where viral and host geometry MUST align:")
    if convergence:
        for c in convergence[:3]:
            print(f"   - RBD-{c['viral_pos']} ↔ ACE2-{c['host_pos']}: distance {c['distance']:.4f}")

    print("\n2. THERAPEUTIC TARGETS (Asymmetric Perturbation)")
    print("   Modifications that disrupt viral binding while preserving host:")
    high_asym = sorted(asymmetric_results, key=lambda x: x["asymmetry"], reverse=True)[:3]
    for a in high_asym:
        print(f"   - {a['modification']} at RBD-{a['viral_position']}: " f"viral shift {a['viral_shift']:.3f} vs host {a['host_shift']:.3f}")

    print("\n3. FORBIDDEN ZONE CANDIDATES")
    print("   Host geometry regions unreachable by viral evolution:")
    print("   (To be computed by sampling host protein space)")


if __name__ == "__main__":
    main()
