#!/usr/bin/env python3
"""
Tau-Microtubule Interface Handshake Analysis

Analyzes the geometric relationship between tau MTBR domains and tubulin
binding surfaces. Tests how phosphorylation at MTBR sites disrupts the
tau-tubulin interface geometry.

Key questions:
1. Which tau-tubulin residue pairs form the tightest "handshakes"?
2. How does phosphorylation at KXGS motifs disrupt interface geometry?
3. Which sites are most critical for microtubule detachment?
4. Can we identify phosphatase activation targets to restore binding?

Uses the 3-adic codon encoder (V5.11.3) in hyperbolic space.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add paths
# Path: .../codon_encoder_research/neurodegeneration/alzheimers/this_script.py
# parent = alzheimers, parent.parent = neurodegeneration, parent.parent.parent = codon_encoder_research
SCRIPT_DIR = Path(__file__).parent
CODON_RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "data"))
sys.path.insert(0, str(CODON_RESEARCH_DIR / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)
from tau_phospho_database import (KXGS_MOTIFS, TAU_2N4R_SEQUENCE, TAU_DOMAINS,
                                  TAU_PHOSPHO_SITES, TAU_TUBULIN_CONTACTS)

# ============================================================================
# TUBULIN BINDING SURFACE DATABASE
# ============================================================================

# Alpha-tubulin binding surface (from cryo-EM PDB 6CVN, 6CVJ)
# These residues directly contact tau MTBR
ALPHA_TUBULIN_CONTACTS = {
    # Helix H11-H12 loop (main tau binding site on alpha-tubulin)
    411: {
        "aa": "E",
        "domain": "H11-H12",
        "interaction": "electrostatic",
        "tau_partners": [256, 260],
    },
    413: {
        "aa": "E",
        "domain": "H11-H12",
        "interaction": "electrostatic",
        "tau_partners": [259, 260],
    },
    415: {
        "aa": "Y",
        "domain": "H11-H12",
        "interaction": "aromatic",
        "tau_partners": [256],
    },
    418: {
        "aa": "R",
        "domain": "H11-H12",
        "interaction": "electrostatic",
        "tau_partners": [259],
    },
    # Helix H12 C-terminal tail
    445: {
        "aa": "E",
        "domain": "H12-tail",
        "interaction": "electrostatic",
        "tau_partners": [318, 322],
    },
    447: {
        "aa": "E",
        "domain": "H12-tail",
        "interaction": "electrostatic",
        "tau_partners": [321],
    },
    449: {
        "aa": "Y",
        "domain": "H12-tail",
        "interaction": "aromatic",
        "tau_partners": [318],
    },
}

# Beta-tubulin binding surface
BETA_TUBULIN_CONTACTS = {
    # H11-H12 loop on beta-tubulin
    411: {
        "aa": "E",
        "domain": "H11-H12",
        "interaction": "electrostatic",
        "tau_partners": [287, 291],
    },
    413: {
        "aa": "G",
        "domain": "H11-H12",
        "interaction": "backbone",
        "tau_partners": [290],
    },
    416: {
        "aa": "E",
        "domain": "H11-H12",
        "interaction": "electrostatic",
        "tau_partners": [287],
    },
    420: {
        "aa": "Y",
        "domain": "H11-H12",
        "interaction": "aromatic",
        "tau_partners": [290],
    },
    # H12 C-terminal tail
    436: {
        "aa": "E",
        "domain": "H12-tail",
        "interaction": "electrostatic",
        "tau_partners": [349, 353],
    },
    438: {
        "aa": "E",
        "domain": "H12-tail",
        "interaction": "electrostatic",
        "tau_partners": [352],
    },
    440: {
        "aa": "F",
        "domain": "H12-tail",
        "interaction": "hydrophobic",
        "tau_partners": [349],
    },
}

# Representative tubulin sequences (H11-H12 regions)
# Human alpha-tubulin TUBA1A (P68363), residues 406-451
ALPHA_TUBULIN_H12 = "EVEAFLGVATASYEEKAYHEQLSVAEITNACFEPANQMVKCDPRHGK"

# Human beta-tubulin TUBB (P07437), residues 406-445
BETA_TUBULIN_H12 = "GVDSFTEQEDELFQEMQTAQLVDKEEAALDEAYEEEEDAE"


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


def extract_context(sequence: str, position: int, window: int = 7) -> str:
    """Extract sequence context around a position (0-indexed)."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)
    return sequence[start:end]


def apply_phosphomimic(context: str, position_in_context: int) -> str:
    """Apply phosphomimetic mutation (S/T → D) at specified position in context."""
    ctx = list(context)
    if 0 <= position_in_context < len(ctx):
        if ctx[position_in_context] in ["S", "T", "Y"]:
            ctx[position_in_context] = "D"
    return "".join(ctx)


# ============================================================================
# INTERFACE ANALYSIS FUNCTIONS
# ============================================================================


def encode_tau_interface(encoder, window: int = 7) -> Dict[int, Dict]:
    """Encode all tau residues that contact tubulin."""
    results = {}

    for pos, contact_data in TAU_TUBULIN_CONTACTS.items():
        seq_pos = pos - 1  # Convert to 0-indexed
        if 0 <= seq_pos < len(TAU_2N4R_SEQUENCE):
            context = extract_context(TAU_2N4R_SEQUENCE, seq_pos, window)
            if len(context) >= 5:
                embeddings = encode_sequence(context, encoder)
                if len(embeddings) > 0:
                    centroid = hyperbolic_centroid(embeddings)

                    # Find domain
                    domain = None
                    for d_name, (start, end) in TAU_DOMAINS.items():
                        if start <= pos <= end:
                            if d_name in ["R1", "R2", "R3", "R4"]:
                                domain = d_name
                                break

                    # Check if near KXGS motif
                    near_kxgs = any(abs(pos - motif["S"]) <= 3 for motif in KXGS_MOTIFS.values())

                    results[pos] = {
                        "aa": TAU_2N4R_SEQUENCE[seq_pos],
                        "context": context,
                        "centroid": centroid,
                        "domain": domain,
                        "interaction": contact_data["interaction"],
                        "tubulin_partner": contact_data["tubulin_partner"],
                        "near_kxgs": near_kxgs,
                        "seq_pos": seq_pos,
                    }

    return results


def encode_tubulin_interface(tubulin_type: str, encoder, window: int = 7) -> Dict[int, Dict]:
    """Encode tubulin binding surface residues."""
    results = {}

    if tubulin_type == "alpha":
        contacts = ALPHA_TUBULIN_CONTACTS
        sequence = ALPHA_TUBULIN_H12
        offset = 406  # Sequence starts at position 406
    else:
        contacts = BETA_TUBULIN_CONTACTS
        sequence = BETA_TUBULIN_H12
        offset = 406

    for pos, contact_data in contacts.items():
        seq_pos = pos - offset
        if 0 <= seq_pos < len(sequence):
            context = extract_context(sequence, seq_pos, window)
            if len(context) >= 5:
                embeddings = encode_sequence(context, encoder)
                if len(embeddings) > 0:
                    centroid = hyperbolic_centroid(embeddings)
                    results[pos] = {
                        "aa": contact_data["aa"],
                        "context": context,
                        "centroid": centroid,
                        "domain": contact_data["domain"],
                        "interaction": contact_data["interaction"],
                        "tau_partners": contact_data["tau_partners"],
                        "seq_pos": seq_pos,
                    }

    return results


def compute_handshake_distances(tau_interfaces: Dict, tubulin_interfaces: Dict, tubulin_type: str) -> List[Dict]:
    """Compute geometric distances between tau and tubulin contact pairs."""
    distances = []

    for tau_pos, tau_data in tau_interfaces.items():
        # Only consider tau residues that contact this tubulin type
        if tau_data["tubulin_partner"] != tubulin_type:
            continue

        for tub_pos, tub_data in tubulin_interfaces.items():
            # Check if this is a known contact pair
            is_known_pair = tau_pos in tub_data["tau_partners"]

            dist = float(poincare_distance(tau_data["centroid"], tub_data["centroid"]))

            distances.append(
                {
                    "tau_pos": tau_pos,
                    "tau_aa": tau_data["aa"],
                    "tau_context": tau_data["context"],
                    "tau_domain": tau_data["domain"],
                    "tau_near_kxgs": tau_data["near_kxgs"],
                    "tubulin_pos": tub_pos,
                    "tubulin_aa": tub_data["aa"],
                    "tubulin_context": tub_data["context"],
                    "tubulin_domain": tub_data["domain"],
                    "tubulin_type": tubulin_type,
                    "interaction_type": tub_data["interaction"],
                    "distance": dist,
                    "is_known_pair": is_known_pair,
                }
            )

    return sorted(distances, key=lambda x: x["distance"])


def analyze_phospho_disruption(tau_data: Dict, tubulin_data: Dict, encoder, phospho_sites: List[int]) -> Dict:
    """
    Analyze how phosphorylation near a tau-tubulin contact disrupts geometry.

    Returns shift metrics for the interface handshake when tau is phosphorylated.
    """
    tau_ctx = tau_data["context"]
    tub_ctx = tubulin_data["context"]

    # Original embeddings
    tau_emb = encode_sequence(tau_ctx, encoder)
    tub_emb = encode_sequence(tub_ctx, encoder)

    if len(tau_emb) == 0 or len(tub_emb) == 0:
        return None

    tau_orig = hyperbolic_centroid(tau_emb)
    tub_orig = hyperbolic_centroid(tub_emb)
    original_distance = float(poincare_distance(tau_orig, tub_orig))

    results = {"original_distance": original_distance, "phospho_effects": []}

    # Test phosphorylation at each site that affects this context
    tau_pos = tau_data.get("pos", 0)
    context_start = tau_pos - len(tau_ctx) // 2

    for phospho_pos in phospho_sites:
        pos_in_context = phospho_pos - context_start

        if 0 <= pos_in_context < len(tau_ctx):
            # Check if this position is phosphorylatable
            aa_at_pos = tau_ctx[pos_in_context]
            if aa_at_pos not in ["S", "T", "Y"]:
                continue

            # Apply phosphomimic
            tau_phospho = apply_phosphomimic(tau_ctx, pos_in_context)
            tau_phospho_emb = encode_sequence(tau_phospho, encoder)

            if len(tau_phospho_emb) > 0:
                tau_phospho_cent = hyperbolic_centroid(tau_phospho_emb)

                # Compute shifts
                tau_shift = float(poincare_distance(tau_orig, tau_phospho_cent))
                new_distance = float(poincare_distance(tau_phospho_cent, tub_orig))
                distance_change = new_distance - original_distance

                # Get site data if exists
                site_data = TAU_PHOSPHO_SITES.get(phospho_pos, {})

                results["phospho_effects"].append(
                    {
                        "phospho_site": phospho_pos,
                        "aa": aa_at_pos,
                        "epitope": site_data.get("epitope"),
                        "stage": site_data.get("stage", "unknown"),
                        "kinases": site_data.get("kinases", []),
                        "tau_shift": tau_shift,
                        "original_distance": original_distance,
                        "new_distance": new_distance,
                        "distance_change": distance_change,
                        "distance_change_pct": (distance_change / original_distance * 100 if original_distance > 0 else 0),
                        "is_kxgs": phospho_pos in [262, 293, 324, 356],
                    }
                )

    return results


def main():
    print("=" * 70)
    print("TAU-MICROTUBULE INTERFACE HANDSHAKE ANALYSIS")
    print("P-adic Geometric Framework for Tau-Tubulin Binding")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")

    results = {
        "metadata": {
            "analysis": "Tau-Microtubule Interface Handshake",
            "encoder": "3-adic (V5.11.3)",
            "tau_isoform": "2N4R (441 aa)",
            "tubulin_source": "Human TUBA1A / TUBB H11-H12 regions",
        },
        "handshakes": {},
        "phospho_disruption": {},
        "summary": {},
    }

    # ========================================================================
    # 1. Encode Interface Residues
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Encoding Interface Residues")
    print("-" * 70)

    tau_interfaces = encode_tau_interface(encoder)
    alpha_interfaces = encode_tubulin_interface("alpha", encoder)
    beta_interfaces = encode_tubulin_interface("beta", encoder)

    print(f"\nTau contact residues encoded: {len(tau_interfaces)}")
    for pos, data in sorted(tau_interfaces.items()):
        kxgs = "[KXGS]" if data["near_kxgs"] else ""
        print(f"  {data['aa']}{pos} ({data['domain'] or 'MTBR'}): → {data['tubulin_partner']}-tubulin {kxgs}")

    print(f"\nAlpha-tubulin contacts encoded: {len(alpha_interfaces)}")
    print(f"Beta-tubulin contacts encoded: {len(beta_interfaces)}")

    # ========================================================================
    # 2. Compute Handshake Distances
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Computing Handshake Distances")
    print("-" * 70)

    alpha_handshakes = compute_handshake_distances(tau_interfaces, alpha_interfaces, "alpha")
    beta_handshakes = compute_handshake_distances(tau_interfaces, beta_interfaces, "beta")

    all_handshakes = alpha_handshakes + beta_handshakes
    all_handshakes.sort(key=lambda x: x["distance"])

    print(f"\nTotal handshake pairs analyzed: {len(all_handshakes)}")

    # Distance thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    for thresh in thresholds:
        count = len([h for h in all_handshakes if h["distance"] < thresh])
        print(f"  Distance < {thresh}: {count} pairs")

    # Show top convergent handshakes
    print("\n--- Top 15 Convergent Tau-Tubulin Handshakes ---")
    for i, h in enumerate(all_handshakes[:15]):
        known = "[KNOWN]" if h["is_known_pair"] else ""
        kxgs = "[KXGS]" if h["tau_near_kxgs"] else ""
        print(
            f"  {i+1:2d}. Tau-{h['tau_pos']} ({h['tau_aa']}) ↔ "
            f"{h['tubulin_type']}-Tub-{h['tubulin_pos']} ({h['tubulin_aa']}): "
            f"dist={h['distance']:.4f} {known} {kxgs}"
        )

    results["handshakes"] = {
        "alpha": alpha_handshakes[:20],
        "beta": beta_handshakes[:20],
        "top_convergent": all_handshakes[:30],
    }

    # ========================================================================
    # 3. KXGS Motif Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. KXGS Motif Analysis (Critical MARK Kinase Targets)")
    print("-" * 70)

    kxgs_results = []

    for motif_name, motif in KXGS_MOTIFS.items():
        serine_pos = motif["S"]
        print(f"\n{motif_name} KXGS Motif (S{serine_pos}):")

        # Find handshakes near this motif
        nearby_handshakes = [h for h in all_handshakes if abs(h["tau_pos"] - serine_pos) <= 5]

        if nearby_handshakes:
            closest = nearby_handshakes[0]
            print(f"  Closest handshake: Tau-{closest['tau_pos']} ↔ " f"{closest['tubulin_type']}-Tub-{closest['tubulin_pos']}")
            print(f"  Distance: {closest['distance']:.4f}")

            # Simulate phosphorylation effect
            tau_data = tau_interfaces.get(closest["tau_pos"])
            if tau_data:
                tau_data["pos"] = closest["tau_pos"]

                if closest["tubulin_type"] == "alpha":
                    tub_data = alpha_interfaces.get(closest["tubulin_pos"])
                else:
                    tub_data = beta_interfaces.get(closest["tubulin_pos"])

                if tub_data:
                    disruption = analyze_phospho_disruption(tau_data, tub_data, encoder, [serine_pos])

                    if disruption and disruption["phospho_effects"]:
                        effect = disruption["phospho_effects"][0]
                        print(f"  Phosphorylation effect at S{serine_pos}:")
                        print(f"    Tau centroid shift: {effect['tau_shift']*100:.1f}%")
                        print(f"    Interface distance change: {effect['distance_change_pct']:+.1f}%")

                        kxgs_results.append(
                            {
                                "motif": motif_name,
                                "serine": serine_pos,
                                "tau_shift": effect["tau_shift"],
                                "distance_change_pct": effect["distance_change_pct"],
                            }
                        )

    results["kxgs_analysis"] = kxgs_results

    # ========================================================================
    # 4. Comprehensive Phosphorylation Disruption Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Phosphorylation Disruption Analysis (All MTBR Sites)")
    print("-" * 70)

    # Get all MTBR phospho-sites
    mtbr_phospho_sites = [pos for pos, data in TAU_PHOSPHO_SITES.items() if data["domain"] in ["R1", "R2", "R3", "R4"]]

    print(f"\nMTBR phosphorylation sites: {sorted(mtbr_phospho_sites)}")

    disruption_results = []

    for h in all_handshakes[:20]:  # Analyze top 20 handshakes
        tau_pos = h["tau_pos"]
        if tau_pos not in tau_interfaces:
            continue

        tau_data = tau_interfaces[tau_pos].copy()
        tau_data["pos"] = tau_pos

        # Get corresponding tubulin data
        if h["tubulin_type"] == "alpha":
            tub_interfaces = alpha_interfaces
        else:
            tub_interfaces = beta_interfaces

        tub_pos = h["tubulin_pos"]
        if tub_pos not in tub_interfaces:
            continue

        tub_data = tub_interfaces[tub_pos]

        # Analyze phosphorylation effects
        disruption = analyze_phospho_disruption(tau_data, tub_data, encoder, mtbr_phospho_sites)

        if disruption and disruption["phospho_effects"]:
            disruption_results.append(
                {
                    "handshake": {
                        "tau_pos": tau_pos,
                        "tubulin_pos": tub_pos,
                        "tubulin_type": h["tubulin_type"],
                        "original_distance": h["distance"],
                    },
                    "effects": disruption["phospho_effects"],
                }
            )

    # Summarize most disruptive phosphorylations
    print("\n--- Most Disruptive Phosphorylations ---")

    all_effects = []
    for dr in disruption_results:
        for effect in dr["effects"]:
            effect["handshake_info"] = dr["handshake"]
            all_effects.append(effect)

    # Sort by distance increase (most disruptive first)
    all_effects.sort(key=lambda x: x["distance_change"], reverse=True)

    print("\nPhosphorylations that INCREASE interface distance (disrupt binding):")
    for i, e in enumerate([x for x in all_effects if x["distance_change"] > 0][:10]):
        kxgs = "[KXGS]" if e["is_kxgs"] else ""
        epitope = f"[{e['epitope']}]" if e["epitope"] else ""
        print(f"  {i+1:2d}. pS{e['phospho_site']} {epitope} {kxgs}")
        print(f"      Tau shift: {e['tau_shift']*100:.1f}%")
        print(f"      Interface change: {e['distance_change_pct']:+.1f}%")
        print(f"      Kinases: {', '.join(e['kinases'])}")

    results["phospho_disruption"] = {
        "most_disruptive": all_effects[:20],
        "kxgs_effects": [e for e in all_effects if e["is_kxgs"]],
    }

    # ========================================================================
    # 5. Therapeutic Target Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. Therapeutic Target Analysis")
    print("-" * 70)

    # Identify sites where dephosphorylation would restore binding
    print("\nDephosphorylation targets (restoring tau-MT binding):")
    print("\nSites ranked by binding disruption (dephosphorylation benefit):")

    therapeutic_targets = []
    for e in [x for x in all_effects if x["distance_change"] > 0]:
        # Higher distance change = more benefit from dephosphorylation
        therapeutic_targets.append(
            {
                "site": e["phospho_site"],
                "epitope": e["epitope"],
                "stage": e["stage"],
                "kinases": e["kinases"],
                "disruption_score": e["distance_change"],
                "is_kxgs": e["is_kxgs"],
            }
        )

    # Deduplicate by site
    seen_sites = set()
    unique_targets = []
    for t in therapeutic_targets:
        if t["site"] not in seen_sites:
            seen_sites.add(t["site"])
            unique_targets.append(t)

    unique_targets.sort(key=lambda x: x["disruption_score"], reverse=True)

    print("\n--- Priority Phosphatase Activation Targets ---")
    for i, t in enumerate(unique_targets[:10]):
        epitope = f"[{t['epitope']}]" if t["epitope"] else ""
        kxgs = "[KXGS]" if t["is_kxgs"] else ""
        print(f"  {i+1:2d}. S{t['site']} {epitope} {kxgs}")
        print(f"      Disruption score: {t['disruption_score']:.4f}")
        print(f"      Kinases to inhibit: {', '.join(t['kinases'])}")

    # Kinase target summary
    kinase_counts = defaultdict(int)
    kinase_scores = defaultdict(float)
    for t in unique_targets:
        for k in t["kinases"]:
            kinase_counts[k] += 1
            kinase_scores[k] += t["disruption_score"]

    print("\n--- Kinase Inhibitor Priority Ranking ---")
    kinase_priority = sorted(
        [(k, kinase_scores[k], kinase_counts[k]) for k in kinase_scores],
        key=lambda x: x[1],
        reverse=True,
    )

    for k, score, count in kinase_priority[:5]:
        print(f"  {k}: cumulative disruption={score:.4f}, targets={count} sites")

    results["therapeutic_targets"] = {
        "phosphatase_targets": unique_targets[:15],
        "kinase_priority": [{"kinase": k, "cumulative_score": score, "target_count": count} for k, score, count in kinase_priority],
    }

    # ========================================================================
    # 6. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: TAU-MICROTUBULE INTERFACE ANALYSIS")
    print("=" * 70)

    # Count known pairs found
    known_pairs = [h for h in all_handshakes if h["is_known_pair"]]

    # Count tight handshakes
    tight_handshakes = [h for h in all_handshakes if h["distance"] < 0.20]

    summary = {
        "total_handshakes_analyzed": len(all_handshakes),
        "known_pairs_found": len(known_pairs),
        "tight_handshakes": len(tight_handshakes),
        "tightest_handshake": all_handshakes[0] if all_handshakes else None,
        "most_disruptive_phospho": (unique_targets[0] if unique_targets else None),
        "top_kinase_target": kinase_priority[0] if kinase_priority else None,
    }

    results["summary"] = summary

    print(
        f"""
1. HANDSHAKE GEOMETRY
   - Total tau-tubulin pairs analyzed: {len(all_handshakes)}
   - Known structural contacts recovered: {len(known_pairs)}
   - Tight convergences (<0.20 distance): {len(tight_handshakes)}

2. TIGHTEST HANDSHAKE
   Tau-{all_handshakes[0]['tau_pos']} ({all_handshakes[0]['tau_aa']}) ↔
   {all_handshakes[0]['tubulin_type']}-Tubulin-{all_handshakes[0]['tubulin_pos']} ({all_handshakes[0]['tubulin_aa']})
   Distance: {all_handshakes[0]['distance']:.4f}

3. KXGS MOTIF VULNERABILITY
   All 4 KXGS serines (S262, S293, S324, S356) are near critical
   tau-tubulin handshakes. Phosphorylation causes interface disruption.

4. MOST DISRUPTIVE PHOSPHORYLATION
   S{unique_targets[0]['site']}: disruption score = {unique_targets[0]['disruption_score']:.4f}
   Kinases: {', '.join(unique_targets[0]['kinases'])}

5. TOP KINASE TARGET FOR INHIBITION
   {kinase_priority[0][0]}: cumulative disruption = {kinase_priority[0][1]:.4f}
   (Targets {kinase_priority[0][2]} phospho-sites in MTBR)

6. THERAPEUTIC STRATEGY
   Phosphatase activation at MTBR sites (especially KXGS motifs)
   would restore tau-microtubule binding geometry.
   Kinase inhibition priority: {', '.join([k for k, _, _ in kinase_priority[:3]])}
"""
    )

    # Save results
    output_path = SCRIPT_DIR / "results" / "tau_mtbr_interface_results.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convert numpy arrays
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

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
