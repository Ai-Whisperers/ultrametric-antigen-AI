#!/usr/bin/env python3
"""
SARS-CoV-2 Spike Glycan Sentinel Analysis using P-adic Goldilocks Model.

Identifies sentinel glycans whose removal shifts epitopes into the immunogenic
Goldilocks Zone (15-30% centroid shift), analogous to HIV glycan shield analysis.

Reference: Wuhan-Hu-1 spike (UniProt P0DTC2)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add paths for imports
# Path: .../codon_encoder_research/sars_cov_2/glycan_shield/this_script.py
# parent = glycan_shield, parent.parent = sars_cov_2, parent.parent.parent = codon_encoder_research
CODON_RESEARCH_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CODON_RESEARCH_DIR / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)

# SARS-CoV-2 Spike protein sequence (Wuhan-Hu-1, UniProt P0DTC2)
# Full length: 1273 amino acids
SPIKE_SEQUENCE = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV"
    "SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPF"
    "LGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPI"
    "NLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYN"
    "ENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASV"
    "YAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD"
    "YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYF"
    "PLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFL"
    "PFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLT"
    "PTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLG"
    "AENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGI"
    "AVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC"
    "LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIG"
    "VTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDI"
    "LSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLM"
    "SFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNT"
    "FVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVA"
    "KNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD"
    "SEPVLKGVKLHYT"
)

# Known N-linked glycosylation sites with annotations
SPIKE_GLYCAN_SITES = [
    {
        "position": 17,
        "name": "N17",
        "domain": "NTD",
        "glycan_type": "Complex",
        "bnab_relevance": "NTD shielding",
    },
    {
        "position": 61,
        "name": "N61",
        "domain": "NTD",
        "glycan_type": "Complex/Hybrid",
        "bnab_relevance": "Antigenic supersite",
    },
    {
        "position": 74,
        "name": "N74",
        "domain": "NTD",
        "glycan_type": "Complex",
        "bnab_relevance": "NTD shielding",
    },
    {
        "position": 122,
        "name": "N122",
        "domain": "NTD",
        "glycan_type": "Complex",
        "bnab_relevance": "Near antigenic site",
    },
    {
        "position": 149,
        "name": "N149",
        "domain": "NTD",
        "glycan_type": "Complex",
        "bnab_relevance": "Antigenic supersite",
    },
    {
        "position": 165,
        "name": "N165",
        "domain": "NTD",
        "glycan_type": "Complex",
        "bnab_relevance": "RBD positioning",
    },
    {
        "position": 234,
        "name": "N234",
        "domain": "RBD",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "ACE2 adjacent",
    },
    {
        "position": 282,
        "name": "N282",
        "domain": "RBD",
        "glycan_type": "Complex",
        "bnab_relevance": "RBD shielding",
    },
    {
        "position": 331,
        "name": "N331",
        "domain": "RBD",
        "glycan_type": "Complex",
        "bnab_relevance": "Critical RBD shield",
    },
    {
        "position": 343,
        "name": "N343",
        "domain": "RBD",
        "glycan_type": "Complex",
        "bnab_relevance": "Critical RBD shield",
    },
    {
        "position": 603,
        "name": "N603",
        "domain": "SD1",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Structural",
    },
    {
        "position": 616,
        "name": "N616",
        "domain": "SD1",
        "glycan_type": "Complex",
        "bnab_relevance": "Structural",
    },
    {
        "position": 657,
        "name": "N657",
        "domain": "SD2",
        "glycan_type": "Complex",
        "bnab_relevance": "Structural",
    },
    {
        "position": 709,
        "name": "N709",
        "domain": "S2",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Fusion machinery",
    },
    {
        "position": 717,
        "name": "N717",
        "domain": "S2",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Fusion machinery",
    },
    {
        "position": 801,
        "name": "N801",
        "domain": "S2",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Near fusion peptide",
    },
    {
        "position": 1074,
        "name": "N1074",
        "domain": "S2",
        "glycan_type": "Complex",
        "bnab_relevance": "S2 shielding",
    },
    {
        "position": 1098,
        "name": "N1098",
        "domain": "S2",
        "glycan_type": "Complex",
        "bnab_relevance": "S2 shielding",
    },
    {
        "position": 1134,
        "name": "N1134",
        "domain": "HR2",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Stem region",
    },
    {
        "position": 1158,
        "name": "N1158",
        "domain": "HR2",
        "glycan_type": "Complex",
        "bnab_relevance": "Stem region",
    },
    {
        "position": 1173,
        "name": "N1173",
        "domain": "HR2",
        "glycan_type": "Oligomannose",
        "bnab_relevance": "Stem region",
    },
    {
        "position": 1194,
        "name": "N1194",
        "domain": "HR2",
        "glycan_type": "Complex",
        "bnab_relevance": "Membrane proximal",
    },
]

# Goldilocks Zone boundaries
GOLDILOCKS_MIN = 0.15  # 15%
GOLDILOCKS_MAX = 0.30  # 30%


def get_context_window(sequence: str, position: int, window_size: int = 11) -> str:
    """Extract sequence context around a position."""
    half_window = window_size // 2
    start = max(0, position - half_window)
    end = min(len(sequence), position + half_window + 1)

    context = sequence[start:end]

    # Pad if needed
    if start == 0:
        context = "X" * (half_window - position) + context
    if end == len(sequence):
        context = context + "X" * (position + half_window + 1 - len(sequence))

    return context


def simulate_deglycosylation(context: str, position_in_context: int = 5) -> str:
    """Simulate deglycosylation by N→Q mutation (removes glycosylation sequon)."""
    context_list = list(context)
    if context_list[position_in_context] == "N":
        context_list[position_in_context] = "Q"
    return "".join(context_list)


def encode_sequence(encoder, sequence: str) -> np.ndarray:
    """Encode amino acid sequence to hyperbolic embeddings."""
    embeddings = []
    for aa in sequence.upper():
        if aa == "X":
            # Placeholder for padding - use glycine
            aa = "G"
        codon = AA_TO_CODON.get(aa, "NNN")
        if codon != "NNN":
            emb = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(emb)
    return np.array(embeddings)


def calculate_centroid_shift(wt_emb: np.ndarray, mut_emb: np.ndarray) -> float:
    """Calculate normalized centroid shift between WT and mutant embeddings."""
    wt_centroid = hyperbolic_centroid(wt_emb)
    mut_centroid = hyperbolic_centroid(mut_emb)

    # Calculate Poincare distance between centroids
    wt_centroid_t = torch.from_numpy(wt_centroid).float().unsqueeze(0)
    mut_centroid_t = torch.from_numpy(mut_centroid).float().unsqueeze(0)
    dist = poincare_distance(wt_centroid_t, mut_centroid_t).item()

    # Normalize by max possible distance (diameter ~2 for unit ball)
    normalized_shift = min(dist / 2.0, 1.0)
    return normalized_shift


def calculate_js_divergence(wt_emb: np.ndarray, mut_emb: np.ndarray) -> float:
    """Calculate Jensen-Shannon divergence proxy using embedding distributions."""
    # Use variance ratio as proxy
    wt_var = np.var(wt_emb)
    mut_var = np.var(mut_emb)

    # Symmetric KL-like measure
    js = 0.5 * (np.log(wt_var / mut_var + 1e-10) ** 2 + np.log(mut_var / wt_var + 1e-10) ** 2)
    return float(min(js, 1.0))


def calculate_entropy_change(wt_emb: np.ndarray, mut_emb: np.ndarray) -> float:
    """Calculate entropy change between distributions."""
    wt_entropy = np.mean(np.var(wt_emb, axis=0))
    mut_entropy = np.mean(np.var(mut_emb, axis=0))
    return float(mut_entropy - wt_entropy)


def classify_goldilocks(shift: float) -> str:
    """Classify centroid shift into Goldilocks zones."""
    if shift < GOLDILOCKS_MIN:
        return "below"
    elif shift <= GOLDILOCKS_MAX:
        return "goldilocks"
    else:
        return "above"


def calculate_goldilocks_score(shift: float, js_div: float, entropy_change: float) -> float:
    """
    Calculate Goldilocks score combining multiple metrics.
    Higher score = better sentinel candidate.
    """
    # Optimal shift is center of Goldilocks Zone (22.5%)
    optimal_shift = (GOLDILOCKS_MIN + GOLDILOCKS_MAX) / 2
    shift_score = 1.0 - abs(shift - optimal_shift) / optimal_shift

    # JS divergence contributes positively (boundary crossing)
    js_score = min(js_div * 10, 1.0)

    # Entropy change (magnitude matters, not sign)
    entropy_score = min(abs(entropy_change) * 5, 1.0)

    # Combined score (weighted)
    score = 0.5 * shift_score + 0.3 * js_score + 0.2 * entropy_score

    return max(0.2, min(score, 1.2))  # Clamp to reasonable range


def analyze_glycan_site(encoder, site_info: dict, sequence: str) -> dict:
    """Analyze a single glycan site for sentinel potential."""
    position = site_info["position"] - 1  # Convert to 0-indexed

    # Get context windows
    wt_context = get_context_window(sequence, position)
    mut_context = simulate_deglycosylation(wt_context)

    # Encode sequences
    wt_embedding = encode_sequence(encoder, wt_context)
    mut_embedding = encode_sequence(encoder, mut_context)

    # Calculate metrics
    centroid_shift = calculate_centroid_shift(wt_embedding, mut_embedding)
    js_divergence = calculate_js_divergence(wt_embedding, mut_embedding)
    entropy_change = calculate_entropy_change(wt_embedding, mut_embedding)

    # Classify and score
    zone = classify_goldilocks(centroid_shift)
    score = calculate_goldilocks_score(centroid_shift, js_divergence, entropy_change)

    return {
        "position": site_info["position"],
        "name": site_info["name"],
        "domain": site_info["domain"],
        "glycan_type": site_info["glycan_type"],
        "bnab_relevance": site_info["bnab_relevance"],
        "centroid_shift": float(centroid_shift),
        "js_divergence": float(js_divergence),
        "entropy_change": float(entropy_change),
        "boundary_crossed": js_divergence > 0.01,
        "goldilocks_zone": zone,
        "goldilocks_score": float(score),
        "wt_context": wt_context,
        "mut_context": mut_context,
    }


def main():
    print("=" * 70)
    print("SARS-CoV-2 Spike Glycan Sentinel Analysis")
    print("P-adic Goldilocks Model (Inverse Goldilocks)")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder(device="cpu", version="3adic")
    print("Encoder loaded successfully")

    # Verify spike sequence
    print(f"\nSpike sequence length: {len(SPIKE_SEQUENCE)} amino acids")
    print(f"Total glycan sites to analyze: {len(SPIKE_GLYCAN_SITES)}")

    # Analyze all glycan sites
    print("\n" + "-" * 70)
    print("Analyzing glycan sites...")
    print("-" * 70)

    results = []
    for site in SPIKE_GLYCAN_SITES:
        result = analyze_glycan_site(encoder, site, SPIKE_SEQUENCE)
        results.append(result)

        zone_marker = "***" if result["goldilocks_zone"] == "goldilocks" else "   "
        print(
            f"{zone_marker} {result['name']:8} ({result['domain']:4}): "
            f"shift={result['centroid_shift']*100:5.1f}%, "
            f"zone={result['goldilocks_zone']:10}, "
            f"score={result['goldilocks_score']:.3f}"
        )

    # Sort by Goldilocks score
    results_sorted = sorted(results, key=lambda x: x["goldilocks_score"], reverse=True)

    # Identify Goldilocks sites
    goldilocks_sites = [r for r in results if r["goldilocks_zone"] == "goldilocks"]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal sites analyzed: {len(results)}")
    print(f"Sites in Goldilocks Zone (15-30%): {len(goldilocks_sites)}")

    print("\n--- Top 10 Sentinel Candidates (by Goldilocks Score) ---")
    for i, r in enumerate(results_sorted[:10], 1):
        zone_marker = "[GOLDILOCKS]" if r["goldilocks_zone"] == "goldilocks" else f"[{r['goldilocks_zone'].upper()}]"
        print(
            f"{i:2}. {r['name']:8} | {r['domain']:4} | "
            f"shift={r['centroid_shift']*100:5.1f}% | "
            f"score={r['goldilocks_score']:.3f} | "
            f"{zone_marker}"
        )

    print("\n--- Goldilocks Zone Sites (Sentinel Candidates) ---")
    if goldilocks_sites:
        for r in sorted(goldilocks_sites, key=lambda x: x["goldilocks_score"], reverse=True):
            print(
                f"  {r['name']:8} ({r['domain']:4}): {r['centroid_shift']*100:.1f}% shift, "
                f"score={r['goldilocks_score']:.3f}, {r['bnab_relevance']}"
            )
    else:
        print("  No sites found in Goldilocks Zone")

    # Domain breakdown
    print("\n--- Analysis by Domain ---")
    domains = {}
    for r in results:
        domain = r["domain"]
        if domain not in domains:
            domains[domain] = {"total": 0, "goldilocks": 0, "sites": []}
        domains[domain]["total"] += 1
        if r["goldilocks_zone"] == "goldilocks":
            domains[domain]["goldilocks"] += 1
        domains[domain]["sites"].append(r["name"])

    for domain, info in domains.items():
        print(f"  {domain}: {info['goldilocks']}/{info['total']} in Goldilocks Zone " f"({', '.join(info['sites'])})")

    # Save results
    output_file = Path(__file__).parent / "spike_analysis_results.json"
    output_data = {
        "metadata": {
            "encoder": "3-adic (V5.11.3)",
            "sequence": "SARS-CoV-2 Spike (Wuhan-Hu-1)",
            "uniprot": "P0DTC2",
            "goldilocks_zone": "[15%, 30%]",
            "total_sites": len(results),
            "goldilocks_count": len(goldilocks_sites),
        },
        "results": results_sorted,
        "top_candidates": [r["name"] for r in results_sorted[:10]],
        "goldilocks_sites": [r["name"] for r in goldilocks_sites],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results_sorted


if __name__ == "__main__":
    main()
