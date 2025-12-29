#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
HIV Glycan Shield Analysis: Identifying Sentinel Glycans via P-Adic Geometry

This script applies the inverse Goldilocks model to predict which glycosylation
sites, when removed, would expose broadly neutralizing antibody (bnAb) epitopes.

Conjecture: Deglycosylation events that shift epitopes into the Goldilocks Zone
(15-30% p-adic shift) should correlate with known bnAb target glycans.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance from origin."""
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


def poincare_distance_np(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance between two points."""
    x_norm_sq = np.clip(np.sum(x ** 2), 0, 0.999)
    y_norm_sq = np.clip(np.sum(y ** 2), 0, 0.999)
    diff_norm_sq = np.sum((x - y) ** 2)
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    arg = 1 + 2 * c * diff_norm_sq / (denom + 1e-10)
    return float(np.arccosh(np.clip(arg, 1.0, 1e10)))


# Import shared utilities from local scripts folder
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from hyperbolic_utils import AA_TO_CODON, codon_to_onehot, load_codon_encoder

# =============================================================================
# HIV ENV SEQUENCE DATA
# =============================================================================

# BG505 SOSIP.664 gp120 sequence (Clade A, widely used in vaccine research)
# This is a stabilized, soluble Env trimer
BG505_GP120 = """
AENLWVTVYYGVPVWKDAETTLFCASDAKAYETEVHNVWATHACVPTDPNPQEIHLENVTEEFNMWKNNMVEQMHT
DIISLWDQSLKPCVKLTPLCVTLQCTNVTNNITDDMRGELKNCSFNMTTELRDKKQKVYSLFYRLDVVQINENQGN
RSNNSNKEYRLINCNTSAITQACPKVSFEPIPIHYCAPAGFAILKCKDKKFNGTGPCPSVSTVQCTHGIKPVVSTQ
LLLNGSLAEEEVMIRSENITNNAKNILVQFNTPVQINCTRPNNNTRKSIRIGPGQAFYATGDIIGDIRQAHCNISR
AKWNNTLQQVAKKLREHFPNKTIKFANSSGGDLEITTHSFNCGGEFFYCNTSGLFNSTWISNTSVQGSNSTGSNDS
ITLPCRIKQIINMWQRIGQAMYAPPIQGVIRCVSNITGLILTRDGGSTNSTTETFRPGGGDMRDNWRSELYKYKVV
KIEPLGVAPTRCKRRVVGRR
""".replace(
    "\n", ""
).replace(
    " ", ""
)

# Known N-linked glycosylation sites in BG505 (HXB2 numbering where applicable)
# Format: (position_in_sequence, HXB2_number, region, bnAb_relevance)
GLYCAN_SITES = [
    # gp120 glycans with known bnAb relevance
    (88, "N88", "C1", "Structural"),
    (133, "N133", "V1", "V1/V2 bnAbs"),
    (137, "N137", "V1", "V1/V2 bnAbs"),
    (156, "N156", "V1/V2", "PG9/PG16 critical"),
    (160, "N160", "V2", "PGT145 critical"),
    (186, "N186", "V2", "V1/V2 shield"),
    (197, "N197", "C2", "CD4bs proximal"),
    (234, "N234", "C2", "CD4bs shield"),
    (241, "N241", "C2", "Structural"),
    (262, "N262", "C2/Loop D", "CD4bs critical"),
    (276, "N276", "C2", "VRC01-class shield"),
    (289, "N289", "C2", "Near CD4bs"),
    (295, "N295", "C2", "CD4bs shield"),
    (301, "N301", "V3", "2G12 high-mannose"),
    (332, "N332", "V3", "PGT121/128 supersite"),
    (339, "N339", "C3", "V3 supersite"),
    (355, "N355", "C3", "Structural"),
    (363, "N363", "C3", "Structural"),
    (386, "N386", "C3", "Core glycan"),
    (392, "N392", "V4", "Variable"),
    (398, "N398", "V4", "Variable"),
    (406, "N406", "V4", "Structural"),
    (411, "N411", "V4", "Structural"),
    (448, "N448", "C4", "gp120-gp41 interface"),
    (460, "N460", "V5", "Variable"),
    (463, "N463", "V5", "Variable"),
]

# bnAb epitope associations for validation
BNAB_GLYCANS = {
    "V1/V2 apex": ["N156", "N160"],
    "V3 supersite": ["N301", "N332", "N339"],
    "CD4 binding site": ["N234", "N276", "N262", "N295"],
    "gp120-gp41 interface": ["N448"],
    "High-mannose patch": ["N295", "N301", "N332", "N339", "N386", "N392"],
}


@dataclass
class GlycanAnalysisResult:
    """Results for a single glycan site analysis."""

    position: int
    name: str
    region: str
    bnab_relevance: str
    wt_sequence: str
    mut_sequence: str
    centroid_shift: float
    js_divergence: float
    entropy_change: float
    boundary_crossed: bool
    goldilocks_zone: str  # 'below', 'goldilocks', 'above'
    goldilocks_score: float


def find_glycan_sequons(sequence: str) -> List[Tuple[int, str]]:
    """Find all N-X-S/T sequons (potential N-glycosylation sites)."""
    sequons = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P":
            if sequence[i + 2] in ["S", "T"]:
                sequons.append((i, f"N{i+1}"))  # 1-indexed naming
    return sequons


def get_context_window(sequence: str, position: int, window: int = 5) -> str:
    """Extract sequence context around a position."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)
    return sequence[start:end]


def mutate_glycan_site(sequence: str, position: int, mutation: str = "Q") -> str:
    """Create deglycosylation mutant (N→Q or N→D)."""
    seq_list = list(sequence)
    seq_list[position] = mutation
    return "".join(seq_list)


def get_embedding_for_sequence(encoder, sequence: str, aa_to_codon: dict) -> np.ndarray:
    """Get mean embedding for an amino acid sequence."""
    embeddings = []
    for aa in sequence:
        codon = aa_to_codon.get(aa)
        if codon:
            x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
            with torch.no_grad():
                emb = encoder.encode(x)[0].numpy()
            embeddings.append(emb)

    if not embeddings:
        return np.zeros(16)
    return np.mean(embeddings, axis=0)


def get_cluster_distribution(encoder, sequence: str, aa_to_codon: dict, n_clusters: int = 21) -> np.ndarray:
    """Get cluster probability distribution for a sequence."""
    cluster_counts = np.zeros(n_clusters)
    for aa in sequence:
        codon = aa_to_codon.get(aa)
        if codon:
            x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
            with torch.no_grad():
                cluster_id, _ = encoder.get_cluster(x)
            cluster_counts[cluster_id.item()] += 1

    total = cluster_counts.sum()
    if total > 0:
        return cluster_counts / total
    return cluster_counts


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * (kl_pm + kl_qm)


def entropy(p: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Shannon entropy."""
    p = p + eps
    p = p / p.sum()
    return -np.sum(p * np.log(p))


def analyze_glycan_site(
    encoder,
    sequence: str,
    position: int,
    name: str,
    region: str,
    bnab_relevance: str,
    aa_to_codon: dict,
    window: int = 5,
) -> GlycanAnalysisResult:
    """Analyze a single glycan site for sentinel potential."""

    # Get context windows
    wt_context = get_context_window(sequence, position, window)
    mut_sequence = mutate_glycan_site(sequence, position, "Q")
    mut_context = get_context_window(mut_sequence, position, window)

    # Get embeddings
    wt_emb = get_embedding_for_sequence(encoder, wt_context, aa_to_codon)
    mut_emb = get_embedding_for_sequence(encoder, mut_context, aa_to_codon)

    # Calculate centroid shift (V5.12.2: use hyperbolic distance)
    wt_radius = hyperbolic_radius(wt_emb)
    if wt_radius > 0:
        centroid_shift = poincare_distance_np(mut_emb, wt_emb) / wt_radius
    else:
        centroid_shift = 0.0

    # Get cluster distributions
    wt_dist = get_cluster_distribution(encoder, wt_context, aa_to_codon)
    mut_dist = get_cluster_distribution(encoder, mut_context, aa_to_codon)

    # Calculate JS divergence
    js_div = jensen_shannon_divergence(wt_dist, mut_dist)

    # Calculate entropy change
    wt_entropy = entropy(wt_dist)
    mut_entropy = entropy(mut_dist)
    entropy_change = mut_entropy - wt_entropy

    # Check boundary crossing (cluster change for the N residue)
    n_codon = aa_to_codon.get("N")
    q_codon = aa_to_codon.get("Q")
    if n_codon and q_codon:
        x_n = torch.from_numpy(np.array([codon_to_onehot(n_codon)])).float()
        x_q = torch.from_numpy(np.array([codon_to_onehot(q_codon)])).float()
        with torch.no_grad():
            cluster_n, _ = encoder.get_cluster(x_n)
            cluster_q, _ = encoder.get_cluster(x_q)
        boundary_crossed = cluster_n.item() != cluster_q.item()
    else:
        boundary_crossed = False

    # Classify Goldilocks zone
    if centroid_shift < 0.15:
        goldilocks_zone = "below"
    elif centroid_shift <= 0.30:
        goldilocks_zone = "goldilocks"
    else:
        goldilocks_zone = "above"

    # Composite Goldilocks score (higher = more likely sentinel)
    # Penalize being outside the zone, reward boundary crossing
    if goldilocks_zone == "goldilocks":
        zone_score = 1.0 - abs(centroid_shift - 0.225) / 0.075  # Peak at 22.5%
    elif goldilocks_zone == "below":
        zone_score = centroid_shift / 0.15 * 0.5  # Partial credit
    else:
        zone_score = max(0, 1.0 - (centroid_shift - 0.30) / 0.20) * 0.5

    boundary_bonus = 0.2 if boundary_crossed else 0.0
    goldilocks_score = zone_score + boundary_bonus

    return GlycanAnalysisResult(
        position=position,
        name=name,
        region=region,
        bnab_relevance=bnab_relevance,
        wt_sequence=wt_context,
        mut_sequence=mut_context,
        centroid_shift=centroid_shift,
        js_divergence=js_div,
        entropy_change=entropy_change,
        boundary_crossed=boundary_crossed,
        goldilocks_zone=goldilocks_zone,
        goldilocks_score=goldilocks_score,
    )


def assign_region(position: int, seq_len: int) -> Tuple[str, str]:
    """Assign approximate region based on position in gp120."""
    # Approximate gp120 domain boundaries (based on typical structure)
    if position < 50:
        return "C1", "Structural"
    elif position < 90:
        return "V1", "V1/V2 shield"
    elif position < 120:
        return "V2", "V1/V2 bnAbs"
    elif position < 180:
        return "C2", "CD4bs proximal"
    elif position < 220:
        return "V3", "V3 supersite"
    elif position < 280:
        return "C3", "Core glycan"
    elif position < 320:
        return "V4", "Variable"
    elif position < 380:
        return "C4", "gp120-gp41 interface"
    elif position < 420:
        return "V5", "Variable"
    else:
        return "C5", "Structural"


def main():
    print("=" * 70)
    print("HIV GLYCAN SHIELD ANALYSIS: SENTINEL GLYCAN IDENTIFICATION")
    print("Using 3-Adic Codon Encoder (V5.11.3)")
    print("=" * 70)

    # Load encoder
    encoder, mapping, native_hyp = load_codon_encoder(device="cpu", version="3adic")
    print(f"\nEncoder loaded (native hyperbolic: {native_hyp})")

    # Find all N-X-S/T sequons in the sequence
    sequons = find_glycan_sequons(BG505_GP120)
    print(f"\nFound {len(sequons)} potential N-glycosylation sites (N-X-S/T sequons)")

    # Analyze all sequons
    results = []
    print("\nAnalyzing all glycosylation sites in BG505 Env...")
    print("-" * 70)

    for pos_idx, name in sequons:
        region, relevance = assign_region(pos_idx, len(BG505_GP120))

        result = analyze_glycan_site(encoder, BG505_GP120, pos_idx, name, region, relevance, AA_TO_CODON)
        results.append(result)

        zone_marker = "***" if result.goldilocks_zone == "goldilocks" else "   "
        print(
            f"  {result.name:6s} [{result.region:10s}]: "
            f"Δ={result.centroid_shift*100:5.1f}% "
            f"({result.goldilocks_zone:9s}) "
            f"Score={result.goldilocks_score:.3f} {zone_marker}"
        )

    # Sort by Goldilocks score
    results.sort(key=lambda x: x.goldilocks_score, reverse=True)

    # Summary
    print("\n" + "=" * 70)
    print("TOP SENTINEL GLYCAN CANDIDATES")
    print("=" * 70)

    goldilocks_sites = [r for r in results if r.goldilocks_zone == "goldilocks"]

    print(f"\nSites in Goldilocks Zone (15-30% shift): {len(goldilocks_sites)}/{len(results)}")
    print("-" * 70)
    print(f"{'Rank':<5} {'Site':<8} {'Region':<12} {'Shift':<10} {'Score':<8} {'bnAb Relevance'}")
    print("-" * 70)

    for i, r in enumerate(results[:15], 1):
        shift_str = f"{r.centroid_shift*100:.1f}%"
        zone_flag = "*" if r.goldilocks_zone == "goldilocks" else " "
        print(f"{i:<5} {r.name:<8} {r.region:<12} {shift_str:<10} {r.goldilocks_score:.3f}{zone_flag}   {r.bnab_relevance}")

    # Validate against known bnAb glycans
    print("\n" + "=" * 70)
    print("VALIDATION: Known bnAb Glycans")
    print("=" * 70)

    for epitope_class, glycans in BNAB_GLYCANS.items():
        print(f"\n{epitope_class}:")
        for glycan_name in glycans:
            matching = [r for r in results if r.name == glycan_name]
            if matching:
                r = matching[0]
                zone_str = "GOLDILOCKS" if r.goldilocks_zone == "goldilocks" else r.goldilocks_zone
                print(f"  {glycan_name}: {r.centroid_shift*100:.1f}% shift -> {zone_str}")
            else:
                print(f"  {glycan_name}: Not found in analysis")

    # Save results to research hiv results folder
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "glycan_analysis_results.json"

    results_dict = {
        "metadata": {
            "encoder": "3-adic (V5.11.3)",
            "sequence": "BG505 SOSIP gp120",
            "goldilocks_zone": "[15%, 30%]",
            "total_sites": len(results),
            "goldilocks_count": len(goldilocks_sites),
        },
        "results": [
            {
                "position": r.position,
                "name": r.name,
                "region": r.region,
                "bnab_relevance": r.bnab_relevance,
                "centroid_shift": float(r.centroid_shift),
                "js_divergence": float(r.js_divergence),
                "entropy_change": float(r.entropy_change),
                "boundary_crossed": r.boundary_crossed,
                "goldilocks_zone": r.goldilocks_zone,
                "goldilocks_score": float(r.goldilocks_score),
                "wt_context": r.wt_sequence,
                "mut_context": r.mut_sequence,
            }
            for r in results
        ],
        "top_candidates": [r.name for r in results[:10]],
        "goldilocks_sites": [r.name for r in goldilocks_sites],
    }

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Return top candidates for AlphaFold3 analysis
    print("\n" + "=" * 70)
    print("RECOMMENDED FOR ALPHAFOLD3 VALIDATION")
    print("=" * 70)
    print("\nTop 10 candidates for structural validation:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. {r.name} ({r.region}): {r.centroid_shift*100:.1f}% shift")

    return results


if __name__ == "__main__":
    results = main()
