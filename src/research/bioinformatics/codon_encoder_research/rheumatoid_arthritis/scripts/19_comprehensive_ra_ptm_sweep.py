#!/usr/bin/env python3
"""
Comprehensive RA PTM Sweep Analysis

Applies ALL PTM types to ALL modifiable sites across ALL ACPA target proteins,
computing centroid shifts using the 3-adic hyperbolic encoder.

Part of Phase 1: RA Extensions (PRIORITY)
See: research/genetic_code/PTM_EXTENSION_PLAN.md

Input: research/bioinformatics/rheumatoid_arthritis/data/acpa_proteins.json
Output: research/bioinformatics/rheumatoid_arthritis/data/ra_ptm_sweep_results.json
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)

# =============================================================================
# PTM DEFINITIONS
# =============================================================================

PTM_TYPES = {
    "R->Q": {
        "name": "Citrullination",
        "source_residue": "R",
        "target_residue": "Q",
        "biological_relevance": "Primary ACPA trigger, PAD4 enzyme",
    },
    "S->D": {
        "name": "Phosphoserine mimic",
        "source_residue": "S",
        "target_residue": "D",
        "biological_relevance": "Kinase signaling, inflammation",
    },
    "T->D": {
        "name": "Phosphothreonine mimic",
        "source_residue": "T",
        "target_residue": "D",
        "biological_relevance": "Kinase cascades",
    },
    "Y->D": {
        "name": "Phosphotyrosine mimic",
        "source_residue": "Y",
        "target_residue": "D",
        "biological_relevance": "Receptor signaling",
    },
    "N->Q": {
        "name": "Deglycosylation",
        "source_residue": "N",
        "target_residue": "Q",
        "biological_relevance": "Glycan removal, epitope exposure",
    },
    "K->Q": {
        "name": "Acetylation mimic",
        "source_residue": "K",
        "target_residue": "Q",
        "biological_relevance": "Histone modification, NETs",
    },
    "M->Q": {
        "name": "Oxidation mimic",
        "source_residue": "M",
        "target_residue": "Q",
        "biological_relevance": "Oxidative stress, Met-sulfoxide",
    },
}

# Goldilocks Zone boundaries
GOLDILOCKS_MIN = 0.15  # 15%
GOLDILOCKS_MAX = 0.30  # 30%


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
        elif aa == "-" or aa == "X":
            # Padding - use glycine as neutral
            emb = encode_codon_hyperbolic("GGC", encoder)
            embeddings.append(emb)
    return np.array(embeddings) if embeddings else np.array([])


def apply_ptm(context: str, position_in_context: int, ptm_type: str) -> str:
    """Apply PTM modification to context at specified position."""
    ptm_info = PTM_TYPES[ptm_type]
    context_list = list(context)

    if 0 <= position_in_context < len(context_list):
        if context_list[position_in_context] == ptm_info["source_residue"]:
            context_list[position_in_context] = ptm_info["target_residue"]

    return "".join(context_list)


def calculate_centroid_shift(encoder, wt_context: str, mod_context: str) -> float:
    """Calculate normalized centroid shift between WT and modified contexts."""
    wt_emb = encode_context(encoder, wt_context)
    mod_emb = encode_context(encoder, mod_context)

    if len(wt_emb) == 0 or len(mod_emb) == 0:
        return 0.0

    wt_centroid = hyperbolic_centroid(wt_emb)
    mod_centroid = hyperbolic_centroid(mod_emb)

    # Calculate Poincare distance between centroids
    wt_centroid_t = torch.from_numpy(wt_centroid).float().unsqueeze(0)
    mod_centroid_t = torch.from_numpy(mod_centroid).float().unsqueeze(0)
    dist = poincare_distance(wt_centroid_t, mod_centroid_t).item()

    # Normalize by max possible distance (diameter ~2 for unit ball)
    normalized_shift = min(dist / 2.0, 1.0)
    return normalized_shift


def classify_goldilocks(shift: float) -> str:
    """Classify shift into Goldilocks zones."""
    if shift < GOLDILOCKS_MIN:
        return "below"
    elif shift <= GOLDILOCKS_MAX:
        return "goldilocks"
    else:
        return "above"


def compute_goldilocks_score(shift: float) -> float:
    """Compute Goldilocks score (peaks at center of zone)."""
    center = (GOLDILOCKS_MIN + GOLDILOCKS_MAX) / 2
    width = (GOLDILOCKS_MAX - GOLDILOCKS_MIN) / 2

    if shift < GOLDILOCKS_MIN:
        # Below zone: score decreases as shift decreases
        return max(0, shift / GOLDILOCKS_MIN)
    elif shift > GOLDILOCKS_MAX:
        # Above zone: score decreases as shift increases
        return max(0, 1 - (shift - GOLDILOCKS_MAX) / GOLDILOCKS_MAX)
    else:
        # In zone: score peaks at center
        distance_from_center = abs(shift - center)
        return 1 - (distance_from_center / width) * 0.5


# =============================================================================
# MAIN SWEEP
# =============================================================================


def run_ptm_sweep(proteins_data: dict, encoder) -> dict:
    """Run comprehensive PTM sweep across all proteins and sites."""

    results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "encoder": "3-adic (V5.11.3)",
            "goldilocks_zone": f"[{GOLDILOCKS_MIN*100:.0f}%, {GOLDILOCKS_MAX*100:.0f}%]",
            "ptm_types": list(PTM_TYPES.keys()),
        },
        "samples": [],
        "statistics": {
            "total_samples": 0,
            "by_protein": {},
            "by_ptm_type": defaultdict(lambda: {"total": 0, "goldilocks": 0}),
            "by_zone": {"below": 0, "goldilocks": 0, "above": 0},
            "known_acpa_analysis": {"total": 0, "in_goldilocks": 0},
        },
    }

    total_sites = sum(p["total_modifiable_sites"] for p in proteins_data["proteins"])
    processed = 0

    for protein in proteins_data["proteins"]:
        protein_name = protein["name"]
        sequence = protein["sequence"]
        known_cit_sites = set(protein.get("known_cit_sites", []))

        print(f"\nProcessing {protein_name} ({protein['length']} aa, {protein['total_modifiable_sites']} sites)...")

        protein_stats = {
            "total": 0,
            "goldilocks": 0,
            "by_ptm": defaultdict(int),
        }

        for residue_type, sites in protein["modifiable_sites"].items():
            # Determine applicable PTM type for this residue
            ptm_type = None
            for pt, info in PTM_TYPES.items():
                if info["source_residue"] == residue_type:
                    ptm_type = pt
                    break

            if ptm_type is None:
                continue

            for site in sites:
                position = site["position"]  # 1-indexed
                context = site["context_11mer"]
                position_in_context = 5  # Center of 11-mer

                # Apply PTM
                mod_context = apply_ptm(context, position_in_context, ptm_type)

                # Skip if no change (shouldn't happen)
                if mod_context == context:
                    continue

                # Calculate shift
                shift = calculate_centroid_shift(encoder, context, mod_context)
                zone = classify_goldilocks(shift)
                score = compute_goldilocks_score(shift)

                # Check if known ACPA site
                is_known_acpa = residue_type == "R" and position in known_cit_sites

                sample = {
                    "protein": protein_name,
                    "uniprot": protein["uniprot"],
                    "position": position,
                    "residue": residue_type,
                    "wt_context": context,
                    "mod_context": mod_context,
                    "ptm_type": ptm_type,
                    "centroid_shift": round(shift, 6),
                    "goldilocks_zone": zone,
                    "goldilocks_score": round(score, 4),
                    "is_known_acpa": is_known_acpa,
                }

                results["samples"].append(sample)

                # Update statistics
                results["statistics"]["total_samples"] += 1
                results["statistics"]["by_zone"][zone] += 1
                results["statistics"]["by_ptm_type"][ptm_type]["total"] += 1
                if zone == "goldilocks":
                    results["statistics"]["by_ptm_type"][ptm_type]["goldilocks"] += 1
                    protein_stats["goldilocks"] += 1

                if is_known_acpa:
                    results["statistics"]["known_acpa_analysis"]["total"] += 1
                    if zone == "goldilocks":
                        results["statistics"]["known_acpa_analysis"]["in_goldilocks"] += 1

                protein_stats["total"] += 1
                protein_stats["by_ptm"][ptm_type] += 1

                processed += 1
                if processed % 500 == 0:
                    print(f"  Processed {processed}/{total_sites} sites...")

        results["statistics"]["by_protein"][protein_name] = {
            "total": protein_stats["total"],
            "goldilocks": protein_stats["goldilocks"],
            "goldilocks_rate": (protein_stats["goldilocks"] / protein_stats["total"] if protein_stats["total"] > 0 else 0),
        }

    # Convert defaultdicts to regular dicts
    results["statistics"]["by_ptm_type"] = dict(results["statistics"]["by_ptm_type"])

    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE RA PTM SWEEP")
    print("Phase 1: RA Extensions (PRIORITY)")
    print("=" * 70)

    # Load ACPA proteins
    data_dir = SCRIPT_DIR.parent / "data"
    proteins_path = data_dir / "acpa_proteins.json"

    if not proteins_path.exists():
        print("ERROR: ACPA proteins not found. Run 18_extract_acpa_proteins.py first.")
        return 1

    print(f"\nLoading ACPA proteins from: {proteins_path}")
    with open(proteins_path) as f:
        proteins_data = json.load(f)

    print(f"  Proteins: {len(proteins_data['proteins'])}")
    print(f"  Total modifiable sites: {proteins_data['summary']['total_modifiable_sites']}")

    # Load encoder
    print("\nLoading 3-adic hyperbolic encoder...")
    try:
        encoder, mapping = load_hyperbolic_encoder(device="cpu", version="3adic")
        print("  Encoder loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load encoder: {e}")
        return 1

    # Run sweep
    print("\nRunning comprehensive PTM sweep...")
    results = run_ptm_sweep(proteins_data, encoder)

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    stats = results["statistics"]
    print(f"\n  Total PTM samples: {stats['total_samples']}")

    print("\n  By Goldilocks zone:")
    for zone, count in stats["by_zone"].items():
        pct = count / stats["total_samples"] * 100 if stats["total_samples"] > 0 else 0
        print(f"    {zone}: {count} ({pct:.1f}%)")

    print("\n  By PTM type:")
    for ptm, data in stats["by_ptm_type"].items():
        goldilocks_rate = data["goldilocks"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"    {ptm}: {data['total']} total, {data['goldilocks']} goldilocks ({goldilocks_rate:.1f}%)")

    print("\n  Known ACPA sites analysis:")
    acpa = stats["known_acpa_analysis"]
    if acpa["total"] > 0:
        acpa_rate = acpa["in_goldilocks"] / acpa["total"] * 100
        print(f"    Total known sites: {acpa['total']}")
        print(f"    In Goldilocks zone: {acpa['in_goldilocks']} ({acpa_rate:.1f}%)")
    else:
        print("    No known ACPA sites found in sweep")

    # Save results
    output_path = data_dir / "ra_ptm_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Save summary for quick reference
    summary = {
        "analysis_date": results["metadata"]["analysis_date"],
        "total_samples": stats["total_samples"],
        "goldilocks_count": stats["by_zone"]["goldilocks"],
        "goldilocks_rate": (stats["by_zone"]["goldilocks"] / stats["total_samples"] if stats["total_samples"] > 0 else 0),
        "known_acpa_in_goldilocks": acpa["in_goldilocks"],
        "known_acpa_total": acpa["total"],
        "known_acpa_goldilocks_rate": (acpa["in_goldilocks"] / acpa["total"] if acpa["total"] > 0 else 0),
        "by_ptm_type": {k: v["goldilocks"] / v["total"] if v["total"] > 0 else 0 for k, v in stats["by_ptm_type"].items()},
    }

    summary_path = data_dir / "ra_ptm_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run 20_ra_handshake_analysis.py for HLA/TCR interface analysis")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
