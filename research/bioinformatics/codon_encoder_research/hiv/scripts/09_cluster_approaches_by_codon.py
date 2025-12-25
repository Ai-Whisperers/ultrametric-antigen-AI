#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Cluster HIV Treatment Approaches by Codon/PTM Signatures

Analyzes approaches from recent HIV research papers (TRT-008, TRT-009, PATH-003, PATH-005)
by encoding their key mutations in hyperbolic space and clustering them.

Papers analyzed:
- TRT-008: VH4524184 Third-Gen INSTI (Phase 1)
- TRT-009: PASO-DOBLE DTG vs BIC head-to-head
- PATH-003: Elite Controllers functional cure
- PATH-005: Tissue reservoir persistence
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot,
                              hyperbolic_centroid, hyperbolic_variance,
                              load_codon_encoder, poincare_distance,
                              poincare_distance_matrix)

# ============================================================================
# APPROACH DEFINITIONS FROM RESEARCH PAPERS
# ============================================================================

APPROACHES = {
    # TRT-008: VH4524184 Third-Generation INSTI (Phase 1)
    # Focus: Activity against resistant mutations
    "TRT008_INSTI_RESISTANT": {
        "paper": "TRT-008",
        "title": "Third-Gen INSTI Resistance",
        "category": "treatment",
        "mutations": [
            ("G", "S", 140, "integrase"),  # G140S accessory
            ("Q", "H", 148, "integrase"),  # Q148H primary
            ("N", "H", 155, "integrase"),  # N155H primary
            ("R", "K", 263, "integrase"),  # R263K DTG-selected
            ("E", "Q", 92, "integrase"),  # E92Q accessory
            ("Q", "R", 148, "integrase"),  # Q148R variant
        ],
        "mechanism": "Novel INSTI with activity against first/second-gen resistant mutants",
    },
    # TRT-009: PASO-DOBLE Study (DTG/3TC vs BIC/FTC/TAF)
    # Focus: 2-drug vs 3-drug regimens
    "TRT009_DTG_BACKBONE": {
        "paper": "TRT-009",
        "title": "Dolutegravir 2-Drug Regimen",
        "category": "treatment",
        "mutations": [
            ("M", "V", 184, "RT"),  # M184V - 3TC resistance
            ("K", "R", 65, "RT"),  # K65R - TAF resistance context
        ],
        "mechanism": "DTG/3TC 2-drug regimen - high barrier to resistance",
    },
    "TRT009_BIC_BACKBONE": {
        "paper": "TRT-009",
        "title": "Bictegravir 3-Drug Regimen",
        "category": "treatment",
        "mutations": [
            ("M", "V", 184, "RT"),  # M184V - FTC resistance
            ("K", "R", 65, "RT"),  # K65R - TAF resistance
        ],
        "mechanism": "BIC/FTC/TAF 3-drug regimen - standard of care",
    },
    # PATH-003: Elite Controllers
    # Focus: HLA-associated CTL epitopes and escape mutations
    "PATH003_CTL_TARGETS": {
        "paper": "PATH-003",
        "title": "Elite Controller CTL Targets",
        "category": "immune",
        "mutations": [
            # HLA-B*57:01 targets (Gag-specific)
            ("T", "N", 242, "gag"),  # TW10 epitope position
            ("G", "E", 248, "gag"),  # Escape mutation
            # HLA-B*27:05 targets
            ("R", "K", 264, "gag"),  # KK10 epitope
            ("L", "M", 268, "gag"),  # Escape position
        ],
        "mechanism": "CTL epitopes under protective HLA alleles",
    },
    "PATH003_ESCAPE_MUTATIONS": {
        "paper": "PATH-003",
        "title": "EC Immune Escape",
        "category": "immune",
        "mutations": [
            ("T", "A", 242, "gag"),  # TW10 escape
            ("G", "D", 248, "gag"),  # Fitness cost escape
            ("K", "R", 70, "RT"),  # Broad escape
        ],
        "mechanism": "Mutations that enable immune evasion in controllers",
    },
    # PATH-005: Tissue Reservoirs
    # Focus: Compartmentalized evolution and tissue-specific adaptation
    "PATH005_LYMPHOID": {
        "paper": "PATH-005",
        "title": "Lymphoid Tissue Reservoir",
        "category": "reservoir",
        "mutations": [
            # T cell adapted sequences
            ("R", "K", 5, "matrix"),  # Tissue tropism
            ("K", "R", 103, "RT"),  # NNRTI context
            ("Y", "C", 181, "RT"),  # Persistent variants
        ],
        "mechanism": "Clonal populations in lymph nodes (>98% reservoir)",
    },
    "PATH005_CNS": {
        "paper": "PATH-005",
        "title": "CNS Sanctuary",
        "category": "reservoir",
        "mutations": [
            # Macrophage tropism
            ("V", "I", 3, "env"),  # gp120 adaptation
            ("R", "Q", 318, "env"),  # CD4 binding
            ("N", "T", 160, "env"),  # Glycan shield variation
        ],
        "mechanism": "BBB-protected microglia with 4+ year lifespan",
    },
}

# ============================================================================
# ENCODING AND ANALYSIS FUNCTIONS
# ============================================================================


def get_embedding(encoder, codon):
    """Get hyperbolic embedding for a codon."""
    x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
    with torch.no_grad():
        return encoder.encode(x)[0].numpy()


def encode_mutation(encoder, wt_aa, mut_aa):
    """Encode a mutation as the difference vector in hyperbolic space."""
    wt_codon = AA_TO_CODON.get(wt_aa)
    mut_codon = AA_TO_CODON.get(mut_aa)

    if not wt_codon or not mut_codon:
        return None, None, None

    wt_emb = get_embedding(encoder, wt_codon)
    mut_emb = get_embedding(encoder, mut_codon)
    dist = poincare_distance(torch.tensor(wt_emb).unsqueeze(0), torch.tensor(mut_emb).unsqueeze(0)).item()

    return wt_emb, mut_emb, dist


def analyze_approach(encoder, approach_name, approach_data):
    """Analyze all mutations in an approach."""
    embeddings = []
    distances = []
    mutation_details = []

    for wt_aa, mut_aa, position, gene in approach_data["mutations"]:
        wt_emb, mut_emb, dist = encode_mutation(encoder, wt_aa, mut_aa)

        if wt_emb is not None:
            embeddings.append(mut_emb)  # Use mutant embedding as signature
            distances.append(dist)
            mutation_details.append(
                {
                    "mutation": f"{wt_aa}{position}{mut_aa}",
                    "gene": gene,
                    "distance": dist,
                }
            )

    if not embeddings:
        return None

    embeddings_arr = np.array(embeddings)
    centroid = hyperbolic_centroid(embeddings_arr)
    variance = hyperbolic_variance(embeddings_arr, centroid)

    return {
        "name": approach_name,
        "paper": approach_data["paper"],
        "title": approach_data["title"],
        "category": approach_data["category"],
        "mechanism": approach_data["mechanism"],
        "centroid": centroid,
        "variance": variance,
        "mean_mutation_distance": np.mean(distances),
        "mutation_details": mutation_details,
        "n_mutations": len(embeddings),
    }


def compute_approach_distances(approach_results):
    """Compute pairwise distances between approach centroids."""
    names = list(approach_results.keys())
    n = len(names)

    centroids = np.array([approach_results[name]["centroid"] for name in names])
    dist_matrix = poincare_distance_matrix(centroids)

    return names, dist_matrix


def cluster_by_category(approach_results):
    """Group approaches by category and compute within/between distances."""
    by_category = defaultdict(list)
    for name, result in approach_results.items():
        by_category[result["category"]].append(name)

    return dict(by_category)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    print("=" * 70)
    print("HIV APPROACH CLUSTERING BY CODON SIGNATURES")
    print("Using 3-adic Hyperbolic Encoder (V5.11.3)")
    print("=" * 70)

    # Load encoder
    encoder, _, _ = load_codon_encoder(device="cpu", version="3adic")

    # Analyze each approach
    print("\n[1] ANALYZING APPROACHES")
    print("-" * 70)

    approach_results = {}
    for name, data in APPROACHES.items():
        result = analyze_approach(encoder, name, data)
        if result:
            approach_results[name] = result
            print(f"\n{result['paper']}: {result['title']}")
            print(f"  Category: {result['category']}")
            print(f"  Mutations: {result['n_mutations']}")
            print(f"  Mean mutation distance: {result['mean_mutation_distance']:.3f}")
            print(f"  Hyperbolic variance: {result['variance']:.4f}")
            for md in result["mutation_details"]:
                print(f"    {md['mutation']:>8s} ({md['gene']:>10s}): d={md['distance']:.3f}")

    # Compute pairwise distances between approaches
    print("\n" + "=" * 70)
    print("[2] PAIRWISE APPROACH DISTANCES (HYPERBOLIC)")
    print("-" * 70)

    names, dist_matrix = compute_approach_distances(approach_results)

    # Print distance matrix header
    short_names = [n.split("_")[0][:7] for n in names]
    header = "            " + "  ".join(f"{s:>7s}" for s in short_names)
    print(header)

    for i, name in enumerate(names):
        row = f"{short_names[i]:>10s}  "
        for j in range(len(names)):
            if i == j:
                row += "   -    "
            else:
                row += f"{dist_matrix[i, j]:7.3f} "
        print(row)

    # Cluster by category
    print("\n" + "=" * 70)
    print("[3] CATEGORY ANALYSIS")
    print("-" * 70)

    categories = cluster_by_category(approach_results)

    for category, members in categories.items():
        print(f"\n{category.upper()} ({len(members)} approaches):")

        if len(members) > 1:
            # Compute within-category distances
            member_centroids = np.array([approach_results[m]["centroid"] for m in members])
            within_dists = poincare_distance_matrix(member_centroids)
            mean_within = within_dists[np.triu_indices(len(members), k=1)].mean()
            print(f"  Within-category mean distance: {mean_within:.3f}")

        for member in members:
            r = approach_results[member]
            print(f"  - {r['title']}: var={r['variance']:.4f}, mean_d={r['mean_mutation_distance']:.3f}")

    # Find closest approaches across categories
    print("\n" + "=" * 70)
    print("[4] CROSS-CATEGORY CONNECTIONS")
    print("-" * 70)

    connections = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                cat_i = approach_results[name_i]["category"]
                cat_j = approach_results[name_j]["category"]
                if cat_i != cat_j:
                    connections.append(
                        {
                            "approach_1": name_i,
                            "approach_2": name_j,
                            "cat_1": cat_i,
                            "cat_2": cat_j,
                            "distance": dist_matrix[i, j],
                        }
                    )

    # Sort by distance (closest first)
    connections.sort(key=lambda x: x["distance"])

    print("\nClosest cross-category pairs:")
    for conn in connections[:5]:
        print(f"  {conn['approach_1']} <-> {conn['approach_2']}")
        print(f"    ({conn['cat_1']} <-> {conn['cat_2']}): d={conn['distance']:.3f}")

    # Key insights
    print("\n" + "=" * 70)
    print("[5] KEY INSIGHTS")
    print("-" * 70)

    # Find which treatment approaches are closest to immune/reservoir
    treatment_names = [n for n in names if approach_results[n]["category"] == "treatment"]
    other_names = [n for n in names if approach_results[n]["category"] != "treatment"]

    for t_name in treatment_names:
        t_idx = names.index(t_name)
        closest = None
        min_dist = float("inf")
        for o_name in other_names:
            o_idx = names.index(o_name)
            if dist_matrix[t_idx, o_idx] < min_dist:
                min_dist = dist_matrix[t_idx, o_idx]
                closest = o_name

        t_result = approach_results[t_name]
        c_result = approach_results[closest]
        print(f"\n{t_result['title']}:")
        print(f"  Closest non-treatment: {c_result['title']} (d={min_dist:.3f})")
        print(f"  Implication: {t_result['mechanism']}")

    # Save results
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Prepare serializable results
    output_data = {
        "metadata": {
            "encoder": "3-adic (V5.11.3)",
            "papers": ["TRT-008", "TRT-009", "PATH-003", "PATH-005"],
        },
        "approaches": {},
        "distance_matrix": {
            "names": names,
            "distances": dist_matrix.tolist(),
        },
        "categories": categories,
        "connections": [{**c, "distance": float(c["distance"])} for c in connections[:10]],
    }

    for name, result in approach_results.items():
        # Convert numpy floats to Python floats for JSON serialization
        mutation_details = []
        for md in result["mutation_details"]:
            mutation_details.append(
                {
                    "mutation": md["mutation"],
                    "gene": md["gene"],
                    "distance": float(md["distance"]),
                }
            )
        output_data["approaches"][name] = {
            "paper": result["paper"],
            "title": result["title"],
            "category": result["category"],
            "mechanism": result["mechanism"],
            "centroid": result["centroid"].tolist(),
            "variance": float(result["variance"]),
            "mean_mutation_distance": float(result["mean_mutation_distance"]),
            "mutation_details": mutation_details,
            "n_mutations": result["n_mutations"],
        }

    output_path = results_dir / "approach_clustering.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
