#!/usr/bin/env python3
"""
HIV-1 CTL Escape Mutation Analysis using Hyperbolic Geometry
Uses shared hyperbolic_utils from RA pipeline (codon_encoder_3adic.pt)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Use local hyperbolic_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperbolic_utils import (
    load_codon_encoder,
    poincare_distance,
    codon_to_onehot,
    AA_TO_CODON,
)

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

HIV_CTL_EPITOPES = {
    "SL9_Gag77": {
        "protein": "Gag p17",
        "hla_restriction": "HLA-A*02:01",
        "position": "77-85",
        "wild_type": {
            "sequence": "SLYNTVATL",
            "codons": ["AGC", "CTG", "TAC", "AAC", "ACC", "GTG", "GCC", "ACC", "CTG"],
        },
        "escape_variants": [
            {
                "name": "Y79F",
                "sequence": "SLFNTVATL",
                "position": 2,
                "escape_efficacy": "high",
                "fitness_cost": "low",
                "frequency": 0.15,
            },
            {
                "name": "T84I",
                "sequence": "SLYNTVAIL",
                "position": 7,
                "escape_efficacy": "moderate",
                "fitness_cost": "moderate",
                "frequency": 0.08,
            },
        ],
    },
    "KK10_Gag263": {
        "protein": "Gag p24",
        "hla_restriction": "HLA-B*27:05",
        "position": "263-272",
        "wild_type": {
            "sequence": "KRWIILGLNK",
            "codons": [
                "AAG",
                "CGG",
                "TGG",
                "ATC",
                "ATC",
                "CTG",
                "GGC",
                "CTG",
                "AAC",
                "AAG",
            ],
        },
        "escape_variants": [
            {
                "name": "R264K",
                "sequence": "KKWIILGLNK",
                "position": 1,
                "escape_efficacy": "high",
                "fitness_cost": "high",
                "frequency": 0.05,
            },
            {
                "name": "L268M",
                "sequence": "KRWIIMGLNK",
                "position": 5,
                "escape_efficacy": "moderate",
                "fitness_cost": "low",
                "frequency": 0.12,
            },
        ],
    },
    "TW10_Gag240": {
        "protein": "Gag p24",
        "hla_restriction": "HLA-B*57:01",
        "position": "240-249",
        "wild_type": {
            "sequence": "TSTLQEQIGW",
            "codons": [
                "ACC",
                "AGC",
                "ACC",
                "CTG",
                "CAG",
                "GAG",
                "CAG",
                "ATC",
                "GGC",
                "TGG",
            ],
        },
        "escape_variants": [
            {
                "name": "T242N",
                "sequence": "TSNLQEQIGW",
                "position": 2,
                "escape_efficacy": "high",
                "fitness_cost": "moderate",
                "frequency": 0.10,
            },
            {
                "name": "G248A",
                "sequence": "TSTLQEQIAW",
                "position": 8,
                "escape_efficacy": "moderate",
                "fitness_cost": "low",
                "frequency": 0.15,
            },
        ],
    },
    "FL8_Nef90": {
        "protein": "Nef",
        "hla_restriction": "HLA-A*24:02",
        "position": "90-97",
        "wild_type": {
            "sequence": "FLKEKGGL",
            "codons": ["TTC", "CTG", "AAG", "GAG", "AAG", "GGC", "GGC", "CTG"],
        },
        "escape_variants": [
            {
                "name": "K94R",
                "sequence": "FLKERGGL",
                "position": 4,
                "escape_efficacy": "high",
                "fitness_cost": "low",
                "frequency": 0.20,
            }
        ],
    },
    "IV9_RT179": {
        "protein": "RT",
        "hla_restriction": "HLA-A*02:01",
        "position": "179-187",
        "wild_type": {
            "sequence": "ILKEPVHGV",
            "codons": ["ATC", "CTG", "AAG", "GAG", "CCG", "GTG", "CAC", "GGC", "GTG"],
        },
        "escape_variants": [
            {
                "name": "V181I",
                "sequence": "ILKEPIHGV",
                "position": 5,
                "escape_efficacy": "moderate",
                "fitness_cost": "low",
                "frequency": 0.18,
            }
        ],
    },
    "RL9_Env311": {
        "protein": "Env gp120",
        "hla_restriction": "HLA-B*08:01",
        "position": "311-319",
        "wild_type": {
            "sequence": "RLRDLLLIW",
            "codons": ["CGG", "CTG", "CGG", "GAC", "CTG", "CTG", "CTG", "ATC", "TGG"],
        },
        "escape_variants": [
            {
                "name": "D314N",
                "sequence": "RLRNLLLIW",
                "position": 3,
                "escape_efficacy": "moderate",
                "fitness_cost": "high",
                "frequency": 0.06,
            }
        ],
    },
}


def get_embedding(encoder, codon):
    x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
    with torch.no_grad():
        return encoder.encode(x)[0].numpy()


def get_codons_for_aa(aa):
    return [c for c, a in CODON_TABLE.items() if a == aa]


def analyze_escape(encoder, wt_codons, escape_info, mapping):
    pos = escape_info["position"]
    wt_codon = wt_codons[pos]
    wt_aa = CODON_TABLE[wt_codon]
    escape_aa = escape_info["sequence"][pos]
    wt_emb = get_embedding(encoder, wt_codon)
    wt_cluster = mapping.get(wt_codon, -1)

    results = []
    for esc_codon in get_codons_for_aa(escape_aa):
        esc_emb = get_embedding(encoder, esc_codon)
        d = poincare_distance(
            torch.tensor(wt_emb).unsqueeze(0), torch.tensor(esc_emb).unsqueeze(0)
        ).item()
        results.append(
            {
                "codon": esc_codon,
                "distance": d,
                "boundary_crossed": wt_cluster != mapping.get(esc_codon, -1),
            }
        )

    best = min(results, key=lambda x: x["distance"])
    return {
        "mutation": escape_info["name"],
        "hyperbolic_distance": best["distance"],
        "boundary_crossed": best["boundary_crossed"],
        "escape_efficacy": escape_info["escape_efficacy"],
        "fitness_cost": escape_info["fitness_cost"],
    }


def main():
    print("HIV-1 CTL ESCAPE ANALYSIS - codon_encoder_3adic")
    print("=" * 60)

    encoder, mapping, _ = load_codon_encoder(device="cpu", version="3adic")
    pos_to_cluster = {
        pos: idx % 21 for idx, pos in enumerate(sorted(set(mapping.values())))
    }
    cluster_map = {c: pos_to_cluster.get(p, -1) for c, p in mapping.items()}

    all_results = []
    epitope_results = {}

    for name, data in HIV_CTL_EPITOPES.items():
        seq = data["wild_type"]["sequence"]
        hla = data["wild_type"].get("hla", "Unknown")
        print(f"\n{name} (HLA: {hla}): {seq}")

        epitope_data = {
            "epitope": name,
            "hla": hla,
            "wild_type_sequence": seq,
            "escape_mutations": [],
        }

        for esc in data["escape_variants"]:
            r = analyze_escape(encoder, data["wild_type"]["codons"], esc, cluster_map)
            status = "CROSSED" if r["boundary_crossed"] else "within"
            print(f"  {r['mutation']}: d={r['hyperbolic_distance']:.3f}, {status}")

            epitope_data["escape_mutations"].append(r)
            all_results.append({"epitope": name, "hla": hla, **r})

        epitope_results[name] = epitope_data

    # Compute summary statistics
    distances = [r["hyperbolic_distance"] for r in all_results]
    boundary_crossed = sum(1 for r in all_results if r["boundary_crossed"])

    summary = {
        "total_mutations": len(all_results),
        "boundary_crossed": boundary_crossed,
        "boundary_crossing_rate": (
            boundary_crossed / len(all_results) if all_results else 0
        ),
        "mean_distance": float(np.mean(distances)) if distances else 0,
        "std_distance": float(np.std(distances)) if distances else 0,
        "min_distance": float(np.min(distances)) if distances else 0,
        "max_distance": float(np.max(distances)) if distances else 0,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total mutations analyzed: {summary['total_mutations']}")
    print(f"  Boundary crossing rate: {summary['boundary_crossing_rate']:.1%}")
    print(f"  Mean hyperbolic distance: {summary['mean_distance']:.3f}")

    # Save results to JSON
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "data" / "metrics"
    results_dir.mkdir(exist_ok=True)

    output = {
        "metadata": {
            "encoder": "3-adic (V5.11.3)",
            "analysis_type": "HIV CTL Escape Mutations",
        },
        "summary": summary,
        "epitopes": epitope_results,
        "all_mutations": all_results,
    }

    output_path = results_dir / "hiv_escape_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
