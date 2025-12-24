#!/usr/bin/env python3
"""
HIV-1 Drug Resistance Mutation Analysis using Hyperbolic Geometry
Uses shared hyperbolic_utils from RA pipeline (codon_encoder_3adic.pt)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Use local hyperbolic_utils
sys.path.insert(0, str(Path(__file__).parent))

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

DRUG_RESISTANCE = {
    # NRTIs (Nucleoside RT Inhibitors)
    "M184V": {
        "wt": "M",
        "mut": "V",
        "drugs": ["3TC", "FTC"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "NRTI",
    },
    "K65R": {
        "wt": "K",
        "mut": "R",
        "drugs": ["TDF", "ABC"],
        "resistance": "moderate",
        "fitness": "moderate_decrease",
        "class": "NRTI",
    },
    "K70R": {
        "wt": "K",
        "mut": "R",
        "drugs": ["AZT", "D4T"],
        "resistance": "moderate",
        "fitness": "moderate_decrease",
        "class": "NRTI",
    },
    "T215Y": {
        "wt": "T",
        "mut": "Y",
        "drugs": ["AZT", "D4T"],
        "resistance": "high",
        "fitness": "minimal",
        "class": "NRTI",
    },
    "L74V": {
        "wt": "L",
        "mut": "V",
        "drugs": ["ABC", "DDI"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "NRTI",
    },
    # NNRTIs (Non-Nucleoside RT Inhibitors)
    "K103N": {
        "wt": "K",
        "mut": "N",
        "drugs": ["EFV", "NVP"],
        "resistance": "high",
        "fitness": "minimal",
        "class": "NNRTI",
    },
    "Y181C": {
        "wt": "Y",
        "mut": "C",
        "drugs": ["NVP", "EFV"],
        "resistance": "high",
        "fitness": "minimal",
        "class": "NNRTI",
    },
    "G190A": {
        "wt": "G",
        "mut": "A",
        "drugs": ["NVP", "EFV"],
        "resistance": "high",
        "fitness": "minimal",
        "class": "NNRTI",
    },
    "K101E": {
        "wt": "K",
        "mut": "E",
        "drugs": ["NVP", "EFV"],
        "resistance": "moderate",
        "fitness": "minimal",
        "class": "NNRTI",
    },
    # PIs (Protease Inhibitors)
    "M46I": {
        "wt": "M",
        "mut": "I",
        "drugs": ["IDV", "NFV"],
        "resistance": "moderate",
        "fitness": "minimal",
        "class": "PI",
    },
    "I84V": {
        "wt": "I",
        "mut": "V",
        "drugs": ["DRV", "ATV"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "PI",
    },
    "V82A": {
        "wt": "V",
        "mut": "A",
        "drugs": ["IDV", "RTV"],
        "resistance": "high",
        "fitness": "minimal",
        "class": "PI",
    },
    "L90M": {
        "wt": "L",
        "mut": "M",
        "drugs": ["SQV", "NFV"],
        "resistance": "moderate",
        "fitness": "minimal",
        "class": "PI",
    },
    # INSTIs (Integrase Strand Transfer Inhibitors)
    "N155H": {
        "wt": "N",
        "mut": "H",
        "drugs": ["RAL", "EVG"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "INSTI",
    },
    "Q148H": {
        "wt": "Q",
        "mut": "H",
        "drugs": ["RAL", "EVG", "DTG"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "INSTI",
    },
    "Y143R": {
        "wt": "Y",
        "mut": "R",
        "drugs": ["RAL"],
        "resistance": "high",
        "fitness": "moderate_decrease",
        "class": "INSTI",
    },
    "R263K": {
        "wt": "R",
        "mut": "K",
        "drugs": ["DTG"],
        "resistance": "low",
        "fitness": "high_decrease",
        "class": "INSTI",
    },
    "E92Q": {
        "wt": "E",
        "mut": "Q",
        "drugs": ["RAL", "EVG"],
        "resistance": "moderate",
        "fitness": "minimal",
        "class": "INSTI",
    },
}


def get_embedding(encoder, codon):
    x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
    with torch.no_grad():
        return encoder.encode(x)[0].numpy()


def get_codons_for_aa(aa):
    return [c for c, a in CODON_TABLE.items() if a == aa]


def analyze_resistance(encoder, mutation_name, mutation_data):
    wt_aa = mutation_data["wt"]
    mut_aa = mutation_data["mut"]
    wt_codon = AA_TO_CODON.get(wt_aa, get_codons_for_aa(wt_aa)[0])
    mut_codon = AA_TO_CODON.get(mut_aa, get_codons_for_aa(mut_aa)[0])
    wt_emb = get_embedding(encoder, wt_codon)
    mut_emb = get_embedding(encoder, mut_codon)
    dist = poincare_distance(
        torch.tensor(wt_emb).unsqueeze(0), torch.tensor(mut_emb).unsqueeze(0)
    ).item()
    return {
        "mutation": mutation_name,
        "hyperbolic_distance": dist,
        "drugs": mutation_data["drugs"],
        "resistance": mutation_data["resistance"],
        "fitness": mutation_data["fitness"],
        "class": mutation_data.get("class", "unknown"),
    }


def main():
    print("HIV-1 DRUG RESISTANCE ANALYSIS - codon_encoder_3adic")
    print("=" * 60)
    encoder, _, _ = load_codon_encoder(device="cpu", version="3adic")

    results = []
    by_class = {}

    for name, data in DRUG_RESISTANCE.items():
        r = analyze_resistance(encoder, name, data)
        results.append(r)

        drug_class = r["class"]
        if drug_class not in by_class:
            by_class[drug_class] = []
        by_class[drug_class].append(r["hyperbolic_distance"])

        drugs_str = ", ".join(r["drugs"])
        print(
            f"{r['mutation']:8s} [{r['class']:5s}]: d={r['hyperbolic_distance']:.3f}, drugs=[{drugs_str}]"
        )

    # Summary by class
    print("\n" + "=" * 60)
    print("SUMMARY BY DRUG CLASS:")
    for cls in ["NRTI", "NNRTI", "PI", "INSTI"]:
        if cls in by_class:
            dists = by_class[cls]
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            print(f"  {cls:6s}: mean={mean_d:.3f} +/- {std_d:.3f} (n={len(dists)})")

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "data" / "metrics"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "hiv_resistance_results.json"

    # Save with summary
    summary = {
        cls: {
            "mean": float(np.mean(by_class[cls])),
            "std": float(np.std(by_class[cls])),
            "n": len(by_class[cls]),
        }
        for cls in by_class
    }

    with open(output_path, "w") as f:
        json.dump(
            {
                "metadata": {"encoder": "3-adic (V5.11.3)"},
                "summary_by_class": summary,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
