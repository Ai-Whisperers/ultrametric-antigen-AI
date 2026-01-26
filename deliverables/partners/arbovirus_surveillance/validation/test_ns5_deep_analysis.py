# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Deep Analysis of DENV NS5 Polymerase Variability.

This script performs detailed analysis of the NS5 protein to identify:
1. Which domains show the highest variability in DENV-4
2. Which specific residues in the RdRp active site differ
3. Known fidelity-affecting positions
4. Evolutionary context of DENV-4 divergence

NS5 Structure (Dengue):
    - Residues 1-265: Methyltransferase (MTase) domain
    - Residues 266-900: RNA-dependent RNA polymerase (RdRp) domain

RdRp Conserved Motifs (known fidelity determinants):
    - Motif A: ~420-435 (palm subdomain)
    - Motif B: ~500-520 (palm subdomain)
    - Motif C: ~530-545 (palm subdomain, catalytic GDD)
    - Motif D: ~560-575 (palm subdomain)
    - Motif E: ~600-615 (palm subdomain)
    - Motif F: ~350-370 (fingers subdomain)
    - Motif G: ~375-395 (fingers subdomain)

Usage:
    python validation/test_ns5_deep_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import datetime

import numpy as np
from scipy.stats import spearmanr

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import BioPython for translation
try:
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


# =============================================================================
# NS5 PROTEIN DOMAIN DEFINITIONS
# =============================================================================

# NS5 gene position in genome (nucleotide)
NS5_GENOME_START = 7570
NS5_GENOME_END = 10269

# NS5 protein domains (amino acid positions, 0-indexed)
NS5_DOMAINS = {
    "MTase": {
        "name": "Methyltransferase",
        "start": 0,
        "end": 265,
        "function": "RNA capping (5' cap formation)",
        "fidelity_role": "Indirect - cap quality affects template stability",
    },
    "Linker": {
        "name": "Interdomain Linker",
        "start": 266,
        "end": 280,
        "function": "Connects MTase to RdRp",
        "fidelity_role": "None known",
    },
    "RdRp_Fingers": {
        "name": "RdRp Fingers Subdomain",
        "start": 281,
        "end": 400,
        "function": "Template entry, NTP binding",
        "fidelity_role": "Affects NTP selectivity",
    },
    "RdRp_Palm": {
        "name": "RdRp Palm Subdomain",
        "start": 401,
        "end": 620,
        "function": "Catalysis, phosphodiester bond formation",
        "fidelity_role": "PRIMARY - contains catalytic site and fidelity determinants",
    },
    "RdRp_Thumb": {
        "name": "RdRp Thumb Subdomain",
        "start": 621,
        "end": 900,
        "function": "Duplex RNA exit, processivity",
        "fidelity_role": "Affects elongation speed/accuracy tradeoff",
    },
}

# Known RdRp conserved motifs (amino acid positions in NS5)
RDRP_MOTIFS = {
    "Motif_F": {
        "start": 350,
        "end": 370,
        "consensus": "ARGWMPMQERVAAMEVKPAQG",
        "function": "NTP entry tunnel",
        "fidelity_role": "Affects incoming NTP positioning",
    },
    "Motif_G": {
        "start": 375,
        "end": 395,
        "consensus": "TDTTPFGQQRVFKEKVDTRTQ",
        "function": "Template positioning",
        "fidelity_role": "Template-primer alignment",
    },
    "Motif_A": {
        "start": 420,
        "end": 435,
        "consensus": "WSSRDGHD",
        "function": "Metal coordination (Mg2+)",
        "fidelity_role": "HIGH - catalytic positioning",
    },
    "Motif_B": {
        "start": 500,
        "end": 520,
        "consensus": "LMSGDDCVVKPLD",
        "function": "NTP binding and selection",
        "fidelity_role": "HIGH - discriminates correct NTP",
    },
    "Motif_C": {
        "start": 530,
        "end": 545,
        "consensus": "GDDSVYH",  # Contains catalytic GDD
        "function": "Catalytic core (Asp-Asp)",
        "fidelity_role": "CRITICAL - catalytic residues",
    },
    "Motif_D": {
        "start": 560,
        "end": 575,
        "consensus": "PRSMAMTG",
        "function": "Translocation",
        "fidelity_role": "Affects processivity/fidelity tradeoff",
    },
    "Motif_E": {
        "start": 600,
        "end": 615,
        "consensus": "TLFGNKFRV",
        "function": "Primer grip",
        "fidelity_role": "Template-primer positioning",
    },
}

# Known fidelity-affecting residues in flavivirus RdRp (literature-based)
# Positions are approximate and based on alignment with other flaviviruses
KNOWN_FIDELITY_RESIDUES = {
    # Poliovirus-like fidelity residues mapped to DENV
    300: {"role": "NTP discrimination", "poliovirus_equivalent": "G64S"},
    421: {"role": "Metal coordination", "poliovirus_equivalent": "Motif A"},
    532: {"role": "Catalytic (GDD)", "poliovirus_equivalent": "Motif C"},
    533: {"role": "Catalytic (GDD)", "poliovirus_equivalent": "Motif C"},
    534: {"role": "Catalytic (GDD)", "poliovirus_equivalent": "Motif C"},
    460: {"role": "Fidelity checkpoint", "poliovirus_equivalent": "H273R"},
    510: {"role": "NTP binding", "poliovirus_equivalent": "Motif B"},
}


def compute_shannon_entropy(column: list[str]) -> float:
    """Compute Shannon entropy for a sequence column."""
    valid = [b for b in column if b in "ACGT"]
    if len(valid) == 0:
        return 2.0
    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_aa_entropy(column: list[str]) -> float:
    """Compute Shannon entropy for amino acid column."""
    valid = [aa for aa in column if aa not in "*X-"]
    if len(valid) == 0:
        return 4.3  # Max entropy for 20 amino acids
    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def translate_sequence(dna_seq: str) -> str:
    """Translate DNA to protein sequence."""
    if BIOPYTHON_AVAILABLE:
        try:
            # Ensure length is multiple of 3
            dna_seq = dna_seq[:len(dna_seq) - len(dna_seq) % 3]
            return str(Seq(dna_seq).translate())
        except Exception:
            pass

    # Fallback: simple translation
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3].upper().replace('U', 'T')
        protein.append(codon_table.get(codon, 'X'))
    return ''.join(protein)


def extract_ns5_proteins(strain_cache: Path) -> dict[str, list[str]]:
    """Extract and translate NS5 protein sequences from all strains."""

    if not strain_cache.exists():
        return {}

    with open(strain_cache) as f:
        all_sequences = json.load(f)

    ns5_proteins = {}

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        sequences = all_sequences.get(serotype, [])

        if sequences and isinstance(sequences[0], list):
            sequences = [(s[0], s[1]) for s in sequences]

        proteins = []
        for acc, seq in sequences:
            if len(seq) >= NS5_GENOME_END:
                ns5_dna = seq[NS5_GENOME_START:NS5_GENOME_END]
                ns5_protein = translate_sequence(ns5_dna)
                if len(ns5_protein) >= 850:  # Minimum expected length
                    proteins.append(ns5_protein)

        ns5_proteins[serotype] = proteins

    return ns5_proteins


def analyze_domain_variability(
    ns5_proteins: dict[str, list[str]],
) -> dict[str, dict]:
    """Analyze entropy per domain for each serotype."""

    results = {}

    for serotype, proteins in ns5_proteins.items():
        if len(proteins) < 5:
            continue

        min_len = min(len(p) for p in proteins)
        domain_results = {}

        for domain_name, domain_info in NS5_DOMAINS.items():
            start = domain_info["start"]
            end = min(domain_info["end"], min_len)

            if start >= min_len:
                continue

            # Compute per-position entropy
            entropies = []
            for pos in range(start, end):
                column = [p[pos] for p in proteins if pos < len(p)]
                entropy = compute_aa_entropy(column)
                entropies.append(entropy)

            if entropies:
                domain_results[domain_name] = {
                    "mean_entropy": np.mean(entropies),
                    "max_entropy": max(entropies),
                    "n_variable": sum(1 for e in entropies if e > 0.5),
                    "n_positions": len(entropies),
                    "pct_variable": sum(1 for e in entropies if e > 0.5) / len(entropies) * 100,
                }

        results[serotype] = domain_results

    return results


def analyze_motif_variability(
    ns5_proteins: dict[str, list[str]],
) -> dict[str, dict]:
    """Analyze entropy at RdRp conserved motifs."""

    results = {}

    for serotype, proteins in ns5_proteins.items():
        if len(proteins) < 5:
            continue

        min_len = min(len(p) for p in proteins)
        motif_results = {}

        for motif_name, motif_info in RDRP_MOTIFS.items():
            start = motif_info["start"]
            end = min(motif_info["end"], min_len)

            if start >= min_len:
                continue

            # Extract motif sequences
            motif_seqs = []
            entropies = []

            for pos in range(start, end):
                column = [p[pos] for p in proteins if pos < len(p)]
                entropy = compute_aa_entropy(column)
                entropies.append(entropy)

            # Get consensus for this serotype
            consensus = []
            for pos in range(start, end):
                column = [p[pos] for p in proteins if pos < len(p)]
                if column:
                    consensus.append(Counter(column).most_common(1)[0][0])

            motif_results[motif_name] = {
                "consensus": ''.join(consensus),
                "expected": motif_info.get("consensus", ""),
                "mean_entropy": np.mean(entropies) if entropies else 0,
                "max_entropy": max(entropies) if entropies else 0,
                "n_variable": sum(1 for e in entropies if e > 0.3),  # Stricter for motifs
                "function": motif_info["function"],
                "fidelity_role": motif_info["fidelity_role"],
            }

        results[serotype] = motif_results

    return results


def analyze_fidelity_residues(
    ns5_proteins: dict[str, list[str]],
) -> dict[str, dict]:
    """Analyze known fidelity-affecting residue positions."""

    results = {}

    for serotype, proteins in ns5_proteins.items():
        if len(proteins) < 5:
            continue

        min_len = min(len(p) for p in proteins)
        residue_results = {}

        for pos, info in KNOWN_FIDELITY_RESIDUES.items():
            if pos >= min_len:
                continue

            # Get amino acids at this position
            aas = [p[pos] for p in proteins if pos < len(p)]
            aa_counts = Counter(aas)
            consensus_aa = aa_counts.most_common(1)[0][0] if aa_counts else 'X'
            entropy = compute_aa_entropy(aas)

            residue_results[str(pos)] = {
                "consensus_aa": consensus_aa,
                "all_aas": dict(aa_counts),
                "entropy": entropy,
                "role": info["role"],
                "is_variable": entropy > 0.3,
            }

        results[serotype] = residue_results

    return results


def find_denv4_specific_differences(
    ns5_proteins: dict[str, list[str]],
) -> list[dict]:
    """Find positions where DENV-4 differs from all other serotypes."""

    differences = []

    if "DENV-4" not in ns5_proteins or len(ns5_proteins["DENV-4"]) < 5:
        return differences

    denv4_proteins = ns5_proteins["DENV-4"]
    other_proteins = []
    for s in ["DENV-1", "DENV-2", "DENV-3"]:
        other_proteins.extend(ns5_proteins.get(s, []))

    if len(other_proteins) < 10:
        return differences

    min_len = min(
        min(len(p) for p in denv4_proteins),
        min(len(p) for p in other_proteins)
    )

    for pos in range(min_len):
        # Get consensus for DENV-4
        denv4_aas = [p[pos] for p in denv4_proteins]
        denv4_consensus = Counter(denv4_aas).most_common(1)[0][0]
        denv4_freq = Counter(denv4_aas)[denv4_consensus] / len(denv4_aas)

        # Get consensus for others
        other_aas = [p[pos] for p in other_proteins]
        other_consensus = Counter(other_aas).most_common(1)[0][0]
        other_freq = Counter(other_aas)[other_consensus] / len(other_aas)

        # Check if DENV-4 differs
        if denv4_consensus != other_consensus and denv4_freq > 0.8 and other_freq > 0.8:
            # Find which domain
            domain = "Unknown"
            for d_name, d_info in NS5_DOMAINS.items():
                if d_info["start"] <= pos < d_info["end"]:
                    domain = d_name
                    break

            # Check if in a known motif
            in_motif = None
            for m_name, m_info in RDRP_MOTIFS.items():
                if m_info["start"] <= pos < m_info["end"]:
                    in_motif = m_name
                    break

            differences.append({
                "position": pos,
                "denv4_aa": denv4_consensus,
                "denv4_frequency": round(denv4_freq, 2),
                "other_aa": other_consensus,
                "other_frequency": round(other_freq, 2),
                "domain": domain,
                "in_motif": in_motif,
                "fidelity_relevant": pos in KNOWN_FIDELITY_RESIDUES,
            })

    return differences


def run_deep_analysis(cache_dir: Path) -> dict:
    """Run comprehensive NS5 deep analysis."""

    print("=" * 70)
    print("NS5 POLYMERASE DEEP ANALYSIS")
    print("=" * 70)
    print()
    print("Investigating molecular basis of DENV-4 polymerase fidelity difference")
    print()

    strain_cache = cache_dir / "dengue_strains.json"

    # Extract NS5 proteins
    print("Extracting and translating NS5 proteins...")
    ns5_proteins = extract_ns5_proteins(strain_cache)

    for serotype, proteins in ns5_proteins.items():
        print(f"  {serotype}: {len(proteins)} proteins extracted")
    print()

    # Domain variability analysis
    print("-" * 70)
    print("DOMAIN VARIABILITY ANALYSIS")
    print("-" * 70)
    print()

    domain_results = analyze_domain_variability(ns5_proteins)

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        if serotype not in domain_results:
            continue
        print(f"{serotype}:")
        for domain, data in domain_results[serotype].items():
            print(f"  {domain}: entropy={data['mean_entropy']:.3f}, "
                  f"variable={data['n_variable']}/{data['n_positions']} ({data['pct_variable']:.1f}%)")
        print()

    # Motif variability analysis
    print("-" * 70)
    print("RdRp CONSERVED MOTIFS ANALYSIS")
    print("-" * 70)
    print()

    motif_results = analyze_motif_variability(ns5_proteins)

    print("Motif variability by serotype:")
    for motif_name in RDRP_MOTIFS.keys():
        print(f"\n{motif_name} ({RDRP_MOTIFS[motif_name]['fidelity_role']}):")
        for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
            if serotype not in motif_results:
                continue
            data = motif_results[serotype].get(motif_name, {})
            if data:
                print(f"  {serotype}: {data['consensus']} (entropy={data['mean_entropy']:.3f})")
    print()

    # Fidelity residue analysis
    print("-" * 70)
    print("KNOWN FIDELITY RESIDUE ANALYSIS")
    print("-" * 70)
    print()

    fidelity_results = analyze_fidelity_residues(ns5_proteins)

    print("Key fidelity positions:")
    for pos in sorted(KNOWN_FIDELITY_RESIDUES.keys()):
        info = KNOWN_FIDELITY_RESIDUES[pos]
        print(f"\nPosition {pos} ({info['role']}):")
        for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
            if serotype not in fidelity_results:
                continue
            data = fidelity_results[serotype].get(str(pos), {})
            if data:
                var = "VARIABLE" if data["is_variable"] else "conserved"
                print(f"  {serotype}: {data['consensus_aa']} ({var}, entropy={data['entropy']:.3f})")
    print()

    # DENV-4 specific differences
    print("-" * 70)
    print("DENV-4 SPECIFIC AMINO ACID DIFFERENCES")
    print("-" * 70)
    print()

    differences = find_denv4_specific_differences(ns5_proteins)

    print(f"Found {len(differences)} positions where DENV-4 consensus differs from others:\n")

    # Group by domain
    by_domain = defaultdict(list)
    for diff in differences:
        by_domain[diff["domain"]].append(diff)

    for domain in ["MTase", "Linker", "RdRp_Fingers", "RdRp_Palm", "RdRp_Thumb"]:
        diffs = by_domain.get(domain, [])
        if diffs:
            print(f"{domain} ({len(diffs)} differences):")
            # Show first 5 and any in motifs
            shown = 0
            for diff in diffs:
                if shown < 5 or diff["in_motif"] or diff["fidelity_relevant"]:
                    motif_note = f" [IN {diff['in_motif']}]" if diff["in_motif"] else ""
                    fidelity_note = " [FIDELITY RESIDUE]" if diff["fidelity_relevant"] else ""
                    print(f"  Pos {diff['position']}: {diff['other_aa']}→{diff['denv4_aa']}"
                          f"{motif_note}{fidelity_note}")
                    shown += 1
            if len(diffs) > 5:
                remaining = len(diffs) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more")
            print()

    # Summary
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print()

    # Calculate key metrics
    denv4_palm = domain_results.get("DENV-4", {}).get("RdRp_Palm", {})
    other_palm_mean = np.mean([
        domain_results.get(s, {}).get("RdRp_Palm", {}).get("mean_entropy", 0)
        for s in ["DENV-1", "DENV-2", "DENV-3"]
    ])

    motif_c_entropy = {
        s: motif_results.get(s, {}).get("Motif_C", {}).get("mean_entropy", 0)
        for s in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]
    }

    n_palm_diffs = len(by_domain.get("RdRp_Palm", []))
    n_fidelity_diffs = sum(1 for d in differences if d["fidelity_relevant"])
    n_motif_diffs = sum(1 for d in differences if d["in_motif"])

    print(f"RdRp Palm domain (catalytic):")
    print(f"  DENV-4 entropy: {denv4_palm.get('mean_entropy', 0):.3f}")
    print(f"  Others mean:    {other_palm_mean:.3f}")
    print(f"  Ratio:          {denv4_palm.get('mean_entropy', 0) / other_palm_mean if other_palm_mean > 0 else 0:.1f}x")
    print()

    print(f"Motif C (catalytic GDD):")
    for s, e in motif_c_entropy.items():
        print(f"  {s}: entropy={e:.3f}")
    print()

    print(f"DENV-4 specific substitutions:")
    print(f"  Total fixed differences:     {len(differences)}")
    print(f"  In RdRp Palm domain:         {n_palm_diffs}")
    print(f"  In conserved motifs:         {n_motif_diffs}")
    print(f"  At known fidelity residues:  {n_fidelity_diffs}")
    print()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "proteins_analyzed": {s: len(p) for s, p in ns5_proteins.items()},
        "domain_variability": domain_results,
        "motif_variability": motif_results,
        "fidelity_residues": fidelity_results,
        "denv4_specific_differences": differences,
        "summary": {
            "total_denv4_differences": len(differences),
            "palm_domain_differences": n_palm_diffs,
            "motif_differences": n_motif_diffs,
            "fidelity_residue_differences": n_fidelity_diffs,
            "denv4_palm_entropy": denv4_palm.get("mean_entropy", 0),
            "other_palm_entropy_mean": other_palm_mean,
        },
    }


def main():
    """Main entry point."""
    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    results = run_deep_analysis(cache_dir)

    # Save results
    output_path = validation_dir / "ns5_deep_analysis_results.json"

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
