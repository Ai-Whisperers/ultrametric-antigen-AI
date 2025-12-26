# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Codon Extraction Utilities for HIV Analysis

Provides functions to extract codons from various HIV data formats and
encode them using the hyperbolic codon encoder for geometric analysis.

Integrates with:
- unified_data_loader.py for data loading
- position_mapper.py for position conversion
- hyperbolic_utils.py for codon encoding
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hyperbolic_utils import (
    AA_TO_CODON,
    HyperbolicCodonEncoder,
    codon_to_onehot,
    load_hyperbolic_encoder,
    poincare_distance,
)
from position_mapper import parse_mutation, sequence_to_codons

# ============================================================================
# CODON TABLES
# ============================================================================

# Standard genetic code: codon -> amino acid
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

# Reverse mapping: amino acid -> list of codons
AA_TO_CODONS = {}
for codon, aa in CODON_TABLE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)


def get_codons_for_aa(amino_acid: str) -> list[str]:
    """Get all codons encoding a given amino acid."""
    return AA_TO_CODONS.get(amino_acid.upper(), [])


def get_synonymous_codons(codon: str) -> list[str]:
    """Get all synonymous codons (same amino acid)."""
    aa = CODON_TABLE.get(codon.upper())
    if aa is None:
        return []
    return [c for c in AA_TO_CODONS[aa] if c != codon.upper()]


# ============================================================================
# MUTATION CODON EXTRACTION
# ============================================================================


def mutation_to_codons(mutation_str: str, wild_type_codon: Optional[str] = None) -> dict:
    """
    Extract wild-type and mutant codon information from a mutation.

    Args:
        mutation_str: Mutation string like "D30N"
        wild_type_codon: Known wild-type codon (if available)

    Returns:
        Dictionary with:
        - position: Mutation position
        - wild_type_aa: Wild-type amino acid
        - mutant_aa: Mutant amino acid
        - wild_type_codons: Possible wild-type codons
        - mutant_codons: Possible mutant codons
        - wild_type_codon: Specific wild-type codon (if provided)
    """
    parsed = parse_mutation(mutation_str)
    if parsed is None:
        return None

    wild_aa = parsed["wild_type"]
    mut_aa = parsed["mutant"]

    result = {
        "position": parsed["position"],
        "wild_type_aa": wild_aa,
        "mutant_aa": mut_aa,
        "wild_type_codons": get_codons_for_aa(wild_aa) if wild_aa else [],
        "mutant_codons": get_codons_for_aa(mut_aa) if mut_aa else [],
        "wild_type_codon": wild_type_codon,
    }

    return result


def get_all_codon_transitions(wild_aa: str, mutant_aa: str) -> list[tuple[str, str]]:
    """
    Get all possible codon transitions between two amino acids.

    Args:
        wild_aa: Wild-type amino acid
        mutant_aa: Mutant amino acid

    Returns:
        List of (wild_codon, mutant_codon) tuples
    """
    wild_codons = get_codons_for_aa(wild_aa)
    mut_codons = get_codons_for_aa(mutant_aa)

    return [(w, m) for w in wild_codons for m in mut_codons]


def get_single_nt_transitions(wild_aa: str, mutant_aa: str) -> list[tuple[str, str]]:
    """
    Get codon transitions requiring only a single nucleotide change.

    Args:
        wild_aa: Wild-type amino acid
        mutant_aa: Mutant amino acid

    Returns:
        List of (wild_codon, mutant_codon) tuples with single nt difference
    """
    all_transitions = get_all_codon_transitions(wild_aa, mutant_aa)

    single_nt = []
    for wild, mut in all_transitions:
        diff_count = sum(1 for w, m in zip(wild, mut) if w != m)
        if diff_count == 1:
            single_nt.append((wild, mut))

    return single_nt


def get_minimum_distance_transition(
    wild_aa: str, mutant_aa: str, encoder: HyperbolicCodonEncoder
) -> tuple[str, str, float]:
    """
    Find the codon transition with minimum hyperbolic distance.

    Args:
        wild_aa: Wild-type amino acid
        mutant_aa: Mutant amino acid
        encoder: Hyperbolic codon encoder

    Returns:
        Tuple of (wild_codon, mutant_codon, distance)
    """
    transitions = get_all_codon_transitions(wild_aa, mutant_aa)

    if not transitions:
        return (None, None, float("inf"))

    min_dist = float("inf")
    best_transition = (None, None)

    for wild, mut in transitions:
        wild_emb = encoder.encode_numpy(codon_to_onehot(wild))
        mut_emb = encoder.encode_numpy(codon_to_onehot(mut))
        dist = poincare_distance(wild_emb, mut_emb).item()

        if dist < min_dist:
            min_dist = dist
            best_transition = (wild, mut)

    return best_transition[0], best_transition[1], min_dist


# ============================================================================
# SEQUENCE CODON EXTRACTION
# ============================================================================


def extract_codons_from_alignment(alignment: dict[str, str], start_pos: int = 1) -> pd.DataFrame:
    """
    Extract codons from aligned nucleotide sequences.

    Args:
        alignment: Dictionary mapping sequence ID to aligned sequence
        start_pos: Starting position number

    Returns:
        DataFrame with columns: seq_id, position, codon, amino_acid
    """
    records = []

    for seq_id, sequence in alignment.items():
        codons = sequence_to_codons(sequence)
        for i, codon in enumerate(codons):
            if "-" not in codon and "N" not in codon:
                aa = CODON_TABLE.get(codon.upper(), "X")
                records.append(
                    {
                        "seq_id": seq_id,
                        "position": start_pos + i,
                        "codon": codon.upper(),
                        "amino_acid": aa,
                    }
                )

    return pd.DataFrame(records)


def extract_epitope_codons(
    sequence: str, epitope_start: int, epitope_length: int = 9, region_start: int = 1
) -> list[str]:
    """
    Extract codons for an epitope region.

    Args:
        sequence: Nucleotide sequence (aligned to region start)
        epitope_start: Starting amino acid position of epitope
        epitope_length: Length of epitope in amino acids
        region_start: Starting position of the sequence region

    Returns:
        List of codon strings for the epitope
    """
    offset = epitope_start - region_start
    nt_start = offset * 3

    if nt_start < 0 or nt_start + epitope_length * 3 > len(sequence):
        return []

    epitope_seq = sequence[nt_start : nt_start + epitope_length * 3]
    return sequence_to_codons(epitope_seq)


def extract_v3_codons(sequence: str, v3_start: int = 296, v3_end: int = 331) -> list[str]:
    """
    Extract codons from the V3 loop region of gp120.

    Args:
        sequence: gp120 nucleotide sequence (aligned to gp120 start)
        v3_start: V3 loop start position (default 296)
        v3_end: V3 loop end position (default 331)

    Returns:
        List of V3 loop codons
    """
    v3_length = v3_end - v3_start + 1
    return extract_epitope_codons(sequence, v3_start, v3_length, region_start=1)


# ============================================================================
# HYPERBOLIC ENCODING
# ============================================================================


def encode_mutation_pair(
    wild_aa: str, mutant_aa: str, encoder: HyperbolicCodonEncoder, method: str = "representative"
) -> dict:
    """
    Encode a mutation and calculate hyperbolic distance.

    Args:
        wild_aa: Wild-type amino acid
        mutant_aa: Mutant amino acid
        encoder: Hyperbolic codon encoder
        method: 'representative' (use AA_TO_CODON), 'minimum' (min distance),
                'average' (mean of all transitions)

    Returns:
        Dictionary with embeddings and distance
    """
    if method == "representative":
        # Use representative codon for each amino acid
        wild_codon = AA_TO_CODON.get(wild_aa, "NNN")
        mut_codon = AA_TO_CODON.get(mutant_aa, "NNN")

        if wild_codon == "NNN" or mut_codon == "NNN":
            return None

        wild_emb = encoder.encode_numpy(codon_to_onehot(wild_codon))
        mut_emb = encoder.encode_numpy(codon_to_onehot(mut_codon))
        distance = poincare_distance(wild_emb, mut_emb).item()

        return {
            "wild_type_codon": wild_codon,
            "mutant_codon": mut_codon,
            "wild_type_embedding": wild_emb.squeeze(),
            "mutant_embedding": mut_emb.squeeze(),
            "hyperbolic_distance": distance,
        }

    elif method == "minimum":
        wild_codon, mut_codon, distance = get_minimum_distance_transition(wild_aa, mutant_aa, encoder)
        if wild_codon is None:
            return None

        wild_emb = encoder.encode_numpy(codon_to_onehot(wild_codon))
        mut_emb = encoder.encode_numpy(codon_to_onehot(mut_codon))

        return {
            "wild_type_codon": wild_codon,
            "mutant_codon": mut_codon,
            "wild_type_embedding": wild_emb.squeeze(),
            "mutant_embedding": mut_emb.squeeze(),
            "hyperbolic_distance": distance,
        }

    elif method == "average":
        transitions = get_all_codon_transitions(wild_aa, mutant_aa)
        if not transitions:
            return None

        distances = []
        for wild, mut in transitions:
            wild_emb = encoder.encode_numpy(codon_to_onehot(wild))
            mut_emb = encoder.encode_numpy(codon_to_onehot(mut))
            distances.append(poincare_distance(wild_emb, mut_emb).item())

        return {
            "wild_type_aa": wild_aa,
            "mutant_aa": mutant_aa,
            "n_transitions": len(transitions),
            "mean_distance": np.mean(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "std_distance": np.std(distances),
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def encode_sequence(sequence: str, encoder: HyperbolicCodonEncoder, is_dna: bool = True) -> np.ndarray:
    """
    Encode a nucleotide sequence to hyperbolic embeddings.

    Args:
        sequence: Nucleotide sequence
        encoder: Hyperbolic codon encoder
        is_dna: If True, sequence is DNA; if False, RNA

    Returns:
        Array of embeddings, shape (n_codons, 16)
    """
    if not is_dna:
        sequence = sequence.replace("U", "T").replace("u", "t")

    codons = sequence_to_codons(sequence)
    embeddings = []

    for codon in codons:
        if "-" not in codon and "N" not in codon:
            onehot = codon_to_onehot(codon)
            emb = encoder.encode_numpy(onehot)
            embeddings.append(emb.squeeze())
        else:
            # Gap or ambiguous - use zero embedding
            embeddings.append(np.zeros(16))

    return np.array(embeddings)


def encode_amino_acid_sequence(aa_sequence: str, encoder: HyperbolicCodonEncoder) -> np.ndarray:
    """
    Encode amino acid sequence using representative codons.

    Args:
        aa_sequence: Amino acid sequence string
        encoder: Hyperbolic codon encoder

    Returns:
        Array of embeddings, shape (n_aa, 16)
    """
    embeddings = []

    for aa in aa_sequence.upper():
        codon = AA_TO_CODON.get(aa)
        if codon:
            onehot = codon_to_onehot(codon)
            emb = encoder.encode_numpy(onehot)
            embeddings.append(emb.squeeze())
        else:
            # Unknown amino acid - use zero embedding
            embeddings.append(np.zeros(16))

    return np.array(embeddings)


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def process_mutation_list(
    mutations: list[str], encoder: HyperbolicCodonEncoder, method: str = "representative"
) -> pd.DataFrame:
    """
    Process a list of mutations and calculate hyperbolic distances.

    Args:
        mutations: List of mutation strings (e.g., ['D30N', 'M46I'])
        encoder: Hyperbolic codon encoder
        method: Encoding method

    Returns:
        DataFrame with mutation details and distances
    """
    records = []

    for mut_str in mutations:
        parsed = parse_mutation(mut_str)
        if parsed is None:
            continue

        result = encode_mutation_pair(parsed["wild_type"], parsed["mutant"], encoder, method)
        if result is None:
            continue

        record = {
            "mutation": mut_str,
            "position": parsed["position"],
            "wild_type_aa": parsed["wild_type"],
            "mutant_aa": parsed["mutant"],
            "hyperbolic_distance": result.get("hyperbolic_distance") or result.get("mean_distance"),
        }

        if method == "average":
            record["min_distance"] = result["min_distance"]
            record["max_distance"] = result["max_distance"]
        else:
            record["wild_type_codon"] = result["wild_type_codon"]
            record["mutant_codon"] = result["mutant_codon"]

        records.append(record)

    return pd.DataFrame(records)


def process_stanford_record(row: pd.Series, protein: str, encoder: HyperbolicCodonEncoder) -> list[dict]:
    """
    Process a Stanford HIVDB record and extract mutation distances.

    Args:
        row: DataFrame row from Stanford data
        protein: 'PR', 'RT', or 'IN'
        encoder: Hyperbolic codon encoder

    Returns:
        List of mutation analysis dictionaries
    """
    mut_list = row.get("CompMutList", "")
    if pd.isna(mut_list) or not mut_list:
        return []

    # Parse mutations
    mutations = []
    for mut in str(mut_list).split(","):
        mut = mut.strip()
        if not mut:
            continue
        parsed = parse_mutation(mut)
        if parsed:
            mutations.append(parsed)

    results = []
    for mut in mutations:
        # Get actual amino acid from sequence columns if available
        col_prefix = {"PR": "P", "RT": "RT", "IN": "IN"}.get(protein.upper(), "P")
        col_name = f"{col_prefix}{mut['position']}"

        actual_aa = row.get(col_name)
        if pd.notna(actual_aa) and actual_aa != "-":
            # Use actual mutant AA from sequence
            wild_aa = mut["wild_type"]
            mut_aa = actual_aa
        else:
            wild_aa = mut["wild_type"]
            mut_aa = mut["mutant"]

        # Encode and calculate distance
        result = encode_mutation_pair(wild_aa, mut_aa, encoder)
        if result:
            results.append(
                {
                    "seq_id": row.get("SeqID"),
                    "protein": protein,
                    "position": mut["position"],
                    "wild_type_aa": wild_aa,
                    "mutant_aa": mut_aa,
                    "hyperbolic_distance": result["hyperbolic_distance"],
                }
            )

    return results


# ============================================================================
# GLYCAN SITE ANALYSIS
# ============================================================================


def find_glycan_sites(aa_sequence: str) -> list[int]:
    """
    Find N-glycosylation sites (N-X-S/T motif where X != P).

    Args:
        aa_sequence: Amino acid sequence

    Returns:
        List of positions (0-indexed) of potential glycan sites
    """
    sites = []
    seq = aa_sequence.upper()

    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in ["S", "T"]:
            sites.append(i)

    return sites


def encode_glycan_sites(
    aa_sequence: str, encoder: HyperbolicCodonEncoder, context_length: int = 3
) -> list[dict]:
    """
    Encode regions around glycan sites.

    Args:
        aa_sequence: Amino acid sequence
        encoder: Hyperbolic codon encoder
        context_length: Number of residues on each side of NXT motif

    Returns:
        List of dictionaries with site info and embeddings
    """
    sites = find_glycan_sites(aa_sequence)
    results = []

    for site in sites:
        start = max(0, site - context_length)
        end = min(len(aa_sequence), site + 3 + context_length)

        context_seq = aa_sequence[start:end]
        embeddings = encode_amino_acid_sequence(context_seq, encoder)

        results.append(
            {
                "position": site,
                "motif": aa_sequence[site : site + 3],
                "context": context_seq,
                "embeddings": embeddings,
                "mean_embedding": embeddings.mean(axis=0),
            }
        )

    return results


# ============================================================================
# MAIN - TEST FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Codon Extraction Utilities - Test Suite")
    print("=" * 60)

    # Test codon tables
    print("\n1. Codon Tables:")
    print(f"   Leucine codons: {get_codons_for_aa('L')}")
    print(f"   Synonymous to CTG: {get_synonymous_codons('CTG')}")

    # Test mutation codon extraction
    print("\n2. Mutation Codon Extraction:")
    result = mutation_to_codons("D30N")
    print("   D30N:")
    print(f"     Wild-type (D) codons: {result['wild_type_codons']}")
    print(f"     Mutant (N) codons: {result['mutant_codons']}")

    # Test single-nt transitions
    print("\n3. Single Nucleotide Transitions:")
    single_nt = get_single_nt_transitions("D", "N")
    print(f"   D -> N with single nt change: {single_nt}")

    # Test glycan site detection
    print("\n4. Glycan Site Detection:")
    test_seq = "MNVTSLLIVNGSQLFLYCVHQRIDV"
    sites = find_glycan_sites(test_seq)
    print(f"   Sequence: {test_seq}")
    print(f"   Glycan sites: {sites}")
    for site in sites:
        print(f"     Position {site}: {test_seq[site:site+3]}")

    # Test hyperbolic encoding (if encoder available)
    print("\n5. Hyperbolic Encoding:")
    try:
        encoder, _ = load_hyperbolic_encoder()
        result = encode_mutation_pair("D", "N", encoder)
        print(f"   D -> N hyperbolic distance: {result['hyperbolic_distance']:.4f}")
        print(f"   Codon transition: {result['wild_type_codon']} -> {result['mutant_codon']}")
    except FileNotFoundError:
        print("   (Codon encoder not found - skipping hyperbolic tests)")

    print("\n" + "=" * 60)
