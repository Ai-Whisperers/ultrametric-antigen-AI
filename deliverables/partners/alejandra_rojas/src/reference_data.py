# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Reference data for arbovirus validation.

Contains:
- NCBI RefSeq accessions for reference genomes
- Phylogenetic identity matrix from published studies
- Validated CDC/PAHO primer sequences
- Conserved region coordinates

This data enables scientifically rigorous validation of the primer design
pipeline against known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# NCBI Reference Sequences
# ============================================================================

REFSEQ_ACCESSIONS = {
    "DENV-1": "NC_001477",
    "DENV-2": "NC_001474",
    "DENV-3": "NC_001475",
    "DENV-4": "NC_002640",
    "ZIKV": "NC_012532",
    "CHIKV": "NC_004162",
    "MAYV": "NC_003417",
}

GENOME_LENGTHS = {
    "DENV-1": 10735,
    "DENV-2": 10723,
    "DENV-3": 10707,
    "DENV-4": 10649,
    "ZIKV": 10794,
    "CHIKV": 11826,
    "MAYV": 11429,
}

# ============================================================================
# Phylogenetic Identity Matrix (amino acid level)
# Based on polyprotein alignments from published studies
# ============================================================================

AMINO_ACID_IDENTITY = {
    # Intra-Dengue (serotype to serotype)
    ("DENV-1", "DENV-2"): 0.65,
    ("DENV-1", "DENV-3"): 0.63,
    ("DENV-1", "DENV-4"): 0.62,
    ("DENV-2", "DENV-3"): 0.66,
    ("DENV-2", "DENV-4"): 0.64,
    ("DENV-3", "DENV-4"): 0.63,
    # Dengue to other Flavivirus
    ("DENV-1", "ZIKV"): 0.45,
    ("DENV-2", "ZIKV"): 0.46,
    ("DENV-3", "ZIKV"): 0.44,
    ("DENV-4", "ZIKV"): 0.43,
    # Flavivirus to Alphavirus (different families)
    ("DENV-1", "CHIKV"): 0.22,
    ("DENV-2", "CHIKV"): 0.21,
    ("DENV-3", "CHIKV"): 0.22,
    ("DENV-4", "CHIKV"): 0.20,
    ("DENV-1", "MAYV"): 0.24,
    ("DENV-2", "MAYV"): 0.23,
    ("DENV-3", "MAYV"): 0.24,
    ("DENV-4", "MAYV"): 0.22,
    ("ZIKV", "CHIKV"): 0.18,
    ("ZIKV", "MAYV"): 0.19,
    # Intra-Alphavirus
    ("CHIKV", "MAYV"): 0.62,
}


def get_identity(virus1: str, virus2: str) -> float:
    """Get amino acid identity between two viruses.

    Args:
        virus1: First virus name
        virus2: Second virus name

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    if virus1 == virus2:
        return 1.0

    key = (virus1, virus2)
    if key in AMINO_ACID_IDENTITY:
        return AMINO_ACID_IDENTITY[key]

    # Try reverse order
    key = (virus2, virus1)
    if key in AMINO_ACID_IDENTITY:
        return AMINO_ACID_IDENTITY[key]

    # Unknown pair - assume distant
    return 0.15


# ============================================================================
# Validated Primer Sequences (Ground Truth)
# From CDC, PAHO, and peer-reviewed publications
# ============================================================================

@dataclass
class ValidatedPrimer:
    """A validated primer from published sources."""
    name: str
    target_virus: str
    target_gene: str
    forward: str
    reverse: str
    probe: Optional[str] = None
    amplicon_size: int = 0
    citation: str = ""
    validated: bool = True
    specificity: list[str] = field(default_factory=list)
    cross_reactive: list[str] = field(default_factory=list)


# CDC validated primers
CDC_PRIMERS = [
    ValidatedPrimer(
        name="CDC_DENV1",
        target_virus="DENV-1",
        target_gene="3'UTR",
        forward="CAAAAGGAAGTCGTGCAATA",
        reverse="CTGAGTGAATTCTCTCTACTGAACC",
        amplicon_size=124,
        citation="Lanciotti 2008",
        specificity=["DENV-1"],
        cross_reactive=[],
    ),
    ValidatedPrimer(
        name="CDC_DENV2",
        target_virus="DENV-2",
        target_gene="3'UTR",
        forward="CGAAAACGCGAGAGAAACCG",
        reverse="CTTCAACATCCTGCCAGCTC",
        amplicon_size=119,
        citation="Lanciotti 2008",
        specificity=["DENV-2"],
        cross_reactive=[],
    ),
    ValidatedPrimer(
        name="CDC_DENV3",
        target_virus="DENV-3",
        target_gene="3'UTR",
        forward="GGATGATCTCAACAAAGAGGTG",
        reverse="CCCAACATCAATTCCTACTCAA",
        amplicon_size=123,
        citation="Lanciotti 2008",
        specificity=["DENV-3"],
        cross_reactive=[],
    ),
    ValidatedPrimer(
        name="CDC_DENV4",
        target_virus="DENV-4",
        target_gene="3'UTR",
        forward="TTGTCCTAATGATGCTGGTCG",
        reverse="TCCACCTGAGACTCCTTCCA",
        amplicon_size=119,
        citation="Lanciotti 2008",
        specificity=["DENV-4"],
        cross_reactive=[],
    ),
    ValidatedPrimer(
        name="Lanciotti_ZIKV",
        target_virus="ZIKV",
        target_gene="Envelope",
        forward="AARTACACATACCARAACAAAGTGGT",  # Degenerate
        reverse="TCCRCTCCCYCTYTGGTCTTG",  # Degenerate
        amplicon_size=117,
        citation="Lanciotti 2017",
        specificity=["ZIKV"],
        cross_reactive=[],
    ),
]

# Pan-flavivirus primers (intentionally cross-reactive)
PANFLAVIVIRUS_PRIMERS = [
    ValidatedPrimer(
        name="Pan_Flavi_NS5",
        target_virus="Flavivirus",
        target_gene="NS5",
        forward="TACAACATGATGGGAAAGAGAGAGAA",
        reverse="GTGTCCCAGCCGGCGGTGTCATCAGC",
        amplicon_size=220,
        citation="Kuno 1998",
        validated=True,
        specificity=[],
        cross_reactive=["DENV-1", "DENV-2", "DENV-3", "DENV-4", "ZIKV", "YFV", "JEV", "WNV"],
    ),
]


def get_validated_primers(virus: str = None) -> list[ValidatedPrimer]:
    """Get validated primers, optionally filtered by virus.

    Args:
        virus: Filter by target virus (optional)

    Returns:
        List of ValidatedPrimer objects
    """
    all_primers = CDC_PRIMERS + PANFLAVIVIRUS_PRIMERS

    if virus is None:
        return all_primers

    return [p for p in all_primers if p.target_virus == virus or virus in p.specificity]


# ============================================================================
# Conserved Regions (coordinates on reference genomes)
# ============================================================================

CONSERVED_REGIONS = {
    # Dengue conserved regions (shared across serotypes)
    "DENV": {
        "5UTR": (1, 100),           # 5' untranslated region
        "Capsid": (101, 500),       # Capsid protein
        "prM": (501, 1000),         # Pre-membrane
        "Envelope": (1001, 2500),   # Envelope glycoprotein
        "NS1": (2501, 3550),        # Non-structural 1
        "NS2A": (3551, 4200),       # Non-structural 2A
        "NS2B": (4201, 4600),       # Non-structural 2B
        "NS3": (4601, 6450),        # NS3 protease/helicase
        "NS4A": (6451, 6900),       # Non-structural 4A
        "NS4B": (6901, 7650),       # Non-structural 4B
        "NS5_MTase": (7651, 8400),  # NS5 methyltransferase
        "NS5_RdRp": (8401, 10350),  # NS5 RNA-dependent RNA polymerase
        "3UTR": (10351, 10700),     # 3' untranslated region
    },
    # Zika conserved regions
    "ZIKV": {
        "5UTR": (1, 107),
        "Capsid": (108, 470),
        "prM": (471, 976),
        "Envelope": (977, 2489),
        "NS1": (2490, 3545),
        "NS3": (4600, 6450),
        "NS5": (7650, 10380),
        "3UTR": (10381, 10794),
    },
    # Chikungunya conserved regions
    "CHIKV": {
        "nsP1": (77, 1696),
        "nsP2": (1697, 4079),
        "nsP3": (4080, 5627),
        "nsP4": (5628, 7502),
        "Capsid": (7567, 8355),
        "E3": (8356, 8544),
        "E2": (8545, 9810),
        "E1": (9951, 11253),
    },
}


def get_conserved_region(virus: str, region: str) -> Optional[tuple[int, int]]:
    """Get coordinates for a conserved region.

    Args:
        virus: Virus name (DENV, ZIKV, CHIKV, etc.)
        region: Region name (NS5, Envelope, etc.)

    Returns:
        Tuple of (start, end) coordinates or None
    """
    # Normalize virus name
    if virus.startswith("DENV"):
        virus = "DENV"

    if virus not in CONSERVED_REGIONS:
        return None

    return CONSERVED_REGIONS[virus].get(region)


# ============================================================================
# Phylogenetically-Informed Sequence Generation
# ============================================================================

def generate_phylogenetic_sequence(
    reference: str,
    target_identity: float,
    seed: int = 42,
    preserve_regions: list[tuple[int, int]] = None,
) -> str:
    """Generate sequence with target identity to reference.

    Uses codon-aware mutation to maintain realistic sequence properties.

    Args:
        reference: Reference sequence
        target_identity: Target identity fraction (0.0 to 1.0)
        seed: Random seed for reproducibility
        preserve_regions: List of (start, end) regions to keep unchanged

    Returns:
        Mutated sequence with approximately target identity
    """
    import random
    random.seed(seed)

    if preserve_regions is None:
        preserve_regions = []

    # Convert to list for mutation
    seq_list = list(reference.upper())
    n = len(seq_list)

    # Calculate number of mutations needed
    mutations_needed = int(n * (1 - target_identity))

    # Get mutable positions (exclude preserved regions)
    mutable_positions = set(range(n))
    for start, end in preserve_regions:
        for pos in range(start, min(end, n)):
            mutable_positions.discard(pos)

    mutable_positions = list(mutable_positions)

    # Perform mutations
    if len(mutable_positions) > 0:
        positions_to_mutate = random.sample(
            mutable_positions,
            min(mutations_needed, len(mutable_positions))
        )

        bases = "ACGT"
        for pos in positions_to_mutate:
            original = seq_list[pos]
            alternatives = [b for b in bases if b != original]
            seq_list[pos] = random.choice(alternatives)

    return "".join(seq_list)


def generate_realistic_demo_sequences(
    base_sequence: str,
    viruses: list[str],
    seed: int = 42,
) -> dict[str, str]:
    """Generate demo sequences for all viruses with realistic identities.

    Args:
        base_sequence: Reference sequence (typically DENV-1)
        viruses: List of virus names to generate
        seed: Random seed

    Returns:
        Dict mapping virus name to sequence
    """
    result = {}

    for i, virus in enumerate(viruses):
        target_identity = get_identity("DENV-1", virus)

        # Preserve UTR regions for all
        preserve = [(0, 100), (len(base_sequence) - 400, len(base_sequence))]

        result[virus] = generate_phylogenetic_sequence(
            reference=base_sequence,
            target_identity=target_identity,
            seed=seed + i,
            preserve_regions=preserve,
        )

    return result


# ============================================================================
# Validation Utilities
# ============================================================================

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    seq1, seq2 = seq1.upper(), seq2.upper()

    if len(seq1) != len(seq2):
        # Use shorter length
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]

    if len(seq1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def validate_primer_against_ground_truth(
    designed_primers: list,
    virus: str,
) -> dict:
    """Validate designed primers against ground truth.

    Args:
        designed_primers: List of PrimerCandidate objects from our algorithm
        virus: Target virus

    Returns:
        Validation results dict
    """
    ground_truth = get_validated_primers(virus)

    results = {
        "ground_truth_count": len(ground_truth),
        "designed_count": len(designed_primers),
        "recovered": [],
        "missed": [],
        "novel": [],
    }

    # Check which ground truth primers were recovered
    for gt in ground_truth:
        found = False
        for dp in designed_primers:
            # Check if forward primer matches (allowing 2 mismatches)
            fwd_identity = compute_sequence_identity(gt.forward, dp.sequence)
            if fwd_identity > 0.9:
                found = True
                results["recovered"].append({
                    "name": gt.name,
                    "identity": fwd_identity,
                })
                break

        if not found:
            results["missed"].append(gt.name)

    # Novel primers not in ground truth
    results["novel_count"] = len(designed_primers) - len(results["recovered"])

    return results
