# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Constants for arbovirus surveillance and primer design.

This module contains virus definitions, NCBI taxonomy IDs,
primer design constraints, conserved region definitions,
and reference data for validation.

Updated 2026-01-03: Added RefSeq accessions, phylogenetic identity matrix,
and validated primer sequences for Phase 1 validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# NCBI Taxonomy IDs for arboviruses
ARBOVIRUS_TAXIDS: dict[str, int] = {
    "DENV-1": 11053,  # Dengue virus 1
    "DENV-2": 11060,  # Dengue virus 2
    "DENV-3": 11069,  # Dengue virus 3
    "DENV-4": 11070,  # Dengue virus 4
    "ZIKV": 64320,    # Zika virus
    "CHIKV": 37124,   # Chikungunya virus
    "MAYV": 59300,    # Mayaro virus
    "YFV": 11089,     # Yellow fever virus
    "WNV": 11082,     # West Nile virus
}

# Arbovirus target definitions
ARBOVIRUS_TARGETS: dict[str, dict] = {
    "DENV-1": {
        "full_name": "Dengue virus serotype 1",
        "taxid": 11053,
        "genome_size": 10700,
        "gene_regions": {
            "5UTR": (1, 100),
            "C": (95, 437),
            "prM": (438, 935),
            "E": (936, 2421),
            "NS1": (2422, 3477),
            "NS2A": (3478, 4132),
            "NS2B": (4133, 4522),
            "NS3": (4523, 6380),
            "NS4A": (6381, 6764),
            "NS4B": (6765, 7521),
            "NS5": (7522, 10270),
            "3UTR": (10271, 10735),
        },
        "conserved_regions": [(100, 400), (2500, 3200), (7600, 8400)],
        "ncbi_query": "Dengue virus 1[Organism] AND complete genome[Title]",
    },
    "DENV-2": {
        "full_name": "Dengue virus serotype 2",
        "taxid": 11060,
        "genome_size": 10700,
        "gene_regions": {
            "5UTR": (1, 96),
            "C": (97, 438),
            "prM": (439, 936),
            "E": (937, 2422),
            "NS1": (2423, 3478),
            "NS3": (4524, 6381),
            "NS5": (7523, 10271),
        },
        "conserved_regions": [(100, 400), (2500, 3200), (7600, 8400)],
        "ncbi_query": "Dengue virus 2[Organism] AND complete genome[Title]",
    },
    "DENV-3": {
        "full_name": "Dengue virus serotype 3",
        "taxid": 11069,
        "genome_size": 10700,
        "conserved_regions": [(100, 400), (2500, 3200), (7600, 8400)],
        "ncbi_query": "Dengue virus 3[Organism] AND complete genome[Title]",
    },
    "DENV-4": {
        "full_name": "Dengue virus serotype 4",
        "taxid": 11070,
        "genome_size": 10700,
        "conserved_regions": [(100, 400), (2500, 3200), (7600, 8400)],
        "ncbi_query": "Dengue virus 4[Organism] AND complete genome[Title]",
    },
    "ZIKV": {
        "full_name": "Zika virus",
        "taxid": 64320,
        "genome_size": 10800,
        "gene_regions": {
            "5UTR": (1, 107),
            "C": (108, 473),
            "prM": (474, 977),
            "E": (978, 2489),
            "NS1": (2490, 3545),
            "NS3": (4600, 6457),
            "NS5": (7598, 10346),
        },
        "conserved_regions": [(150, 450), (2600, 3300), (7700, 8500)],
        "ncbi_query": "Zika virus[Organism] AND complete genome[Title]",
    },
    "CHIKV": {
        "full_name": "Chikungunya virus",
        "taxid": 37124,
        "genome_size": 11800,
        "gene_regions": {
            "nsP1": (77, 1672),
            "nsP2": (1673, 4039),
            "nsP3": (4040, 5584),
            "nsP4": (5585, 7399),
            "C": (7566, 8345),
            "E3": (8346, 8534),
            "E2": (8535, 9812),
            "E1": (9967, 11271),
        },
        "conserved_regions": [(200, 500), (4200, 4800), (8600, 9200)],
        "ncbi_query": "Chikungunya virus[Organism] AND complete genome[Title]",
    },
    "MAYV": {
        "full_name": "Mayaro virus",
        "taxid": 59300,
        "genome_size": 11400,
        "conserved_regions": [(200, 500), (4100, 4700), (8400, 9000)],
        "ncbi_query": "Mayaro virus[Organism] AND complete genome[Title]",
    },
}

# Primer design constraints for RT-PCR
PRIMER_CONSTRAINTS: dict[str, dict] = {
    "length": {
        "min": 18,
        "max": 25,
        "optimal": 20,
    },
    "gc_content": {
        "min": 0.40,
        "max": 0.60,
        "optimal": 0.50,
    },
    "tm": {
        "min": 55.0,
        "max": 65.0,
        "optimal": 60.0,
        "max_diff": 2.0,  # Between F and R primers
    },
    "amplicon": {
        "min": 80,
        "max": 300,
        "optimal": 150,
    },
    "self_complementarity": {
        "max_3prime": 3,  # Max 3' self-complementarity
        "max_any": 6,     # Max overall self-complementarity
    },
    "gc_clamp": {
        "required": True,
        "min_gc_3prime": 1,  # At least 1 G/C in last 5 bases
        "max_gc_3prime": 3,  # At most 3 G/C in last 5 bases
    },
}

# Conserved regions for universal primer design
CONSERVED_REGIONS: dict[str, list[dict]] = {
    "flavivirus": [
        {
            "name": "NS5_RdRp",
            "description": "RNA-dependent RNA polymerase (highly conserved)",
            "approximate_position": (7500, 8500),
            "conservation_score": 0.95,
        },
        {
            "name": "NS3_helicase",
            "description": "NS3 helicase domain",
            "approximate_position": (5000, 5800),
            "conservation_score": 0.90,
        },
        {
            "name": "E_fusion",
            "description": "E protein fusion loop",
            "approximate_position": (1800, 2200),
            "conservation_score": 0.85,
        },
    ],
    "alphavirus": [
        {
            "name": "nsP4_RdRp",
            "description": "RNA-dependent RNA polymerase",
            "approximate_position": (5500, 7300),
            "conservation_score": 0.92,
        },
        {
            "name": "nsP1_methyl",
            "description": "Methyltransferase domain",
            "approximate_position": (100, 1500),
            "conservation_score": 0.88,
        },
    ],
}

# Geographic regions for surveillance
SURVEILLANCE_REGIONS: dict[str, list[str]] = {
    "south_america": [
        "Paraguay", "Brazil", "Argentina", "Colombia", "Peru",
        "Ecuador", "Venezuela", "Bolivia", "Uruguay", "Chile",
    ],
    "central_america": [
        "Mexico", "Guatemala", "Honduras", "El Salvador",
        "Nicaragua", "Costa Rica", "Panama",
    ],
    "caribbean": [
        "Puerto Rico", "Cuba", "Dominican Republic", "Jamaica",
        "Haiti", "Trinidad and Tobago",
    ],
    "asia_pacific": [
        "Thailand", "Vietnam", "Philippines", "Indonesia",
        "Malaysia", "Singapore", "India", "Sri Lanka",
    ],
}

# Nucleotide scoring matrices
NUCLEOTIDE_COMPLEMENT: dict[str, str] = {
    "A": "T", "T": "A", "G": "C", "C": "G",
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D", "N": "N",
}

# IUPAC ambiguity codes
IUPAC_CODES: dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}

# ============================================================================
# REFERENCE SEQUENCES (NCBI RefSeq)
# ============================================================================

REFSEQ_ACCESSIONS: dict[str, str] = {
    "DENV-1": "NC_001477",
    "DENV-2": "NC_001474",
    "DENV-3": "NC_001475",
    "DENV-4": "NC_002640",
    "ZIKV": "NC_012532",
    "CHIKV": "NC_004162",
    "MAYV": "NC_003417",
}

# ============================================================================
# PHYLOGENETIC IDENTITY MATRIX (amino acid level)
# Based on polyprotein alignments from published studies
# ============================================================================

PHYLOGENETIC_IDENTITY: dict[tuple[str, str], float] = {
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


def get_phylogenetic_identity(virus1: str, virus2: str) -> float:
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
    if key in PHYLOGENETIC_IDENTITY:
        return PHYLOGENETIC_IDENTITY[key]

    # Try reverse order
    key = (virus2, virus1)
    if key in PHYLOGENETIC_IDENTITY:
        return PHYLOGENETIC_IDENTITY[key]

    # Unknown pair - assume distant
    return 0.15


# ============================================================================
# VALIDATED PRIMER SEQUENCES (Ground Truth from CDC/PAHO)
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


# CDC validated primers (Lanciotti et al., 2008; Santiago 2013)
# NOTE: Gene targets verified against RefSeq genomes 2026-01-03
CDC_PRIMERS: list[ValidatedPrimer] = [
    ValidatedPrimer(
        name="CDC_DENV1",
        target_virus="DENV-1",
        target_gene="NS5",  # Verified: matches at pos ~8972 in NS5 region
        forward="CAAAAGGAAGTCGTGCAATA",
        reverse="CTGAGTGAATTCTCTCTACTGAACC",
        amplicon_size=124,
        citation="Lanciotti 1992; Santiago 2013",
        specificity=["DENV-1"],
    ),
    ValidatedPrimer(
        name="CDC_DENV2",
        target_virus="DENV-2",
        target_gene="5'UTR/C",  # Verified: matches at pos ~141 in 5' region
        forward="CGAAAACGCGAGAGAAACCG",
        reverse="CTTCAACATCCTGCCAGCTC",
        amplicon_size=119,
        citation="Lanciotti 1992; Santiago 2013",
        specificity=["DENV-2"],
    ),
    ValidatedPrimer(
        name="CDC_DENV3",
        target_virus="DENV-3",
        target_gene="NS5",  # Low match - may require strain-specific update
        forward="GGATGATCTCAACAAAGAGGTG",
        reverse="CCCAACATCAATTCCTACTCAA",
        amplicon_size=123,
        citation="Lanciotti 1992; Santiago 2013",
        specificity=["DENV-3"],
    ),
    ValidatedPrimer(
        name="CDC_DENV4",
        target_virus="DENV-4",
        target_gene="prM/E",  # Verified: 100% match at pos 903-972
        forward="TTGTCCTAATGATGCTGGTCG",
        reverse="TCCACCTGAGACTCCTTCCA",
        amplicon_size=119,
        citation="Lanciotti 1992; Santiago 2013",
        specificity=["DENV-4"],
    ),
    ValidatedPrimer(
        name="Lanciotti_ZIKV",
        target_virus="ZIKV",
        target_gene="NS5",  # Verified: 100% match at pos 9364 in NS5
        forward="AARTACACATACCARAACAAAGTGGT",  # Degenerate
        reverse="TCCRCTCCCYCTYTGGTCTTG",  # Degenerate
        amplicon_size=117,
        citation="Lanciotti 2008",
        specificity=["ZIKV"],
    ),
]

# Pan-flavivirus primers (intentionally cross-reactive, for negative control)
PANFLAVIVIRUS_PRIMERS: list[ValidatedPrimer] = [
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
