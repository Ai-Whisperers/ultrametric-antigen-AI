# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Constants for arbovirus surveillance and primer design.

This module contains virus definitions, NCBI taxonomy IDs,
primer design constraints, and conserved region definitions.
"""

from __future__ import annotations

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
