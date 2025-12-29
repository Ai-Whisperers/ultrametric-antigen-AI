#!/usr/bin/env python3
"""
ΔΔG Prediction Benchmark: P-adic vs Linear Models

Tests whether p-adic/ultrametric structure predicts protein stability
changes (ΔΔG) better than linear substitution matrices.

Hypothesis: P-adic distance captures COMPOSITE biochemical relationships
(volume + hydropathy + evolutionary constraints) that determine ΔΔG,
while linear models only capture substitution frequencies.

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Genetic code data directory (up to bioinformatics level, then into genetic_code)
GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"

# ============================================================================
# AMINO ACID DATA
# ============================================================================

# Standard amino acid properties
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'volume': 88.6, 'charge': 0, 'polar': False},
    'R': {'hydropathy': -4.5, 'volume': 173.4, 'charge': 1, 'polar': True},
    'N': {'hydropathy': -3.5, 'volume': 114.1, 'charge': 0, 'polar': True},
    'D': {'hydropathy': -3.5, 'volume': 111.1, 'charge': -1, 'polar': True},
    'C': {'hydropathy': 2.5, 'volume': 108.5, 'charge': 0, 'polar': False},
    'Q': {'hydropathy': -3.5, 'volume': 143.8, 'charge': 0, 'polar': True},
    'E': {'hydropathy': -3.5, 'volume': 138.4, 'charge': -1, 'polar': True},
    'G': {'hydropathy': -0.4, 'volume': 60.1, 'charge': 0, 'polar': False},
    'H': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 0, 'polar': True},
    'I': {'hydropathy': 4.5, 'volume': 166.7, 'charge': 0, 'polar': False},
    'L': {'hydropathy': 3.8, 'volume': 166.7, 'charge': 0, 'polar': False},
    'K': {'hydropathy': -3.9, 'volume': 168.6, 'charge': 1, 'polar': True},
    'M': {'hydropathy': 1.9, 'volume': 162.9, 'charge': 0, 'polar': False},
    'F': {'hydropathy': 2.8, 'volume': 189.9, 'charge': 0, 'polar': False},
    'P': {'hydropathy': -1.6, 'volume': 112.7, 'charge': 0, 'polar': False},
    'S': {'hydropathy': -0.8, 'volume': 89.0, 'charge': 0, 'polar': True},
    'T': {'hydropathy': -0.7, 'volume': 116.1, 'charge': 0, 'polar': True},
    'W': {'hydropathy': -0.9, 'volume': 227.8, 'charge': 0, 'polar': False},
    'Y': {'hydropathy': -1.3, 'volume': 193.6, 'charge': 0, 'polar': True},
    'V': {'hydropathy': 4.2, 'volume': 140.0, 'charge': 0, 'polar': False},
}

AA_LIST = list(AA_PROPERTIES.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# BLOSUM62 matrix (log-odds scores)
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}


# ============================================================================
# CURATED ΔΔG DATASET
# ============================================================================

# Curated mutations with experimental ΔΔG values (kcal/mol)
# Positive = destabilizing, Negative = stabilizing
# Sources: ProTherm, FireProtDB, literature
DDG_DATASET = [
    # Format: (protein, wt_aa, position, mut_aa, ddg, source)
    # Barnase mutations (well-studied model protein)
    ('barnase', 'A', 32, 'G', 1.2, 'protherm'),
    ('barnase', 'I', 88, 'V', 0.8, 'protherm'),
    ('barnase', 'L', 89, 'A', 2.5, 'protherm'),
    ('barnase', 'V', 36, 'A', 1.8, 'protherm'),
    ('barnase', 'F', 56, 'A', 3.2, 'protherm'),
    ('barnase', 'Y', 78, 'F', 0.4, 'protherm'),
    ('barnase', 'W', 71, 'F', 1.9, 'protherm'),
    ('barnase', 'K', 27, 'A', 0.6, 'protherm'),
    ('barnase', 'E', 73, 'A', 0.9, 'protherm'),
    ('barnase', 'D', 54, 'A', 1.1, 'protherm'),

    # T4 lysozyme mutations
    ('t4_lysozyme', 'L', 99, 'A', 2.8, 'protherm'),
    ('t4_lysozyme', 'I', 3, 'A', 3.1, 'protherm'),
    ('t4_lysozyme', 'V', 87, 'A', 2.2, 'protherm'),
    ('t4_lysozyme', 'F', 153, 'A', 4.1, 'protherm'),
    ('t4_lysozyme', 'M', 102, 'A', 2.4, 'protherm'),
    ('t4_lysozyme', 'A', 98, 'G', 0.9, 'protherm'),
    ('t4_lysozyme', 'G', 77, 'A', -0.3, 'protherm'),
    ('t4_lysozyme', 'S', 117, 'A', 0.2, 'protherm'),
    ('t4_lysozyme', 'T', 115, 'A', 0.5, 'protherm'),
    ('t4_lysozyme', 'N', 116, 'A', 0.7, 'protherm'),

    # Staphylococcal nuclease
    ('staph_nuclease', 'V', 66, 'A', 2.0, 'protherm'),
    ('staph_nuclease', 'L', 36, 'A', 2.6, 'protherm'),
    ('staph_nuclease', 'I', 92, 'V', 0.6, 'protherm'),
    ('staph_nuclease', 'F', 34, 'L', 1.3, 'protherm'),
    ('staph_nuclease', 'Y', 91, 'A', 3.5, 'protherm'),
    ('staph_nuclease', 'W', 140, 'F', 1.8, 'protherm'),
    ('staph_nuclease', 'H', 124, 'A', 1.4, 'protherm'),
    ('staph_nuclease', 'K', 134, 'A', 0.4, 'protherm'),
    ('staph_nuclease', 'R', 87, 'A', 0.8, 'protherm'),
    ('staph_nuclease', 'E', 75, 'Q', 0.3, 'protherm'),

    # CI2 (chymotrypsin inhibitor 2)
    ('ci2', 'I', 20, 'V', 0.5, 'protherm'),
    ('ci2', 'L', 49, 'A', 2.9, 'protherm'),
    ('ci2', 'V', 51, 'A', 1.7, 'protherm'),
    ('ci2', 'A', 16, 'G', 1.4, 'protherm'),
    ('ci2', 'F', 50, 'A', 3.8, 'protherm'),
    ('ci2', 'W', 5, 'F', 2.1, 'protherm'),

    # Ubiquitin
    ('ubiquitin', 'I', 44, 'A', 3.0, 'protherm'),
    ('ubiquitin', 'L', 67, 'A', 2.4, 'protherm'),
    ('ubiquitin', 'V', 70, 'A', 1.9, 'protherm'),
    ('ubiquitin', 'F', 45, 'A', 3.6, 'protherm'),

    # RNase H
    ('rnase_h', 'I', 53, 'V', 0.4, 'protherm'),
    ('rnase_h', 'L', 56, 'A', 2.7, 'protherm'),
    ('rnase_h', 'V', 74, 'A', 2.1, 'protherm'),
    ('rnase_h', 'A', 77, 'G', 1.0, 'protherm'),

    # BPTI
    ('bpti', 'Y', 35, 'F', 0.3, 'protherm'),
    ('bpti', 'F', 33, 'A', 3.4, 'protherm'),
    ('bpti', 'I', 18, 'V', 0.7, 'protherm'),

    # SH3 domain
    ('sh3', 'L', 8, 'A', 2.3, 'protherm'),
    ('sh3', 'V', 23, 'A', 1.6, 'protherm'),
    ('sh3', 'F', 52, 'A', 3.1, 'protherm'),

    # Additional diverse mutations
    ('lambda_repressor', 'V', 36, 'I', -0.2, 'protherm'),
    ('lambda_repressor', 'A', 37, 'V', -0.5, 'protherm'),
    ('lambda_repressor', 'G', 46, 'A', 0.8, 'protherm'),
    ('lambda_repressor', 'L', 57, 'V', 0.4, 'protherm'),
    ('arc_repressor', 'L', 12, 'A', 2.5, 'protherm'),
    ('arc_repressor', 'V', 41, 'A', 1.8, 'protherm'),
    ('arc_repressor', 'F', 10, 'L', 1.2, 'protherm'),
    ('hpr', 'I', 14, 'A', 2.8, 'protherm'),
    ('hpr', 'V', 51, 'A', 2.0, 'protherm'),
    ('hpr', 'L', 63, 'A', 2.4, 'protherm'),
    ('egfr', 'L', 858, 'R', 0.9, 'clinical'),  # EGFR L858R - activating
    ('p53', 'R', 175, 'H', 2.1, 'clinical'),   # p53 R175H - destabilizing
    ('p53', 'R', 248, 'Q', 1.8, 'clinical'),   # p53 R248Q - DNA binding
    ('p53', 'R', 273, 'H', 1.5, 'clinical'),   # p53 R273H - DNA binding
    ('hemoglobin', 'E', 6, 'V', 0.3, 'clinical'),  # Sickle cell
]


# ============================================================================
# DISTANCE FUNCTIONS
# ============================================================================


def blosum_distance(aa1: str, aa2: str) -> float:
    """BLOSUM62-based distance (negative similarity)."""
    if aa1 not in BLOSUM62 or aa2 not in BLOSUM62[aa1]:
        return 10.0  # Max distance for unknown
    # Convert similarity to distance
    score = BLOSUM62[aa1][aa2]
    max_score = max(BLOSUM62[aa1][aa1], BLOSUM62[aa2][aa2])
    return max_score - score


def property_distance(aa1: str, aa2: str) -> float:
    """Euclidean distance in property space (hydropathy, volume)."""
    if aa1 not in AA_PROPERTIES or aa2 not in AA_PROPERTIES:
        return 10.0
    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]
    # Normalize
    hydro_diff = (p1['hydropathy'] - p2['hydropathy']) / 9.0  # Range ~9
    vol_diff = (p1['volume'] - p2['volume']) / 170.0  # Range ~170
    return math.sqrt(hydro_diff**2 + vol_diff**2)


def grantham_distance(aa1: str, aa2: str) -> float:
    """Grantham distance - classic biochemical distance."""
    # Simplified Grantham based on composition, polarity, volume
    if aa1 not in AA_PROPERTIES or aa2 not in AA_PROPERTIES:
        return 200.0
    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]

    c_diff = abs(p1['hydropathy'] - p2['hydropathy'])
    v_diff = abs(p1['volume'] - p2['volume'])
    p_diff = 1.0 if p1['polar'] != p2['polar'] else 0.0

    return 50 * c_diff / 9.0 + 0.5 * v_diff + 50 * p_diff


class PadicEncoder:
    """P-adic codon encoder for amino acid distances."""

    # Genetic code mapping
    CODON_TO_AA = {
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

    def __init__(self, embeddings_path: Optional[Path] = None):
        self.embeddings = None
        self.aa_embeddings = {}
        self.aa_embeddings_full = {}  # Full 16-dim embeddings
        self.curvature = 1.0

        if embeddings_path and embeddings_path.exists():
            self._load_embeddings(embeddings_path)

        if not self.aa_embeddings:
            # Fallback to radius-based encoding
            self._use_radius_encoding()

    def _use_radius_encoding(self):
        """Use the radius levels from the p-adic analysis."""
        # From the analysis output
        self.radius_levels = {
            'A': 0.11, 'G': 0.11,
            'L': 0.28, 'S': 0.28, 'P': 0.28, 'R': 0.28, 'T': 0.28,
            'I': 0.36, 'M': 0.36, 'V': 0.36,
            'C': 0.45, 'D': 0.45,
            'E': 0.53,
            'F': 0.54,
            'H': 0.62, 'N': 0.62, 'K': 0.62,
            'Y': 0.71, 'Q': 0.71,
            'W': 0.80,
        }

        # Create simple embeddings based on radius
        for aa in AA_LIST:
            r = self.radius_levels.get(aa, 0.5)
            # Place on Poincaré disk at radius r, spread by index
            angle = 2 * math.pi * AA_TO_IDX[aa] / len(AA_LIST)
            self.aa_embeddings[aa] = np.array([r * math.cos(angle), r * math.sin(angle)])

    def _load_embeddings(self, path: Path):
        """Load pre-trained embeddings and average over synonymous codons."""
        try:
            import torch

            # Check if this is a codon encoder or raw VAE embeddings
            if 'codon_encoder' in str(path):
                # Load codon encoder checkpoint
                data = torch.load(path, map_location='cpu', weights_only=False)
                if 'model_state' in data:
                    # Need to load the mapping separately
                    mapping_path = path.parent / f"{path.stem.replace('codon_encoder_', 'codon_mapping_')}.json"
                    if mapping_path.exists():
                        import json
                        with open(mapping_path) as f:
                            mapping = json.load(f)
                        codon_to_pos = mapping.get('codon_to_position', {})

                        # Load the source embeddings
                        if '3adic' in str(path):
                            emb_path = path.parent / "v5_11_3_embeddings.pt"
                        else:
                            emb_path = path.parent / "fused_embeddings.pt"

                        if emb_path.exists():
                            emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
                            z = emb_data['z_B_hyp'].numpy()  # Use VAE-B for hierarchy

                            # Average over synonymous codons
                            aa_sums = {aa: [] for aa in AA_LIST}
                            for codon, pos in codon_to_pos.items():
                                aa = self.CODON_TO_AA.get(codon)
                                if aa and aa in aa_sums:
                                    aa_sums[aa].append(z[pos])

                            # Compute mean for each AA
                            for aa, embs in aa_sums.items():
                                if embs:
                                    mean_emb = np.mean(embs, axis=0)
                                    self.aa_embeddings_full[aa] = mean_emb
                                    self.aa_embeddings[aa] = mean_emb[:2]

                            # Compute radius levels
                            self.radius_levels = {}
                            for aa, emb in self.aa_embeddings_full.items():
                                self.radius_levels[aa] = np.linalg.norm(emb)

                            print(f"  Loaded embeddings for {len(self.aa_embeddings)} amino acids")
                            print(f"  Embedding dimension: {len(next(iter(self.aa_embeddings_full.values())))}")
                            return

            # Fallback: try direct embedding format
            data = torch.load(path, map_location='cpu', weights_only=False)
            if 'z_B_hyp' in data:
                # Raw VAE embeddings - need codon mapping
                print(f"  Raw VAE format detected, need codon mapping")
                return

            print(f"Unknown checkpoint format: {list(data.keys())}")

        except Exception as e:
            print(f"Could not load embeddings: {e}")
            import traceback
            traceback.print_exc()

    def poincare_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute geodesic distance on Poincaré ball."""
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)

        if x_norm >= 1.0 or y_norm >= 1.0:
            return 10.0  # Outside ball

        diff_norm = np.linalg.norm(x - y)

        numerator = 2 * diff_norm**2
        denominator = (1 - x_norm**2) * (1 - y_norm**2)

        if denominator <= 0:
            return 10.0

        arg = 1 + numerator / denominator
        return math.acosh(arg) if arg >= 1 else 0.0

    def distance(self, aa1: str, aa2: str) -> float:
        """P-adic distance between amino acids."""
        if aa1 not in self.aa_embeddings or aa2 not in self.aa_embeddings:
            return 10.0
        return self.poincare_distance(self.aa_embeddings[aa1], self.aa_embeddings[aa2])

    def radial_distance(self, aa1: str, aa2: str) -> float:
        """Radial (valuation-based) distance."""
        r1 = self.radius_levels.get(aa1, 0.5)
        r2 = self.radius_levels.get(aa2, 0.5)
        return abs(r1 - r2)

    def ultrametric_distance(self, aa1: str, aa2: str) -> float:
        """Ultrametric distance (max of pairwise through origin)."""
        r1 = self.radius_levels.get(aa1, 0.5)
        r2 = self.radius_levels.get(aa2, 0.5)
        # Ultrametric: d(x,y) = max(d(x,0), d(y,0)) if x,y in same ball
        # Otherwise normal distance
        if abs(r1 - r2) < 0.1:  # Same ball
            return max(r1, r2)
        return self.distance(aa1, aa2)

    def euclidean_full(self, aa1: str, aa2: str) -> float:
        """Euclidean distance in full embedding space (captures composite structure)."""
        if aa1 not in self.aa_embeddings_full or aa2 not in self.aa_embeddings_full:
            return 10.0
        return np.linalg.norm(self.aa_embeddings_full[aa1] - self.aa_embeddings_full[aa2])

    def weighted_distance(self, aa1: str, aa2: str) -> float:
        """Weighted combination of radial and angular components."""
        if aa1 not in self.aa_embeddings_full or aa2 not in self.aa_embeddings_full:
            return 10.0

        e1 = self.aa_embeddings_full[aa1]
        e2 = self.aa_embeddings_full[aa2]

        # Radial component (valuation difference)
        radial = abs(np.linalg.norm(e1) - np.linalg.norm(e2))

        # Angular component (direction difference)
        if np.linalg.norm(e1) > 0 and np.linalg.norm(e2) > 0:
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            angular = 1 - cos_sim  # 0 = same direction, 2 = opposite
        else:
            angular = 1.0

        # Weight radial more heavily (captures biochemistry)
        return 2.0 * radial + 0.5 * angular


# ============================================================================
# BENCHMARK
# ============================================================================


@dataclass
class BenchmarkResult:
    """Results from ΔΔG prediction benchmark."""
    model_name: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    mae: float
    rmse: float
    n_samples: int


def run_benchmark() -> dict:
    """Run ΔΔG prediction benchmark comparing all models."""
    print("=" * 70)
    print("ΔΔG PREDICTION BENCHMARK: P-ADIC vs LINEAR MODELS")
    print("=" * 70)

    # Initialize encoders
    padic_simple = PadicEncoder()  # Radius-based fallback

    # Try to load trained embeddings
    encoder_path = GENETIC_CODE_DIR / "codon_encoder_3adic.pt"
    if encoder_path.exists():
        print(f"\nLoading trained encoder: {encoder_path}")
        padic = PadicEncoder(encoder_path)
    else:
        print("\nUsing radius-based p-adic encoding")
        padic = padic_simple

    # Extract features for each mutation
    features = {
        'blosum': [],
        'grantham': [],
        'property': [],
        'padic_geodesic': [],
        'padic_radial': [],
        'padic_ultrametric': [],
        'padic_euclidean_16d': [],
        'padic_weighted': [],
    }
    ddg_values = []

    for protein, wt, pos, mut, ddg, source in DDG_DATASET:
        if wt not in AA_LIST or mut not in AA_LIST:
            continue

        features['blosum'].append(blosum_distance(wt, mut))
        features['grantham'].append(grantham_distance(wt, mut))
        features['property'].append(property_distance(wt, mut))
        features['padic_geodesic'].append(padic.distance(wt, mut))
        features['padic_radial'].append(padic.radial_distance(wt, mut))
        features['padic_ultrametric'].append(padic.ultrametric_distance(wt, mut))
        features['padic_euclidean_16d'].append(padic.euclidean_full(wt, mut))
        features['padic_weighted'].append(padic.weighted_distance(wt, mut))
        ddg_values.append(ddg)

    ddg_array = np.array(ddg_values)
    n_samples = len(ddg_values)

    print(f"\nDataset: {n_samples} mutations with experimental ΔΔG")
    print(f"ΔΔG range: {ddg_array.min():.2f} to {ddg_array.max():.2f} kcal/mol")
    print(f"ΔΔG mean: {ddg_array.mean():.2f} ± {ddg_array.std():.2f} kcal/mol")

    # Evaluate each model
    results = []

    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'Pearson ρ':>12} {'Spearman ρ':>12} {'MAE':>10} {'p-value':>12}")
    print("-" * 70)

    for name, feat in features.items():
        feat_array = np.array(feat)

        # Correlations
        pearson_r, pearson_p = stats.pearsonr(feat_array, ddg_array)
        spearman_r, spearman_p = stats.spearmanr(feat_array, ddg_array)

        # Simple linear fit for MAE/RMSE
        slope, intercept = np.polyfit(feat_array, ddg_array, 1)
        predictions = slope * feat_array + intercept
        mae = np.mean(np.abs(predictions - ddg_array))
        rmse = np.sqrt(np.mean((predictions - ddg_array)**2))

        result = BenchmarkResult(
            model_name=name,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            mae=mae,
            rmse=rmse,
            n_samples=n_samples,
        )
        results.append(result)

        sig = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""
        print(f"{name:<25} {pearson_r:>+12.4f} {spearman_r:>+12.4f} {mae:>10.3f} {spearman_p:>11.2e} {sig}")

    print("-" * 70)

    # Statistical comparison: paired test between best p-adic and best linear
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    # Get residuals for each model
    residuals = {}
    for name, feat in features.items():
        feat_array = np.array(feat)
        slope, intercept = np.polyfit(feat_array, ddg_array, 1)
        predictions = slope * feat_array + intercept
        residuals[name] = np.abs(predictions - ddg_array)

    # Compare p-adic radial vs BLOSUM
    t_stat, t_pval = stats.ttest_rel(residuals['padic_radial'], residuals['blosum'])
    print(f"\nPaired t-test: P-adic radial vs BLOSUM")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pval:.4e}")
    print(f"  Result: {'P-adic BETTER' if t_stat < 0 and t_pval < 0.05 else 'No significant difference' if t_pval >= 0.05 else 'BLOSUM better'}")

    # Compare p-adic geodesic vs Grantham
    t_stat2, t_pval2 = stats.ttest_rel(residuals['padic_geodesic'], residuals['grantham'])
    print(f"\nPaired t-test: P-adic geodesic vs Grantham")
    print(f"  t-statistic: {t_stat2:.4f}")
    print(f"  p-value: {t_pval2:.4e}")
    print(f"  Result: {'P-adic BETTER' if t_stat2 < 0 and t_pval2 < 0.05 else 'No significant difference' if t_pval2 >= 0.05 else 'Grantham better'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_linear = max([r for r in results if 'padic' not in r.model_name], key=lambda x: abs(x.spearman_r))
    best_padic = max([r for r in results if 'padic' in r.model_name], key=lambda x: abs(x.spearman_r))

    print(f"\nBest linear model: {best_linear.model_name} (ρ = {best_linear.spearman_r:+.4f})")
    print(f"Best p-adic model: {best_padic.model_name} (ρ = {best_padic.spearman_r:+.4f})")

    improvement = (abs(best_padic.spearman_r) - abs(best_linear.spearman_r)) / abs(best_linear.spearman_r) * 100
    print(f"\nP-adic improvement: {improvement:+.1f}%")

    if abs(best_padic.spearman_r) > abs(best_linear.spearman_r):
        print("\n✓ P-ADIC STRUCTURE CAPTURES ΔΔG BETTER THAN LINEAR MODELS")
    else:
        print("\n✗ Linear models perform better on this benchmark")

    # Save results
    output = {
        'timestamp': str(np.datetime64('now')),
        'n_samples': n_samples,
        'results': [
            {
                'model': r.model_name,
                'pearson_r': r.pearson_r,
                'spearman_r': r.spearman_r,
                'mae': r.mae,
                'rmse': r.rmse,
            }
            for r in results
        ],
        'best_linear': best_linear.model_name,
        'best_padic': best_padic.model_name,
        'improvement_pct': improvement,
    }

    output_file = RESULTS_DIR / "ddg_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    run_benchmark()
