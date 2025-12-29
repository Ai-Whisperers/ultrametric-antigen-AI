#!/usr/bin/env python3
"""
PyTorch ΔΔG Predictor with Hyperparameter Search

Extensive hyperparameter search for p-adic-based ΔΔG prediction.
Designed to run for hours and produce valuable architecture insights.

Features:
- Multiple neural network architectures
- Grid search over hyperparameters
- Cross-validation with variance estimates
- Checkpointing and comprehensive logging
- GPU support when available

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance from origin for a Poincare ball embedding.

    V5.12.2: Use proper hyperbolic distance formula instead of Euclidean norm.
    """
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


# Set up paths
SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = VALIDATION_DIR / "results" / "ddg_pytorch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(run_name: str) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('ddg_pytorch')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    log_file = LOGS_DIR / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# DATA (imported from ddg_predictor_training.py concepts)
# ============================================================================

AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'volume': 88.6, 'mass': 89.09, 'charge': 0},
    'R': {'hydropathy': -4.5, 'volume': 173.4, 'mass': 174.20, 'charge': 1},
    'N': {'hydropathy': -3.5, 'volume': 114.1, 'mass': 132.12, 'charge': 0},
    'D': {'hydropathy': -3.5, 'volume': 111.1, 'mass': 133.10, 'charge': -1},
    'C': {'hydropathy': 2.5, 'volume': 108.5, 'mass': 121.16, 'charge': 0},
    'Q': {'hydropathy': -3.5, 'volume': 143.8, 'mass': 146.15, 'charge': 0},
    'E': {'hydropathy': -3.5, 'volume': 138.4, 'mass': 147.13, 'charge': -1},
    'G': {'hydropathy': -0.4, 'volume': 60.1, 'mass': 75.07, 'charge': 0},
    'H': {'hydropathy': -3.2, 'volume': 153.2, 'mass': 155.16, 'charge': 0},
    'I': {'hydropathy': 4.5, 'volume': 166.7, 'mass': 131.17, 'charge': 0},
    'L': {'hydropathy': 3.8, 'volume': 166.7, 'mass': 131.17, 'charge': 0},
    'K': {'hydropathy': -3.9, 'volume': 168.6, 'mass': 146.19, 'charge': 1},
    'M': {'hydropathy': 1.9, 'volume': 162.9, 'mass': 149.21, 'charge': 0},
    'F': {'hydropathy': 2.8, 'volume': 189.9, 'mass': 165.19, 'charge': 0},
    'P': {'hydropathy': -1.6, 'volume': 112.7, 'mass': 115.13, 'charge': 0},
    'S': {'hydropathy': -0.8, 'volume': 89.0, 'mass': 105.09, 'charge': 0},
    'T': {'hydropathy': -0.7, 'volume': 116.1, 'mass': 119.12, 'charge': 0},
    'W': {'hydropathy': -0.9, 'volume': 227.8, 'mass': 204.23, 'charge': 0},
    'Y': {'hydropathy': -1.3, 'volume': 193.6, 'mass': 181.19, 'charge': 0},
    'V': {'hydropathy': 4.2, 'volume': 140.0, 'mass': 117.15, 'charge': 0},
}

# Full mutation database (same as ddg_predictor_training.py)
DDG_DATABASE = [
    # Barnase mutations
    ("barnase", "A", 32, "G", 1.8, "Serrano1992"),
    ("barnase", "I", 51, "V", 1.2, "Serrano1992"),
    ("barnase", "I", 51, "A", 3.5, "Serrano1992"),
    ("barnase", "I", 51, "G", 5.2, "Serrano1992"),
    ("barnase", "L", 14, "A", 2.8, "Serrano1992"),
    ("barnase", "L", 14, "V", 1.1, "Serrano1992"),
    ("barnase", "V", 36, "A", 2.1, "Serrano1992"),
    ("barnase", "V", 36, "G", 3.8, "Serrano1992"),
    ("barnase", "F", 56, "A", 4.2, "Serrano1992"),
    ("barnase", "Y", 78, "F", 0.8, "Serrano1992"),
    ("barnase", "Y", 78, "A", 3.1, "Serrano1992"),
    ("barnase", "W", 71, "F", 2.5, "Serrano1992"),
    # T4 lysozyme
    ("T4_lysozyme", "L", 99, "A", 3.2, "Eriksson1992"),
    ("T4_lysozyme", "L", 99, "V", 1.4, "Eriksson1992"),
    ("T4_lysozyme", "L", 99, "I", 0.3, "Eriksson1992"),
    ("T4_lysozyme", "L", 99, "F", 0.8, "Eriksson1992"),
    ("T4_lysozyme", "L", 99, "M", 0.5, "Eriksson1992"),
    ("T4_lysozyme", "L", 118, "A", 2.9, "Eriksson1992"),
    ("T4_lysozyme", "L", 121, "A", 3.5, "Eriksson1992"),
    ("T4_lysozyme", "L", 133, "A", 2.7, "Eriksson1992"),
    ("T4_lysozyme", "I", 3, "A", 3.1, "Eriksson1992"),
    ("T4_lysozyme", "I", 3, "V", 1.0, "Eriksson1992"),
    ("T4_lysozyme", "V", 87, "A", 2.4, "Eriksson1992"),
    ("T4_lysozyme", "F", 153, "A", 4.5, "Eriksson1992"),
    ("T4_lysozyme", "M", 102, "A", 1.8, "Eriksson1992"),
    ("T4_lysozyme", "M", 102, "L", 0.2, "Eriksson1992"),
    # Staphylococcal nuclease
    ("staph_nuclease", "V", 66, "A", 2.2, "Shortle1992"),
    ("staph_nuclease", "V", 66, "G", 4.1, "Shortle1992"),
    ("staph_nuclease", "V", 66, "L", 0.4, "Shortle1992"),
    ("staph_nuclease", "I", 72, "A", 2.8, "Shortle1992"),
    ("staph_nuclease", "I", 72, "V", 1.0, "Shortle1992"),
    ("staph_nuclease", "L", 25, "A", 2.5, "Shortle1992"),
    ("staph_nuclease", "L", 36, "A", 3.0, "Shortle1992"),
    ("staph_nuclease", "L", 37, "A", 2.6, "Shortle1992"),
    ("staph_nuclease", "F", 34, "A", 3.8, "Shortle1992"),
    ("staph_nuclease", "Y", 91, "A", 3.2, "Shortle1992"),
    # CI2
    ("CI2", "I", 20, "A", 2.9, "Jackson1993"),
    ("CI2", "I", 20, "V", 0.9, "Jackson1993"),
    ("CI2", "I", 20, "L", 0.2, "Jackson1993"),
    ("CI2", "L", 32, "A", 3.1, "Jackson1993"),
    ("CI2", "L", 32, "V", 1.5, "Jackson1993"),
    ("CI2", "V", 47, "A", 2.3, "Jackson1993"),
    ("CI2", "V", 47, "G", 4.0, "Jackson1993"),
    ("CI2", "V", 51, "A", 2.0, "Jackson1993"),
    ("CI2", "A", 16, "G", 1.5, "Jackson1993"),
    ("CI2", "F", 50, "A", 4.1, "Jackson1993"),
    ("CI2", "F", 50, "L", 1.8, "Jackson1993"),
    # SH3 domain
    ("SH3", "L", 10, "A", 2.4, "Viguera1994"),
    ("SH3", "L", 10, "V", 0.8, "Viguera1994"),
    ("SH3", "I", 28, "A", 2.7, "Viguera1994"),
    ("SH3", "I", 28, "V", 0.7, "Viguera1994"),
    ("SH3", "V", 44, "A", 2.1, "Viguera1994"),
    ("SH3", "F", 52, "A", 3.6, "Viguera1994"),
    ("SH3", "Y", 8, "A", 2.9, "Viguera1994"),
    ("SH3", "Y", 8, "F", 0.5, "Viguera1994"),
    ("SH3", "W", 36, "F", 2.2, "Viguera1994"),
    # Ubiquitin
    ("ubiquitin", "I", 3, "A", 2.6, "Khorasanizadeh1996"),
    ("ubiquitin", "I", 3, "V", 0.8, "Khorasanizadeh1996"),
    ("ubiquitin", "L", 15, "A", 2.8, "Khorasanizadeh1996"),
    ("ubiquitin", "L", 43, "A", 3.0, "Khorasanizadeh1996"),
    ("ubiquitin", "V", 5, "A", 2.2, "Khorasanizadeh1996"),
    ("ubiquitin", "V", 17, "A", 1.9, "Khorasanizadeh1996"),
    ("ubiquitin", "V", 26, "A", 2.4, "Khorasanizadeh1996"),
    ("ubiquitin", "F", 45, "A", 4.0, "Khorasanizadeh1996"),
    ("ubiquitin", "Y", 59, "F", 0.6, "Khorasanizadeh1996"),
    ("ubiquitin", "I", 23, "V", 0.9, "Khorasanizadeh1996"),
    ("ubiquitin", "I", 30, "V", 0.7, "Khorasanizadeh1996"),
    ("ubiquitin", "I", 36, "V", 1.0, "Khorasanizadeh1996"),
    ("ubiquitin", "I", 44, "V", 0.8, "Khorasanizadeh1996"),
    # Lambda repressor
    ("lambda_repressor", "L", 18, "A", 2.5, "Lim1994"),
    ("lambda_repressor", "L", 18, "V", 0.9, "Lim1994"),
    ("lambda_repressor", "I", 54, "A", 2.8, "Lim1994"),
    ("lambda_repressor", "V", 36, "A", 2.0, "Lim1994"),
    ("lambda_repressor", "A", 37, "G", 1.4, "Lim1994"),
    ("lambda_repressor", "F", 51, "A", 3.5, "Lim1994"),
    ("lambda_repressor", "Y", 22, "F", 0.7, "Lim1994"),
    # BPTI
    ("BPTI", "Y", 21, "A", 3.3, "Kim1993"),
    ("BPTI", "Y", 21, "F", 0.6, "Kim1993"),
    ("BPTI", "Y", 23, "F", 0.8, "Kim1993"),
    ("BPTI", "F", 22, "A", 3.8, "Kim1993"),
    ("BPTI", "F", 33, "A", 4.2, "Kim1993"),
    ("BPTI", "F", 45, "A", 3.5, "Kim1993"),
    # Protein G
    ("protein_G", "L", 5, "A", 2.7, "Smith1994"),
    ("protein_G", "L", 5, "V", 1.1, "Smith1994"),
    ("protein_G", "V", 21, "A", 2.3, "Smith1994"),
    ("protein_G", "V", 29, "A", 1.8, "Smith1994"),
    ("protein_G", "V", 39, "A", 2.0, "Smith1994"),
    ("protein_G", "A", 24, "G", 1.6, "Smith1994"),
    ("protein_G", "F", 30, "A", 3.9, "Smith1994"),
    ("protein_G", "Y", 33, "F", 0.5, "Smith1994"),
    ("protein_G", "W", 43, "F", 2.0, "Smith1994"),
    # RNase H
    ("RNase_H", "I", 7, "A", 2.5, "Ishikawa1993"),
    ("RNase_H", "I", 7, "V", 0.8, "Ishikawa1993"),
    ("RNase_H", "L", 56, "A", 2.9, "Ishikawa1993"),
    ("RNase_H", "V", 74, "A", 2.1, "Ishikawa1993"),
    ("RNase_H", "V", 98, "A", 1.9, "Ishikawa1993"),
    ("RNase_H", "F", 35, "A", 3.7, "Ishikawa1993"),
    # Trp repressor
    ("trp_repressor", "L", 75, "A", 2.6, "Mann1993"),
    ("trp_repressor", "I", 57, "A", 2.8, "Mann1993"),
    ("trp_repressor", "V", 58, "A", 2.0, "Mann1993"),
    ("trp_repressor", "F", 22, "A", 3.4, "Mann1993"),
    ("trp_repressor", "W", 19, "F", 2.3, "Mann1993"),
    # Arc repressor
    ("arc_repressor", "L", 12, "A", 2.4, "Milla1994"),
    ("arc_repressor", "I", 9, "A", 2.6, "Milla1994"),
    ("arc_repressor", "V", 41, "A", 1.8, "Milla1994"),
    ("arc_repressor", "F", 10, "A", 3.5, "Milla1994"),
    # Stabilizing mutations
    ("T4_lysozyme", "A", 82, "P", -0.5, "Matthews1995"),
    ("T4_lysozyme", "G", 77, "A", -0.8, "Matthews1995"),
    ("barnase", "G", 53, "A", -0.4, "Serrano1992"),
    ("CI2", "A", 16, "V", -0.3, "Jackson1993"),
    ("staph_nuclease", "G", 79, "A", -0.6, "Shortle1992"),
    # Charge mutations
    ("barnase", "K", 27, "A", 1.2, "Serrano1992"),
    ("barnase", "E", 73, "A", 0.9, "Serrano1992"),
    ("barnase", "D", 54, "A", 0.7, "Serrano1992"),
    ("barnase", "R", 83, "A", 1.5, "Serrano1992"),
    ("T4_lysozyme", "K", 16, "A", 0.8, "Eriksson1992"),
    ("T4_lysozyme", "E", 22, "A", 0.6, "Eriksson1992"),
    ("T4_lysozyme", "D", 70, "A", 0.5, "Eriksson1992"),
    # Polar mutations
    ("barnase", "S", 57, "A", 0.4, "Serrano1992"),
    ("barnase", "T", 79, "A", 0.6, "Serrano1992"),
    ("barnase", "N", 23, "A", 0.8, "Serrano1992"),
    ("barnase", "Q", 31, "A", 0.7, "Serrano1992"),
    ("T4_lysozyme", "S", 44, "A", 0.3, "Eriksson1992"),
    ("T4_lysozyme", "T", 109, "A", 0.5, "Eriksson1992"),
    ("T4_lysozyme", "N", 68, "A", 0.6, "Eriksson1992"),
    # Proline mutations
    ("barnase", "P", 64, "A", 1.8, "Serrano1992"),
    ("barnase", "P", 64, "G", 2.5, "Serrano1992"),
    ("T4_lysozyme", "P", 37, "A", 1.5, "Eriksson1992"),
    ("CI2", "P", 25, "A", 2.0, "Jackson1993"),
    # Additional entries
    ("lysozyme_hen", "W", 62, "F", 2.1, "Takano1995"),
    ("lysozyme_hen", "W", 62, "Y", 1.8, "Takano1995"),
    ("lysozyme_hen", "W", 63, "F", 1.9, "Takano1995"),
    ("lysozyme_hen", "W", 108, "F", 2.4, "Takano1995"),
    ("lysozyme_hen", "W", 111, "F", 2.0, "Takano1995"),
    ("lysozyme_hen", "W", 123, "F", 2.2, "Takano1995"),
    ("lysozyme_hen", "I", 55, "A", 2.7, "Takano1995"),
    ("lysozyme_hen", "I", 55, "V", 0.9, "Takano1995"),
    ("lysozyme_hen", "L", 56, "A", 2.9, "Takano1995"),
    ("lysozyme_hen", "V", 92, "A", 2.3, "Takano1995"),
    ("lysozyme_hen", "F", 34, "A", 3.6, "Takano1995"),
    ("lysozyme_hen", "F", 38, "A", 3.4, "Takano1995"),
    ("lysozyme_hen", "Y", 20, "F", 0.7, "Takano1995"),
    ("lysozyme_hen", "Y", 23, "F", 0.8, "Takano1995"),
    # Myoglobin
    ("myoglobin", "V", 68, "A", 2.0, "Quillin1993"),
    ("myoglobin", "V", 68, "G", 3.5, "Quillin1993"),
    ("myoglobin", "V", 68, "I", 0.4, "Quillin1993"),
    ("myoglobin", "V", 68, "L", 0.3, "Quillin1993"),
    ("myoglobin", "L", 29, "A", 2.6, "Quillin1993"),
    ("myoglobin", "L", 29, "V", 1.0, "Quillin1993"),
    ("myoglobin", "I", 107, "A", 2.5, "Quillin1993"),
    ("myoglobin", "I", 107, "V", 0.7, "Quillin1993"),
    ("myoglobin", "F", 43, "A", 3.8, "Quillin1993"),
    ("myoglobin", "F", 46, "A", 3.5, "Quillin1993"),
    # Cytochrome c
    ("cyt_c", "F", 82, "A", 3.2, "Luntz1989"),
    ("cyt_c", "F", 82, "L", 1.2, "Luntz1989"),
    ("cyt_c", "Y", 67, "F", 0.5, "Luntz1989"),
    ("cyt_c", "L", 94, "A", 2.5, "Luntz1989"),
    ("cyt_c", "I", 75, "A", 2.3, "Luntz1989"),
    ("cyt_c", "V", 20, "A", 1.8, "Luntz1989"),
    # FKBP12
    ("FKBP12", "F", 36, "A", 3.4, "Main1999"),
    ("FKBP12", "F", 36, "L", 1.0, "Main1999"),
    ("FKBP12", "F", 46, "A", 3.1, "Main1999"),
    ("FKBP12", "Y", 26, "F", 0.4, "Main1999"),
    ("FKBP12", "W", 59, "F", 1.8, "Main1999"),
    ("FKBP12", "I", 56, "A", 2.4, "Main1999"),
    ("FKBP12", "I", 56, "V", 0.7, "Main1999"),
    ("FKBP12", "L", 50, "A", 2.6, "Main1999"),
    ("FKBP12", "V", 55, "A", 1.9, "Main1999"),
    # Tendamistat
    ("tendamistat", "Y", 15, "F", 0.6, "Schreiber1994"),
    ("tendamistat", "Y", 15, "A", 2.8, "Schreiber1994"),
    ("tendamistat", "W", 18, "F", 2.0, "Schreiber1994"),
    ("tendamistat", "F", 48, "A", 3.3, "Schreiber1994"),
    ("tendamistat", "I", 34, "A", 2.4, "Schreiber1994"),
    ("tendamistat", "I", 34, "V", 0.8, "Schreiber1994"),
    ("tendamistat", "L", 12, "A", 2.5, "Schreiber1994"),
    ("tendamistat", "V", 52, "A", 1.8, "Schreiber1994"),
]


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def load_padic_embeddings() -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Load p-adic radii and full embeddings."""
    mapping_path = GENETIC_CODE_DIR / "codon_mapping_3adic.json"
    emb_path = GENETIC_CODE_DIR / "v5_11_3_embeddings.pt"

    if not mapping_path.exists() or not emb_path.exists():
        return {}, {}

    with open(mapping_path) as f:
        mapping = json.load(f)

    codon_to_pos = mapping['codon_to_position']
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    z = emb_data['z_B_hyp'].numpy()

    CODON_TO_AA = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
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

    aa_embs = {}
    for codon, pos in codon_to_pos.items():
        aa = CODON_TO_AA.get(codon)
        if aa:
            if aa not in aa_embs:
                aa_embs[aa] = []
            aa_embs[aa].append(z[pos])

    radii = {}
    embeddings = {}
    for aa in aa_embs:
        mean_emb = np.mean(aa_embs[aa], axis=0)
        radii[aa] = hyperbolic_radius(mean_emb)
        embeddings[aa] = mean_emb

    return radii, embeddings


def prepare_data(
    padic_radii: Dict[str, float],
    padic_embeddings: Dict[str, np.ndarray]
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare feature tensors for all feature sets."""
    features_raw = {
        'mass': [], 'property': [], 'padic_radius': [],
        'padic_embedding': [], 'padic_mass': [], 'all': []
    }
    targets = []

    for protein, wt_aa, position, mut_aa, ddg, source in DDG_DATABASE:
        if wt_aa not in AA_PROPERTIES or mut_aa not in AA_PROPERTIES:
            continue
        if wt_aa not in padic_radii or mut_aa not in padic_radii:
            continue

        targets.append(ddg)

        # Mass features
        mass_wt = AA_PROPERTIES[wt_aa]['mass']
        mass_mut = AA_PROPERTIES[mut_aa]['mass']
        delta_mass = mass_mut - mass_wt
        mass_ratio = mass_mut / mass_wt
        features_raw['mass'].append([delta_mass, mass_ratio, abs(delta_mass)])

        # Property features
        delta_h = AA_PROPERTIES[mut_aa]['hydropathy'] - AA_PROPERTIES[wt_aa]['hydropathy']
        delta_v = AA_PROPERTIES[mut_aa]['volume'] - AA_PROPERTIES[wt_aa]['volume']
        delta_c = AA_PROPERTIES[mut_aa]['charge'] - AA_PROPERTIES[wt_aa]['charge']
        features_raw['property'].append([delta_h, delta_v, delta_c])

        # P-adic radius features
        r_wt = padic_radii[wt_aa]
        r_mut = padic_radii[mut_aa]
        delta_r = r_mut - r_wt
        features_raw['padic_radius'].append([delta_r, abs(delta_r), r_wt, r_mut])

        # P-adic embedding features
        emb_diff = padic_embeddings[mut_aa] - padic_embeddings[wt_aa]
        features_raw['padic_embedding'].append(emb_diff)

        # P-adic + mass
        features_raw['padic_mass'].append(np.concatenate([emb_diff, [delta_mass, mass_ratio, abs(delta_mass)]]))

        # All combined
        features_raw['all'].append(np.concatenate([
            [delta_mass, mass_ratio, abs(delta_mass)],
            [delta_h, delta_v, delta_c],
            [delta_r, abs(delta_r), r_wt, r_mut],
        ]))

    # Convert to tensors
    y = torch.tensor(targets, dtype=torch.float32)

    datasets = {}
    for name, feats in features_raw.items():
        X = torch.tensor(np.array(feats), dtype=torch.float32)
        datasets[name] = (X, y)

    return datasets


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================


class MLPPredictor(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = 'relu',
        batch_norm: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # Build layers
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        return self.output(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


class ResidualMLP(nn.Module):
    """MLP with residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Train for one epoch."""
    model.train()
    n_samples = len(y)
    indices = torch.randperm(n_samples)

    total_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]

        X_batch = X[batch_idx].to(DEVICE)
        y_batch = y[batch_idx].to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        X_dev = X.to(DEVICE)
        pred = model(X_dev).cpu().numpy()
        y_np = y.numpy()

    spearman, sp_p = stats.spearmanr(y_np, pred)
    pearson, pe_p = stats.pearsonr(y_np, pred)
    rmse = np.sqrt(np.mean((y_np - pred) ** 2))
    mae = np.mean(np.abs(y_np - pred))

    return {
        'spearman': float(spearman),
        'pearson': float(pearson),
        'rmse': float(rmse),
        'mae': float(mae),
    }


def cross_validate_model(
    X: torch.Tensor,
    y: torch.Tensor,
    model_fn,
    model_kwargs: Dict,
    optimizer_fn,
    optimizer_kwargs: Dict,
    n_epochs: int = 200,
    n_folds: int = 5,
    batch_size: int = 32,
    early_stopping: int = 20,
    verbose: bool = False,
) -> Dict:
    """Cross-validation with a model."""
    n_samples = len(y)
    fold_size = n_samples // n_folds

    all_preds = np.zeros(n_samples)
    fold_metrics = []

    for fold in range(n_folds):
        # Split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples

        test_idx = list(range(test_start, test_end))
        train_idx = list(range(0, test_start)) + list(range(test_end, n_samples))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize features
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # Normalize targets
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train_norm = (y_train - y_mean) / y_std

        # Create model
        model = model_fn(**model_kwargs).to(DEVICE)
        optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)

        # Training
        best_val_loss = float('inf')
        patience = 0
        best_state = None

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, optimizer, X_train_norm, y_train_norm, batch_size)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_norm.to(DEVICE)).cpu()
                val_loss = nn.functional.mse_loss(val_pred, (y_test - y_mean) / y_std).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)

        # Final predictions
        model.eval()
        with torch.no_grad():
            pred = model(X_test_norm.to(DEVICE)).cpu().numpy()
            pred = pred * y_std.item() + y_mean.item()

        all_preds[test_idx] = pred

        # Fold metrics
        r, _ = stats.spearmanr(y_test.numpy(), pred)
        rmse = np.sqrt(np.mean((y_test.numpy() - pred) ** 2))
        fold_metrics.append({'spearman': r, 'rmse': rmse})

    # Overall metrics
    y_np = y.numpy()
    spearman, sp_p = stats.spearmanr(y_np, all_preds)
    pearson, _ = stats.pearsonr(y_np, all_preds)
    rmse = np.sqrt(np.mean((y_np - all_preds) ** 2))
    mae = np.mean(np.abs(y_np - all_preds))

    return {
        'spearman': float(spearman),
        'spearman_p': float(sp_p),
        'pearson': float(pearson),
        'rmse': float(rmse),
        'mae': float(mae),
        'fold_metrics': fold_metrics,
        'spearman_std': float(np.std([f['spearman'] for f in fold_metrics])),
    }


# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================


@dataclass
class HyperparamConfig:
    """Hyperparameter configuration."""
    hidden_dims: List[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    activation: str
    batch_norm: bool
    batch_size: int
    n_epochs: int

    def to_dict(self) -> Dict:
        return {
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
        }


def generate_hyperparam_grid() -> List[HyperparamConfig]:
    """Generate hyperparameter grid."""
    configs = []

    # Architecture variations
    hidden_dims_options = [
        [32],
        [64],
        [128],
        [32, 32],
        [64, 32],
        [64, 64],
        [128, 64],
        [128, 64, 32],
        [64, 64, 64],
        [256, 128, 64],
    ]

    # Other hyperparameters
    dropout_options = [0.0, 0.1, 0.2, 0.3]
    lr_options = [0.001, 0.003, 0.01]
    wd_options = [0.0, 0.0001, 0.001]
    activation_options = ['relu', 'gelu']
    batch_norm_options = [False, True]
    batch_size_options = [16, 32]

    # Full grid (reduced for practical runtime)
    for hidden_dims in hidden_dims_options:
        for dropout in dropout_options:
            for lr in lr_options:
                for wd in wd_options:
                    for activation in activation_options:
                        for batch_norm in batch_norm_options:
                            for batch_size in batch_size_options:
                                configs.append(HyperparamConfig(
                                    hidden_dims=hidden_dims,
                                    dropout=dropout,
                                    learning_rate=lr,
                                    weight_decay=wd,
                                    activation=activation,
                                    batch_norm=batch_norm,
                                    batch_size=batch_size,
                                    n_epochs=300,
                                ))

    return configs


def run_hyperparameter_search(
    datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    feature_sets: List[str],
    logger: logging.Logger,
    max_configs: int = 100,
    n_folds: int = 5,
) -> Dict:
    """Run hyperparameter search."""
    all_configs = generate_hyperparam_grid()

    # Subsample if too many
    if len(all_configs) > max_configs:
        np.random.seed(42)
        indices = np.random.choice(len(all_configs), max_configs, replace=False)
        configs = [all_configs[i] for i in indices]
    else:
        configs = all_configs

    logger.info(f"Testing {len(configs)} hyperparameter configurations")

    results = {}
    best_per_feature = {}

    for feat_name in feature_sets:
        X, y = datasets[feat_name]
        input_dim = X.shape[1]

        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Set: {feat_name} (dim={input_dim})")
        logger.info(f"{'='*60}")

        results[feat_name] = []
        best_spearman = -1
        best_config = None
        best_result = None

        for i, config in enumerate(configs):
            try:
                # Create model function
                def model_fn(input_dim=input_dim, **kwargs):
                    return MLPPredictor(
                        input_dim=input_dim,
                        hidden_dims=config.hidden_dims,
                        dropout=config.dropout,
                        activation=config.activation,
                        batch_norm=config.batch_norm,
                    )

                # Run cross-validation
                result = cross_validate_model(
                    X, y,
                    model_fn=model_fn,
                    model_kwargs={'input_dim': input_dim},
                    optimizer_fn=optim.AdamW,
                    optimizer_kwargs={
                        'lr': config.learning_rate,
                        'weight_decay': config.weight_decay,
                    },
                    n_epochs=config.n_epochs,
                    n_folds=n_folds,
                    batch_size=config.batch_size,
                    early_stopping=30,
                )

                result['config'] = config.to_dict()
                results[feat_name].append(result)

                if result['spearman'] > best_spearman:
                    best_spearman = result['spearman']
                    best_config = config
                    best_result = result

                # Progress logging (every 10 configs)
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"  [{i+1}/{len(configs)}] Current best: ρ={best_spearman:.4f} "
                        f"(h={best_config.hidden_dims}, d={best_config.dropout:.1f}, "
                        f"lr={best_config.learning_rate})"
                    )

            except Exception as e:
                logger.warning(f"  Config {i} failed: {e}")
                continue

        best_per_feature[feat_name] = {
            'spearman': best_spearman,
            'config': best_config.to_dict() if best_config else None,
            'full_result': best_result,
        }

        logger.info(f"\n  BEST for {feat_name}: ρ={best_spearman:.4f}")
        if best_config:
            logger.info(f"    hidden={best_config.hidden_dims}, dropout={best_config.dropout}")
            logger.info(f"    lr={best_config.learning_rate}, wd={best_config.weight_decay}")
            logger.info(f"    activation={best_config.activation}, batch_norm={best_config.batch_norm}")

    return {
        'all_results': results,
        'best_per_feature': best_per_feature,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    run_name = f"ddg_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(run_name)

    logger.info("=" * 70)
    logger.info("PyTorch ΔΔG Predictor - Hyperparameter Search")
    logger.info("=" * 70)
    logger.info(f"Run: {run_name}")
    logger.info(f"Device: {DEVICE}")

    # Load data
    logger.info("\nLoading p-adic embeddings...")
    padic_radii, padic_embeddings = load_padic_embeddings()
    logger.info(f"  Loaded embeddings for {len(padic_radii)} amino acids")

    # Prepare datasets
    logger.info("Preparing feature datasets...")
    datasets = prepare_data(padic_radii, padic_embeddings)

    for name, (X, y) in datasets.items():
        logger.info(f"  {name}: {X.shape[0]} samples, {X.shape[1]} features")

    # Feature sets to test
    feature_sets = ['mass', 'property', 'padic_radius', 'padic_embedding', 'padic_mass', 'all']

    # Run hyperparameter search
    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 70)

    start_time = time.time()

    search_results = run_hyperparameter_search(
        datasets=datasets,
        feature_sets=feature_sets,
        logger=logger,
        max_configs=100,  # Limit for practical runtime
        n_folds=5,
    )

    elapsed = time.time() - start_time
    logger.info(f"\nTotal search time: {elapsed/60:.1f} minutes")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    for feat_name, best in search_results['best_per_feature'].items():
        logger.info(f"  {feat_name:<20} ρ={best['spearman']:.4f}")
        if best['config']:
            logger.info(f"    → hidden={best['config']['hidden_dims']}, "
                       f"dropout={best['config']['dropout']}, "
                       f"lr={best['config']['learning_rate']}")

    # Save results
    output = {
        'run_name': run_name,
        'device': str(DEVICE),
        'n_mutations': len(DDG_DATABASE),
        'elapsed_minutes': elapsed / 60,
        'best_per_feature': {
            k: {'spearman': v['spearman'], 'config': v['config']}
            for k, v in search_results['best_per_feature'].items()
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_file = RESULTS_DIR / f"{run_name}_summary.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Also save detailed results
    detailed_file = RESULTS_DIR / f"{run_name}_detailed.pkl"
    with open(detailed_file, 'wb') as f:
        pickle.dump(search_results, f)

    logger.info(f"Detailed results saved to: {detailed_file}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
