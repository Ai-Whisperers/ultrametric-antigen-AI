#!/usr/bin/env python3
"""
Deep Physics Benchmark: How Deep Are P-adic Invariants?

Tests whether the p-adic codon structure encodes physics at different depths:

Level 0: Biochemistry (hydropathy, charge) - traditional view
Level 1: Classical mechanics (mass, inertia) - established
Level 2: Statistical mechanics (entropy, partition functions)
Level 3: Vibrational physics (IR/Raman, normal modes)
Level 4: Quantum mechanics (zero-point energy)
Level 5: Force field fundamentals (bond lengths, force constants)

If p-adic correlates with Level 3+ invariants, the genetic code encodes
physics deep enough to predict dynamics and 3D structure over time.

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"

# ============================================================================
# LEVEL 0: BIOCHEMISTRY (Traditional)
# ============================================================================

# Standard amino acid properties
AA_BIOCHEMISTRY = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'R': {'hydropathy': -4.5, 'charge': 1, 'polarity': 1, 'aromaticity': 0},
    'N': {'hydropathy': -3.5, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'D': {'hydropathy': -3.5, 'charge': -1, 'polarity': 1, 'aromaticity': 0},
    'C': {'hydropathy': 2.5, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'E': {'hydropathy': -3.5, 'charge': -1, 'polarity': 1, 'aromaticity': 0},
    'G': {'hydropathy': -0.4, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'H': {'hydropathy': -3.2, 'charge': 0, 'polarity': 1, 'aromaticity': 1},
    'I': {'hydropathy': 4.5, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'L': {'hydropathy': 3.8, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'K': {'hydropathy': -3.9, 'charge': 1, 'polarity': 1, 'aromaticity': 0},
    'M': {'hydropathy': 1.9, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'F': {'hydropathy': 2.8, 'charge': 0, 'polarity': 0, 'aromaticity': 1},
    'P': {'hydropathy': -1.6, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
    'S': {'hydropathy': -0.8, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'T': {'hydropathy': -0.7, 'charge': 0, 'polarity': 1, 'aromaticity': 0},
    'W': {'hydropathy': -0.9, 'charge': 0, 'polarity': 0, 'aromaticity': 1},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polarity': 1, 'aromaticity': 1},
    'V': {'hydropathy': 4.2, 'charge': 0, 'polarity': 0, 'aromaticity': 0},
}

# ============================================================================
# LEVEL 1: CLASSICAL MECHANICS
# ============================================================================

# Molecular masses (Da) and derived classical properties
AA_CLASSICAL = {
    'A': {'mass': 89.09, 'volume': 88.6, 'surface_area': 115},
    'R': {'mass': 174.20, 'volume': 173.4, 'surface_area': 225},
    'N': {'mass': 132.12, 'volume': 114.1, 'surface_area': 160},
    'D': {'mass': 133.10, 'volume': 111.1, 'surface_area': 150},
    'C': {'mass': 121.16, 'volume': 108.5, 'surface_area': 135},
    'Q': {'mass': 146.15, 'volume': 143.8, 'surface_area': 180},
    'E': {'mass': 147.13, 'volume': 138.4, 'surface_area': 190},
    'G': {'mass': 75.07, 'volume': 60.1, 'surface_area': 75},
    'H': {'mass': 155.16, 'volume': 153.2, 'surface_area': 195},
    'I': {'mass': 131.17, 'volume': 166.7, 'surface_area': 175},
    'L': {'mass': 131.17, 'volume': 166.7, 'surface_area': 170},
    'K': {'mass': 146.19, 'volume': 168.6, 'surface_area': 200},
    'M': {'mass': 149.21, 'volume': 162.9, 'surface_area': 185},
    'F': {'mass': 165.19, 'volume': 189.9, 'surface_area': 210},
    'P': {'mass': 115.13, 'volume': 112.7, 'surface_area': 145},
    'S': {'mass': 105.09, 'volume': 89.0, 'surface_area': 115},
    'T': {'mass': 119.12, 'volume': 116.1, 'surface_area': 140},
    'W': {'mass': 204.23, 'volume': 227.8, 'surface_area': 255},
    'Y': {'mass': 181.19, 'volume': 193.6, 'surface_area': 230},
    'V': {'mass': 117.15, 'volume': 140.0, 'surface_area': 155},
}

# Add derived classical quantities
for aa in AA_CLASSICAL:
    m = AA_CLASSICAL[aa]['mass']
    # Inertia proxy (moment of inertia scales with mass * r^2)
    AA_CLASSICAL[aa]['inertia'] = m * (AA_CLASSICAL[aa]['volume'] ** (2/3))
    # Diffusion coefficient (Stokes-Einstein: D ∝ 1/r, r ∝ V^(1/3))
    AA_CLASSICAL[aa]['diffusion_coeff'] = 1.0 / (AA_CLASSICAL[aa]['volume'] ** (1/3))

# ============================================================================
# LEVEL 2: STATISTICAL MECHANICS
# ============================================================================

# Conformational entropy (from MD simulations and NMR)
# Higher = more conformational freedom
AA_STAT_MECH = {
    # Side chain entropy (cal/mol/K) from Doig & Sternberg 1995
    'A': {'sidechain_entropy': 0.0, 'backbone_flexibility': 1.0},
    'R': {'sidechain_entropy': 2.7, 'backbone_flexibility': 0.9},
    'N': {'sidechain_entropy': 1.4, 'backbone_flexibility': 0.95},
    'D': {'sidechain_entropy': 1.0, 'backbone_flexibility': 0.95},
    'C': {'sidechain_entropy': 0.5, 'backbone_flexibility': 0.9},
    'Q': {'sidechain_entropy': 2.2, 'backbone_flexibility': 0.9},
    'E': {'sidechain_entropy': 1.8, 'backbone_flexibility': 0.9},
    'G': {'sidechain_entropy': 0.0, 'backbone_flexibility': 1.5},  # Most flexible
    'H': {'sidechain_entropy': 1.2, 'backbone_flexibility': 0.85},
    'I': {'sidechain_entropy': 1.4, 'backbone_flexibility': 0.7},
    'L': {'sidechain_entropy': 1.4, 'backbone_flexibility': 0.8},
    'K': {'sidechain_entropy': 2.7, 'backbone_flexibility': 0.85},
    'M': {'sidechain_entropy': 2.0, 'backbone_flexibility': 0.85},
    'F': {'sidechain_entropy': 1.0, 'backbone_flexibility': 0.75},
    'P': {'sidechain_entropy': 0.0, 'backbone_flexibility': 0.3},  # Least flexible
    'S': {'sidechain_entropy': 0.5, 'backbone_flexibility': 1.0},
    'T': {'sidechain_entropy': 0.7, 'backbone_flexibility': 0.9},
    'W': {'sidechain_entropy': 1.2, 'backbone_flexibility': 0.7},
    'Y': {'sidechain_entropy': 1.0, 'backbone_flexibility': 0.75},
    'V': {'sidechain_entropy': 0.7, 'backbone_flexibility': 0.75},
}

# Add partition function proxy (Z ∝ number of accessible states)
for aa in AA_STAT_MECH:
    # Rotamer counts from Dunbrack library
    rotamer_counts = {
        'A': 1, 'R': 81, 'N': 9, 'D': 9, 'C': 3, 'Q': 27, 'E': 27,
        'G': 1, 'H': 9, 'I': 9, 'L': 9, 'K': 81, 'M': 27, 'F': 6,
        'P': 2, 'S': 3, 'T': 3, 'W': 6, 'Y': 6, 'V': 3
    }
    AA_STAT_MECH[aa]['rotamer_count'] = rotamer_counts[aa]
    AA_STAT_MECH[aa]['log_partition'] = np.log(rotamer_counts[aa])

# ============================================================================
# LEVEL 3: VIBRATIONAL PHYSICS
# ============================================================================

# Amino acid characteristic vibrational frequencies (cm^-1)
# From IR/Raman spectroscopy databases
AA_VIBRATIONAL = {
    # Amide I (~1650), Amide II (~1550), C-H stretch (~2900), specific modes
    'A': {'amide_I': 1654, 'CH_stretch': 2960, 'specific_mode': 893},  # CH3 rock
    'R': {'amide_I': 1658, 'CH_stretch': 2935, 'specific_mode': 1672},  # C=NH2+
    'N': {'amide_I': 1648, 'CH_stretch': 2940, 'specific_mode': 1678},  # C=O amide
    'D': {'amide_I': 1645, 'CH_stretch': 2945, 'specific_mode': 1716},  # C=O acid
    'C': {'amide_I': 1652, 'CH_stretch': 2955, 'specific_mode': 2551},  # S-H
    'Q': {'amide_I': 1650, 'CH_stretch': 2938, 'specific_mode': 1670},  # C=O amide
    'E': {'amide_I': 1647, 'CH_stretch': 2942, 'specific_mode': 1712},  # C=O acid
    'G': {'amide_I': 1660, 'CH_stretch': 2970, 'specific_mode': 1033},  # C-N
    'H': {'amide_I': 1655, 'CH_stretch': 2930, 'specific_mode': 1596},  # ring
    'I': {'amide_I': 1651, 'CH_stretch': 2965, 'specific_mode': 1170},  # CH3
    'L': {'amide_I': 1653, 'CH_stretch': 2958, 'specific_mode': 1130},  # CH3
    'K': {'amide_I': 1656, 'CH_stretch': 2933, 'specific_mode': 1526},  # NH3+
    'M': {'amide_I': 1652, 'CH_stretch': 2920, 'specific_mode': 700},   # C-S
    'F': {'amide_I': 1649, 'CH_stretch': 3030, 'specific_mode': 1004},  # ring
    'P': {'amide_I': 1635, 'CH_stretch': 2950, 'specific_mode': 918},   # ring
    'S': {'amide_I': 1657, 'CH_stretch': 2945, 'specific_mode': 1030},  # C-O
    'T': {'amide_I': 1655, 'CH_stretch': 2948, 'specific_mode': 1075},  # C-O
    'W': {'amide_I': 1646, 'CH_stretch': 3055, 'specific_mode': 1340},  # indole
    'Y': {'amide_I': 1648, 'CH_stretch': 3040, 'specific_mode': 1517},  # ring
    'V': {'amide_I': 1652, 'CH_stretch': 2962, 'specific_mode': 1130},  # CH3
}

# Add derived vibrational quantities
for aa in AA_VIBRATIONAL:
    m = AA_CLASSICAL[aa]['mass']
    # Effective force constant from ω = √(k/m) → k = m * ω²
    omega = AA_VIBRATIONAL[aa]['specific_mode']
    AA_VIBRATIONAL[aa]['force_constant'] = m * (omega ** 2) / 1e6  # Normalized
    # Mean vibrational frequency
    AA_VIBRATIONAL[aa]['mean_freq'] = np.mean([
        AA_VIBRATIONAL[aa]['amide_I'],
        AA_VIBRATIONAL[aa]['CH_stretch'],
        AA_VIBRATIONAL[aa]['specific_mode']
    ])

# ============================================================================
# LEVEL 4: QUANTUM MECHANICS
# ============================================================================

# Quantum mechanical properties
AA_QUANTUM = {}
h = 6.626e-34  # Planck constant
c = 3e10  # Speed of light (cm/s)
k_B = 1.381e-23  # Boltzmann constant

for aa in AA_VIBRATIONAL:
    m = AA_CLASSICAL[aa]['mass'] * 1.66e-27  # Convert Da to kg
    omega = AA_VIBRATIONAL[aa]['specific_mode']  # cm^-1

    # Zero-point energy: E₀ = (1/2)ℏω
    E_0 = 0.5 * h * c * omega  # Joules

    # De Broglie wavelength at room temperature: λ = h/√(2mkT)
    T = 300  # K
    lambda_dB = h / np.sqrt(2 * m * k_B * T) * 1e10  # Angstroms

    # Tunneling probability proxy (exponential in √(m*barrier))
    # Higher mass = lower tunneling
    tunneling_proxy = np.exp(-np.sqrt(AA_CLASSICAL[aa]['mass']) / 10)

    AA_QUANTUM[aa] = {
        'zero_point_energy': E_0 * 6.022e23 / 4184,  # kcal/mol
        'de_broglie_wavelength': lambda_dB,
        'tunneling_proxy': tunneling_proxy,
        'quantum_mass_factor': 1.0 / np.sqrt(AA_CLASSICAL[aa]['mass']),
    }

# ============================================================================
# LEVEL 5: FORCE FIELD FUNDAMENTALS
# ============================================================================

# AMBER/CHARMM force field parameters
AA_FORCEFIELD = {
    # Bond lengths (Å), angles (degrees), and torsional barriers (kcal/mol)
    'A': {'CA_CB_length': 1.52, 'N_CA_C_angle': 111.2, 'chi1_barrier': 0.0},
    'R': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 2.1},
    'N': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.1, 'chi1_barrier': 1.8},
    'D': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.1, 'chi1_barrier': 1.5},
    'C': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.8, 'chi1_barrier': 1.0},
    'Q': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 2.0},
    'E': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 1.8},
    'G': {'CA_CB_length': 0.0, 'N_CA_C_angle': 112.5, 'chi1_barrier': 0.0},  # No CB
    'H': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 1.2},
    'I': {'CA_CB_length': 1.54, 'N_CA_C_angle': 109.5, 'chi1_barrier': 2.5},
    'L': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.5, 'chi1_barrier': 2.2},
    'K': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 2.3},
    'M': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.8, 'chi1_barrier': 1.8},
    'F': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.5, 'chi1_barrier': 1.0},
    'P': {'CA_CB_length': 1.53, 'N_CA_C_angle': 103.0, 'chi1_barrier': 3.5},  # Proline ring
    'S': {'CA_CB_length': 1.53, 'N_CA_C_angle': 111.0, 'chi1_barrier': 0.8},
    'T': {'CA_CB_length': 1.54, 'N_CA_C_angle': 109.8, 'chi1_barrier': 1.2},
    'W': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.5, 'chi1_barrier': 0.8},
    'Y': {'CA_CB_length': 1.53, 'N_CA_C_angle': 110.5, 'chi1_barrier': 0.9},
    'V': {'CA_CB_length': 1.54, 'N_CA_C_angle': 109.5, 'chi1_barrier': 2.0},
}

# Add derived force field quantities
for aa in AA_FORCEFIELD:
    if AA_FORCEFIELD[aa]['CA_CB_length'] > 0:
        m = AA_CLASSICAL[aa]['mass']
        # Harmonic force constant proxy from bond length and mass
        # k ∝ m / r² (simplified)
        r = AA_FORCEFIELD[aa]['CA_CB_length']
        AA_FORCEFIELD[aa]['bond_force_constant'] = m / (r ** 2) * 10
    else:
        AA_FORCEFIELD[aa]['bond_force_constant'] = 0

# ============================================================================
# EXPERIMENTAL: B-FACTORS AND NMR DATA
# ============================================================================

# Average B-factors by amino acid type (from PDB statistics)
# Higher = more disorder/flexibility
AA_BFACTORS = {
    'A': 25.3, 'R': 32.1, 'N': 31.5, 'D': 30.8, 'C': 24.2,
    'Q': 33.2, 'E': 32.5, 'G': 35.8, 'H': 29.5, 'I': 23.8,
    'L': 25.1, 'K': 35.2, 'M': 27.3, 'F': 24.5, 'P': 30.2,
    'S': 29.8, 'T': 27.5, 'W': 23.2, 'Y': 25.8, 'V': 23.5,
}

# NMR order parameters (S²) - higher = more rigid
# From Lipari-Szabo analysis of relaxation data
AA_NMR_ORDER = {
    'A': 0.88, 'R': 0.82, 'N': 0.84, 'D': 0.85, 'C': 0.87,
    'Q': 0.81, 'E': 0.82, 'G': 0.75, 'H': 0.85, 'I': 0.90,
    'L': 0.88, 'K': 0.80, 'M': 0.85, 'F': 0.89, 'P': 0.92,
    'S': 0.85, 'T': 0.86, 'W': 0.90, 'Y': 0.88, 'V': 0.89,
}

# ============================================================================
# LOAD P-ADIC DATA
# ============================================================================


def load_padic_data() -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Load p-adic radii and full embeddings."""
    import torch

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
        radii[aa] = np.linalg.norm(mean_emb)
        embeddings[aa] = mean_emb

    return radii, embeddings


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def test_level_correlation(
    padic_radii: Dict[str, float],
    property_dict: Dict[str, Dict],
    property_name: str,
    level_name: str
) -> Dict:
    """Test correlation between p-adic radius and a physical property."""
    aa_list = sorted(set(padic_radii.keys()) & set(property_dict.keys()))

    if len(aa_list) < 10:
        return {'error': 'insufficient data'}

    radii = np.array([padic_radii[aa] for aa in aa_list])
    values = np.array([property_dict[aa][property_name] for aa in aa_list])

    # Remove any NaN or infinite values
    valid = np.isfinite(radii) & np.isfinite(values)
    radii = radii[valid]
    values = values[valid]

    if len(radii) < 5:
        return {'error': 'insufficient valid data'}

    r, p = stats.spearmanr(radii, values)

    return {
        'spearman': float(r),
        'p_value': float(p),
        'n': len(radii),
        'level': level_name,
        'property': property_name,
    }


def benchmark_all_levels(padic_radii: Dict[str, float]) -> Dict:
    """Run benchmarks at all physics levels."""
    results = {}

    # Level 0: Biochemistry
    print("\n" + "=" * 70)
    print("LEVEL 0: BIOCHEMISTRY (Traditional)")
    print("=" * 70)

    results['level_0_biochemistry'] = {}
    for prop in ['hydropathy', 'charge', 'polarity', 'aromaticity']:
        res = test_level_correlation(padic_radii, AA_BIOCHEMISTRY, prop, 'biochemistry')
        results['level_0_biochemistry'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Level 1: Classical Mechanics
    print("\n" + "=" * 70)
    print("LEVEL 1: CLASSICAL MECHANICS")
    print("=" * 70)

    results['level_1_classical'] = {}
    for prop in ['mass', 'volume', 'surface_area', 'inertia', 'diffusion_coeff']:
        res = test_level_correlation(padic_radii, AA_CLASSICAL, prop, 'classical')
        results['level_1_classical'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Level 2: Statistical Mechanics
    print("\n" + "=" * 70)
    print("LEVEL 2: STATISTICAL MECHANICS")
    print("=" * 70)

    results['level_2_stat_mech'] = {}
    for prop in ['sidechain_entropy', 'backbone_flexibility', 'rotamer_count', 'log_partition']:
        res = test_level_correlation(padic_radii, AA_STAT_MECH, prop, 'stat_mech')
        results['level_2_stat_mech'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Level 3: Vibrational Physics
    print("\n" + "=" * 70)
    print("LEVEL 3: VIBRATIONAL PHYSICS")
    print("=" * 70)

    results['level_3_vibrational'] = {}
    for prop in ['amide_I', 'CH_stretch', 'specific_mode', 'force_constant', 'mean_freq']:
        res = test_level_correlation(padic_radii, AA_VIBRATIONAL, prop, 'vibrational')
        results['level_3_vibrational'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Level 4: Quantum Mechanics
    print("\n" + "=" * 70)
    print("LEVEL 4: QUANTUM MECHANICS")
    print("=" * 70)

    results['level_4_quantum'] = {}
    for prop in ['zero_point_energy', 'de_broglie_wavelength', 'tunneling_proxy', 'quantum_mass_factor']:
        res = test_level_correlation(padic_radii, AA_QUANTUM, prop, 'quantum')
        results['level_4_quantum'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Level 5: Force Field Fundamentals
    print("\n" + "=" * 70)
    print("LEVEL 5: FORCE FIELD FUNDAMENTALS")
    print("=" * 70)

    results['level_5_forcefield'] = {}
    for prop in ['CA_CB_length', 'N_CA_C_angle', 'chi1_barrier', 'bond_force_constant']:
        res = test_level_correlation(padic_radii, AA_FORCEFIELD, prop, 'forcefield')
        results['level_5_forcefield'][prop] = res
        if 'error' not in res:
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
            print(f"  {prop:<25} ρ = {res['spearman']:+.4f}  p = {res['p_value']:.2e} {sig}")

    # Experimental: B-factors and NMR
    print("\n" + "=" * 70)
    print("EXPERIMENTAL: CRYSTALLOGRAPHIC & NMR")
    print("=" * 70)

    results['experimental'] = {}

    # B-factors
    aa_list = sorted(set(padic_radii.keys()) & set(AA_BFACTORS.keys()))
    radii = np.array([padic_radii[aa] for aa in aa_list])
    bfactors = np.array([AA_BFACTORS[aa] for aa in aa_list])
    r, p = stats.spearmanr(radii, bfactors)
    results['experimental']['b_factor'] = {'spearman': float(r), 'p_value': float(p), 'n': len(aa_list)}
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {'B-factor (disorder)':<25} ρ = {r:+.4f}  p = {p:.2e} {sig}")

    # NMR order parameters
    order_params = np.array([AA_NMR_ORDER[aa] for aa in aa_list])
    r, p = stats.spearmanr(radii, order_params)
    results['experimental']['nmr_order_S2'] = {'spearman': float(r), 'p_value': float(p), 'n': len(aa_list)}
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {'NMR order (S²)':<25} ρ = {r:+.4f}  p = {p:.2e} {sig}")

    return results


def identify_deepest_invariant(results: Dict) -> Dict:
    """Identify the deepest physics level that p-adic structure encodes."""
    level_summary = {}

    for level_name, level_results in results.items():
        if level_name in ['experimental']:
            continue

        # Find best correlation at this level
        best_corr = 0
        best_prop = None
        best_p = 1.0

        for prop, res in level_results.items():
            if 'error' in res:
                continue
            if abs(res['spearman']) > abs(best_corr) and res['p_value'] < 0.05:
                best_corr = res['spearman']
                best_prop = prop
                best_p = res['p_value']

        level_summary[level_name] = {
            'best_property': best_prop,
            'best_correlation': best_corr,
            'best_p_value': best_p,
            'significant': best_p < 0.05
        }

    return level_summary


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("DEEP PHYSICS BENCHMARK: How Deep Are P-adic Invariants?")
    print("=" * 70)
    print("""
Testing whether p-adic codon structure encodes physics at multiple depths:

  Level 0: Biochemistry (hydropathy, charge)
  Level 1: Classical mechanics (mass, inertia)
  Level 2: Statistical mechanics (entropy, partition functions)
  Level 3: Vibrational physics (IR/Raman frequencies)
  Level 4: Quantum mechanics (zero-point energy, tunneling)
  Level 5: Force field fundamentals (bond lengths, force constants)

If p-adic correlates significantly at Level 3+, the genetic code encodes
physics deep enough to predict dynamics and 3D structure over time.
""")

    # Load p-adic data
    print("Loading p-adic embeddings...")
    padic_radii, padic_embeddings = load_padic_data()

    if not padic_radii:
        print("ERROR: Could not load p-adic data!")
        return

    print(f"  Loaded radii for {len(padic_radii)} amino acids")

    # Run benchmarks
    results = benchmark_all_levels(padic_radii)

    # Identify deepest invariant
    level_summary = identify_deepest_invariant(results)
    results['level_summary'] = level_summary

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Deepest Physics Level Encoded")
    print("=" * 70)

    deepest_level = None
    deepest_corr = 0

    for level, summary in level_summary.items():
        status = "✓" if summary['significant'] else "✗"
        if summary['best_property']:
            print(f"  {status} {level}: {summary['best_property']} (ρ = {summary['best_correlation']:+.3f})")
            if summary['significant'] and abs(summary['best_correlation']) > abs(deepest_corr):
                deepest_level = level
                deepest_corr = summary['best_correlation']
        else:
            print(f"  ✗ {level}: No significant correlation")

    print(f"\n  DEEPEST SIGNIFICANT LEVEL: {deepest_level}")

    # Physical interpretation
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)

    interpretations = {
        'level_0_biochemistry': """
  The p-adic structure only encodes biochemistry (hydropathy, charge).
  This is the traditional view - genetic code optimized for biochemistry only.
  Implications: Limited predictive power for dynamics.
""",
        'level_1_classical': """
  The p-adic structure encodes classical mechanics (mass, inertia).
  The genetic code "knows" about Newtonian physics.
  Implications: Can predict equilibrium properties, diffusion.
""",
        'level_2_stat_mech': """
  The p-adic structure encodes statistical mechanics (entropy, partition functions).
  The genetic code "knows" about thermodynamics at the ensemble level.
  Implications: Can predict conformational landscapes, free energies.
""",
        'level_3_vibrational': """
  The p-adic structure encodes vibrational physics (normal modes, IR/Raman).
  The genetic code "knows" about molecular vibrations and force constants.
  Implications: Can predict dynamics, spectroscopy, allostery.

  THIS IS THE KEY THRESHOLD FOR PREDICTING 3D OVER TIME.
""",
        'level_4_quantum': """
  The p-adic structure encodes quantum mechanics (ZPE, tunneling).
  The genetic code "knows" about quantum effects.
  Implications: Can predict proton transfer, enzyme catalysis.
""",
        'level_5_forcefield': """
  The p-adic structure encodes force field fundamentals (bonds, angles).
  The genetic code "knows" about the deepest level of molecular structure.
  Implications: Could serve as a generative prior for structure prediction.
""",
    }

    if deepest_level and deepest_level in interpretations:
        print(interpretations[deepest_level])

    # Implications for 3D prediction
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR 3D STRUCTURE PREDICTION")
    print("=" * 70)

    vibrational_corr = level_summary.get('level_3_vibrational', {}).get('best_correlation', 0)

    if abs(vibrational_corr) > 0.3 and level_summary.get('level_3_vibrational', {}).get('significant'):
        print("""
  ✓ VIBRATIONAL PHYSICS ENCODED (ρ = {:.3f})

  The p-adic codon structure correlates with vibrational frequencies.
  This means it encodes information about:
    - Force constants (k) that determine normal modes
    - Effective masses for each degree of freedom
    - Potential energy surface curvature

  FOR 3D OVER TIME PREDICTION:
    1. Use p-adic embeddings as priors for normal mode analysis
    2. Integrate with elastic network models (ANM/GNM)
    3. Couple to MD simulations for dynamics prediction
    4. Potential for predicting allosteric pathways
""".format(vibrational_corr))
    else:
        print("""
  ✗ VIBRATIONAL PHYSICS NOT SIGNIFICANTLY ENCODED

  The p-adic structure encodes shallower physics (mass, entropy).
  Still useful for:
    - Equilibrium structure prediction (static)
    - Thermodynamic stability (ΔΔG)
    - Conformational entropy estimates

  May need to:
    - Look at dimension-specific encoding (not just radius)
    - Test with different p-adic embeddings
    - Consider codon-level (not AA-level) correlations
""")

    # Save results
    output_file = RESULTS_DIR / "deep_physics_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
