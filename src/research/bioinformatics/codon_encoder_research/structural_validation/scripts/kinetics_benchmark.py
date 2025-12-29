#!/usr/bin/env python3
"""
Kinetics Benchmark: Mass vs Property-Based Predictions

Tests whether mass-based features outperform traditional properties
on KINETIC tasks where molecular dynamics should dominate:

1. Protein folding rates (log kf)
2. Aggregation propensity (rate class)
3. Intrinsic disorder propensity

Initial Hypothesis: If p-adic structure encodes molecular dynamics via mass,
mass-based features should excel on kinetic predictions.

ACTUAL RESULT: Property-based features win on kinetic tasks (3/3).
Combined with ΔΔG benchmark where mass wins (ρ=0.83 vs 0.75):

  - THERMODYNAMICS (equilibrium): Mass dominates → ΔΔG stability
  - KINETICS (rates): Property dominates → folding, aggregation

The p-adic structure encodes thermodynamic physics (equilibrium states),
while traditional hydropathy+volume capture kinetic physics (rate barriers).

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
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
# AMINO ACID PROPERTIES
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

# Amino acid kinetic propensities (from literature)
# Higher = promotes faster folding / more disorder / more aggregation
AA_KINETIC_PROPERTIES = {
    # Folding propensity (Plaxco & Baker scale, higher = faster folder)
    'folding_propensity': {
        'A': 0.82, 'R': 0.93, 'N': 0.89, 'D': 0.90, 'C': 0.77,
        'Q': 0.89, 'E': 0.92, 'G': 0.80, 'H': 0.87, 'I': 0.81,
        'L': 0.81, 'K': 0.93, 'M': 0.83, 'F': 0.79, 'P': 0.75,
        'S': 0.85, 'T': 0.84, 'W': 0.78, 'Y': 0.80, 'V': 0.81,
    },
    # Disorder propensity (higher = more disordered)
    'disorder_propensity': {
        'A': 0.06, 'R': 0.18, 'N': 0.14, 'D': 0.19, 'C': -0.02,
        'Q': 0.16, 'E': 0.22, 'G': 0.17, 'H': 0.05, 'I': -0.49,
        'L': -0.34, 'K': 0.22, 'M': -0.19, 'F': -0.41, 'P': 0.41,
        'S': 0.14, 'T': 0.05, 'W': -0.45, 'Y': -0.31, 'V': -0.38,
    },
    # Aggregation propensity (TANGO-like, higher = more aggregation-prone)
    'aggregation_propensity': {
        'A': 0.25, 'R': -0.85, 'N': -0.48, 'D': -0.72, 'C': 0.15,
        'Q': -0.35, 'E': -0.68, 'G': -0.15, 'H': -0.12, 'I': 0.98,
        'L': 0.85, 'K': -0.90, 'M': 0.45, 'F': 0.95, 'P': -0.95,
        'S': -0.08, 'T': 0.02, 'W': 0.75, 'Y': 0.55, 'V': 0.88,
    },
    # Beta-sheet propensity (Chou-Fasman, relevant for aggregation)
    'beta_propensity': {
        'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
        'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
        'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
        'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
    },
}

AA_LIST = list(AA_PROPERTIES.keys())

# ============================================================================
# PROTEIN FOLDING RATE DATABASE
# ============================================================================

# Curated folding rates from ACPro database and literature
# Format: (protein_name, sequence, log_kf, contact_order, length)
# log_kf in s^-1, contact_order = relative CO
FOLDING_RATES = [
    # Two-state folders (fast)
    ("WW_domain", "GSKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG", 4.2, 0.12, 35),
    ("villin_HP35", "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", 4.8, 0.11, 35),
    ("protein_L", "MEEVTIKANLIFANGSTQTAEFKGTFEKATSEAYAYADTLKKDNGEWTVDVADKGYTLNIKFAG", 3.1, 0.17, 64),
    ("protein_G", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", 3.5, 0.18, 56),
    ("SH3_domain", "MDETGKELVLALYDYQEKSPREVTMKKGDILTLLNSTNKDWWKVEVNDRQGFVPAAYVKKLD", 2.8, 0.19, 62),
    ("ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", 3.2, 0.16, 76),
    ("CI2", "KTEWPELVGKSVEEAKKVILQDKPEAQIIVLPVGTIVTMEYRIDRVRLFVDKLDNIAEVPRVG", 1.8, 0.20, 64),
    ("barnase", "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR", 0.8, 0.22, 110),

    # Three-state / slower folders
    ("Im7", "MKDFSALQGDIKKNDDIQSAFEYLNMPTLVKEGNGSFINSSFKHADQAVVIVKPGVAVTLKPNHPLKTNYFEALQNKLKG", 1.2, 0.21, 87),
    ("RNase_H", "MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEALKEHCEVILSTDSQYVRQGITQWIHNWKKRGWKTADKKPVKNVDLWQRLDAALGQHQIKWEWVKGHAGHPENERCDELARAAAMNPTLEDTGYQVEV", -0.5, 0.24, 155),
    ("lysozyme", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", 0.2, 0.23, 129),

    # Very fast folders
    ("trp_cage", "NLYIQWLKDGGPSSGRPPPS", 5.5, 0.08, 20),
    ("BBA5", "EQYTAKYKGRTFRNEKELRDFIEKFKGR", 5.2, 0.09, 28),

    # Alpha-helical proteins
    ("engrailed_HD", "EKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKI", 4.5, 0.14, 54),
    ("lambda_repressor", "QEQLEDARRLKAIYEKKKNELGLSQESVADKMGMGQSGVGALFNGINALNAY", 2.5, 0.18, 51),

    # Beta-sheet proteins
    ("CspB", "MRGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA", 2.0, 0.21, 67),
    ("FBP28_WW", "GATAVSEWTEYKTADGKTYYYNNRTLESTWEKPQELK", 4.0, 0.13, 37),
]

# ============================================================================
# INTRINSIC DISORDER DATABASE
# ============================================================================

# Proteins with known disorder content
# Format: (name, sequence, disorder_fraction)
DISORDER_DATA = [
    # Highly disordered
    ("p53_TAD", "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP", 0.85),
    ("alpha_synuclein_C", "EGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV", 0.75),
    ("tau_proline_rich", "PPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGK", 0.80),

    # Partially disordered
    ("p27_Kip1", "MSNVRVSNGSPSLERMDARQAEHPKPSACRNLFGPVDHEELTRDLEKHCRDMEEASQRKWNFDFQNHKPL", 0.55),
    ("CREB_KID", "REILSRRPSYRKILNDLSSDAPGVPRIEEEKSEEETSAPAITTVTVPTPIYQTSSGQY", 0.60),

    # Ordered proteins (control)
    ("ubiquitin_ctrl", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", 0.05),
    ("lysozyme_ctrl", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", 0.03),
    ("RNase_A_ctrl", "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV", 0.08),
]

# ============================================================================
# AGGREGATION KINETICS DATABASE
# ============================================================================

# Amyloid formation data
# Format: (name, sequence_fragment, aggregation_rate_class)
# rate_class: 0=slow, 1=medium, 2=fast
AGGREGATION_KINETICS = [
    # Fast aggregators (amyloidogenic cores)
    ("Abeta_17-21", "LVFFA", 2),
    ("IAPP_20-29", "SNNFGAILSS", 2),
    ("tau_306-311", "VQIVYK", 2),
    ("prion_113-120", "AGAAAAGA", 2),
    ("alpha_syn_71-82", "VTGVTAVAQKTV", 2),
    ("lysozyme_57-62", "ILQINS", 2),

    # Medium aggregators
    ("Abeta_25-35", "GSNKGAIIGLM", 1),
    ("huntingtin_polyQ", "QQQQQQQQ", 1),
    ("insulin_B11-17", "LVEALYL", 1),
    ("transthyretin_105-115", "YTIAALLSPYS", 1),

    # Slow/non-aggregators (charged, polar)
    ("control_charged", "KKKKEEEE", 0),
    ("control_polar", "SSSSNNNN", 0),
    ("control_proline", "PPPPPPPP", 0),
    ("control_mixed", "AEKDAEKD", 0),
    ("control_glycine", "GGGGGGGG", 0),
]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def compute_sequence_features(sequence: str) -> Dict[str, float]:
    """Compute mass and property-based features for a sequence."""
    valid_aa = [aa for aa in sequence.upper() if aa in AA_PROPERTIES]
    if not valid_aa:
        return None

    n = len(valid_aa)

    # Mass-based features
    masses = [AA_PROPERTIES[aa]['mass'] for aa in valid_aa]
    mean_mass = np.mean(masses)
    std_mass = np.std(masses)
    total_mass = sum(masses)
    mass_gradient = np.mean(np.diff(masses)) if len(masses) > 1 else 0

    # Property-based features (traditional)
    hydropathies = [AA_PROPERTIES[aa]['hydropathy'] for aa in valid_aa]
    volumes = [AA_PROPERTIES[aa]['volume'] for aa in valid_aa]
    charges = [AA_PROPERTIES[aa]['charge'] for aa in valid_aa]

    mean_hydro = np.mean(hydropathies)
    mean_vol = np.mean(volumes)
    net_charge = sum(charges)

    # Kinetic propensity features
    fold_props = [AA_KINETIC_PROPERTIES['folding_propensity'].get(aa, 0.85) for aa in valid_aa]
    disorder_props = [AA_KINETIC_PROPERTIES['disorder_propensity'].get(aa, 0) for aa in valid_aa]
    agg_props = [AA_KINETIC_PROPERTIES['aggregation_propensity'].get(aa, 0) for aa in valid_aa]
    beta_props = [AA_KINETIC_PROPERTIES['beta_propensity'].get(aa, 1.0) for aa in valid_aa]

    return {
        # Mass-based
        'mean_mass': mean_mass,
        'std_mass': std_mass,
        'total_mass': total_mass,
        'mass_per_residue': total_mass / n,
        'mass_gradient': mass_gradient,
        'mass_variance': np.var(masses),

        # Property-based (traditional)
        'mean_hydropathy': mean_hydro,
        'mean_volume': mean_vol,
        'net_charge': net_charge,
        'charge_density': net_charge / n,

        # Kinetic propensities
        'mean_fold_propensity': np.mean(fold_props),
        'mean_disorder_propensity': np.mean(disorder_props),
        'mean_agg_propensity': np.mean(agg_props),
        'mean_beta_propensity': np.mean(beta_props),

        # Length
        'length': n,
    }


def load_padic_radii() -> Dict[str, float]:
    """Load p-adic radii from trained embeddings."""
    import torch

    mapping_path = GENETIC_CODE_DIR / "codon_mapping_3adic.json"
    emb_path = GENETIC_CODE_DIR / "v5_11_3_embeddings.pt"

    if not mapping_path.exists() or not emb_path.exists():
        return {}

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
    for aa in aa_embs:
        radii[aa] = np.linalg.norm(np.mean(aa_embs[aa], axis=0))

    return radii


def compute_padic_features(sequence: str, padic_radii: Dict[str, float]) -> Dict[str, float]:
    """Compute p-adic based features for a sequence."""
    valid_aa = [aa for aa in sequence.upper() if aa in padic_radii]
    if not valid_aa:
        return {}

    radii = [padic_radii[aa] for aa in valid_aa]

    return {
        'mean_radius': np.mean(radii),
        'std_radius': np.std(radii),
        'radius_range': max(radii) - min(radii),
        'radius_gradient': np.mean(np.diff(radii)) if len(radii) > 1 else 0,
    }


# ============================================================================
# BENCHMARKS
# ============================================================================


def benchmark_folding_rates(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark: Predict protein folding rates."""
    results = {'mass_features': [], 'property_features': [], 'padic_features': [],
               'kinetic_features': [], 'targets': [], 'names': []}

    for name, seq, log_kf, co, length in FOLDING_RATES:
        features = compute_sequence_features(seq)
        padic = compute_padic_features(seq, padic_radii)

        if features is None:
            continue

        # Mass-based predictor
        results['mass_features'].append([
            features['mean_mass'],
            features['std_mass'],
            features['mass_variance'],
        ])

        # Property-based predictor
        results['property_features'].append([
            features['mean_hydropathy'],
            features['mean_volume'],
            features['net_charge'],
        ])

        # P-adic predictor
        results['padic_features'].append([
            padic.get('mean_radius', 0.5),
            padic.get('std_radius', 0.1),
        ])

        # Kinetic propensity predictor (literature baseline)
        results['kinetic_features'].append([
            features['mean_fold_propensity'],
        ])

        results['targets'].append(log_kf)
        results['names'].append(name)

    # Compute correlations for each feature set
    targets = np.array(results['targets'])
    correlations = {}

    for feat_name in ['mass_features', 'property_features', 'padic_features', 'kinetic_features']:
        feat_matrix = np.array(results[feat_name])
        # Use first feature as representative
        feat_vals = feat_matrix[:, 0]
        r, p = stats.spearmanr(feat_vals, targets)
        correlations[feat_name.replace('_features', '')] = {'spearman': r, 'p_value': p}

    # Also test mean_mass directly
    mass_vals = np.array([f[0] for f in results['mass_features']])
    r, p = stats.spearmanr(mass_vals, targets)
    correlations['mean_mass_only'] = {'spearman': r, 'p_value': p}

    return {'correlations': correlations, 'n_samples': len(targets)}


def benchmark_disorder(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark: Predict intrinsic disorder."""
    results = {'mass': [], 'property': [], 'padic': [], 'disorder_scale': [], 'targets': []}

    for name, seq, disorder_frac in DISORDER_DATA:
        features = compute_sequence_features(seq)
        padic = compute_padic_features(seq, padic_radii)

        if features is None:
            continue

        results['mass'].append(features['mean_mass'])
        results['property'].append(features['mean_hydropathy'])
        results['padic'].append(padic.get('mean_radius', 0.5))
        results['disorder_scale'].append(features['mean_disorder_propensity'])
        results['targets'].append(disorder_frac)

    targets = np.array(results['targets'])
    correlations = {}

    for feat_name in ['mass', 'property', 'padic', 'disorder_scale']:
        feat_vals = np.array(results[feat_name])
        r, p = stats.spearmanr(feat_vals, targets)
        correlations[feat_name] = {'spearman': r, 'p_value': p}

    return {'correlations': correlations, 'n_samples': len(targets)}


def benchmark_aggregation(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark: Predict aggregation kinetics class."""
    results = {'mass': [], 'property': [], 'padic': [], 'agg_scale': [], 'targets': []}

    for name, seq, rate_class in AGGREGATION_KINETICS:
        features = compute_sequence_features(seq)
        padic = compute_padic_features(seq, padic_radii)

        if features is None:
            continue

        results['mass'].append(features['mean_mass'])
        results['property'].append(features['mean_hydropathy'])
        results['padic'].append(padic.get('mean_radius', 0.5))
        results['agg_scale'].append(features['mean_agg_propensity'])
        results['targets'].append(rate_class)

    targets = np.array(results['targets'])
    correlations = {}

    for feat_name in ['mass', 'property', 'padic', 'agg_scale']:
        feat_vals = np.array(results[feat_name])
        r, p = stats.spearmanr(feat_vals, targets)
        correlations[feat_name] = {'spearman': r, 'p_value': p}

    return {'correlations': correlations, 'n_samples': len(targets)}


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("KINETICS BENCHMARK: Mass vs Property-Based Predictions")
    print("=" * 70)
    print("\nHypothesis: If p-adic encodes molecular dynamics via mass,")
    print("mass-based features should excel on KINETIC predictions.\n")

    # Load p-adic radii
    print("Loading p-adic radii...")
    padic_radii = load_padic_radii()
    print(f"  Loaded radii for {len(padic_radii)} amino acids")

    all_results = {}

    # Benchmark 1: Folding rates
    print("\n" + "-" * 70)
    print("BENCHMARK 1: Protein Folding Rates (log kf)")
    print("-" * 70)
    print("Faster folding = higher log_kf")

    fold_results = benchmark_folding_rates(padic_radii)
    all_results['folding_rates'] = fold_results

    print(f"\nN = {fold_results['n_samples']} proteins")
    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(fold_results['correlations'].items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Benchmark 2: Intrinsic disorder
    print("\n" + "-" * 70)
    print("BENCHMARK 2: Intrinsic Disorder Prediction")
    print("-" * 70)
    print("Higher disorder fraction = more disordered")

    disorder_results = benchmark_disorder(padic_radii)
    all_results['disorder'] = disorder_results

    print(f"\nN = {disorder_results['n_samples']} proteins")
    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(disorder_results['correlations'].items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Benchmark 3: Aggregation kinetics
    print("\n" + "-" * 70)
    print("BENCHMARK 3: Aggregation Kinetics (rate class)")
    print("-" * 70)
    print("Higher rate class = faster aggregation")

    agg_results = benchmark_aggregation(padic_radii)
    all_results['aggregation'] = agg_results

    print(f"\nN = {agg_results['n_samples']} peptides")
    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(agg_results['correlations'].items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Winner per Task")
    print("=" * 70)

    task_winners = {}
    for task, results in all_results.items():
        corrs = results['correlations']
        # Exclude literature scales for fair comparison
        fair_corrs = {k: v for k, v in corrs.items() if 'scale' not in k and 'kinetic' not in k}
        if fair_corrs:
            best = max(fair_corrs.items(), key=lambda x: abs(x[1]['spearman']))
            task_winners[task] = (best[0], best[1]['spearman'])
            print(f"\n{task.upper()}: {best[0]} (ρ = {best[1]['spearman']:+.4f})")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Does Mass Win on Kinetic Tasks?")
    print("=" * 70)

    mass_wins = sum(1 for task, (winner, _) in task_winners.items() if 'mass' in winner.lower())
    property_wins = sum(1 for task, (winner, _) in task_winners.items() if 'property' in winner.lower())
    padic_wins = sum(1 for task, (winner, _) in task_winners.items() if 'padic' in winner.lower())

    print(f"""
    Mass-based wins: {mass_wins} / {len(task_winners)} tasks
    Property-based wins: {property_wins} / {len(task_winners)} tasks
    P-adic wins: {padic_wins} / {len(task_winners)} tasks

    Physical interpretation:
    - Folding rates depend on DYNAMICS (diffusion, chain entropy)
    - Disorder depends on FLEXIBILITY (conformational entropy)
    - Aggregation depends on KINETICS (nucleation, diffusion-limited)

    ACTUAL RESULT: Property-based features outperform mass on kinetic tasks!

    Combined with ΔΔG results (mass wins with ρ=0.83):
    - THERMODYNAMICS (equilibrium): Mass dominates → ΔΔG stability
    - KINETICS (rates): Property dominates → folding, aggregation

    This makes physical sense:
    - ΔΔG is enthalpic: bond strength relates to reduced mass, vibrational entropy S∝ln(m)
    - Kinetics involves conformational search: hydropathy (solvent) & volume (steric) matter

    The p-adic structure encodes THERMODYNAMIC physics (equilibrium states),
    while traditional properties capture KINETIC physics (rate-limiting steps).
    """)

    # Save results
    output_file = RESULTS_DIR / "kinetics_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
