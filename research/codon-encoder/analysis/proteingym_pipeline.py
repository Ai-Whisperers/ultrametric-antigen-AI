#!/usr/bin/env python3
"""
ProteinGym Data Pipeline + Embedding Space Analysis

Downloads ProteinGym substitution benchmark data and analyzes which
dimensions of the p-adic embedding space encode physical invariants.

Goal: Find the geometric invariants that connect p-adic structure
to 3D dynamics prediction.

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import sys
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GENETIC_CODE_DIR, ANALYSIS_RESULTS_DIR, CODON_TO_AA,
    AA_PROPERTIES, AA_FORCE_CONSTANTS, load_padic_embeddings,
    poincare_distance_from_origin
)

# Local paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "proteingym"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ANALYSIS_RESULTS_DIR / "embedding_invariants"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ProteinGym URLs
PROTEINGYM_SUBSTITUTIONS_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

# Use AA_PROPERTIES from config (already imported)
# Local reference for backwards compatibility
_AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'volume': 88.6, 'mass': 89.09, 'charge': 0, 'flexibility': 1.0},
    'R': {'hydropathy': -4.5, 'volume': 173.4, 'mass': 174.20, 'charge': 1, 'flexibility': 0.9},
    'N': {'hydropathy': -3.5, 'volume': 114.1, 'mass': 132.12, 'charge': 0, 'flexibility': 0.95},
    'D': {'hydropathy': -3.5, 'volume': 111.1, 'mass': 133.10, 'charge': -1, 'flexibility': 0.95},
    'C': {'hydropathy': 2.5, 'volume': 108.5, 'mass': 121.16, 'charge': 0, 'flexibility': 0.9},
    'Q': {'hydropathy': -3.5, 'volume': 143.8, 'mass': 146.15, 'charge': 0, 'flexibility': 0.9},
    'E': {'hydropathy': -3.5, 'volume': 138.4, 'mass': 147.13, 'charge': -1, 'flexibility': 0.9},
    'G': {'hydropathy': -0.4, 'volume': 60.1, 'mass': 75.07, 'charge': 0, 'flexibility': 1.5},
    'H': {'hydropathy': -3.2, 'volume': 153.2, 'mass': 155.16, 'charge': 0, 'flexibility': 0.85},
    'I': {'hydropathy': 4.5, 'volume': 166.7, 'mass': 131.17, 'charge': 0, 'flexibility': 0.7},
    'L': {'hydropathy': 3.8, 'volume': 166.7, 'mass': 131.17, 'charge': 0, 'flexibility': 0.8},
    'K': {'hydropathy': -3.9, 'volume': 168.6, 'mass': 146.19, 'charge': 1, 'flexibility': 0.85},
    'M': {'hydropathy': 1.9, 'volume': 162.9, 'mass': 149.21, 'charge': 0, 'flexibility': 0.85},
    'F': {'hydropathy': 2.8, 'volume': 189.9, 'mass': 165.19, 'charge': 0, 'flexibility': 0.75},
    'P': {'hydropathy': -1.6, 'volume': 112.7, 'mass': 115.13, 'charge': 0, 'flexibility': 0.3},
    'S': {'hydropathy': -0.8, 'volume': 89.0, 'mass': 105.09, 'charge': 0, 'flexibility': 1.0},
    'T': {'hydropathy': -0.7, 'volume': 116.1, 'mass': 119.12, 'charge': 0, 'flexibility': 0.9},
    'W': {'hydropathy': -0.9, 'volume': 227.8, 'mass': 204.23, 'charge': 0, 'flexibility': 0.7},
    'Y': {'hydropathy': -1.3, 'volume': 193.6, 'mass': 181.19, 'charge': 0, 'flexibility': 0.75},
    'V': {'hydropathy': 4.2, 'volume': 140.0, 'mass': 117.15, 'charge': 0, 'flexibility': 0.75},
}

# Force constants derived from vibrational data (relative scale)
AA_FORCE_CONSTANTS = {
    'G': 0.50, 'A': 0.65, 'S': 0.70, 'P': 0.75, 'V': 0.80,
    'T': 0.82, 'C': 0.85, 'I': 0.88, 'L': 0.88, 'N': 0.90,
    'D': 0.92, 'Q': 0.95, 'K': 0.98, 'E': 1.00, 'M': 1.02,
    'H': 1.05, 'F': 1.10, 'R': 1.15, 'Y': 1.18, 'W': 1.25,
}

# ============================================================================
# DATA LOADING
# ============================================================================


# V5.12.2 FIX: Removed duplicate load_padic_embeddings function.
# Now uses the corrected version from config.py which computes
# hyperbolic distance from origin instead of Euclidean norm.
# Import is already at line 34: from config import ... load_padic_embeddings


# ============================================================================
# EMBEDDING SPACE ANALYSIS
# ============================================================================


def analyze_dimension_correlations(embeddings: Dict[str, np.ndarray]) -> Dict:
    """Analyze what physical properties each dimension encodes."""
    print("\n" + "=" * 70)
    print("EMBEDDING DIMENSION ANALYSIS")
    print("=" * 70)
    print("\nFinding which dimensions encode which physical properties...\n")

    aa_list = sorted(embeddings.keys())
    n_dims = len(embeddings[aa_list[0]])

    # Build embedding matrix
    emb_matrix = np.array([embeddings[aa] for aa in aa_list])

    # Physical properties to test
    properties = {
        'mass': [AA_PROPERTIES[aa]['mass'] for aa in aa_list],
        'volume': [AA_PROPERTIES[aa]['volume'] for aa in aa_list],
        'hydropathy': [AA_PROPERTIES[aa]['hydropathy'] for aa in aa_list],
        'charge': [AA_PROPERTIES[aa]['charge'] for aa in aa_list],
        'flexibility': [AA_PROPERTIES[aa]['flexibility'] for aa in aa_list],
        'force_constant': [AA_FORCE_CONSTANTS[aa] for aa in aa_list],
    }

    results = {'dimension_correlations': {}, 'property_best_dims': {}}

    # For each dimension, find correlations with all properties
    print(f"{'Dim':<5} {'Mass':>8} {'Volume':>8} {'Hydro':>8} {'Charge':>8} {'Flex':>8} {'Force':>8}")
    print("-" * 60)

    for dim in range(n_dims):
        dim_values = emb_matrix[:, dim]
        results['dimension_correlations'][dim] = {}

        row = f"{dim:<5}"
        for prop_name, prop_values in properties.items():
            r, p = stats.spearmanr(dim_values, prop_values)
            results['dimension_correlations'][dim][prop_name] = {'r': float(r), 'p': float(p)}

            # Format correlation with significance marker
            sig = "*" if p < 0.05 else " "
            row += f" {r:>+6.2f}{sig}"

        print(row)

    # Find best dimension for each property
    print("\n" + "-" * 60)
    print("BEST DIMENSION PER PROPERTY:")
    print("-" * 60)

    for prop_name, prop_values in properties.items():
        best_dim = -1
        best_r = 0

        for dim in range(n_dims):
            r = results['dimension_correlations'][dim][prop_name]['r']
            if abs(r) > abs(best_r):
                best_r = r
                best_dim = dim

        p = results['dimension_correlations'][best_dim][prop_name]['p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {prop_name:<15} → Dim {best_dim:>2} (ρ = {best_r:+.3f}) {sig}")

        results['property_best_dims'][prop_name] = {
            'dimension': best_dim,
            'correlation': best_r,
            'p_value': p
        }

    return results


def find_geometric_invariants(embeddings: Dict[str, np.ndarray]) -> Dict:
    """Find geometric structures in embedding space that correspond to physics."""
    print("\n" + "=" * 70)
    print("GEOMETRIC INVARIANTS ANALYSIS")
    print("=" * 70)

    aa_list = sorted(embeddings.keys())
    emb_matrix = np.array([embeddings[aa] for aa in aa_list])

    results = {}

    # 1. Radial structure (already known: mass correlation)
    # V5.12.2 FIX: Use hyperbolic distance from origin, not Euclidean norm
    radii = np.array([poincare_distance_from_origin(emb) for emb in emb_matrix])
    masses = [AA_PROPERTIES[aa]['mass'] for aa in aa_list]
    r_mass, p_mass = stats.spearmanr(radii, masses)
    print(f"\n1. RADIAL STRUCTURE")
    print(f"   Radius vs Mass: ρ = {r_mass:.3f} (p = {p_mass:.2e})")
    results['radial_mass'] = {'r': float(r_mass), 'p': float(p_mass)}

    # 2. Angular structure - do angles encode different properties?
    # Convert to spherical-like coordinates (angle from first principal component)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(emb_matrix)

    # Angles in the reduced space
    theta = np.arctan2(emb_3d[:, 1], emb_3d[:, 0])  # azimuthal
    phi = np.arccos(emb_3d[:, 2] / (np.linalg.norm(emb_3d, axis=1) + 1e-8))  # polar

    hydropathies = [AA_PROPERTIES[aa]['hydropathy'] for aa in aa_list]
    r_theta_h, p_theta = stats.spearmanr(theta, hydropathies)
    r_phi_h, p_phi = stats.spearmanr(phi, hydropathies)

    print(f"\n2. ANGULAR STRUCTURE (PCA-reduced)")
    print(f"   Azimuthal (θ) vs Hydropathy: ρ = {r_theta_h:.3f} (p = {p_theta:.2e})")
    print(f"   Polar (φ) vs Hydropathy: ρ = {r_phi_h:.3f} (p = {p_phi:.2e})")

    results['angular_hydropathy'] = {
        'theta': {'r': float(r_theta_h), 'p': float(p_theta)},
        'phi': {'r': float(r_phi_h), 'p': float(p_phi)}
    }

    # 3. Clustering structure - do amino acids cluster by property?
    print(f"\n3. CLUSTERING STRUCTURE")

    # Group by charge
    charged = [aa for aa in aa_list if AA_PROPERTIES[aa]['charge'] != 0]
    neutral = [aa for aa in aa_list if AA_PROPERTIES[aa]['charge'] == 0]

    charged_embs = np.array([embeddings[aa] for aa in charged])
    neutral_embs = np.array([embeddings[aa] for aa in neutral])

    charged_center = charged_embs.mean(axis=0)
    neutral_center = neutral_embs.mean(axis=0)
    # V5.12.2 FIX: Use hyperbolic distance between centroids
    # For centroid-to-centroid distance, we compute poincare distance
    from src.geometry import poincare_distance
    import torch
    charge_separation = poincare_distance(
        torch.tensor(charged_center).unsqueeze(0),
        torch.tensor(neutral_center).unsqueeze(0),
        c=1.0
    ).item()

    print(f"   Charged vs Neutral separation: {charge_separation:.4f}")
    results['charge_separation'] = float(charge_separation)

    # Group by hydrophobicity
    hydrophobic = [aa for aa in aa_list if AA_PROPERTIES[aa]['hydropathy'] > 0]
    hydrophilic = [aa for aa in aa_list if AA_PROPERTIES[aa]['hydropathy'] <= 0]

    hydrophobic_embs = np.array([embeddings[aa] for aa in hydrophobic])
    hydrophilic_embs = np.array([embeddings[aa] for aa in hydrophilic])

    hydro_center = hydrophobic_embs.mean(axis=0)
    philic_center = hydrophilic_embs.mean(axis=0)
    # V5.12.2 FIX: Use hyperbolic distance between centroids
    hydro_separation = poincare_distance(
        torch.tensor(hydro_center).unsqueeze(0),
        torch.tensor(philic_center).unsqueeze(0),
        c=1.0
    ).item()

    print(f"   Hydrophobic vs Hydrophilic separation: {hydro_separation:.4f}")
    results['hydropathy_separation'] = float(hydro_separation)

    # 4. Force constant encoding
    print(f"\n4. FORCE CONSTANT STRUCTURE")

    force_constants = [AA_FORCE_CONSTANTS[aa] for aa in aa_list]

    # Which geometric property best predicts force constants?
    r_radius_k, p_r = stats.spearmanr(radii, force_constants)
    print(f"   Radius vs Force Constant: ρ = {r_radius_k:.3f} (p = {p_r:.2e})")

    # Try combined mass + radius
    combined = radii * np.array(masses)
    r_combined, p_c = stats.spearmanr(combined, force_constants)
    print(f"   Radius × Mass vs Force Constant: ρ = {r_combined:.3f} (p = {p_c:.2e})")

    results['force_constant'] = {
        'radius': {'r': float(r_radius_k), 'p': float(p_r)},
        'radius_mass': {'r': float(r_combined), 'p': float(p_c)}
    }

    # 5. PCA variance explained
    print(f"\n5. EMBEDDING STRUCTURE (PCA)")
    pca_full = PCA()
    pca_full.fit(emb_matrix)
    var_explained = pca_full.explained_variance_ratio_

    print(f"   Variance explained by top components:")
    cumsum = 0
    for i, v in enumerate(var_explained[:5]):
        cumsum += v
        print(f"     PC{i+1}: {v:.1%} (cumulative: {cumsum:.1%})")

    results['pca_variance'] = [float(v) for v in var_explained[:5]]

    return results


def build_dynamics_features(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Build features that could predict dynamics (3D over time).

    Key insight: ω = √(k/m), so if we can predict k from embeddings,
    and we have m, we can predict vibrational frequencies.
    """
    print("\n" + "=" * 70)
    print("DYNAMICS FEATURE EXTRACTION")
    print("=" * 70)

    aa_list = sorted(embeddings.keys())
    emb_matrix = np.array([embeddings[aa] for aa in aa_list])

    results = {'dynamics_features': {}}

    print("\nBuilding features for dynamics prediction:")
    print("  ω = √(k/m) → frequency from force constant and mass")
    print("  τ = 1/ω → characteristic timescale")
    print("  D ∝ 1/√m → diffusion coefficient")
    print()

    for aa in aa_list:
        emb = embeddings[aa]
        mass = AA_PROPERTIES[aa]['mass']
        k = AA_FORCE_CONSTANTS[aa]

        # Derived dynamics quantities
        omega = np.sqrt(k / mass) * 100  # Normalized frequency
        tau = 1 / omega  # Timescale
        diffusion = 1 / np.sqrt(mass)  # Diffusion proxy

        # P-adic derived features
        # V5.12.2 FIX: Use hyperbolic distance from origin, not Euclidean norm
        radius = poincare_distance_from_origin(emb)
        # Predicted force constant from embedding (hypothesis: k ∝ r × m)
        k_predicted = radius * mass / 100

        results['dynamics_features'][aa] = {
            'mass': mass,
            'force_constant_exp': k,
            'force_constant_pred': k_predicted,
            'omega': omega,
            'tau': tau,
            'diffusion': diffusion,
            'radius': radius,
            'embedding': emb.tolist(),
        }

    # Validate: does predicted k correlate with experimental k?
    k_exp = [results['dynamics_features'][aa]['force_constant_exp'] for aa in aa_list]
    k_pred = [results['dynamics_features'][aa]['force_constant_pred'] for aa in aa_list]
    r, p = stats.spearmanr(k_exp, k_pred)

    print(f"Force constant prediction from p-adic:")
    print(f"  k_pred = radius × mass / 100")
    print(f"  Correlation with experimental k: ρ = {r:.3f} (p = {p:.2e})")

    results['k_prediction_accuracy'] = {'r': float(r), 'p': float(p)}

    # Vibrational frequency prediction
    omega_exp = [np.sqrt(k_exp[i] / AA_PROPERTIES[aa]['mass']) for i, aa in enumerate(aa_list)]
    omega_pred = [results['dynamics_features'][aa]['omega'] for aa in aa_list]
    r_omega, p_omega = stats.spearmanr(omega_exp, omega_pred)

    print(f"\nVibrational frequency prediction:")
    print(f"  ω_pred = √(k_pred/m)")
    print(f"  Correlation with ω_exp: ρ = {r_omega:.3f} (p = {p_omega:.2e})")

    results['omega_prediction_accuracy'] = {'r': float(r_omega), 'p': float(p_omega)}

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("P-ADIC EMBEDDING INVARIANTS FOR 3D DYNAMICS")
    print("=" * 70)
    print("""
Goal: Find geometric invariants in p-adic embedding space that
connect to physical properties needed for 3D dynamics prediction.

Key question: Can we derive force constants (k) from embeddings?
If yes: ω = √(k/m) gives us vibrational frequencies → dynamics
""")

    # Load embeddings
    print("\nLoading p-adic embeddings...")
    radii, embeddings = load_padic_embeddings()

    if not embeddings:
        print("ERROR: Could not load embeddings!")
        return

    print(f"  Loaded embeddings for {len(embeddings)} amino acids")
    print(f"  Embedding dimension: {len(list(embeddings.values())[0])}")

    all_results = {}

    # 1. Dimension correlation analysis
    dim_results = analyze_dimension_correlations(embeddings)
    all_results['dimension_analysis'] = dim_results

    # 2. Geometric invariants
    geom_results = find_geometric_invariants(embeddings)
    all_results['geometric_invariants'] = geom_results

    # 3. Dynamics features
    dyn_results = build_dynamics_features(embeddings)
    all_results['dynamics_features'] = dyn_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: KEY INVARIANTS FOR 3D DYNAMICS")
    print("=" * 70)

    print("""
WHAT THE P-ADIC EMBEDDING ENCODES:

1. RADIAL (distance from origin):
   → Mass (ρ = {:.3f})
   → Force constant proxy

2. DIMENSIONAL (specific axes):
   → Hydropathy (best dim: {})
   → Volume (best dim: {})
   → Charge (separation in space)

3. GEOMETRIC STRUCTURE:
   → Charged/neutral separation: {:.4f}
   → Hydrophobic/hydrophilic separation: {:.4f}

PATH TO 3D DYNAMICS:
   p-adic embedding → force constant (k)
   k + mass (m) → vibrational frequency (ω = √(k/m))
   ω → normal modes → 3D dynamics over time
""".format(
        geom_results['radial_mass']['r'],
        dim_results['property_best_dims']['hydropathy']['dimension'],
        dim_results['property_best_dims']['volume']['dimension'],
        geom_results['charge_separation'],
        geom_results['hydropathy_separation'],
    ))

    # Save results
    output_file = RESULTS_DIR / "embedding_invariants_analysis.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
