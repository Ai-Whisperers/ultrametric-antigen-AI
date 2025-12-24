#!/usr/bin/env python3
"""
AlphaFold 3 Results Analysis

Analyze AlphaFold 3 predictions comparing native vs citrullinated peptide-HLA complexes.
Correlate structural predictions with hyperbolic entropy changes.

Output directory: results/alphafold3/22_analysis/

Version: 1.0
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NUM = "22"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_analysis"
PREDICTIONS_DIRS = [
    "predictions/folds_2025_12_17_21_02",
    "predictions/folds_2025_12_17_23_14",
]

# Epitope entropy changes from our analysis
EPITOPE_ENTROPY = {
    'vim_r71': 0.049,
    'fga_r38': 0.041,
    'fgb_r406': 0.038,
}

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "alphafold3" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_predictions_dirs() -> List[Path]:
    """Get all predictions directories."""
    script_dir = Path(__file__).parent
    return [script_dir.parent / "results" / "alphafold3" / d for d in PREDICTIONS_DIRS]


# ============================================================================
# PARSE PREDICTIONS
# ============================================================================

def parse_prediction_folder(folder: Path) -> Dict:
    """
    Parse a single prediction folder.

    Returns dict with confidence scores and metadata.
    """
    name = folder.name

    # Parse name to extract epitope, state (native/cit), and HLA allele
    parts = name.split('_')

    # Find epitope (e.g., vim_r71, fga_r38)
    epitope = None
    state = None
    hla = None

    for i, part in enumerate(parts):
        if part in ['vim', 'fga', 'fgb'] and i + 1 < len(parts):
            epitope = f"{part}_{parts[i+1]}"
        if part in ['native', 'cit']:
            state = part
        if part == 'drb1' and i + 1 < len(parts):
            hla = f"drb1_{parts[i+1]}"

    if not all([epitope, state, hla]):
        return None

    # Load confidence scores (use model 0 as representative)
    confidence_files = list(folder.glob("*_summary_confidences_0.json"))
    if not confidence_files:
        return None

    with open(confidence_files[0]) as f:
        confidences = json.load(f)

    # Load all 5 models for averaging
    all_confidences = []
    for i in range(5):
        conf_file = folder / f"fold_{name}_summary_confidences_{i}.json"
        if conf_file.exists():
            with open(conf_file) as f:
                all_confidences.append(json.load(f))

    # Average across models
    avg_iptm = np.mean([c['iptm'] for c in all_confidences])
    avg_ptm = np.mean([c['ptm'] for c in all_confidences])
    avg_ranking = np.mean([c['ranking_score'] for c in all_confidences])

    # Chain-specific metrics (chain A = peptide, chain B = HLA)
    peptide_ptm = np.mean([c['chain_ptm'][0] for c in all_confidences])
    hla_ptm = np.mean([c['chain_ptm'][1] for c in all_confidences])

    # Interface metrics
    peptide_hla_iptm = np.mean([c['chain_pair_iptm'][0][1] for c in all_confidences])

    return {
        'folder': name,
        'epitope': epitope,
        'state': state,
        'hla': hla,
        'iptm': avg_iptm,
        'ptm': avg_ptm,
        'ranking_score': avg_ranking,
        'peptide_ptm': peptide_ptm,
        'hla_ptm': hla_ptm,
        'peptide_hla_iptm': peptide_hla_iptm,
        'fraction_disordered': confidences.get('fraction_disordered', 0),
        'has_clash': confidences.get('has_clash', 0),
        'n_models': len(all_confidences),
    }


def load_all_predictions(predictions_dirs: List[Path]) -> List[Dict]:
    """Load all prediction results from multiple directories."""
    results = []

    for predictions_dir in predictions_dirs:
        if not predictions_dir.exists():
            continue

        for folder in predictions_dir.iterdir():
            if not folder.is_dir() or folder.name in ['msas', 'templates']:
                continue

            # Skip duplicate runs (ending in _2)
            if folder.name.endswith('_2'):
                continue

            parsed = parse_prediction_folder(folder)
            if parsed:
                results.append(parsed)

    return results


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_native_vs_citrullinated(results: List[Dict]) -> pd.DataFrame:
    """
    Compare native vs citrullinated predictions for each epitope-HLA pair.
    """
    # Group by epitope and HLA
    grouped = defaultdict(dict)

    for r in results:
        key = (r['epitope'], r['hla'])
        grouped[key][r['state']] = r

    comparisons = []

    for (epitope, hla), states in grouped.items():
        if 'native' not in states or 'cit' not in states:
            # Only native available
            if 'native' in states:
                comparisons.append({
                    'epitope': epitope,
                    'hla': hla,
                    'native_iptm': states['native']['iptm'],
                    'native_ranking': states['native']['ranking_score'],
                    'native_peptide_hla_iptm': states['native']['peptide_hla_iptm'],
                    'cit_iptm': None,
                    'cit_ranking': None,
                    'cit_peptide_hla_iptm': None,
                    'delta_iptm': None,
                    'delta_ranking': None,
                    'entropy_change': EPITOPE_ENTROPY.get(epitope, None),
                })
            continue

        native = states['native']
        cit = states['cit']

        comparisons.append({
            'epitope': epitope,
            'hla': hla,
            'native_iptm': native['iptm'],
            'native_ranking': native['ranking_score'],
            'native_peptide_hla_iptm': native['peptide_hla_iptm'],
            'cit_iptm': cit['iptm'],
            'cit_ranking': cit['ranking_score'],
            'cit_peptide_hla_iptm': cit['peptide_hla_iptm'],
            'delta_iptm': cit['iptm'] - native['iptm'],
            'delta_ranking': cit['ranking_score'] - native['ranking_score'],
            'delta_peptide_hla_iptm': cit['peptide_hla_iptm'] - native['peptide_hla_iptm'],
            'entropy_change': EPITOPE_ENTROPY.get(epitope, None),
        })

    return pd.DataFrame(comparisons)


def analyze_binding_changes(comparisons: pd.DataFrame) -> Dict:
    """Analyze how citrullination affects HLA binding."""

    # Filter to pairs with both native and cit
    complete = comparisons.dropna(subset=['delta_iptm'])

    if len(complete) == 0:
        return {'error': 'No complete native/cit pairs found'}

    analysis = {
        'n_comparisons': len(complete),
        'mean_delta_iptm': float(complete['delta_iptm'].mean()),
        'mean_delta_ranking': float(complete['delta_ranking'].mean()),
        'increased_binding': int((complete['delta_iptm'] > 0).sum()),
        'decreased_binding': int((complete['delta_iptm'] < 0).sum()),
        'percent_increased': float((complete['delta_iptm'] > 0).mean() * 100),
    }

    # Correlation with entropy change
    if 'entropy_change' in complete.columns and complete['entropy_change'].notna().any():
        valid = complete.dropna(subset=['entropy_change'])
        if len(valid) > 1:
            corr = valid['delta_iptm'].corr(valid['entropy_change'])
            analysis['entropy_iptm_correlation'] = float(corr) if not np.isnan(corr) else None

    # Per-epitope summary
    epitope_summary = {}
    for epitope in complete['epitope'].unique():
        ep_data = complete[complete['epitope'] == epitope]
        epitope_summary[epitope] = {
            'mean_delta_iptm': float(ep_data['delta_iptm'].mean()),
            'mean_delta_ranking': float(ep_data['delta_ranking'].mean()),
            'n_hla_tested': len(ep_data),
            'entropy_change': EPITOPE_ENTROPY.get(epitope),
        }
    analysis['epitope_summary'] = epitope_summary

    return analysis


# ============================================================================
# CIF STRUCTURE ANALYSIS
# ============================================================================

def parse_cif_coordinates(cif_path: Path) -> Dict:
    """
    Parse CIF file to extract atom coordinates.

    Returns dict with chain -> residue -> atom -> (x, y, z)
    """
    coords = defaultdict(lambda: defaultdict(dict))

    with open(cif_path) as f:
        in_atom_site = False
        headers = []

        for line in f:
            line = line.strip()

            if line.startswith('_atom_site.'):
                headers.append(line.split('.')[1])
                in_atom_site = True
                continue

            if in_atom_site and line.startswith('_'):
                in_atom_site = False
                continue

            if in_atom_site and line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= len(headers):
                    try:
                        data = dict(zip(headers, parts))
                        chain = data.get('auth_asym_id', data.get('label_asym_id', 'A'))
                        res_id = data.get('auth_seq_id', data.get('label_seq_id', '1'))
                        atom_name = data.get('label_atom_id', data.get('auth_atom_id', 'CA'))
                        x = float(data.get('Cartn_x', 0))
                        y = float(data.get('Cartn_y', 0))
                        z = float(data.get('Cartn_z', 0))
                        coords[chain][res_id][atom_name] = (x, y, z)
                    except (ValueError, KeyError):
                        continue

    return dict(coords)


def compute_rmsd(coords1: Dict, coords2: Dict, chain: str = 'A') -> Optional[float]:
    """Compute RMSD between two structures for a given chain (CA atoms only)."""
    if chain not in coords1 or chain not in coords2:
        return None

    c1 = coords1[chain]
    c2 = coords2[chain]

    # Find common residues
    common_res = set(c1.keys()) & set(c2.keys())

    if len(common_res) == 0:
        return None

    # Collect CA coordinates
    diffs = []
    for res in common_res:
        if 'CA' in c1[res] and 'CA' in c2[res]:
            p1 = np.array(c1[res]['CA'])
            p2 = np.array(c2[res]['CA'])
            diffs.append(np.sum((p1 - p2) ** 2))

    if len(diffs) == 0:
        return None

    return float(np.sqrt(np.mean(diffs)))


def analyze_structures(predictions_dirs: List[Path], results: List[Dict]) -> Dict:
    """Analyze structural differences between native and citrullinated."""

    # Group results
    grouped = defaultdict(dict)
    for r in results:
        key = (r['epitope'], r['hla'])
        grouped[key][r['state']] = r

    structural_analysis = []

    for (epitope, hla), states in grouped.items():
        if 'native' not in states or 'cit' not in states:
            continue

        # Find folders in any of the prediction directories
        native_folder = None
        cit_folder = None
        for pred_dir in predictions_dirs:
            nf = pred_dir / states['native']['folder']
            cf = pred_dir / states['cit']['folder']
            if nf.exists():
                native_folder = nf
            if cf.exists():
                cit_folder = cf

        if not native_folder or not cit_folder:
            continue

        # Load model 0 structures
        native_cif = list(native_folder.glob("*_model_0.cif"))
        cit_cif = list(cit_folder.glob("*_model_0.cif"))

        if not native_cif or not cit_cif:
            continue

        native_coords = parse_cif_coordinates(native_cif[0])
        cit_coords = parse_cif_coordinates(cit_cif[0])

        # Compute peptide RMSD (chain A)
        peptide_rmsd = compute_rmsd(native_coords, cit_coords, 'A')

        # Compute HLA RMSD (chain B)
        hla_rmsd = compute_rmsd(native_coords, cit_coords, 'B')

        structural_analysis.append({
            'epitope': epitope,
            'hla': hla,
            'peptide_rmsd': peptide_rmsd,
            'hla_rmsd': hla_rmsd,
            'delta_iptm': states['cit']['iptm'] - states['native']['iptm'],
        })

    return structural_analysis


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("ALPHAFOLD 3 RESULTS ANALYSIS")
    print("Comparing native vs citrullinated peptide-HLA complexes")
    print("=" * 80)

    predictions_dirs = get_predictions_dirs()
    output_dir = get_output_dir()

    print(f"\nPredictions directories:")
    for pred_dir in predictions_dirs:
        print(f"  {pred_dir}")
    print(f"Output directory: {output_dir}")

    # Load predictions
    print("\n[1] Loading AlphaFold 3 predictions...")
    results = load_all_predictions(predictions_dirs)
    print(f"  Loaded {len(results)} prediction results")

    # Create results dataframe
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "all_predictions.csv", index=False)
    print(f"  Saved: all_predictions.csv")

    # Compare native vs citrullinated
    print("\n[2] Comparing native vs citrullinated...")
    comparisons = compare_native_vs_citrullinated(results)
    comparisons.to_csv(output_dir / "native_vs_citrullinated.csv", index=False)
    print(f"  Saved: native_vs_citrullinated.csv")

    # Print comparison table
    print("\n  Comparison Results:")
    print("-" * 80)
    for _, row in comparisons.iterrows():
        epitope = row['epitope'].upper()
        hla = row['hla'].upper().replace('_', '*')

        if pd.notna(row['delta_iptm']):
            delta = row['delta_iptm']
            direction = "↑" if delta > 0 else "↓"
            print(f"  {epitope} + {hla}:")
            print(f"    Native iPTM: {row['native_iptm']:.3f} -> Cit iPTM: {row['cit_iptm']:.3f} ({direction}{abs(delta):.3f})")
            print(f"    Entropy change: {row['entropy_change']:.4f}")
        else:
            print(f"  {epitope} + {hla}: Native only (iPTM: {row['native_iptm']:.3f})")

    # Binding analysis
    print("\n[3] Analyzing binding changes...")
    binding_analysis = analyze_binding_changes(comparisons)

    with open(output_dir / "binding_analysis.json", 'w') as f:
        json.dump(binding_analysis, f, indent=2)
    print(f"  Saved: binding_analysis.json")

    print("\n  Binding Analysis Summary:")
    print(f"    Comparisons with both states: {binding_analysis.get('n_comparisons', 0)}")
    print(f"    Mean Δ iPTM: {binding_analysis.get('mean_delta_iptm', 0):.4f}")
    print(f"    Increased binding: {binding_analysis.get('increased_binding', 0)} ({binding_analysis.get('percent_increased', 0):.1f}%)")
    print(f"    Decreased binding: {binding_analysis.get('decreased_binding', 0)}")

    # Structural analysis
    print("\n[4] Analyzing structures...")
    structural = analyze_structures(predictions_dirs, results)
    if structural:
        df_structural = pd.DataFrame(structural)
        df_structural.to_csv(output_dir / "structural_analysis.csv", index=False)
        print(f"  Saved: structural_analysis.csv")

        print("\n  Structural Differences (RMSD in Å):")
        for s in structural:
            print(f"    {s['epitope'].upper()} + {s['hla'].upper()}:")
            print(f"      Peptide RMSD: {s['peptide_rmsd']:.2f} Å" if s['peptide_rmsd'] else "      Peptide RMSD: N/A")
            print(f"      Δ iPTM: {s['delta_iptm']:.3f}")

    # Summary report
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    complete = comparisons.dropna(subset=['delta_iptm'])
    if len(complete) > 0:
        if binding_analysis.get('mean_delta_iptm', 0) > 0:
            print("\n✓ CITRULLINATION INCREASES HLA BINDING (on average)")
            print(f"  Mean iPTM increase: +{binding_analysis['mean_delta_iptm']:.3f}")
        else:
            print("\n✗ Citrullination decreases HLA binding (on average)")

        # Per-epitope findings
        print("\n  Per-epitope findings:")
        for epitope, data in binding_analysis.get('epitope_summary', {}).items():
            direction = "increases" if data['mean_delta_iptm'] > 0 else "decreases"
            print(f"    {epitope.upper()}: Citrullination {direction} binding by {abs(data['mean_delta_iptm']):.3f} iPTM")
            print(f"      Entropy change: {data['entropy_change']:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    return {
        'results': results,
        'comparisons': comparisons,
        'binding_analysis': binding_analysis,
    }


if __name__ == '__main__':
    main()
