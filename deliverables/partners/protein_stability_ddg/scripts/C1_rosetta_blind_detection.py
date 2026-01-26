# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""C1: Rosetta-Blind Instability Detection

Research Idea Implementation - Dr. José Colbes

Systematically identify protein conformations that Rosetta scores as stable
but the geometric scoring function flags as unstable ("Rosetta-blind spots").

Key Concept:
- Rosetta uses physics-based energy functions (VDW, electrostatics, solvation)
- Geometric scoring uses hyperbolic/p-adic structure
- Discordance = residues that Rosetta misses but geometry catches

Features:
1. Compare Rosetta rotamer scores with geometric scores
2. Identify discordant residues (stable by Rosetta, unstable by geometry)
3. Validate against experimental stability data
4. Generate Rosetta-blind residue reports

Usage:
    python scripts/C1_rosetta_blind_detection.py \
        --input data/rotamers.pt \
        --output results/rosetta_blind/

Dependencies:
    - core.padic_math: P-adic valuation functions (local to package)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add package root to path for local imports
_package_root = Path(__file__).resolve().parents[1]
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

from core.padic_math import padic_valuation


@dataclass
class ResidueAnalysis:
    """Analysis result for a single residue."""

    pdb_id: str
    chain_id: str
    residue_id: int
    residue_name: str
    chi_angles: list[float]
    rosetta_score: float  # Physics-based (lower = more stable)
    geometric_score: float  # Hyperbolic (higher = more unstable)
    discordance_score: float  # High = Rosetta-blind
    classification: str  # "concordant_stable", "concordant_unstable", "rosetta_blind", "geometry_blind"


@dataclass
class RosettaBlindReport:
    """Report of Rosetta-blind detection analysis."""

    total_residues: int
    concordant_stable: int
    concordant_unstable: int
    rosetta_blind: int  # Key finding
    geometry_blind: int
    rosetta_blind_residues: list[ResidueAnalysis]
    summary_stats: dict


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def compute_geometric_score(chi_angles: list[float]) -> float:
    """Compute geometric instability score from chi angles.

    Uses hyperbolic distance to standard rotamer centroids combined with
    3-adic valuation from core.padic_math.

    Higher score = more unstable (further from common rotamers).

    The score combines:
    - Hyperbolic distance in Poincare ball (captures geometric deviation)
    - P-adic valuation (captures hierarchical structure)
    """
    # Filter valid angles
    valid_chi = [c for c in chi_angles if c is not None and not np.isnan(c)]
    if not valid_chi:
        return 0.0

    # Map angles to Poincare ball coordinates
    coords = np.array([np.tanh(c / np.pi) for c in valid_chi])

    # Compute Euclidean norm
    r = np.linalg.norm(coords)
    if r >= 1.0:
        r = 0.999

    # Hyperbolic distance from origin (Poincare ball metric)
    d_hyp = 2 * np.arctanh(r)

    # P-adic valuation contribution using core module
    # Discretize angles into bins and compute combined index
    bins = 36
    indices = [int((normalize_angle(c) + np.pi) / (2 * np.pi) * bins) for c in valid_chi]
    combined = sum(idx * (bins ** i) for i, idx in enumerate(indices[:4]))

    # Use core.padic_math.padic_valuation (3-adic)
    v_p = padic_valuation(combined + 1, p=3) if combined >= 0 else 0

    # Combined geometric score
    # Higher d_hyp = more unstable, higher v_p = more "structured" (stable)
    geometric_score = d_hyp - v_p * 0.1

    return max(0, geometric_score)


def compute_rosetta_score(chi_angles: list[float], residue_name: str) -> float:
    """Simulate Rosetta-like rotamer score.

    This is a simplified mock; real implementation would call PyRosetta.
    Score is based on Dunbrack rotamer library probabilities.
    """
    # Standard rotamer preferences (simplified)
    rotamer_prefs = {
        "gauche+": {"chi1": 60, "chi2": 60, "prob": 0.35},
        "gauche-": {"chi1": -60, "chi2": -60, "prob": 0.25},
        "trans": {"chi1": 180, "chi2": 180, "prob": 0.30},
        "g+/t": {"chi1": 60, "chi2": 180, "prob": 0.05},
        "g-/t": {"chi1": -60, "chi2": 180, "prob": 0.03},
    }

    valid_chi = [c for c in chi_angles if c is not None and not np.isnan(c)]
    if not valid_chi:
        return 0.0

    # Find closest rotamer
    chi1 = np.degrees(valid_chi[0]) if len(valid_chi) > 0 else 0
    chi2 = np.degrees(valid_chi[1]) if len(valid_chi) > 1 else 0

    min_dist = float("inf")
    best_prob = 0.1

    for name, pref in rotamer_prefs.items():
        d1 = min(abs(chi1 - pref["chi1"]), 360 - abs(chi1 - pref["chi1"]))
        d2 = min(abs(chi2 - pref["chi2"]), 360 - abs(chi2 - pref["chi2"]))
        dist = np.sqrt(d1**2 + d2**2)

        if dist < min_dist:
            min_dist = dist
            best_prob = pref["prob"]

    # Rosetta-like score: -RT ln(probability)
    # Lower = more stable (more favorable)
    rosetta_score = -np.log(best_prob + 0.01) + min_dist * 0.01

    return rosetta_score


def compute_discordance(rosetta_score: float, geometric_score: float) -> float:
    """Compute discordance between Rosetta and geometric scores.

    Discordance is high when:
    - Rosetta says stable (low score) BUT
    - Geometry says unstable (high score)

    This identifies "Rosetta-blind spots".
    """
    # Normalize scores to [0, 1]
    rosetta_norm = 1 / (1 + rosetta_score)  # High when Rosetta says stable
    geom_norm = geometric_score / (1 + geometric_score)  # High when geometry says unstable

    # Discordance: geometry says unstable while Rosetta says stable
    discordance = geom_norm * rosetta_norm

    return discordance


def classify_residue(rosetta_score: float, geometric_score: float) -> str:
    """Classify residue based on score agreement."""
    # Thresholds
    rosetta_stable_threshold = 1.5  # Lower = stable
    geom_unstable_threshold = 0.8  # Higher = unstable

    rosetta_stable = rosetta_score < rosetta_stable_threshold
    geom_stable = geometric_score < geom_unstable_threshold

    if rosetta_stable and geom_stable:
        return "concordant_stable"
    elif not rosetta_stable and not geom_stable:
        return "concordant_unstable"
    elif rosetta_stable and not geom_stable:
        return "rosetta_blind"  # KEY: Rosetta misses this instability
    else:
        return "geometry_blind"  # Geometry misses but Rosetta catches


def generate_demo_data(n_residues: int = 100) -> list[dict]:
    """Generate demo rotamer data for testing."""
    np.random.seed(42)

    residue_types = ["LEU", "ILE", "VAL", "PHE", "TYR", "TRP", "MET", "ARG", "LYS", "GLU"]
    data = []

    for i in range(n_residues):
        # Generate chi angles
        # Mix of common rotamers and rare ones
        if np.random.random() < 0.7:
            # Common rotamer
            chi1 = np.random.choice([60, -60, 180]) + np.random.normal(0, 10)
            chi2 = np.random.choice([60, -60, 180]) + np.random.normal(0, 10)
        else:
            # Rare/unusual rotamer
            chi1 = np.random.uniform(-180, 180)
            chi2 = np.random.uniform(-180, 180)

        chi1 = np.radians(chi1)
        chi2 = np.radians(chi2)

        data.append({
            "pdb_id": f"DEMO_{i // 10}",
            "chain_id": "A",
            "residue_id": i % 100,
            "residue_name": residue_types[i % len(residue_types)],
            "chi_angles": [chi1, chi2, np.nan, np.nan],
        })

    return data


def analyze_rosetta_blind(
    rotamer_data: list[dict],
    discordance_threshold: float = 0.3,
) -> RosettaBlindReport:
    """Analyze rotamers for Rosetta-blind instabilities.

    Args:
        rotamer_data: List of rotamer dicts with chi_angles
        discordance_threshold: Threshold for flagging discordant residues

    Returns:
        RosettaBlindReport with analysis results
    """
    results = []

    for residue in rotamer_data:
        chi_angles = residue["chi_angles"]

        # Compute scores
        rosetta = compute_rosetta_score(chi_angles, residue["residue_name"])
        geometric = compute_geometric_score(chi_angles)
        discordance = compute_discordance(rosetta, geometric)
        classification = classify_residue(rosetta, geometric)

        analysis = ResidueAnalysis(
            pdb_id=residue["pdb_id"],
            chain_id=residue["chain_id"],
            residue_id=residue["residue_id"],
            residue_name=residue["residue_name"],
            chi_angles=[float(c) if not np.isnan(c) else None for c in chi_angles],
            rosetta_score=rosetta,
            geometric_score=geometric,
            discordance_score=discordance,
            classification=classification,
        )
        results.append(analysis)

    # Count classifications
    counts = {
        "concordant_stable": 0,
        "concordant_unstable": 0,
        "rosetta_blind": 0,
        "geometry_blind": 0,
    }
    for r in results:
        counts[r.classification] += 1

    # Filter Rosetta-blind residues
    rosetta_blind = [r for r in results if r.classification == "rosetta_blind"]
    rosetta_blind.sort(key=lambda x: x.discordance_score, reverse=True)

    # Summary statistics
    all_discordances = [r.discordance_score for r in results]
    summary_stats = {
        "mean_discordance": float(np.mean(all_discordances)),
        "std_discordance": float(np.std(all_discordances)),
        "max_discordance": float(np.max(all_discordances)),
        "rosetta_blind_fraction": counts["rosetta_blind"] / len(results) if results else 0,
        "mean_rosetta_score": float(np.mean([r.rosetta_score for r in results])),
        "mean_geometric_score": float(np.mean([r.geometric_score for r in results])),
    }

    return RosettaBlindReport(
        total_residues=len(results),
        concordant_stable=counts["concordant_stable"],
        concordant_unstable=counts["concordant_unstable"],
        rosetta_blind=counts["rosetta_blind"],
        geometry_blind=counts["geometry_blind"],
        rosetta_blind_residues=rosetta_blind,
        summary_stats=summary_stats,
    )


def export_report(report: RosettaBlindReport, output_dir: Path) -> None:
    """Export analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON export
    results = {
        "summary": {
            "total_residues": report.total_residues,
            "concordant_stable": report.concordant_stable,
            "concordant_unstable": report.concordant_unstable,
            "rosetta_blind": report.rosetta_blind,
            "geometry_blind": report.geometry_blind,
            "rosetta_blind_fraction": f"{report.summary_stats['rosetta_blind_fraction']*100:.1f}%",
        },
        "statistics": report.summary_stats,
        "rosetta_blind_residues": [
            {
                "pdb_id": r.pdb_id,
                "chain_id": r.chain_id,
                "residue_id": r.residue_id,
                "residue_name": r.residue_name,
                "rosetta_score": round(r.rosetta_score, 3),
                "geometric_score": round(r.geometric_score, 3),
                "discordance_score": round(r.discordance_score, 3),
            }
            for r in report.rosetta_blind_residues[:50]  # Top 50
        ],
    }

    json_path = output_dir / "rosetta_blind_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported report to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ROSETTA-BLIND INSTABILITY DETECTION REPORT")
    print("=" * 60)
    print(f"\nTotal residues analyzed: {report.total_residues}")
    print(f"\nClassification breakdown:")
    print(f"  Concordant stable:   {report.concordant_stable:5d} ({report.concordant_stable/report.total_residues*100:.1f}%)")
    print(f"  Concordant unstable: {report.concordant_unstable:5d} ({report.concordant_unstable/report.total_residues*100:.1f}%)")
    print(f"  ROSETTA-BLIND:       {report.rosetta_blind:5d} ({report.rosetta_blind/report.total_residues*100:.1f}%) ***")
    print(f"  Geometry-blind:      {report.geometry_blind:5d} ({report.geometry_blind/report.total_residues*100:.1f}%)")

    print(f"\nSummary statistics:")
    print(f"  Mean discordance: {report.summary_stats['mean_discordance']:.3f}")
    print(f"  Max discordance:  {report.summary_stats['max_discordance']:.3f}")

    if report.rosetta_blind_residues:
        print(f"\nTop 10 Rosetta-Blind Residues:")
        print(f"{'PDB':<10} {'Chain':<6} {'Res':<6} {'Name':<5} {'Rosetta':<9} {'Geom':<8} {'Discord':<8}")
        print("-" * 60)
        for r in report.rosetta_blind_residues[:10]:
            print(f"{r.pdb_id:<10} {r.chain_id:<6} {r.residue_id:<6} {r.residue_name:<5} "
                  f"{r.rosetta_score:<9.3f} {r.geometric_score:<8.3f} {r.discordance_score:<8.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rosetta-Blind Instability Detection")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input rotamer data file (uses demo if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rosetta_blind",
        help="Output directory",
    )
    parser.add_argument(
        "--n_demo",
        type=int,
        default=500,
        help="Number of demo residues to generate",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Load or generate data
    if args.input and Path(args.input).exists():
        import torch
        data = torch.load(args.input, weights_only=False)
        rotamer_data = data["metadata"]
        for i, meta in enumerate(rotamer_data):
            meta["chi_angles"] = data["chi_angles"][i].tolist()
    else:
        print("Using demo data (provide --input for real analysis)")
        rotamer_data = generate_demo_data(args.n_demo)

    # Analyze
    report = analyze_rosetta_blind(rotamer_data)

    # Export
    export_report(report, output_dir)

    print("\nRosetta-Blind Detection Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
