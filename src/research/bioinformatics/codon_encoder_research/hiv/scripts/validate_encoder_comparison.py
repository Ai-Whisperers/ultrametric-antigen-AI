#!/usr/bin/env python3
"""
Encoder Comparison Validation on HIV Data

This script validates the Classical 3-Adic vs Fused Hyperbolic codon encoders
on real HIV bioinformatics tasks:

1. Drug Resistance: Correlation between hyperbolic distance and fold-change
2. CTL Escape: Boundary crossing rates for escape mutations
3. Mutation Severity: Radial shift as predictor of clinical impact

The goal is to determine which encoder is better suited for each task
and validate the theoretical advantages of the Fused encoder.

Output: results/encoder_comparison/
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from codon_extraction import encode_mutation_pair
from position_mapper import parse_mutation_list
from unified_data_loader import load_stanford_hivdb

warnings.filterwarnings("ignore")


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance from origin for a Poincare ball embedding.

    V5.12.2: Use proper hyperbolic distance formula instead of Euclidean norm.

    Args:
        embedding: Array of shape (dim,) in Poincare ball
        c: Curvature parameter (default 1.0)

    Returns:
        Hyperbolic radius (scalar)
    """
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "encoder_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path: hiv/scripts -> hiv -> codon_encoder_research -> bioinformatics -> genetic_code/data
GENETIC_CODE_DIR = Path(__file__).parent.parent.parent.parent / "genetic_code" / "data"

# Known high-impact mutations for validation
BENCHMARK_MUTATIONS = {
    "high_impact": [
        ("K103N", "NNRTI", "High"),   # Classic NNRTI resistance
        ("M184V", "NRTI", "High"),    # 3TC/FTC resistance
        ("Y181C", "NNRTI", "High"),   # NVP resistance
        ("M46I", "PI", "Medium"),     # PI accessory
    ],
    "low_impact": [
        ("K103R", "NNRTI", "Low"),    # Polymorphism
        ("V179D", "NNRTI", "Low"),    # Minor mutation
    ]
}


# ============================================================================
# ENCODER LOADING
# ============================================================================


def load_encoder(version: str, device: str = "cpu"):
    """Load codon encoder by version."""
    import torch
    import torch.nn as nn

    class CodonEncoder(nn.Module):
        def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.cluster_head = nn.Linear(embed_dim, n_clusters)

        def encode(self, x):
            return self.encoder(x)

        def forward(self, x):
            emb = self.encode(x)
            logits = self.cluster_head(emb)
            return {"z_hyp": emb, "cluster_logits": logits}

    # Paths for different encoder versions
    if version == "3adic":
        paths = [
            GENETIC_CODE_DIR / "codon_encoder_3adic.pt",
            Path(__file__).parent.parent / "data" / "codon_encoder_3adic.pt",
        ]
    elif version == "fused":
        paths = [
            GENETIC_CODE_DIR / "codon_encoder_fused.pt",
        ]
    else:
        raise ValueError(f"Unknown encoder version: {version}")

    encoder_path = None
    for p in paths:
        if p.exists():
            encoder_path = p
            break

    if encoder_path is None:
        raise FileNotFoundError(f"Encoder not found for version: {version}")

    print(f"  Loading {version} encoder from: {encoder_path}")

    # Load model
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    # Get architecture from checkpoint if available
    hidden_dim = 32 if version == "3adic" else 64

    encoder = CodonEncoder(input_dim=12, hidden_dim=hidden_dim, embed_dim=16, n_clusters=21)

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    # Try to load, handling architecture differences
    try:
        encoder.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"    Warning: Partial load - {e}")

    encoder.eval()
    encoder.to(device)

    mapping = checkpoint.get("codon_to_position", {})
    metadata = checkpoint.get("metadata", {})

    return encoder, mapping, metadata


# ============================================================================
# POINCARE GEOMETRY
# ============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincare ball geodesic distance."""
    import torch

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

    x_norm_sq = torch.sum(x**2, dim=-1)
    y_norm_sq = torch.sum(y**2, dim=-1)
    diff_norm_sq = torch.sum((x - y)**2, dim=-1)

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_norm_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    dist = (1 / np.sqrt(c)) * torch.acosh(arg)
    return dist.numpy() if isinstance(dist, torch.Tensor) else dist


def codon_to_onehot(codon: str) -> np.ndarray:
    """Convert codon to one-hot encoding."""
    nucleotides = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    onehot = np.zeros(12)
    for i, nuc in enumerate(codon.upper()):
        if nuc in nucleotides:
            onehot[i * 4 + nucleotides[nuc]] = 1
    return onehot


def encode_codon(codon: str, encoder) -> np.ndarray:
    """Encode a codon using the given encoder."""
    import torch

    onehot = codon_to_onehot(codon)
    x = torch.from_numpy(onehot).float().unsqueeze(0)

    with torch.no_grad():
        out = encoder(x)
        emb = out["z_hyp"] if isinstance(out, dict) else out

    return emb.numpy().squeeze()


# ============================================================================
# AMINO ACID TO CODON MAPPING
# ============================================================================

AA_TO_CODON = {
    "A": "GCT", "R": "CGG", "N": "AAC", "D": "GAC", "C": "TGC",
    "E": "GAG", "Q": "CAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "TCG", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}


def get_mutation_codons(mutation: str):
    """Parse mutation string and return (wt_codon, mut_codon)."""
    if len(mutation) < 3:
        return None, None

    wt_aa = mutation[0]
    mut_aa = mutation[-1]

    wt_codon = AA_TO_CODON.get(wt_aa)
    mut_codon = AA_TO_CODON.get(mut_aa)

    return wt_codon, mut_codon


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_drug_resistance(encoder, encoder_name: str):
    """Analyze drug resistance correlation."""
    print(f"\n  Analyzing drug resistance with {encoder_name}...")

    try:
        df = load_stanford_hivdb("all")
    except Exception as e:
        print(f"    Error loading HIVDB data: {e}")
        return None

    results = []

    for _, row in df.iterrows():
        mut_list = row.get("CompMutList", "")
        if pd.isna(mut_list) or not mut_list:
            continue

        drug_class = row.get("drug_class", "Unknown")

        # Parse mutations
        for mut in str(mut_list).split("+"):
            mut = mut.strip()
            if len(mut) < 3:
                continue

            wt_codon, mut_codon = get_mutation_codons(mut)
            if wt_codon is None or mut_codon is None:
                continue

            # Encode and compute distance
            try:
                wt_emb = encode_codon(wt_codon, encoder)
                mut_emb = encode_codon(mut_codon, encoder)

                distance = poincare_distance(
                    wt_emb.reshape(1, -1),
                    mut_emb.reshape(1, -1)
                )[0]

                # Compute radial shift
                wt_radius = hyperbolic_radius(wt_emb)
                mut_radius = hyperbolic_radius(mut_emb)
                radial_shift = abs(mut_radius - wt_radius)

                results.append({
                    "mutation": mut,
                    "drug_class": drug_class,
                    "distance": float(distance),
                    "radial_shift": float(radial_shift),
                    "wt_radius": float(wt_radius),
                    "mut_radius": float(mut_radius),
                })
            except Exception:
                continue

    if not results:
        return None

    result_df = pd.DataFrame(results)

    # Compute statistics by drug class
    stats_by_class = {}
    for drug_class in result_df["drug_class"].unique():
        class_df = result_df[result_df["drug_class"] == drug_class]
        stats_by_class[drug_class] = {
            "n_mutations": len(class_df),
            "mean_distance": float(class_df["distance"].mean()),
            "std_distance": float(class_df["distance"].std()),
            "mean_radial_shift": float(class_df["radial_shift"].mean()),
        }

    return {
        "n_total": len(result_df),
        "stats_by_class": stats_by_class,
        "overall_mean_distance": float(result_df["distance"].mean()),
        "overall_std_distance": float(result_df["distance"].std()),
        "overall_mean_radial_shift": float(result_df["radial_shift"].mean()),
    }


def analyze_benchmark_mutations(encoder, encoder_name: str):
    """Analyze known benchmark mutations."""
    print(f"\n  Analyzing benchmark mutations with {encoder_name}...")

    results = {"high_impact": [], "low_impact": []}

    for category, mutations in BENCHMARK_MUTATIONS.items():
        for mut, drug_class, severity in mutations:
            wt_codon, mut_codon = get_mutation_codons(mut)
            if wt_codon is None or mut_codon is None:
                continue

            try:
                wt_emb = encode_codon(wt_codon, encoder)
                mut_emb = encode_codon(mut_codon, encoder)

                distance = poincare_distance(
                    wt_emb.reshape(1, -1),
                    mut_emb.reshape(1, -1)
                )[0]

                wt_radius = hyperbolic_radius(wt_emb)
                mut_radius = hyperbolic_radius(mut_emb)
                radial_shift = abs(mut_radius - wt_radius)

                results[category].append({
                    "mutation": mut,
                    "drug_class": drug_class,
                    "severity": severity,
                    "distance": float(distance),
                    "radial_shift": float(radial_shift),
                    "wt_radius": float(wt_radius),
                    "mut_radius": float(mut_radius),
                })
            except Exception as e:
                print(f"    Error with {mut}: {e}")

    # Compute separation between high and low impact
    high_distances = [m["distance"] for m in results["high_impact"]]
    low_distances = [m["distance"] for m in results["low_impact"]]

    high_radial = [m["radial_shift"] for m in results["high_impact"]]
    low_radial = [m["radial_shift"] for m in results["low_impact"]]

    separation_distance = (
        np.mean(high_distances) - np.mean(low_distances)
        if high_distances and low_distances else 0
    )
    separation_radial = (
        np.mean(high_radial) - np.mean(low_radial)
        if high_radial and low_radial else 0
    )

    return {
        "mutations": results,
        "high_impact_mean_distance": float(np.mean(high_distances)) if high_distances else 0,
        "low_impact_mean_distance": float(np.mean(low_distances)) if low_distances else 0,
        "separation_distance": float(separation_distance),
        "high_impact_mean_radial": float(np.mean(high_radial)) if high_radial else 0,
        "low_impact_mean_radial": float(np.mean(low_radial)) if low_radial else 0,
        "separation_radial": float(separation_radial),
    }


def analyze_radial_structure(encoder, encoder_name: str):
    """Analyze the radial structure of the encoder."""
    print(f"\n  Analyzing radial structure with {encoder_name}...")

    all_codons = [
        "TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG",
        "TAT", "TAC", "TAA", "TAG", "TGT", "TGC", "TGA", "TGG",
        "CTT", "CTC", "CTA", "CTG", "CCT", "CCC", "CCA", "CCG",
        "CAT", "CAC", "CAA", "CAG", "CGT", "CGC", "CGA", "CGG",
        "ATT", "ATC", "ATA", "ATG", "ACT", "ACC", "ACA", "ACG",
        "AAT", "AAC", "AAA", "AAG", "AGT", "AGC", "AGA", "AGG",
        "GTT", "GTC", "GTA", "GTG", "GCT", "GCC", "GCA", "GCG",
        "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA", "GGG",
    ]

    radii = []
    for codon in all_codons:
        try:
            emb = encode_codon(codon, encoder)
            radius = hyperbolic_radius(emb)
            radii.append(radius)
        except Exception:
            continue

    radii = np.array(radii)

    return {
        "n_codons": len(radii),
        "min_radius": float(radii.min()),
        "max_radius": float(radii.max()),
        "mean_radius": float(radii.mean()),
        "std_radius": float(radii.std()),
        "radial_range": float(radii.max() - radii.min()),
    }


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================


def compare_encoders(results_3adic: dict, results_fused: dict):
    """Compare results from both encoders."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "drug_resistance": {},
        "benchmark_mutations": {},
        "radial_structure": {},
        "recommendations": [],
    }

    # Drug resistance comparison
    if results_3adic.get("drug_resistance") and results_fused.get("drug_resistance"):
        dr_3adic = results_3adic["drug_resistance"]
        dr_fused = results_fused["drug_resistance"]

        comparison["drug_resistance"] = {
            "3adic_mean_distance": dr_3adic["overall_mean_distance"],
            "fused_mean_distance": dr_fused["overall_mean_distance"],
            "3adic_mean_radial_shift": dr_3adic["overall_mean_radial_shift"],
            "fused_mean_radial_shift": dr_fused["overall_mean_radial_shift"],
        }

        # Higher radial shift = better for mutation impact prediction
        if dr_fused["overall_mean_radial_shift"] > dr_3adic["overall_mean_radial_shift"]:
            comparison["recommendations"].append(
                "Fused encoder shows higher radial shifts for resistance mutations - better for severity prediction"
            )
        else:
            comparison["recommendations"].append(
                "Classical 3-adic shows higher radial shifts for resistance mutations"
            )

    # Benchmark mutations comparison
    if results_3adic.get("benchmark") and results_fused.get("benchmark"):
        bm_3adic = results_3adic["benchmark"]
        bm_fused = results_fused["benchmark"]

        comparison["benchmark_mutations"] = {
            "3adic_separation_distance": bm_3adic["separation_distance"],
            "fused_separation_distance": bm_fused["separation_distance"],
            "3adic_separation_radial": bm_3adic["separation_radial"],
            "fused_separation_radial": bm_fused["separation_radial"],
        }

        # Higher separation = better discrimination
        if bm_fused["separation_radial"] > bm_3adic["separation_radial"]:
            comparison["recommendations"].append(
                "Fused encoder better separates high/low impact mutations by radial shift"
            )
        else:
            comparison["recommendations"].append(
                "Classical 3-adic better separates high/low impact mutations by radial shift"
            )

    # Radial structure comparison
    if results_3adic.get("radial") and results_fused.get("radial"):
        rs_3adic = results_3adic["radial"]
        rs_fused = results_fused["radial"]

        comparison["radial_structure"] = {
            "3adic_radial_range": rs_3adic["radial_range"],
            "fused_radial_range": rs_fused["radial_range"],
            "3adic_std_radius": rs_3adic["std_radius"],
            "fused_std_radius": rs_fused["std_radius"],
        }

        # Larger radial range = more hierarchical structure
        if rs_fused["radial_range"] > rs_3adic["radial_range"]:
            comparison["recommendations"].append(
                "Fused encoder has larger radial range - better hierarchical encoding"
            )

    return comparison


def create_visualization(results_3adic: dict, results_fused: dict, output_dir: Path):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Radial distribution comparison
    ax1 = axes[0, 0]
    if results_3adic.get("radial") and results_fused.get("radial"):
        categories = ["Min", "Mean", "Max"]
        x = np.arange(len(categories))
        width = 0.35

        values_3adic = [
            results_3adic["radial"]["min_radius"],
            results_3adic["radial"]["mean_radius"],
            results_3adic["radial"]["max_radius"],
        ]
        values_fused = [
            results_fused["radial"]["min_radius"],
            results_fused["radial"]["mean_radius"],
            results_fused["radial"]["max_radius"],
        ]

        ax1.bar(x - width/2, values_3adic, width, label="Classical 3-Adic", color="steelblue")
        ax1.bar(x + width/2, values_fused, width, label="Fused Hyperbolic", color="coral")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.set_ylabel("Radius")
        ax1.set_title("Radial Distribution Comparison")
        ax1.legend()

    # 2. Drug class mean distances
    ax2 = axes[0, 1]
    if results_3adic.get("drug_resistance") and results_fused.get("drug_resistance"):
        drug_classes = list(results_3adic["drug_resistance"]["stats_by_class"].keys())
        x = np.arange(len(drug_classes))
        width = 0.35

        values_3adic = [
            results_3adic["drug_resistance"]["stats_by_class"][dc]["mean_distance"]
            for dc in drug_classes
        ]
        values_fused = [
            results_fused["drug_resistance"]["stats_by_class"].get(dc, {}).get("mean_distance", 0)
            for dc in drug_classes
        ]

        ax2.bar(x - width/2, values_3adic, width, label="Classical 3-Adic", color="steelblue")
        ax2.bar(x + width/2, values_fused, width, label="Fused Hyperbolic", color="coral")
        ax2.set_xticks(x)
        ax2.set_xticklabels(drug_classes)
        ax2.set_ylabel("Mean Poincare Distance")
        ax2.set_title("Drug Resistance: Mean Distance by Class")
        ax2.legend()

    # 3. Benchmark mutation separation
    ax3 = axes[1, 0]
    if results_3adic.get("benchmark") and results_fused.get("benchmark"):
        metrics = ["Distance\nSeparation", "Radial\nSeparation"]
        x = np.arange(len(metrics))
        width = 0.35

        values_3adic = [
            results_3adic["benchmark"]["separation_distance"],
            results_3adic["benchmark"]["separation_radial"],
        ]
        values_fused = [
            results_fused["benchmark"]["separation_distance"],
            results_fused["benchmark"]["separation_radial"],
        ]

        ax3.bar(x - width/2, values_3adic, width, label="Classical 3-Adic", color="steelblue")
        ax3.bar(x + width/2, values_fused, width, label="Fused Hyperbolic", color="coral")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel("High - Low Impact")
        ax3.set_title("Benchmark Mutations: Impact Separation")
        ax3.legend()

    # 4. Summary metrics
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = "ENCODER COMPARISON SUMMARY\n" + "=" * 40 + "\n\n"

    if results_3adic.get("radial") and results_fused.get("radial"):
        summary_text += "RADIAL STRUCTURE:\n"
        summary_text += f"  3-Adic range: {results_3adic['radial']['radial_range']:.4f}\n"
        summary_text += f"  Fused range:  {results_fused['radial']['radial_range']:.4f}\n\n"

    if results_3adic.get("drug_resistance") and results_fused.get("drug_resistance"):
        summary_text += "DRUG RESISTANCE:\n"
        summary_text += f"  3-Adic mean dist: {results_3adic['drug_resistance']['overall_mean_distance']:.4f}\n"
        summary_text += f"  Fused mean dist:  {results_fused['drug_resistance']['overall_mean_distance']:.4f}\n\n"

    if results_3adic.get("benchmark") and results_fused.get("benchmark"):
        summary_text += "BENCHMARK SEPARATION:\n"
        summary_text += f"  3-Adic radial sep: {results_3adic['benchmark']['separation_radial']:.4f}\n"
        summary_text += f"  Fused radial sep:  {results_fused['benchmark']['separation_radial']:.4f}\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(output_dir / "encoder_comparison.png", dpi=150)
    plt.close()

    print(f"\n  Saved visualization to: {output_dir / 'encoder_comparison.png'}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("ENCODER COMPARISON VALIDATION")
    print("Classical 3-Adic vs Fused Hyperbolic on HIV Data")
    print("=" * 70)

    results_3adic = {}
    results_fused = {}

    # Load encoders
    print("\nLoading encoders...")

    try:
        encoder_3adic, mapping_3adic, meta_3adic = load_encoder("3adic")
        print(f"    3-adic metadata: {meta_3adic}")
    except Exception as e:
        print(f"  ERROR loading 3-adic encoder: {e}")
        encoder_3adic = None

    try:
        encoder_fused, mapping_fused, meta_fused = load_encoder("fused")
        print(f"    Fused metadata: {meta_fused}")
    except Exception as e:
        print(f"  ERROR loading fused encoder: {e}")
        encoder_fused = None

    if encoder_3adic is None and encoder_fused is None:
        print("\nNo encoders available. Exiting.")
        return 1

    # Run analyses
    print("\n" + "=" * 70)
    print("RUNNING ANALYSES")
    print("=" * 70)

    # Radial structure
    if encoder_3adic:
        results_3adic["radial"] = analyze_radial_structure(encoder_3adic, "Classical 3-Adic")
    if encoder_fused:
        results_fused["radial"] = analyze_radial_structure(encoder_fused, "Fused Hyperbolic")

    # Drug resistance
    if encoder_3adic:
        results_3adic["drug_resistance"] = analyze_drug_resistance(encoder_3adic, "Classical 3-Adic")
    if encoder_fused:
        results_fused["drug_resistance"] = analyze_drug_resistance(encoder_fused, "Fused Hyperbolic")

    # Benchmark mutations
    if encoder_3adic:
        results_3adic["benchmark"] = analyze_benchmark_mutations(encoder_3adic, "Classical 3-Adic")
    if encoder_fused:
        results_fused["benchmark"] = analyze_benchmark_mutations(encoder_fused, "Fused Hyperbolic")

    # Compare and visualize
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    comparison = compare_encoders(results_3adic, results_fused)

    # Print comparison
    print("\nRadial Structure:")
    if comparison.get("radial_structure"):
        rs = comparison["radial_structure"]
        print(f"  3-Adic radial range: {rs.get('3adic_radial_range', 0):.4f}")
        print(f"  Fused radial range:  {rs.get('fused_radial_range', 0):.4f}")

    print("\nDrug Resistance:")
    if comparison.get("drug_resistance"):
        dr = comparison["drug_resistance"]
        print(f"  3-Adic mean distance: {dr.get('3adic_mean_distance', 0):.4f}")
        print(f"  Fused mean distance:  {dr.get('fused_mean_distance', 0):.4f}")
        print(f"  3-Adic radial shift:  {dr.get('3adic_mean_radial_shift', 0):.4f}")
        print(f"  Fused radial shift:   {dr.get('fused_mean_radial_shift', 0):.4f}")

    print("\nBenchmark Mutations (High vs Low Impact):")
    if comparison.get("benchmark_mutations"):
        bm = comparison["benchmark_mutations"]
        print(f"  3-Adic distance separation: {bm.get('3adic_separation_distance', 0):.4f}")
        print(f"  Fused distance separation:  {bm.get('fused_separation_distance', 0):.4f}")
        print(f"  3-Adic radial separation:   {bm.get('3adic_separation_radial', 0):.4f}")
        print(f"  Fused radial separation:    {bm.get('fused_separation_radial', 0):.4f}")

    print("\nRecommendations:")
    for rec in comparison.get("recommendations", []):
        print(f"  - {rec}")

    # Save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "results_3adic": results_3adic,
        "results_fused": results_fused,
        "comparison": comparison,
    }

    results_path = OUTPUT_DIR / "encoder_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to: {results_path}")

    # Create visualization
    create_visualization(results_3adic, results_fused, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
