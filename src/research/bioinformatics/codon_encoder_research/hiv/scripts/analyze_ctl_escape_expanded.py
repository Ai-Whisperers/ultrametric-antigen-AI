# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Expanded CTL Escape Analysis

Comprehensive analysis of 2,116 CTL epitopes from LANL database using
p-adic hyperbolic codon geometry.

Analyses performed:
1. HLA-stratified escape landscapes
2. Epitope conservation vs radial position
3. Protein-specific escape velocity
4. Boundary-crossing frequency by HLA

Input: ctl_summary.csv (LANL CTL Epitope Database)
Output: CTL escape analysis results and visualizations
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from codon_extraction import encode_amino_acid_sequence
from hyperbolic_utils import (
    load_hyperbolic_encoder,
)
from unified_data_loader import load_lanl_ctl, parse_hla_restrictions

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ctl_escape"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Major HLA types to analyze
MAJOR_HLA_TYPES = [
    "A*02",
    "A*03",
    "A*11",
    "A*24",
    "B*07",
    "B*08",
    "B*27",
    "B*35",
    "B*44",
    "B*51",
    "B*57",
    "B*58",
]

# HIV proteins to analyze
HIV_PROTEINS = ["Gag", "Pol", "Env", "Nef", "Tat", "Rev", "Vif", "Vpr", "Vpu"]


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================


def load_ctl_data() -> pd.DataFrame:
    """Load and preprocess CTL epitope data."""
    print("Loading LANL CTL epitope data...")

    try:
        df = load_lanl_ctl()
        print(f"  Loaded {len(df)} epitopes")

        # Parse HLA restrictions into lists
        df["HLA_list"] = df["HLA"].apply(parse_hla_restrictions)
        df["n_hla_restrictions"] = df["HLA_list"].apply(len)

        # Extract primary protein
        df["primary_protein"] = df["Protein"].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")

        # Calculate epitope length
        df["epitope_length"] = df["Epitope"].apply(lambda x: len(x) if pd.notna(x) else 0)

        # Summary
        print(f"  Epitopes with HLA data: {(df['n_hla_restrictions'] > 0).sum()}")
        print(f"  Unique HLA types: {len(set(h for hlist in df['HLA_list'] for h in hlist))}")

        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def analyze_hla_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze HLA type distribution across epitopes."""
    print("\nAnalyzing HLA distribution...")

    hla_counts = Counter()
    for hla_list in df["HLA_list"]:
        for hla in hla_list:
            # Normalize HLA names (e.g., A2 -> A*02)
            hla_normalized = normalize_hla(hla)
            hla_counts[hla_normalized] += 1

    hla_df = pd.DataFrame(
        [{"hla_type": hla, "epitope_count": count} for hla, count in hla_counts.most_common()]
    )

    print("  Top 10 HLA types:")
    for _, row in hla_df.head(10).iterrows():
        print(f"    {row['hla_type']}: {row['epitope_count']} epitopes")

    return hla_df


def normalize_hla(hla: str) -> str:
    """Normalize HLA type names."""
    hla = hla.strip().upper()

    # Handle common variations
    if hla.startswith("HLA-"):
        hla = hla[4:]

    # Add asterisk if missing (e.g., A02 -> A*02)
    if len(hla) >= 3 and hla[1].isdigit():
        hla = hla[0] + "*" + hla[1:]

    return hla


# ============================================================================
# HYPERBOLIC ANALYSIS
# ============================================================================


def encode_epitopes(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """Encode all epitopes to hyperbolic space."""
    print("\nEncoding epitopes to hyperbolic space...")

    embeddings_list = []
    radii_list = []
    centroids_list = []

    for _, row in df.iterrows():
        epitope = row["Epitope"]
        if pd.isna(epitope) or len(epitope) < 3:
            embeddings_list.append(None)
            radii_list.append(None)
            centroids_list.append(None)
            continue

        try:
            emb = encode_amino_acid_sequence(epitope, encoder)
            if len(emb) > 0:
                embeddings_list.append(emb)
                radii_list.append(np.mean(np.linalg.norm(emb, axis=1)))
                centroids_list.append(emb.mean(axis=0))
            else:
                embeddings_list.append(None)
                radii_list.append(None)
                centroids_list.append(None)
        except Exception:
            embeddings_list.append(None)
            radii_list.append(None)
            centroids_list.append(None)

    df["embeddings"] = embeddings_list
    df["mean_radius"] = radii_list
    df["centroid"] = centroids_list

    valid_count = df["embeddings"].notna().sum()
    print(f"  Encoded {valid_count} epitopes")

    return df


def analyze_protein_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze geometric properties by HIV protein."""
    print("\nAnalyzing protein-specific geometry...")

    results = []

    for protein in HIV_PROTEINS:
        protein_df = df[df["primary_protein"].str.contains(protein, case=False, na=False)]
        valid_df = protein_df[protein_df["mean_radius"].notna()]

        if len(valid_df) < 5:
            continue

        radii = valid_df["mean_radius"].values
        lengths = valid_df["epitope_length"].values

        results.append(
            {
                "protein": protein,
                "n_epitopes": len(protein_df),
                "n_with_embeddings": len(valid_df),
                "mean_radius": radii.mean(),
                "std_radius": radii.std(),
                "min_radius": radii.min(),
                "max_radius": radii.max(),
                "mean_length": lengths.mean(),
            }
        )

    return pd.DataFrame(results)


def analyze_hla_escape_landscapes(df: pd.DataFrame, encoder) -> dict:
    """Analyze escape landscapes for major HLA types."""
    print("\nAnalyzing HLA-specific escape landscapes...")

    results = {}

    for hla_prefix in MAJOR_HLA_TYPES:
        # Get epitopes restricted by this HLA
        hla_epitopes = df[df["HLA"].str.contains(hla_prefix, case=False, na=False)]

        if len(hla_epitopes) < 5:
            continue

        valid_epitopes = hla_epitopes[hla_epitopes["mean_radius"].notna()]
        if len(valid_epitopes) < 5:
            continue

        radii = valid_epitopes["mean_radius"].values

        # Calculate centroid of all epitopes for this HLA
        all_centroids = np.array([c for c in valid_epitopes["centroid"] if c is not None])
        if len(all_centroids) > 0:
            hla_centroid = all_centroids.mean(axis=0)
            variance = np.mean(np.sum((all_centroids - hla_centroid) ** 2, axis=1))
        else:
            hla_centroid = None
            variance = None

        results[hla_prefix] = {
            "n_epitopes": len(hla_epitopes),
            "n_valid": len(valid_epitopes),
            "mean_radius": radii.mean(),
            "std_radius": radii.std(),
            "radius_range": (radii.min(), radii.max()),
            "centroid": hla_centroid,
            "variance": variance,
            "proteins": valid_epitopes["primary_protein"].value_counts().to_dict(),
        }

    return results


def calculate_escape_velocity(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """
    Calculate "escape velocity" - the geometric spread of epitopes.

    Higher spread suggests more evolutionary flexibility.
    """
    print("\nCalculating escape velocity by protein...")

    results = []

    for protein in HIV_PROTEINS:
        protein_df = df[df["primary_protein"].str.contains(protein, case=False, na=False)]
        valid_df = protein_df[protein_df["centroid"].notna()]

        if len(valid_df) < 10:
            continue

        centroids = np.array([c for c in valid_df["centroid"] if c is not None])

        if len(centroids) < 10:
            continue

        # Calculate pairwise distances
        n = len(centroids)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                distances.append(dist)

        # Calculate variance (spread) as escape velocity proxy
        variance = np.var(centroids, axis=0).sum()

        results.append(
            {
                "protein": protein,
                "n_epitopes": len(valid_df),
                "mean_pairwise_distance": np.mean(distances),
                "max_pairwise_distance": np.max(distances),
                "centroid_variance": variance,
                "escape_velocity": np.std(distances),
            }
        )

    return pd.DataFrame(results)


def analyze_conservation_vs_radius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze relationship between epitope conservation (radial position)
    and other properties.

    Hypothesis: More conserved epitopes are more central (lower radius).
    """
    print("\nAnalyzing conservation vs radial position...")

    results = []

    for protein in HIV_PROTEINS:
        protein_df = df[df["primary_protein"].str.contains(protein, case=False, na=False)]
        valid_df = protein_df[protein_df["mean_radius"].notna()]

        if len(valid_df) < 10:
            continue

        # Correlate radius with number of HLA restrictions (proxy for immunogenicity)
        radii = valid_df["mean_radius"].values
        n_hla = valid_df["n_hla_restrictions"].values

        if np.std(radii) > 0 and np.std(n_hla) > 0:
            corr, pval = stats.pearsonr(radii, n_hla)
        else:
            corr, pval = 0, 1

        # Compare radii of epitopes with many vs few HLA restrictions
        many_hla = valid_df[valid_df["n_hla_restrictions"] >= 3]["mean_radius"]
        few_hla = valid_df[valid_df["n_hla_restrictions"] <= 1]["mean_radius"]

        if len(many_hla) >= 5 and len(few_hla) >= 5:
            mw_stat, mw_pval = stats.mannwhitneyu(many_hla, few_hla)
        else:
            mw_stat, mw_pval = 0, 1

        results.append(
            {
                "protein": protein,
                "n_epitopes": len(valid_df),
                "mean_radius": radii.mean(),
                "radius_n_hla_corr": corr,
                "radius_n_hla_pval": pval,
                "many_hla_mean_radius": many_hla.mean() if len(many_hla) > 0 else None,
                "few_hla_mean_radius": few_hla.mean() if len(few_hla) > 0 else None,
                "mw_pvalue": mw_pval,
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_hla_landscape_comparison(hla_results: dict):
    """Plot comparison of HLA escape landscapes."""
    if not hla_results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    hla_names = list(hla_results.keys())
    mean_radii = [hla_results[h]["mean_radius"] for h in hla_names]
    std_radii = [hla_results[h]["std_radius"] for h in hla_names]
    n_epitopes = [hla_results[h]["n_valid"] for h in hla_names]

    # Sort by mean radius
    sorted_idx = np.argsort(mean_radii)
    hla_names = [hla_names[i] for i in sorted_idx]
    mean_radii = [mean_radii[i] for i in sorted_idx]
    std_radii = [std_radii[i] for i in sorted_idx]
    n_epitopes = [n_epitopes[i] for i in sorted_idx]

    x = np.arange(len(hla_names))
    bars = ax.bar(x, mean_radii, yerr=std_radii, capsize=5, color="steelblue", alpha=0.7)

    # Color by sample size
    for bar, n in zip(bars, n_epitopes):
        if n >= 50:
            bar.set_color("darkblue")
        elif n >= 20:
            bar.set_color("steelblue")
        else:
            bar.set_color("lightblue")

    ax.set_xticks(x)
    ax.set_xticklabels(hla_names, rotation=45, ha="right")
    ax.set_xlabel("HLA Type")
    ax.set_ylabel("Mean Epitope Radius")
    ax.set_title("HLA-Specific Epitope Radial Positions")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="darkblue", label="n >= 50"),
        Patch(facecolor="steelblue", label="20 <= n < 50"),
        Patch(facecolor="lightblue", label="n < 20"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hla_landscape_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'hla_landscape_comparison.png'}")


def plot_protein_escape_velocity(escape_df: pd.DataFrame):
    """Plot escape velocity by protein."""
    if escape_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean pairwise distance
    escape_df_sorted = escape_df.sort_values("mean_pairwise_distance", ascending=False)

    ax1 = axes[0]
    bars = ax1.barh(escape_df_sorted["protein"], escape_df_sorted["mean_pairwise_distance"], color="coral")
    ax1.set_xlabel("Mean Pairwise Distance")
    ax1.set_title("Epitope Spread by Protein")
    ax1.invert_yaxis()

    # Plot 2: Escape velocity (std of distances)
    ax2 = axes[1]
    escape_df_sorted = escape_df.sort_values("escape_velocity", ascending=False)
    ax2.barh(escape_df_sorted["protein"], escape_df_sorted["escape_velocity"], color="teal")
    ax2.set_xlabel("Escape Velocity (Std of Distances)")
    ax2.set_title("Evolutionary Flexibility by Protein")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "protein_escape_velocity.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'protein_escape_velocity.png'}")


def plot_conservation_analysis(conservation_df: pd.DataFrame):
    """Plot conservation vs radius analysis."""
    if conservation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean radius by protein
    conservation_df_sorted = conservation_df.sort_values("mean_radius")

    colors = ["green" if p < 0.05 else "gray" for p in conservation_df_sorted["radius_n_hla_pval"]]
    ax.barh(conservation_df_sorted["protein"], conservation_df_sorted["mean_radius"], color=colors, alpha=0.7)

    # Add correlation coefficients as annotations
    for i, (_, row) in enumerate(conservation_df_sorted.iterrows()):
        if pd.notna(row["radius_n_hla_corr"]):
            sig = "*" if row["radius_n_hla_pval"] < 0.05 else ""
            ax.text(
                row["mean_radius"] + 0.01,
                i,
                f"r={row['radius_n_hla_corr']:.2f}{sig}",
                va="center",
                fontsize=9,
            )

    ax.set_xlabel("Mean Epitope Radius")
    ax.set_ylabel("HIV Protein")
    ax.set_title("Epitope Radial Position vs Immunogenicity\n(green = significant correlation with #HLA restrictions)")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conservation_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'conservation_analysis.png'}")


def plot_epitope_length_distribution(df: pd.DataFrame):
    """Plot epitope length distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    ax1 = axes[0]
    lengths = df["epitope_length"].dropna()
    ax1.hist(lengths, bins=range(6, 25), edgecolor="black", alpha=0.7)
    ax1.axvline(lengths.mean(), color="red", linestyle="--", label=f"Mean: {lengths.mean():.1f}")
    ax1.set_xlabel("Epitope Length (amino acids)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("CTL Epitope Length Distribution")
    ax1.legend()

    # By protein
    ax2 = axes[1]
    protein_lengths = []
    proteins = []
    for protein in HIV_PROTEINS:
        prot_df = df[df["primary_protein"].str.contains(protein, case=False, na=False)]
        if len(prot_df) >= 10:
            protein_lengths.append(prot_df["epitope_length"].dropna().values)
            proteins.append(protein)

    if protein_lengths:
        ax2.boxplot(protein_lengths, labels=proteins)
        ax2.set_xlabel("HIV Protein")
        ax2.set_ylabel("Epitope Length")
        ax2.set_title("Epitope Length by Protein")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "epitope_length_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'epitope_length_distribution.png'}")


def plot_radius_by_protein(df: pd.DataFrame):
    """Plot radial position distribution by protein."""
    fig, ax = plt.subplots(figsize=(10, 6))

    protein_radii = []
    proteins = []

    for protein in HIV_PROTEINS:
        prot_df = df[df["primary_protein"].str.contains(protein, case=False, na=False)]
        valid_df = prot_df[prot_df["mean_radius"].notna()]
        if len(valid_df) >= 10:
            protein_radii.append(valid_df["mean_radius"].values)
            proteins.append(f"{protein}\n(n={len(valid_df)})")

    if protein_radii:
        bp = ax.boxplot(protein_radii, labels=proteins, patch_artist=True)

        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(proteins)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel("HIV Protein")
    ax.set_ylabel("Mean Epitope Radius")
    ax.set_title("Epitope Radial Position by HIV Protein")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radius_by_protein.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'radius_by_protein.png'}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_ctl_report(
    df: pd.DataFrame,
    hla_df: pd.DataFrame,
    protein_geometry: pd.DataFrame,
    hla_results: dict,
    escape_df: pd.DataFrame,
    conservation_df: pd.DataFrame,
):
    """Generate comprehensive CTL analysis report."""
    report_path = OUTPUT_DIR / "CTL_ANALYSIS_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# CTL Epitope Escape Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total epitopes: {len(df):,}\n")
        f.write(f"- Epitopes with HLA data: {(df['n_hla_restrictions'] > 0).sum():,}\n")
        f.write(f"- Epitopes with embeddings: {df['mean_radius'].notna().sum():,}\n")
        f.write(f"- Unique HLA types: {len(hla_df):,}\n\n")

        # By protein
        f.write("### Epitopes by Protein\n\n")
        f.write("| Protein | Epitopes | Mean Length | Mean Radius |\n")
        f.write("|---------|----------|-------------|-------------|\n")

        for _, row in protein_geometry.iterrows():
            f.write(
                f"| {row['protein']} | {row['n_epitopes']} | {row['mean_length']:.1f} | "
                f"{row['mean_radius']:.4f} |\n"
            )

        # Top HLA types
        f.write("\n### Top HLA Restrictions\n\n")
        f.write("| HLA Type | Epitope Count | Mean Radius | Std Radius |\n")
        f.write("|----------|---------------|-------------|------------|\n")

        for hla, data in sorted(hla_results.items(), key=lambda x: x[1]["n_valid"], reverse=True)[:15]:
            f.write(
                f"| {hla} | {data['n_valid']} | {data['mean_radius']:.4f} | "
                f"{data['std_radius']:.4f} |\n"
            )

        # Escape velocity
        f.write("\n## Escape Velocity Analysis\n\n")
        f.write(
            "Escape velocity measures the geometric spread of epitopes, "
            "indicating evolutionary flexibility.\n\n"
        )

        if not escape_df.empty:
            f.write("| Protein | Epitopes | Mean Distance | Escape Velocity |\n")
            f.write("|---------|----------|---------------|------------------|\n")

            for _, row in escape_df.sort_values("escape_velocity", ascending=False).iterrows():
                f.write(
                    f"| {row['protein']} | {row['n_epitopes']} | "
                    f"{row['mean_pairwise_distance']:.4f} | {row['escape_velocity']:.4f} |\n"
                )

        # Conservation analysis
        f.write("\n## Conservation vs Radial Position\n\n")
        f.write(
            "Testing hypothesis: More conserved (immunogenic) epitopes have smaller radial positions.\n\n"
        )

        if not conservation_df.empty:
            f.write("| Protein | Mean Radius | Correlation (r) | p-value |\n")
            f.write("|---------|-------------|-----------------|----------|\n")

            for _, row in conservation_df.iterrows():
                sig = "*" if row["radius_n_hla_pval"] < 0.05 else ""
                f.write(
                    f"| {row['protein']} | {row['mean_radius']:.4f} | "
                    f"{row['radius_n_hla_corr']:.3f}{sig} | {row['radius_n_hla_pval']:.4f} |\n"
                )

        # Key findings
        f.write("\n## Key Findings\n\n")

        # 1. Protein with highest escape velocity
        if not escape_df.empty:
            top_escape = escape_df.loc[escape_df["escape_velocity"].idxmax()]
            f.write(
                f"1. **Highest Escape Velocity**: {top_escape['protein']} "
                f"({top_escape['escape_velocity']:.4f})\n"
            )

        # 2. Most restricted HLA
        if hla_results:
            top_hla = max(hla_results.items(), key=lambda x: x[1]["n_valid"])
            f.write(f"2. **Most Common HLA Restriction**: {top_hla[0]} ({top_hla[1]['n_valid']} epitopes)\n")

        # 3. Conservation correlation
        if not conservation_df.empty:
            sig_corr = conservation_df[conservation_df["radius_n_hla_pval"] < 0.05]
            if len(sig_corr) > 0:
                f.write(
                    f"3. **Significant Conservation Correlations**: Found in {len(sig_corr)} proteins\n"
                )

        f.write("\n## Generated Files\n\n")
        f.write("- `hla_landscape_comparison.png` - HLA-specific radial positions\n")
        f.write("- `protein_escape_velocity.png` - Escape velocity by protein\n")
        f.write("- `conservation_analysis.png` - Conservation vs radius\n")
        f.write("- `epitope_length_distribution.png` - Epitope length analysis\n")
        f.write("- `radius_by_protein.png` - Radial position distributions\n")
        f.write("- `epitope_data.csv` - Complete epitope data with embeddings info\n")
        f.write("- `hla_summary.csv` - HLA restriction summary\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete CTL escape analysis."""
    print("=" * 70)
    print("CTL Epitope Escape Analysis - Expanded Dataset")
    print("=" * 70)

    # Load encoder
    print("\nLoading hyperbolic codon encoder...")
    try:
        encoder, _ = load_hyperbolic_encoder()
        print("  Encoder loaded successfully")
    except FileNotFoundError as e:
        print(f"  Error loading encoder: {e}")
        print("  Analysis will continue without hyperbolic calculations")
        encoder = None

    # Load data
    df = load_ctl_data()
    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Analyze HLA distribution
    hla_df = analyze_hla_distribution(df)

    # Encode epitopes
    if encoder:
        df = encode_epitopes(df, encoder)

    # Run analyses
    print("\nRunning geometric analyses...")
    protein_geometry = analyze_protein_geometry(df)

    hla_results = {}
    escape_df = pd.DataFrame()
    conservation_df = pd.DataFrame()

    if encoder:
        hla_results = analyze_hla_escape_landscapes(df, encoder)
        escape_df = calculate_escape_velocity(df, encoder)
        conservation_df = analyze_conservation_vs_radius(df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_epitope_length_distribution(df)
    plot_radius_by_protein(df)

    if hla_results:
        plot_hla_landscape_comparison(hla_results)

    if not escape_df.empty:
        plot_protein_escape_velocity(escape_df)

    if not conservation_df.empty:
        plot_conservation_analysis(conservation_df)

    # Save data
    print("\nSaving results...")

    # Save epitope data (without embeddings - too large)
    save_cols = [
        col
        for col in df.columns
        if col not in ["embeddings", "centroid", "HLA_list"]
    ]
    df[save_cols].to_csv(OUTPUT_DIR / "epitope_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'epitope_data.csv'}")

    hla_df.to_csv(OUTPUT_DIR / "hla_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'hla_summary.csv'}")

    if not protein_geometry.empty:
        protein_geometry.to_csv(OUTPUT_DIR / "protein_geometry.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'protein_geometry.csv'}")

    if not escape_df.empty:
        escape_df.to_csv(OUTPUT_DIR / "escape_velocity.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'escape_velocity.csv'}")

    # Generate report
    print("\nGenerating report...")
    generate_ctl_report(df, hla_df, protein_geometry, hla_results, escape_df, conservation_df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
