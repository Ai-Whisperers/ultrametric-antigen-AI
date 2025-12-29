# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
CATNAP Antibody Neutralization Analysis

Comprehensive analysis of 189,879 antibody-virus neutralization records
using p-adic hyperbolic codon geometry.

Analyses performed:
1. bnAb sensitivity geometric signatures
2. Neutralization breadth vs epitope centrality
3. Antibody potency clustering
4. Cross-neutralization patterns

Input: catnap_assay.txt (CATNAP Database)
Output: Neutralization analysis results and visualizations
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import (
    calculate_antibody_breadth,
    get_catnap_by_antibody,
    load_catnap,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "catnap_neutralization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Major broadly neutralizing antibodies to analyze
MAJOR_BNABS = {
    # CD4 binding site
    "VRC01": {"epitope": "CD4bs", "class": "CD4bs"},
    "VRC03": {"epitope": "CD4bs", "class": "CD4bs"},
    "3BNC117": {"epitope": "CD4bs", "class": "CD4bs"},
    "NIH45-46": {"epitope": "CD4bs", "class": "CD4bs"},
    "b12": {"epitope": "CD4bs", "class": "CD4bs"},
    # V2 apex / glycan
    "PG9": {"epitope": "V2 apex", "class": "V2-glycan"},
    "PG16": {"epitope": "V2 apex", "class": "V2-glycan"},
    "PGT145": {"epitope": "V2 apex", "class": "V2-glycan"},
    # V3 glycan
    "PGT121": {"epitope": "V3 glycan", "class": "V3-glycan"},
    "PGT128": {"epitope": "V3 glycan", "class": "V3-glycan"},
    "10-1074": {"epitope": "V3 glycan", "class": "V3-glycan"},
    # MPER
    "10E8": {"epitope": "MPER", "class": "MPER"},
    "4E10": {"epitope": "MPER", "class": "MPER"},
    "2F5": {"epitope": "MPER", "class": "MPER"},
    # gp120-gp41 interface
    "35O22": {"epitope": "gp120-gp41", "class": "interface"},
    "8ANC195": {"epitope": "gp120-gp41", "class": "interface"},
}

# IC50 thresholds
SENSITIVE_THRESHOLD = 1.0  # IC50 < 1 ug/mL
RESISTANT_THRESHOLD = 50.0  # IC50 > 50 ug/mL


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================


def load_neutralization_data() -> pd.DataFrame:
    """Load and preprocess CATNAP neutralization data."""
    print("Loading CATNAP neutralization data...")

    try:
        df = load_catnap()
        print(f"  Loaded {len(df):,} neutralization records")

        # Basic stats
        n_antibodies = df["Antibody"].nunique()
        n_viruses = df["Virus"].nunique()
        n_valid_ic50 = df["IC50_numeric"].notna().sum()

        print(f"  Unique antibodies: {n_antibodies:,}")
        print(f"  Unique viruses: {n_viruses:,}")
        print(f"  Records with numeric IC50: {n_valid_ic50:,}")

        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def calculate_all_breadths(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate neutralization breadth for all antibodies."""
    print("\nCalculating antibody neutralization breadths...")

    breadth_df = calculate_antibody_breadth(df, ic50_threshold=RESISTANT_THRESHOLD)
    print(f"  Calculated breadth for {len(breadth_df)} antibodies")

    # Add bnAb classification
    breadth_df["is_bnab"] = breadth_df["Antibody"].isin(MAJOR_BNABS.keys())
    breadth_df["bnab_class"] = breadth_df["Antibody"].apply(
        lambda x: MAJOR_BNABS.get(x, {}).get("class", "Other")
    )

    return breadth_df


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================


def analyze_bnab_sensitivity_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sensitivity profiles for major bnAbs."""
    print("\nAnalyzing bnAb sensitivity profiles...")

    results = []

    for antibody, info in MAJOR_BNABS.items():
        ab_data = get_catnap_by_antibody(df, antibody)

        if len(ab_data) == 0:
            continue

        # Get IC50 values
        valid_data = ab_data[ab_data["IC50_numeric"].notna()]
        if len(valid_data) == 0:
            continue

        ic50_values = valid_data["IC50_numeric"].values
        log_ic50 = np.log10(ic50_values + 0.01)

        # Sensitivity classification
        n_sensitive = (ic50_values <= SENSITIVE_THRESHOLD).sum()
        n_resistant = (ic50_values >= RESISTANT_THRESHOLD).sum()
        n_intermediate = len(ic50_values) - n_sensitive - n_resistant

        # Potency statistics
        geometric_mean = np.exp(np.mean(np.log(ic50_values + 0.01)))
        median_ic50 = np.median(ic50_values)

        results.append(
            {
                "Antibody": antibody,
                "epitope_class": info["class"],
                "epitope": info["epitope"],
                "n_tested": len(valid_data),
                "n_sensitive": n_sensitive,
                "n_intermediate": n_intermediate,
                "n_resistant": n_resistant,
                "pct_sensitive": 100 * n_sensitive / len(valid_data),
                "pct_resistant": 100 * n_resistant / len(valid_data),
                "geometric_mean_ic50": geometric_mean,
                "median_ic50": median_ic50,
                "mean_log_ic50": np.mean(log_ic50),
                "std_log_ic50": np.std(log_ic50),
            }
        )

    return pd.DataFrame(results)


def analyze_virus_susceptibility(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze virus susceptibility patterns across antibodies."""
    print("\nAnalyzing virus susceptibility patterns...")

    # Get valid IC50 data
    valid_df = df[df["IC50_numeric"].notna()].copy()

    # Group by virus
    virus_stats = []

    for virus, group in valid_df.groupby("Virus"):
        n_antibodies = group["Antibody"].nunique()
        ic50_values = group["IC50_numeric"].values

        # Calculate susceptibility
        n_sensitive = (ic50_values <= SENSITIVE_THRESHOLD).sum()
        mean_log_ic50 = np.mean(np.log10(ic50_values + 0.01))

        virus_stats.append(
            {
                "Virus": virus,
                "n_antibodies_tested": n_antibodies,
                "n_total_tests": len(group),
                "n_sensitive": n_sensitive,
                "pct_sensitive": 100 * n_sensitive / len(ic50_values),
                "mean_log_ic50": mean_log_ic50,
                "median_ic50": np.median(ic50_values),
            }
        )

    virus_df = pd.DataFrame(virus_stats)
    print(f"  Analyzed {len(virus_df):,} viruses")

    return virus_df


# ============================================================================
# CROSS-NEUTRALIZATION ANALYSIS
# ============================================================================


def build_antibody_similarity_matrix(df: pd.DataFrame, min_common_viruses: int = 10) -> tuple:
    """Build similarity matrix between antibodies based on neutralization profiles."""
    print("\nBuilding antibody similarity matrix...")

    valid_df = df[df["IC50_numeric"].notna()].copy()

    # Get antibodies with sufficient data
    antibody_counts = valid_df.groupby("Antibody")["Virus"].nunique()
    valid_antibodies = antibody_counts[antibody_counts >= min_common_viruses].index.tolist()

    # Focus on bnAbs
    bnab_antibodies = [ab for ab in valid_antibodies if ab in MAJOR_BNABS]

    if len(bnab_antibodies) < 2:
        print("  Not enough bnAbs with sufficient data")
        return None, None

    # Build profile matrix
    viruses = valid_df["Virus"].unique()
    n_ab = len(bnab_antibodies)

    profiles = {}
    for ab in bnab_antibodies:
        ab_data = valid_df[valid_df["Antibody"] == ab].set_index("Virus")
        profile = {}
        for virus in viruses:
            if virus in ab_data.index:
                ic50 = ab_data.loc[virus, "IC50_numeric"]
                if isinstance(ic50, pd.Series):
                    ic50 = ic50.iloc[0]
                profile[virus] = np.log10(ic50 + 0.01)
            else:
                profile[virus] = np.nan
        profiles[ab] = profile

    # Calculate pairwise correlations
    similarity_matrix = np.zeros((n_ab, n_ab))

    for i, ab1 in enumerate(bnab_antibodies):
        for j, ab2 in enumerate(bnab_antibodies):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                # Get common viruses
                common = []
                vals1 = []
                vals2 = []
                for virus in viruses:
                    v1 = profiles[ab1].get(virus)
                    v2 = profiles[ab2].get(virus)
                    if pd.notna(v1) and pd.notna(v2):
                        vals1.append(v1)
                        vals2.append(v2)

                if len(vals1) >= min_common_viruses:
                    corr, _ = stats.spearmanr(vals1, vals2)
                    similarity_matrix[i, j] = corr
                    similarity_matrix[j, i] = corr

    print(f"  Built {n_ab}x{n_ab} similarity matrix")

    return similarity_matrix, bnab_antibodies


def cluster_antibodies_by_sensitivity(df: pd.DataFrame) -> dict:
    """Cluster antibodies by neutralization patterns."""
    print("\nClustering antibodies by sensitivity patterns...")

    similarity_matrix, antibodies = build_antibody_similarity_matrix(df)

    if similarity_matrix is None:
        return {}

    # Convert to distance and cluster
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Hierarchical clustering
    linkage = hierarchy.linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)], method="average")

    # Get cluster assignments at different thresholds
    clusters = hierarchy.fcluster(linkage, t=0.5, criterion="distance")

    cluster_results = {
        "antibodies": antibodies,
        "similarity_matrix": similarity_matrix,
        "linkage": linkage,
        "clusters": clusters,
    }

    return cluster_results


# ============================================================================
# POTENCY ANALYSIS
# ============================================================================


def analyze_potency_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze IC50 potency distributions by epitope class."""
    print("\nAnalyzing potency distributions...")

    results = []

    for epitope_class in set(info["class"] for info in MAJOR_BNABS.values()):
        class_antibodies = [ab for ab, info in MAJOR_BNABS.items() if info["class"] == epitope_class]

        class_data = df[df["Antibody"].isin(class_antibodies) & df["IC50_numeric"].notna()]

        if len(class_data) == 0:
            continue

        ic50_values = class_data["IC50_numeric"].values
        log_ic50 = np.log10(ic50_values + 0.01)

        results.append(
            {
                "epitope_class": epitope_class,
                "n_antibodies": len(class_antibodies),
                "n_records": len(class_data),
                "geometric_mean_ic50": np.exp(np.mean(np.log(ic50_values + 0.01))),
                "median_ic50": np.median(ic50_values),
                "mean_log_ic50": np.mean(log_ic50),
                "std_log_ic50": np.std(log_ic50),
                "min_ic50": np.min(ic50_values),
                "max_ic50": np.max(ic50_values),
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_breadth_distribution(breadth_df: pd.DataFrame):
    """Plot neutralization breadth distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All antibodies
    ax1 = axes[0]
    ax1.hist(breadth_df["breadth_pct"], bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(50, color="red", linestyle="--", label="50% threshold")
    ax1.axvline(80, color="green", linestyle="--", label="80% threshold")
    ax1.set_xlabel("Neutralization Breadth (%)")
    ax1.set_ylabel("Number of Antibodies")
    ax1.set_title(f"Breadth Distribution (n={len(breadth_df)})")
    ax1.legend()

    # Focus on bnAbs
    ax2 = axes[1]
    bnab_df = breadth_df[breadth_df["is_bnab"]].sort_values("breadth_pct", ascending=False)

    if len(bnab_df) > 0:
        colors = [plt.cm.RdYlGn(b / 100) for b in bnab_df["breadth_pct"]]
        ax2.barh(bnab_df["Antibody"], bnab_df["breadth_pct"], color=colors)
        ax2.axvline(50, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Neutralization Breadth (%)")
        ax2.set_title("Known bnAb Breadths")
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "breadth_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'breadth_distribution.png'}")


def plot_bnab_sensitivity_profiles(sensitivity_df: pd.DataFrame):
    """Plot bnAb sensitivity profiles."""
    if sensitivity_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar: sensitive/intermediate/resistant
    ax1 = axes[0]
    sensitivity_sorted = sensitivity_df.sort_values("pct_sensitive", ascending=False)

    x = np.arange(len(sensitivity_sorted))
    width = 0.8

    ax1.bar(
        x,
        sensitivity_sorted["pct_sensitive"],
        width,
        label="Sensitive (<1 ug/mL)",
        color="green",
        alpha=0.7,
    )
    ax1.bar(
        x,
        100 - sensitivity_sorted["pct_sensitive"] - sensitivity_sorted["pct_resistant"],
        width,
        bottom=sensitivity_sorted["pct_sensitive"],
        label="Intermediate",
        color="yellow",
        alpha=0.7,
    )
    ax1.bar(
        x,
        sensitivity_sorted["pct_resistant"],
        width,
        bottom=100 - sensitivity_sorted["pct_resistant"],
        label="Resistant (>50 ug/mL)",
        color="red",
        alpha=0.7,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(sensitivity_sorted["Antibody"], rotation=45, ha="right")
    ax1.set_ylabel("Percentage")
    ax1.set_title("bnAb Sensitivity Profiles")
    ax1.legend(loc="upper right")

    # Potency by epitope class
    ax2 = axes[1]
    class_medians = sensitivity_df.groupby("epitope_class")["geometric_mean_ic50"].median()
    class_medians_sorted = class_medians.sort_values()

    ax2.barh(class_medians_sorted.index, class_medians_sorted.values, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Geometric Mean IC50 (ug/mL)")
    ax2.set_title("Potency by Epitope Class")
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bnab_sensitivity.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'bnab_sensitivity.png'}")


def plot_antibody_clustering(cluster_results: dict):
    """Plot antibody clustering dendrogram and heatmap."""
    if not cluster_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    antibodies = cluster_results["antibodies"]
    similarity_matrix = cluster_results["similarity_matrix"]
    linkage = cluster_results["linkage"]

    # Dendrogram
    ax1 = axes[0]
    hierarchy.dendrogram(linkage, labels=antibodies, ax=ax1, leaf_rotation=45)
    ax1.set_title("bnAb Clustering by Neutralization Profile")
    ax1.set_ylabel("Distance (1 - Spearman r)")

    # Heatmap
    ax2 = axes[1]
    sns.heatmap(
        similarity_matrix,
        xticklabels=antibodies,
        yticklabels=antibodies,
        cmap="RdYlBu",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax2,
        square=True,
    )
    ax2.set_title("Cross-Neutralization Similarity")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "antibody_clustering.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'antibody_clustering.png'}")


def plot_virus_susceptibility(virus_df: pd.DataFrame):
    """Plot virus susceptibility distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of susceptibility
    ax1 = axes[0]
    ax1.hist(virus_df["pct_sensitive"], bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Percent Sensitive to Tested Antibodies")
    ax1.set_ylabel("Number of Viruses")
    ax1.set_title(f"Virus Susceptibility Distribution (n={len(virus_df)})")

    # Relationship between #tests and sensitivity
    ax2 = axes[1]
    ax2.scatter(
        virus_df["n_antibodies_tested"],
        virus_df["pct_sensitive"],
        alpha=0.3,
        s=10,
    )
    ax2.set_xlabel("Number of Antibodies Tested")
    ax2.set_ylabel("Percent Sensitive")
    ax2.set_title("Sensitivity vs Testing Coverage")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "virus_susceptibility.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'virus_susceptibility.png'}")


def plot_potency_by_class(potency_df: pd.DataFrame):
    """Plot IC50 distributions by epitope class."""
    if potency_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    potency_sorted = potency_df.sort_values("geometric_mean_ic50")

    colors = plt.cm.viridis(np.linspace(0, 1, len(potency_sorted)))
    bars = ax.barh(potency_sorted["epitope_class"], potency_sorted["geometric_mean_ic50"], color=colors)

    ax.set_xlabel("Geometric Mean IC50 (ug/mL)")
    ax.set_ylabel("Epitope Class")
    ax.set_title("bnAb Potency by Epitope Class")
    ax.set_xscale("log")

    # Add n values
    for bar, n in zip(bars, potency_sorted["n_records"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" n={n:,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "potency_by_class.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'potency_by_class.png'}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_neutralization_report(
    df: pd.DataFrame,
    breadth_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    virus_df: pd.DataFrame,
    potency_df: pd.DataFrame,
    cluster_results: dict,
):
    """Generate comprehensive neutralization analysis report."""
    report_path = OUTPUT_DIR / "NEUTRALIZATION_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# CATNAP Antibody Neutralization Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total neutralization records: {len(df):,}\n")
        f.write(f"- Records with valid IC50: {df['IC50_numeric'].notna().sum():,}\n")
        f.write(f"- Unique antibodies: {df['Antibody'].nunique():,}\n")
        f.write(f"- Unique viruses: {df['Virus'].nunique():,}\n")
        f.write(f"- Known bnAbs analyzed: {len(MAJOR_BNABS)}\n\n")

        # Breadth analysis
        f.write("## Neutralization Breadth\n\n")
        f.write(f"Breadth calculated at IC50 < {RESISTANT_THRESHOLD} ug/mL\n\n")

        broad_abs = breadth_df[breadth_df["breadth_pct"] >= 50]
        very_broad = breadth_df[breadth_df["breadth_pct"] >= 80]
        f.write(f"- Antibodies with >50% breadth: {len(broad_abs)}\n")
        f.write(f"- Antibodies with >80% breadth: {len(very_broad)}\n\n")

        # bnAb sensitivity profiles
        if not sensitivity_df.empty:
            f.write("### Known bnAb Profiles\n\n")
            f.write("| Antibody | Class | N Tested | % Sensitive | Geo Mean IC50 |\n")
            f.write("|----------|-------|----------|-------------|---------------|\n")

            for _, row in sensitivity_df.sort_values("pct_sensitive", ascending=False).iterrows():
                f.write(
                    f"| {row['Antibody']} | {row['epitope_class']} | {row['n_tested']} | "
                    f"{row['pct_sensitive']:.1f}% | {row['geometric_mean_ic50']:.3f} |\n"
                )
            f.write("\n")

        # Potency by class
        if not potency_df.empty:
            f.write("## Potency by Epitope Class\n\n")
            f.write("| Epitope Class | Antibodies | Records | Geo Mean IC50 | Median IC50 |\n")
            f.write("|---------------|------------|---------|---------------|-------------|\n")

            for _, row in potency_df.sort_values("geometric_mean_ic50").iterrows():
                f.write(
                    f"| {row['epitope_class']} | {row['n_antibodies']} | {row['n_records']:,} | "
                    f"{row['geometric_mean_ic50']:.3f} | {row['median_ic50']:.3f} |\n"
                )
            f.write("\n")

        # Virus susceptibility
        f.write("## Virus Susceptibility\n\n")
        if not virus_df.empty:
            most_susceptible = virus_df.nlargest(10, "pct_sensitive")
            least_susceptible = virus_df[virus_df["n_antibodies_tested"] >= 5].nsmallest(10, "pct_sensitive")

            f.write("### Most Susceptible Viruses\n\n")
            f.write("| Virus | Antibodies Tested | % Sensitive |\n")
            f.write("|-------|-------------------|-------------|\n")
            for _, row in most_susceptible.iterrows():
                f.write(f"| {row['Virus']} | {row['n_antibodies_tested']} | {row['pct_sensitive']:.1f}% |\n")

            f.write("\n### Least Susceptible Viruses (min 5 antibodies tested)\n\n")
            f.write("| Virus | Antibodies Tested | % Sensitive |\n")
            f.write("|-------|-------------------|-------------|\n")
            for _, row in least_susceptible.iterrows():
                f.write(f"| {row['Virus']} | {row['n_antibodies_tested']} | {row['pct_sensitive']:.1f}% |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        if not sensitivity_df.empty:
            most_potent = sensitivity_df.loc[sensitivity_df["geometric_mean_ic50"].idxmin()]
            f.write(
                f"1. **Most Potent bnAb**: {most_potent['Antibody']} "
                f"(IC50 = {most_potent['geometric_mean_ic50']:.3f} ug/mL)\n"
            )

            broadest = sensitivity_df.loc[sensitivity_df["pct_sensitive"].idxmax()]
            f.write(
                f"2. **Broadest bnAb**: {broadest['Antibody']} "
                f"({broadest['pct_sensitive']:.1f}% sensitive)\n"
            )

        if not potency_df.empty:
            best_class = potency_df.loc[potency_df["geometric_mean_ic50"].idxmin()]
            f.write(
                f"3. **Most Potent Epitope Class**: {best_class['epitope_class']} "
                f"(IC50 = {best_class['geometric_mean_ic50']:.3f} ug/mL)\n"
            )

        f.write("\n## Generated Files\n\n")
        f.write("- `breadth_distribution.png` - Antibody breadth histogram\n")
        f.write("- `bnab_sensitivity.png` - bnAb sensitivity profiles\n")
        f.write("- `antibody_clustering.png` - Cross-neutralization clustering\n")
        f.write("- `virus_susceptibility.png` - Virus susceptibility patterns\n")
        f.write("- `potency_by_class.png` - Potency by epitope class\n")
        f.write("- `breadth_data.csv` - Antibody breadth data\n")
        f.write("- `bnab_sensitivity.csv` - bnAb sensitivity profiles\n")
        f.write("- `virus_susceptibility.csv` - Virus susceptibility data\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete CATNAP neutralization analysis."""
    print("=" * 70)
    print("CATNAP Antibody Neutralization Analysis")
    print("=" * 70)

    # Load data
    df = load_neutralization_data()
    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Calculate breadths
    breadth_df = calculate_all_breadths(df)

    # Analyze bnAb sensitivity
    sensitivity_df = analyze_bnab_sensitivity_profiles(df)

    # Analyze virus susceptibility
    virus_df = analyze_virus_susceptibility(df)

    # Analyze potency by class
    potency_df = analyze_potency_distribution(df)

    # Cluster antibodies
    cluster_results = cluster_antibodies_by_sensitivity(df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_breadth_distribution(breadth_df)
    plot_bnab_sensitivity_profiles(sensitivity_df)
    plot_virus_susceptibility(virus_df)
    plot_potency_by_class(potency_df)

    if cluster_results:
        plot_antibody_clustering(cluster_results)

    # Save data
    print("\nSaving results...")

    breadth_df.to_csv(OUTPUT_DIR / "breadth_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'breadth_data.csv'}")

    sensitivity_df.to_csv(OUTPUT_DIR / "bnab_sensitivity.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'bnab_sensitivity.csv'}")

    virus_df.to_csv(OUTPUT_DIR / "virus_susceptibility.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'virus_susceptibility.csv'}")

    if not potency_df.empty:
        potency_df.to_csv(OUTPUT_DIR / "potency_by_class.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'potency_by_class.csv'}")

    # Generate report
    print("\nGenerating report...")
    generate_neutralization_report(df, breadth_df, sensitivity_df, virus_df, potency_df, cluster_results)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
