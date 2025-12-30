# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Stanford HIVDB Drug Resistance Analysis

Comprehensive analysis of 7,154 drug resistance records using p-adic
hyperbolic codon geometry.

Analyses performed:
1. Fold-change vs hyperbolic distance correlation
2. Primary vs accessory mutation classification
3. Cross-resistance pattern mapping
4. Drug class-specific geometric signatures

Input: Stanford HIVDB files (PI, NRTI, NNRTI, INSTI)
Output: Resistance analysis results and visualizations
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

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from codon_extraction import encode_mutation_pair
from hyperbolic_utils import load_hyperbolic_encoder
from position_mapper import parse_mutation_list
from unified_data_loader import load_stanford_hivdb

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "stanford_resistance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Drug class configurations
DRUG_CLASSES = {
    "PI": {"protein": "PR", "drugs": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"]},
    "NRTI": {"protein": "RT", "drugs": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"]},
    "NNRTI": {"protein": "RT", "drugs": ["DOR", "EFV", "ETR", "NVP", "RPV"]},
    "INI": {"protein": "IN", "drugs": ["BIC", "CAB", "DTG", "EVG", "RAL"]},
}

# Known primary resistance mutations (from Stanford HIVDB)
PRIMARY_MUTATIONS = {
    "PI": [
        "D30N",
        "V32I",
        "M46I",
        "M46L",
        "I47V",
        "I47A",
        "G48V",
        "I50V",
        "I50L",
        "I54L",
        "I54M",
        "L76V",
        "V82A",
        "V82F",
        "V82T",
        "V82S",
        "I84V",
        "N88S",
        "L90M",
    ],
    "NRTI": [
        "M41L",
        "K65R",
        "K65N",
        "D67N",
        "K70R",
        "K70E",
        "L74V",
        "L74I",
        "Y115F",
        "Q151M",
        "M184V",
        "M184I",
        "L210W",
        "T215Y",
        "T215F",
        "K219Q",
        "K219E",
    ],
    "NNRTI": ["L100I", "K101E", "K101P", "K103N", "K103S", "V106A", "V106M", "E138K", "E138A", "Y181C", "Y181I", "Y188L", "Y188C", "G190A", "G190S"],
    "INI": ["T66I", "T66A", "E92Q", "G118R", "E138K", "E138A", "G140S", "G140A", "Y143R", "Y143C", "S147G", "Q148H", "Q148K", "Q148R", "N155H", "R263K"],
}


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================


def load_all_resistance_data() -> pd.DataFrame:
    """Load and combine all Stanford HIVDB drug resistance data."""
    print("Loading Stanford HIVDB data...")

    try:
        df = load_stanford_hivdb("all")
        print(f"  Loaded {len(df)} total records")

        for drug_class in ["PI", "NRTI", "NNRTI", "INI"]:
            class_count = len(df[df["drug_class"] == drug_class])
            print(f"    {drug_class}: {class_count} records")

        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def extract_mutations_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and parse all mutations from the dataset."""
    print("\nExtracting mutations...")

    mutation_records = []

    for _, row in df.iterrows():
        mut_list = row.get("CompMutList", "")
        if pd.isna(mut_list) or not mut_list:
            continue

        drug_class = row.get("drug_class", "Unknown")
        protein = DRUG_CLASSES.get(drug_class, {}).get("protein", "Unknown")
        drugs = DRUG_CLASSES.get(drug_class, {}).get("drugs", [])

        # Parse mutations
        mutations = parse_mutation_list(str(mut_list))

        for mut in mutations:
            record = {
                "seq_id": row.get("SeqID"),
                "drug_class": drug_class,
                "protein": protein,
                "position": mut["position"],
                "wild_type_aa": mut["wild_type"],
                "mutant_aa": mut["mutant"],
                "mutation_str": f"{mut['wild_type']}{mut['position']}{mut['mutant']}",
            }

            # Add drug fold-change values
            for drug in drugs:
                fc_value = row.get(drug)
                if pd.notna(fc_value) and str(fc_value) not in ["NA", ""]:
                    try:
                        record[f"{drug}_fc"] = float(fc_value)
                    except ValueError:
                        pass

            # Check if primary mutation
            record["is_primary"] = record["mutation_str"] in PRIMARY_MUTATIONS.get(drug_class, [])

            mutation_records.append(record)

    mutation_df = pd.DataFrame(mutation_records)
    print(f"  Extracted {len(mutation_df)} mutation occurrences")
    print(f"  Unique mutations: {mutation_df['mutation_str'].nunique()}")

    return mutation_df


# ============================================================================
# HYPERBOLIC ANALYSIS
# ============================================================================


def calculate_mutation_distances(mutation_df: pd.DataFrame, encoder) -> pd.DataFrame:
    """Calculate hyperbolic distances for all mutations."""
    print("\nCalculating hyperbolic distances...")

    # Get unique mutations
    unique_muts = mutation_df[["wild_type_aa", "mutant_aa", "mutation_str"]].drop_duplicates()
    print(f"  Processing {len(unique_muts)} unique mutations...")

    distance_map = {}

    for _, row in unique_muts.iterrows():
        mut_str = row["mutation_str"]
        wild_aa = row["wild_type_aa"]
        mut_aa = row["mutant_aa"]

        if pd.isna(wild_aa) or pd.isna(mut_aa):
            continue

        result = encode_mutation_pair(wild_aa, mut_aa, encoder, method="representative")
        if result:
            distance_map[mut_str] = result["hyperbolic_distance"]

    # Map distances back to full DataFrame
    mutation_df["hyperbolic_distance"] = mutation_df["mutation_str"].map(distance_map)

    valid_count = mutation_df["hyperbolic_distance"].notna().sum()
    print(f"  Calculated distances for {valid_count} mutation occurrences")

    return mutation_df


def analyze_distance_resistance_correlation(mutation_df: pd.DataFrame) -> dict:
    """Analyze correlation between hyperbolic distance and drug resistance."""
    print("\nAnalyzing distance-resistance correlations...")

    results = {}

    for drug_class, config in DRUG_CLASSES.items():
        drugs = config["drugs"]
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()

        if len(class_df) == 0:
            continue

        class_results = {"drug_class": drug_class, "correlations": {}}

        for drug in drugs:
            fc_col = f"{drug}_fc"
            if fc_col not in class_df.columns:
                continue

            # Filter valid values
            valid = class_df[[fc_col, "hyperbolic_distance"]].dropna()
            if len(valid) < 10:
                continue

            # Calculate correlation
            corr, pval = stats.pearsonr(valid["hyperbolic_distance"], valid[fc_col])
            spearman_corr, spearman_pval = stats.spearmanr(valid["hyperbolic_distance"], valid[fc_col])

            class_results["correlations"][drug] = {
                "n_samples": len(valid),
                "pearson_r": corr,
                "pearson_pval": pval,
                "spearman_r": spearman_corr,
                "spearman_pval": spearman_pval,
            }

        results[drug_class] = class_results

    return results


def analyze_primary_vs_accessory(mutation_df: pd.DataFrame) -> pd.DataFrame:
    """Compare hyperbolic distances of primary vs accessory mutations."""
    print("\nAnalyzing primary vs accessory mutations...")

    summary = []

    for drug_class in DRUG_CLASSES.keys():
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()

        if len(class_df) == 0:
            continue

        primary = class_df[class_df["is_primary"]]
        accessory = class_df[~class_df["is_primary"]]

        primary_dist = primary["hyperbolic_distance"].dropna()
        accessory_dist = accessory["hyperbolic_distance"].dropna()

        if len(primary_dist) > 0 and len(accessory_dist) > 0:
            # Statistical test
            stat, pval = stats.mannwhitneyu(primary_dist, accessory_dist, alternative="two-sided")

            summary.append(
                {
                    "drug_class": drug_class,
                    "n_primary": len(primary_dist),
                    "n_accessory": len(accessory_dist),
                    "primary_mean_dist": primary_dist.mean(),
                    "primary_std_dist": primary_dist.std(),
                    "accessory_mean_dist": accessory_dist.mean(),
                    "accessory_std_dist": accessory_dist.std(),
                    "mann_whitney_stat": stat,
                    "pvalue": pval,
                }
            )

    return pd.DataFrame(summary)


def analyze_cross_resistance(mutation_df: pd.DataFrame) -> pd.DataFrame:
    """Identify mutations conferring resistance to multiple drugs."""
    print("\nAnalyzing cross-resistance patterns...")

    cross_resistance = []

    for drug_class, config in DRUG_CLASSES.items():
        drugs = config["drugs"]
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()

        if len(class_df) == 0:
            continue

        # Group by mutation
        for mut_str, group in class_df.groupby("mutation_str"):
            fc_values = {}
            for drug in drugs:
                fc_col = f"{drug}_fc"
                if fc_col in group.columns:
                    values = group[fc_col].dropna()
                    if len(values) > 0:
                        fc_values[drug] = values.median()

            # Count drugs with significant resistance (FC > 3)
            resistant_drugs = [d for d, fc in fc_values.items() if fc > 3]

            if len(resistant_drugs) >= 2:
                cross_resistance.append(
                    {
                        "drug_class": drug_class,
                        "mutation": mut_str,
                        "n_drugs_resistant": len(resistant_drugs),
                        "resistant_drugs": ", ".join(resistant_drugs),
                        "hyperbolic_distance": group["hyperbolic_distance"].iloc[0],
                        "mean_fc": np.mean(list(fc_values.values())),
                        "max_fc": max(fc_values.values()) if fc_values else 0,
                    }
                )

    return pd.DataFrame(cross_resistance).sort_values("n_drugs_resistant", ascending=False)


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_distance_distributions(mutation_df: pd.DataFrame):
    """Plot hyperbolic distance distributions by drug class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (drug_class, config) in zip(axes.flat, DRUG_CLASSES.items()):
        class_df = mutation_df[mutation_df["drug_class"] == drug_class]
        distances = class_df["hyperbolic_distance"].dropna()

        if len(distances) > 0:
            ax.hist(distances, bins=30, edgecolor="black", alpha=0.7)
            ax.axvline(distances.mean(), color="red", linestyle="--", label=f"Mean: {distances.mean():.2f}")
            ax.axvline(distances.median(), color="green", linestyle="--", label=f"Median: {distances.median():.2f}")
            ax.set_xlabel("Hyperbolic Distance")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{drug_class} Mutations (n={len(distances)})")
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'distance_distributions.png'}")


def plot_primary_vs_accessory(mutation_df: pd.DataFrame):
    """Plot primary vs accessory mutation distances."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (drug_class, config) in zip(axes.flat, DRUG_CLASSES.items()):
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()

        if len(class_df) == 0:
            continue

        primary = class_df[class_df["is_primary"]]["hyperbolic_distance"].dropna()
        accessory = class_df[~class_df["is_primary"]]["hyperbolic_distance"].dropna()

        if len(primary) > 0 and len(accessory) > 0:
            data = [primary.values, accessory.values]
            bp = ax.boxplot(data, labels=["Primary", "Accessory"], patch_artist=True)
            bp["boxes"][0].set_facecolor("coral")
            bp["boxes"][1].set_facecolor("lightblue")

            # Add significance
            _, pval = stats.mannwhitneyu(primary, accessory)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{drug_class} (p={pval:.4f} {sig})")
            ax.set_ylabel("Hyperbolic Distance")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "primary_vs_accessory.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'primary_vs_accessory.png'}")


def plot_distance_vs_resistance(mutation_df: pd.DataFrame):
    """Plot hyperbolic distance vs fold-change for each drug class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (drug_class, config) in zip(axes.flat, DRUG_CLASSES.items()):
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()
        drugs = config["drugs"]

        if len(class_df) == 0:
            continue

        # Collect all drug data
        all_dists = []
        all_fcs = []
        all_drugs = []

        for drug in drugs:
            fc_col = f"{drug}_fc"
            if fc_col not in class_df.columns:
                continue

            valid = class_df[["hyperbolic_distance", fc_col]].dropna()
            if len(valid) > 0:
                all_dists.extend(valid["hyperbolic_distance"].values)
                all_fcs.extend(np.log10(valid[fc_col].values + 0.1))  # Log scale
                all_drugs.extend([drug] * len(valid))

        if len(all_dists) > 0:
            ax.scatter(all_dists, all_fcs, alpha=0.3, s=10)
            ax.set_xlabel("Hyperbolic Distance")
            ax.set_ylabel("log10(Fold-Change)")
            ax.set_title(f"{drug_class}")

            # Add correlation line
            if len(all_dists) > 10:
                z = np.polyfit(all_dists, all_fcs, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(all_dists), max(all_dists), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.7)

                corr, _ = stats.pearsonr(all_dists, all_fcs)
                ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, verticalalignment="top")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_vs_resistance.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'distance_vs_resistance.png'}")


def plot_cross_resistance_heatmap(mutation_df: pd.DataFrame):
    """Plot cross-resistance patterns as heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (drug_class, config) in zip(axes.flat, DRUG_CLASSES.items()):
        class_df = mutation_df[mutation_df["drug_class"] == drug_class].copy()
        drugs = config["drugs"]

        if len(class_df) == 0:
            continue

        # Build correlation matrix between drugs
        fc_cols = [f"{d}_fc" for d in drugs if f"{d}_fc" in class_df.columns]
        if len(fc_cols) < 2:
            continue

        corr_matrix = class_df[fc_cols].corr()
        corr_matrix.columns = [c.replace("_fc", "") for c in corr_matrix.columns]
        corr_matrix.index = [i.replace("_fc", "") for i in corr_matrix.index]

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax, square=True)
        ax.set_title(f"{drug_class} Cross-Resistance")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_resistance_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'cross_resistance_heatmap.png'}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_summary_report(
    mutation_df: pd.DataFrame,
    correlation_results: dict,
    primary_accessory_df: pd.DataFrame,
    cross_resistance_df: pd.DataFrame,
):
    """Generate comprehensive analysis report."""
    report_path = OUTPUT_DIR / "ANALYSIS_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Stanford HIVDB Drug Resistance Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total mutation occurrences: {len(mutation_df):,}\n")
        f.write(f"- Unique mutations: {mutation_df['mutation_str'].nunique():,}\n")
        f.write(f"- Mutations with valid distances: {mutation_df['hyperbolic_distance'].notna().sum():,}\n\n")

        f.write("### By Drug Class\n\n")
        f.write("| Drug Class | Mutations | Unique | Mean Distance | Std Distance |\n")
        f.write("|------------|-----------|--------|---------------|---------------|\n")

        for drug_class in DRUG_CLASSES.keys():
            class_df = mutation_df[mutation_df["drug_class"] == drug_class]
            dists = class_df["hyperbolic_distance"].dropna()
            f.write(
                f"| {drug_class} | {len(class_df):,} | {class_df['mutation_str'].nunique()} | "
                f"{dists.mean():.3f} | {dists.std():.3f} |\n"
            )

        # Correlation results
        f.write("\n## Distance-Resistance Correlations\n\n")

        for drug_class, results in correlation_results.items():
            if "correlations" not in results or not results["correlations"]:
                continue

            f.write(f"### {drug_class}\n\n")
            f.write("| Drug | N | Pearson r | p-value | Spearman r | p-value |\n")
            f.write("|------|---|-----------|---------|------------|----------|\n")

            for drug, corr in results["correlations"].items():
                p_sig = "***" if corr["pearson_pval"] < 0.001 else "**" if corr["pearson_pval"] < 0.01 else "*" if corr["pearson_pval"] < 0.05 else ""
                f.write(
                    f"| {drug} | {corr['n_samples']} | {corr['pearson_r']:.3f} | "
                    f"{corr['pearson_pval']:.4f}{p_sig} | {corr['spearman_r']:.3f} | "
                    f"{corr['spearman_pval']:.4f} |\n"
                )
            f.write("\n")

        # Primary vs accessory
        f.write("## Primary vs Accessory Mutations\n\n")
        f.write("| Drug Class | N Primary | N Accessory | Primary Mean | Accessory Mean | p-value |\n")
        f.write("|------------|-----------|-------------|--------------|----------------|----------|\n")

        for _, row in primary_accessory_df.iterrows():
            sig = "***" if row["pvalue"] < 0.001 else "**" if row["pvalue"] < 0.01 else "*" if row["pvalue"] < 0.05 else ""
            f.write(
                f"| {row['drug_class']} | {row['n_primary']} | {row['n_accessory']} | "
                f"{row['primary_mean_dist']:.3f} | {row['accessory_mean_dist']:.3f} | "
                f"{row['pvalue']:.4f}{sig} |\n"
            )

        # Cross-resistance mutations
        f.write("\n## Top Cross-Resistance Mutations\n\n")
        f.write("Mutations conferring resistance to 3+ drugs:\n\n")
        f.write("| Class | Mutation | Drugs | Distance | Max FC |\n")
        f.write("|-------|----------|-------|----------|--------|\n")

        top_cross = cross_resistance_df[cross_resistance_df["n_drugs_resistant"] >= 3].head(20)
        for _, row in top_cross.iterrows():
            f.write(
                f"| {row['drug_class']} | {row['mutation']} | {row['resistant_drugs']} | "
                f"{row['hyperbolic_distance']:.3f} | {row['max_fc']:.1f} |\n"
            )

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Calculate average correlation
        all_corrs = []
        for results in correlation_results.values():
            for corr in results.get("correlations", {}).values():
                all_corrs.append(corr["spearman_r"])

        if all_corrs:
            f.write(f"1. **Distance-Resistance Correlation**: Mean Spearman r = {np.mean(all_corrs):.3f}\n")

        # Primary vs accessory difference
        if not primary_accessory_df.empty:
            avg_primary = primary_accessory_df["primary_mean_dist"].mean()
            avg_accessory = primary_accessory_df["accessory_mean_dist"].mean()
            f.write(f"2. **Primary vs Accessory**: Primary mutations have {avg_primary:.3f} mean distance "
                    f"vs {avg_accessory:.3f} for accessory\n")

        # Cross-resistance
        if not cross_resistance_df.empty:
            high_cross = len(cross_resistance_df[cross_resistance_df["n_drugs_resistant"] >= 3])
            f.write(f"3. **Cross-Resistance**: {high_cross} mutations confer resistance to 3+ drugs\n")

        f.write("\n## Generated Files\n\n")
        f.write("- `distance_distributions.png` - Hyperbolic distance histograms by drug class\n")
        f.write("- `primary_vs_accessory.png` - Comparison of primary and accessory mutations\n")
        f.write("- `distance_vs_resistance.png` - Scatter plots of distance vs fold-change\n")
        f.write("- `cross_resistance_heatmap.png` - Drug cross-resistance correlation matrix\n")
        f.write("- `mutation_distances.csv` - Complete mutation data with distances\n")
        f.write("- `cross_resistance.csv` - Cross-resistance mutation details\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete Stanford resistance analysis."""
    print("=" * 70)
    print("Stanford HIVDB Drug Resistance - Hyperbolic Analysis")
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
    resistance_df = load_all_resistance_data()
    if resistance_df.empty:
        print("No data loaded. Exiting.")
        return

    # Extract mutations
    mutation_df = extract_mutations_from_data(resistance_df)
    if mutation_df.empty:
        print("No mutations extracted. Exiting.")
        return

    # Calculate hyperbolic distances
    if encoder:
        mutation_df = calculate_mutation_distances(mutation_df, encoder)

    # Run analyses
    correlation_results = analyze_distance_resistance_correlation(mutation_df)
    primary_accessory_df = analyze_primary_vs_accessory(mutation_df)
    cross_resistance_df = analyze_cross_resistance(mutation_df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_distance_distributions(mutation_df)
    plot_primary_vs_accessory(mutation_df)
    plot_distance_vs_resistance(mutation_df)
    plot_cross_resistance_heatmap(mutation_df)

    # Save data
    print("\nSaving results...")
    mutation_df.to_csv(OUTPUT_DIR / "mutation_distances.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'mutation_distances.csv'}")

    cross_resistance_df.to_csv(OUTPUT_DIR / "cross_resistance.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'cross_resistance.csv'}")

    primary_accessory_df.to_csv(OUTPUT_DIR / "primary_vs_accessory.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'primary_vs_accessory.csv'}")

    # Generate report
    print("\nGenerating report...")
    generate_summary_report(mutation_df, correlation_results, primary_accessory_df, cross_resistance_df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
