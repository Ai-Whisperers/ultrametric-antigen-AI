# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Cross-Dataset Integration Analysis

Integrates multiple HIV datasets for combined analyses:
1. Drug resistance vs immune escape trade-offs
2. Multi-pressure constraint mapping
3. Universal vaccine target identification

Integrates:
- Stanford HIVDB (drug resistance)
- LANL CTL (immune escape)
- CATNAP (antibody neutralization)
- V3/gp120 (tropism)

Output: Integrated analysis results and vaccine target rankings
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from position_mapper import (
    find_overlapping_epitopes,
    get_region,
    parse_mutation_list,
    protein_pos_to_hxb2,
)
from unified_data_loader import (
    load_catnap,
    load_lanl_ctl,
    load_stanford_hivdb,
    parse_hla_restrictions,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "integrated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constraint thresholds
RESISTANCE_FOLD_CHANGE_THRESHOLD = 3.0  # FC > 3 = resistant
CTL_EPITOPE_CONSTRAINT = 3  # At least 3 HLA restrictions
BNAB_BREADTH_THRESHOLD = 50  # >50% breadth = broad


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_datasets() -> tuple:
    """Load all required datasets."""
    print("Loading all datasets...")

    stanford_df = None
    ctl_df = None
    catnap_df = None

    try:
        stanford_df = load_stanford_hivdb("all")
        print(f"  Stanford HIVDB: {len(stanford_df):,} records")
    except FileNotFoundError as e:
        print(f"  Stanford HIVDB: Failed - {e}")

    try:
        ctl_df = load_lanl_ctl()
        print(f"  LANL CTL: {len(ctl_df):,} epitopes")
    except FileNotFoundError as e:
        print(f"  LANL CTL: Failed - {e}")

    try:
        catnap_df = load_catnap()
        print(f"  CATNAP: {len(catnap_df):,} records")
    except FileNotFoundError as e:
        print(f"  CATNAP: Failed - {e}")

    return stanford_df, ctl_df, catnap_df


# ============================================================================
# RESISTANCE-IMMUNITY TRADE-OFF ANALYSIS
# ============================================================================


def analyze_resistance_immunity_overlap(stanford_df: pd.DataFrame, ctl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find positions where drug resistance mutations overlap with CTL epitopes.

    These represent potential trade-offs where resistance may increase immune vulnerability.
    """
    print("\nAnalyzing resistance-immunity overlaps...")

    if stanford_df is None or ctl_df is None:
        return pd.DataFrame()

    overlaps = []

    # Get unique mutations from Stanford
    for _, row in stanford_df.iterrows():
        mut_list = row.get("CompMutList", "")
        if pd.isna(mut_list) or not mut_list:
            continue

        drug_class = row.get("drug_class", "Unknown")
        protein_map = {"PI": "Pol", "NRTI": "Pol", "NNRTI": "Pol", "INI": "Pol"}
        pol_region_map = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INI": "IN"}

        hiv_protein = protein_map.get(drug_class, "Pol")
        enzyme = pol_region_map.get(drug_class, "PR")

        mutations = parse_mutation_list(str(mut_list))

        for mut in mutations:
            position = mut["position"]

            # Convert to HXB2 position
            hxb2_pos = protein_pos_to_hxb2(position, enzyme.lower())
            if hxb2_pos is None:
                continue

            # Find overlapping CTL epitopes
            pol_epitopes = ctl_df[ctl_df["Protein"].str.contains("Pol", case=False, na=False)].to_dict("records")

            overlapping = find_overlapping_epitopes(hiv_protein, hxb2_pos, pol_epitopes)

            for epi in overlapping:
                hla_list = parse_hla_restrictions(epi.get("HLA", ""))

                overlaps.append(
                    {
                        "drug_class": drug_class,
                        "enzyme": enzyme,
                        "position": position,
                        "mutation": f"{mut['wild_type']}{position}{mut['mutant']}",
                        "hxb2_position": hxb2_pos,
                        "epitope": epi.get("Epitope"),
                        "epitope_start": epi.get("HXB2_start"),
                        "n_hla_restrictions": len(hla_list),
                        "hla_types": ", ".join(hla_list[:5]),
                    }
                )

    overlap_df = pd.DataFrame(overlaps)

    if not overlap_df.empty:
        # Remove duplicates
        overlap_df = overlap_df.drop_duplicates(subset=["mutation", "epitope"])
        print(f"  Found {len(overlap_df)} resistance-epitope overlaps")
        print(f"  Unique mutations: {overlap_df['mutation'].nunique()}")
        print(f"  Unique epitopes: {overlap_df['epitope'].nunique()}")

    return overlap_df


def calculate_tradeoff_scores(overlap_df: pd.DataFrame, stanford_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trade-off scores for each overlapping position.

    High score = strong drug resistance + strong immune pressure
    """
    print("\nCalculating trade-off scores...")

    if overlap_df.empty or stanford_df is None:
        return pd.DataFrame()

    # Get mean fold-change for each mutation
    mutation_fc = {}

    for _, row in stanford_df.iterrows():
        mut_list = row.get("CompMutList", "")
        if pd.isna(mut_list):
            continue

        drug_class = row.get("drug_class", "")
        drug_cols = {
            "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
            "NRTI": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
            "NNRTI": ["DOR", "EFV", "ETR", "NVP", "RPV"],
            "INI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
        }.get(drug_class, [])

        for mut_str in str(mut_list).split(","):
            mut = mut_str.strip()
            if not mut:
                continue

            # Collect FC values
            fc_values = []
            for drug in drug_cols:
                fc = row.get(drug)
                if pd.notna(fc):
                    try:
                        fc_values.append(float(fc))
                    except (ValueError, TypeError):
                        pass

            if fc_values:
                key = (drug_class, mut)
                if key not in mutation_fc:
                    mutation_fc[key] = []
                mutation_fc[key].extend(fc_values)

    # Calculate trade-off scores
    scores = []

    for _, row in overlap_df.iterrows():
        key = (row["drug_class"], row["mutation"])
        fc_list = mutation_fc.get(key, [])

        mean_fc = np.mean(fc_list) if fc_list else 0
        max_fc = np.max(fc_list) if fc_list else 0

        # Trade-off score: geometric mean of resistance and immune pressure
        immune_pressure = row["n_hla_restrictions"]
        resistance_score = np.log10(mean_fc + 1) if mean_fc > 0 else 0
        tradeoff_score = np.sqrt(resistance_score * (immune_pressure + 1))

        scores.append(
            {
                "drug_class": row["drug_class"],
                "mutation": row["mutation"],
                "epitope": row["epitope"],
                "n_hla_restrictions": immune_pressure,
                "mean_fold_change": mean_fc,
                "max_fold_change": max_fc,
                "resistance_score": resistance_score,
                "tradeoff_score": tradeoff_score,
            }
        )

    score_df = pd.DataFrame(scores)

    if not score_df.empty:
        score_df = score_df.sort_values("tradeoff_score", ascending=False)
        print(f"  Calculated scores for {len(score_df)} overlaps")

    return score_df


# ============================================================================
# MULTI-PRESSURE CONSTRAINT MAPPING
# ============================================================================


def map_constraint_landscape(stanford_df: pd.DataFrame, ctl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map constraints across the Pol region from both drug and immune pressure.
    """
    print("\nMapping constraint landscape...")

    constraints = []

    # Drug resistance constraints (from Stanford)
    if stanford_df is not None:
        for drug_class in ["PI", "NRTI", "NNRTI", "INI"]:
            class_df = stanford_df[stanford_df["drug_class"] == drug_class]

            # Get position columns
            enzyme = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INI": "IN"}[drug_class]
            prefix = {"PI": "P", "NRTI": "RT", "NNRTI": "RT", "INI": "IN"}[drug_class]

            pos_cols = [c for c in class_df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]

            for col in pos_cols:
                position = int(col[len(prefix) :])

                # Count mutations at this position
                aa_counts = class_df[col].value_counts()
                total = len(class_df[col].dropna())

                if total == 0:
                    continue

                # Calculate conservation (frequency of most common AA)
                conservation = aa_counts.max() / total if total > 0 else 0

                # Calculate mutation frequency
                reference_aa = aa_counts.idxmax() if len(aa_counts) > 0 else "-"
                mutation_freq = 1 - conservation

                constraints.append(
                    {
                        "region": enzyme,
                        "position": position,
                        "constraint_type": "drug_resistance",
                        "conservation": conservation,
                        "mutation_frequency": mutation_freq,
                        "reference_aa": reference_aa,
                    }
                )

    # CTL epitope constraints
    if ctl_df is not None:
        pol_epitopes = ctl_df[ctl_df["Protein"].str.contains("Pol", case=False, na=False)]

        for _, epi in pol_epitopes.iterrows():
            start = epi.get("HXB2_start")
            end = epi.get("HXB2_end")
            epitope_seq = epi.get("Epitope", "")

            if pd.isna(start):
                continue

            # Determine region
            for region_name, region in [("PR", get_region("pr")), ("RT", get_region("rt")), ("IN", get_region("in"))]:
                if region and region.aa_start <= start <= region.aa_end:
                    local_pos = start - region.aa_start + 1

                    hla_count = len(parse_hla_restrictions(epi.get("HLA", "")))

                    constraints.append(
                        {
                            "region": region_name,
                            "position": local_pos,
                            "constraint_type": "ctl_epitope",
                            "n_hla": hla_count,
                            "epitope_length": len(epitope_seq),
                        }
                    )
                    break

    constraint_df = pd.DataFrame(constraints)

    if not constraint_df.empty:
        print(f"  Mapped {len(constraint_df)} constraint records")
        print(f"  Drug resistance: {(constraint_df['constraint_type'] == 'drug_resistance').sum()}")
        print(f"  CTL epitopes: {(constraint_df['constraint_type'] == 'ctl_epitope').sum()}")

    return constraint_df


# ============================================================================
# VACCINE TARGET IDENTIFICATION
# ============================================================================


def identify_vaccine_targets(
    constraint_df: pd.DataFrame, overlap_df: pd.DataFrame, ctl_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify optimal vaccine targets based on multi-constraint analysis.

    Criteria:
    1. High conservation (low mutation frequency)
    2. Multiple HLA restrictions (broad immune coverage)
    3. Minimal overlap with resistance positions
    """
    print("\nIdentifying vaccine targets...")

    if ctl_df is None or constraint_df.empty:
        return pd.DataFrame()

    targets = []

    # Get epitopes with strong HLA restrictions
    ctl_df["HLA_list"] = ctl_df["HLA"].apply(parse_hla_restrictions)
    ctl_df["n_hla"] = ctl_df["HLA_list"].apply(len)

    strong_epitopes = ctl_df[ctl_df["n_hla"] >= CTL_EPITOPE_CONSTRAINT].copy()

    if len(strong_epitopes) == 0:
        print("  No strongly restricted epitopes found")
        return pd.DataFrame()

    # Get resistance positions
    resistance_positions = set()
    if not overlap_df.empty:
        resistance_positions = set(overlap_df["mutation"].unique())

    # Score each epitope
    for _, epi in strong_epitopes.iterrows():
        epitope_seq = epi.get("Epitope", "")
        start = epi.get("HXB2_start")
        protein = epi.get("Protein", "")
        n_hla = epi.get("n_hla", 0)

        if pd.isna(start) or not epitope_seq:
            continue

        # Check for resistance overlap
        has_resistance_overlap = False
        if not overlap_df.empty:
            overlaps = overlap_df[overlap_df["epitope"] == epitope_seq]
            has_resistance_overlap = len(overlaps) > 0

        # Calculate conservation from constraint data
        conservation_score = 0
        if not constraint_df.empty:
            region_cons = constraint_df[constraint_df["constraint_type"] == "drug_resistance"]
            # Average conservation in this region
            conservation_score = region_cons["conservation"].mean() if len(region_cons) > 0 else 0.5

        # Vaccine target score
        # Higher score = better target
        # Penalize resistance overlap, reward HLA breadth and conservation
        overlap_penalty = 0.5 if has_resistance_overlap else 1.0
        target_score = (n_hla / 10) * conservation_score * overlap_penalty

        targets.append(
            {
                "epitope": epitope_seq,
                "protein": protein,
                "hxb2_start": start,
                "length": len(epitope_seq),
                "n_hla_restrictions": n_hla,
                "hla_types": ", ".join(epi.get("HLA_list", [])[:5]),
                "has_resistance_overlap": has_resistance_overlap,
                "conservation_score": conservation_score,
                "target_score": target_score,
            }
        )

    target_df = pd.DataFrame(targets)

    if not target_df.empty:
        target_df = target_df.sort_values("target_score", ascending=False)
        print(f"  Identified {len(target_df)} potential vaccine targets")
        print(f"  Top targets without resistance overlap: {(~target_df['has_resistance_overlap']).sum()}")

    return target_df


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_tradeoff_landscape(score_df: pd.DataFrame):
    """Plot resistance-immunity trade-off landscape."""
    if score_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: resistance vs immunity
    ax1 = axes[0]
    colors = {"PI": "red", "NRTI": "blue", "NNRTI": "green", "INI": "purple"}

    for drug_class, group in score_df.groupby("drug_class"):
        ax1.scatter(
            group["mean_fold_change"],
            group["n_hla_restrictions"],
            c=colors.get(drug_class, "gray"),
            alpha=0.5,
            label=drug_class,
            s=30,
        )

    ax1.set_xlabel("Mean Fold-Change (Drug Resistance)")
    ax1.set_ylabel("# HLA Restrictions (Immune Pressure)")
    ax1.set_title("Drug Resistance vs Immune Pressure Trade-offs")
    ax1.set_xscale("log")
    ax1.legend()

    # Top trade-offs by class
    ax2 = axes[1]
    top_by_class = score_df.groupby("drug_class")["tradeoff_score"].max()

    ax2.bar(top_by_class.index, top_by_class.values, color=[colors.get(c, "gray") for c in top_by_class.index])
    ax2.set_ylabel("Maximum Trade-off Score")
    ax2.set_title("Peak Trade-off by Drug Class")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tradeoff_landscape.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'tradeoff_landscape.png'}")


def plot_constraint_map(constraint_df: pd.DataFrame):
    """Plot constraint landscape across Pol region."""
    if constraint_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    regions = ["PR", "RT", "IN"]

    for ax, region in zip(axes, regions):
        region_data = constraint_df[constraint_df["region"] == region]

        # Drug resistance constraints
        dr_data = region_data[region_data["constraint_type"] == "drug_resistance"]
        if not dr_data.empty:
            ax.bar(
                dr_data["position"],
                dr_data["mutation_frequency"],
                alpha=0.7,
                label="Drug Resistance Pressure",
                color="red",
                width=1,
            )

        # CTL constraints
        ctl_data = region_data[region_data["constraint_type"] == "ctl_epitope"]
        if not ctl_data.empty:
            # Aggregate by position
            ctl_agg = ctl_data.groupby("position")["n_hla"].sum().reset_index()
            ax2 = ax.twinx()
            ax2.bar(
                ctl_agg["position"],
                ctl_agg["n_hla"],
                alpha=0.5,
                label="CTL Pressure",
                color="blue",
                width=1,
            )
            ax2.set_ylabel("CTL Pressure (sum HLA)", color="blue")

        ax.set_xlabel("Position")
        ax.set_ylabel("Mutation Frequency", color="red")
        ax.set_title(f"{region} Constraint Landscape")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "constraint_map.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'constraint_map.png'}")


def plot_vaccine_targets(target_df: pd.DataFrame):
    """Plot vaccine target rankings."""
    if target_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 20 targets
    ax1 = axes[0]
    top_targets = target_df.head(20)

    colors = ["red" if r else "green" for r in top_targets["has_resistance_overlap"]]
    ax1.barh(
        range(len(top_targets)),
        top_targets["target_score"],
        color=colors,
        alpha=0.7,
    )
    ax1.set_yticks(range(len(top_targets)))
    ax1.set_yticklabels([f"{e[:15]}..." if len(e) > 15 else e for e in top_targets["epitope"]])
    ax1.set_xlabel("Target Score")
    ax1.set_title("Top 20 Vaccine Targets\n(Green=No Resistance Overlap, Red=Has Overlap)")
    ax1.invert_yaxis()

    # Score distribution by protein
    ax2 = axes[1]
    protein_scores = target_df.groupby("protein")["target_score"].agg(["mean", "std", "count"])
    protein_scores = protein_scores[protein_scores["count"] >= 5].sort_values("mean", ascending=False)

    if len(protein_scores) > 0:
        ax2.barh(protein_scores.index, protein_scores["mean"], xerr=protein_scores["std"], capsize=5, alpha=0.7)
        ax2.set_xlabel("Mean Target Score")
        ax2.set_title("Average Target Score by Protein")
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vaccine_targets.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'vaccine_targets.png'}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_integration_report(
    overlap_df: pd.DataFrame,
    score_df: pd.DataFrame,
    constraint_df: pd.DataFrame,
    target_df: pd.DataFrame,
):
    """Generate comprehensive integration report."""
    report_path = OUTPUT_DIR / "INTEGRATION_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Cross-Dataset Integration Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("This report integrates multiple HIV datasets to identify:\n")
        f.write("1. Drug resistance vs immune escape trade-offs\n")
        f.write("2. Multi-pressure constraint landscape\n")
        f.write("3. Optimal vaccine targets\n\n")

        # Trade-off analysis
        f.write("## Resistance-Immunity Trade-offs\n\n")

        if not overlap_df.empty:
            f.write(f"- Resistance-epitope overlaps found: {len(overlap_df)}\n")
            f.write(f"- Unique mutations with overlaps: {overlap_df['mutation'].nunique()}\n")
            f.write(f"- Unique epitopes affected: {overlap_df['epitope'].nunique()}\n\n")

            f.write("### Top Trade-off Positions\n\n")
            f.write("| Mutation | Drug Class | Epitope | HLAs | Fold-Change | Score |\n")
            f.write("|----------|------------|---------|------|-------------|-------|\n")

            if not score_df.empty:
                for _, row in score_df.head(15).iterrows():
                    f.write(
                        f"| {row['mutation']} | {row['drug_class']} | {row['epitope'][:15]}... | "
                        f"{row['n_hla_restrictions']} | {row['mean_fold_change']:.1f} | "
                        f"{row['tradeoff_score']:.3f} |\n"
                    )
            f.write("\n")

        # Constraint landscape
        f.write("## Constraint Landscape\n\n")

        if not constraint_df.empty:
            for region in ["PR", "RT", "IN"]:
                region_data = constraint_df[constraint_df["region"] == region]
                dr = region_data[region_data["constraint_type"] == "drug_resistance"]
                ctl = region_data[region_data["constraint_type"] == "ctl_epitope"]

                f.write(f"### {region}\n")
                f.write(f"- Positions analyzed: {len(dr)}\n")
                f.write(f"- CTL epitopes overlapping: {len(ctl)}\n")
                if len(dr) > 0:
                    f.write(f"- Mean conservation: {dr['conservation'].mean():.3f}\n")
                f.write("\n")

        # Vaccine targets
        f.write("## Vaccine Target Identification\n\n")

        if not target_df.empty:
            no_overlap = target_df[~target_df["has_resistance_overlap"]]
            f.write(f"- Total candidates: {len(target_df)}\n")
            f.write(f"- Without resistance overlap: {len(no_overlap)}\n")
            f.write(f"- Minimum HLA restrictions: {CTL_EPITOPE_CONSTRAINT}\n\n")

            f.write("### Top 20 Vaccine Targets\n\n")
            f.write("| Rank | Epitope | Protein | HLAs | Resistance Overlap | Score |\n")
            f.write("|------|---------|---------|------|--------------------|-------|\n")

            for i, (_, row) in enumerate(target_df.head(20).iterrows(), 1):
                overlap_mark = "Yes" if row["has_resistance_overlap"] else "No"
                f.write(
                    f"| {i} | {row['epitope']} | {row['protein']} | "
                    f"{row['n_hla_restrictions']} | {overlap_mark} | {row['target_score']:.3f} |\n"
                )
            f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")

        if not overlap_df.empty:
            n_overlaps = len(overlap_df)
            f.write(f"1. **{n_overlaps} resistance-epitope overlaps** identified, "
                    f"representing potential evolutionary trade-offs\n")

        if not target_df.empty:
            top_target = target_df.iloc[0]
            f.write(
                f"2. **Top vaccine target**: {top_target['epitope']} in {top_target['protein']} "
                f"(Score: {top_target['target_score']:.3f})\n"
            )

            clean_targets = target_df[~target_df["has_resistance_overlap"]]
            f.write(f"3. **{len(clean_targets)} targets** without resistance mutation overlap\n")

        f.write("\n## Generated Files\n\n")
        f.write("- `tradeoff_landscape.png` - Resistance vs immunity visualization\n")
        f.write("- `constraint_map.png` - Multi-pressure constraint landscape\n")
        f.write("- `vaccine_targets.png` - Vaccine target rankings\n")
        f.write("- `resistance_epitope_overlaps.csv` - Detailed overlap data\n")
        f.write("- `tradeoff_scores.csv` - Trade-off scoring results\n")
        f.write("- `vaccine_targets.csv` - Ranked vaccine target list\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete cross-dataset integration analysis."""
    print("=" * 70)
    print("Cross-Dataset Integration Analysis")
    print("=" * 70)

    # Load all datasets
    stanford_df, ctl_df, catnap_df = load_all_datasets()

    # Resistance-immunity overlap analysis
    overlap_df = analyze_resistance_immunity_overlap(stanford_df, ctl_df)

    # Trade-off scoring
    score_df = pd.DataFrame()
    if not overlap_df.empty:
        score_df = calculate_tradeoff_scores(overlap_df, stanford_df)

    # Constraint mapping
    constraint_df = map_constraint_landscape(stanford_df, ctl_df)

    # Vaccine target identification
    target_df = identify_vaccine_targets(constraint_df, overlap_df, ctl_df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_tradeoff_landscape(score_df)
    plot_constraint_map(constraint_df)
    plot_vaccine_targets(target_df)

    # Save data
    print("\nSaving results...")

    if not overlap_df.empty:
        overlap_df.to_csv(OUTPUT_DIR / "resistance_epitope_overlaps.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'resistance_epitope_overlaps.csv'}")

    if not score_df.empty:
        score_df.to_csv(OUTPUT_DIR / "tradeoff_scores.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'tradeoff_scores.csv'}")

    if not constraint_df.empty:
        constraint_df.to_csv(OUTPUT_DIR / "constraint_landscape.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'constraint_landscape.csv'}")

    if not target_df.empty:
        target_df.to_csv(OUTPUT_DIR / "vaccine_targets.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'vaccine_targets.csv'}")

    # Generate report
    print("\nGenerating report...")
    generate_integration_report(overlap_df, score_df, constraint_df, target_df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
