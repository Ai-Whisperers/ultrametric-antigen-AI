#!/usr/bin/env python3
"""
Deep Structural Analysis of Citrullination Sites

Analyze disorder regions, domain boundaries, and structural correlates
of immunogenicity using AlphaFold2 predictions.

Key analyses:
1. Disorder prediction from pLDDT scores
2. Domain boundary detection
3. Spatial clustering of immunodominant sites
4. Surface exposure and burial patterns
5. Structural motif analysis

Output directory: results/structural/20_deep_analysis/

Version: 1.0
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

matplotlib.use("Agg")

# ============================================================================
# CONFIGURATION
# ============================================================================

# pLDDT thresholds (AlphaFold confidence)
PLDDT_VERY_HIGH = 90  # Very high confidence
PLDDT_HIGH = 70  # Confident
PLDDT_LOW = 50  # Low confidence (possibly disordered)
# Below 50 = very low confidence (likely disordered)

# Output configuration
SCRIPT_NUM = "20"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_deep_analysis"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "structural" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_input_dir() -> Path:
    """Get input directory from previous script."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "structural" / "19_alphafold_mapping"


# ============================================================================
# PDB PARSING (Enhanced)
# ============================================================================


def parse_pdb_full(pdb_path: Path) -> Dict:
    """
    Parse PDB file with full residue information.

    Returns comprehensive structural data including:
    - All atom coordinates
    - B-factors (pLDDT in AlphaFold)
    - Sequence
    - CA trace
    """
    residues = {}
    sequence = []

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                bfactor = float(line[60:66].strip()) if len(line) > 66 else 0.0

                if resnum not in residues:
                    residues[resnum] = {
                        "resname": resname,
                        "resnum": resnum,
                        "atoms": {},
                        "plddt": bfactor,  # pLDDT stored in B-factor
                    }
                    sequence.append((resnum, resname))

                residues[resnum]["atoms"][atom_name] = (x, y, z)

    # Sort sequence by residue number
    sequence.sort(key=lambda x: x[0])

    return {
        "residues": residues,
        "sequence": sequence,
        "n_residues": len(sequence),
    }


# ============================================================================
# DISORDER ANALYSIS
# ============================================================================


def classify_disorder(plddt: float) -> str:
    """Classify disorder based on pLDDT score."""
    if plddt >= PLDDT_VERY_HIGH:
        return "ordered_high"
    elif plddt >= PLDDT_HIGH:
        return "ordered"
    elif plddt >= PLDDT_LOW:
        return "flexible"
    else:
        return "disordered"


def analyze_disorder_regions(structure: Dict) -> Dict:
    """
    Identify and characterize disorder regions.

    Returns:
    - Disorder profile (per-residue)
    - Disordered regions (contiguous stretches)
    - Statistics
    """
    residues = structure["residues"]

    # Per-residue disorder classification
    disorder_profile = []
    for resnum in sorted(residues.keys()):
        res = residues[resnum]
        plddt = res["plddt"]
        disorder_profile.append(
            {
                "resnum": resnum,
                "resname": res["resname"],
                "plddt": plddt,
                "disorder_class": classify_disorder(plddt),
                "is_disordered": plddt < PLDDT_LOW,
            }
        )

    # Find contiguous disordered regions
    disordered_regions = []
    current_region = None

    for i, res in enumerate(disorder_profile):
        if res["is_disordered"]:
            if current_region is None:
                current_region = {
                    "start": res["resnum"],
                    "end": res["resnum"],
                    "residues": [res],
                }
            else:
                current_region["end"] = res["resnum"]
                current_region["residues"].append(res)
        else:
            if current_region is not None:
                current_region["length"] = len(current_region["residues"])
                current_region["mean_plddt"] = np.mean([r["plddt"] for r in current_region["residues"]])
                disordered_regions.append(current_region)
                current_region = None

    # Don't forget last region
    if current_region is not None:
        current_region["length"] = len(current_region["residues"])
        current_region["mean_plddt"] = np.mean([r["plddt"] for r in current_region["residues"]])
        disordered_regions.append(current_region)

    # Statistics
    n_disordered = sum(1 for r in disorder_profile if r["is_disordered"])
    n_flexible = sum(1 for r in disorder_profile if r["disorder_class"] == "flexible")
    n_ordered = sum(1 for r in disorder_profile if r["disorder_class"] in ["ordered", "ordered_high"])

    return {
        "profile": disorder_profile,
        "regions": disordered_regions,
        "stats": {
            "n_residues": len(disorder_profile),
            "n_disordered": n_disordered,
            "n_flexible": n_flexible,
            "n_ordered": n_ordered,
            "pct_disordered": (100 * n_disordered / len(disorder_profile) if disorder_profile else 0),
            "n_disordered_regions": len(disordered_regions),
            "mean_plddt": (np.mean([r["plddt"] for r in disorder_profile]) if disorder_profile else 0),
        },
    }


# ============================================================================
# DOMAIN ANALYSIS
# ============================================================================


def detect_domain_boundaries(structure: Dict, window_size: int = 20) -> List[int]:
    """
    Detect potential domain boundaries based on pLDDT transitions
    and structural compactness.

    Domain boundaries often show:
    1. Transition from high to low pLDDT
    2. Increased local flexibility
    3. Extended linker regions
    """
    residues = structure["residues"]
    resnums = sorted(residues.keys())

    if len(resnums) < window_size * 2:
        return []

    # Compute pLDDT gradient
    plddt_values = [residues[r]["plddt"] for r in resnums]

    # Smooth pLDDT with running average
    smoothed = np.convolve(plddt_values, np.ones(window_size) / window_size, mode="valid")

    # Find significant drops in pLDDT (potential linkers)
    gradient = np.gradient(smoothed)

    # Find local minima in smoothed pLDDT (linker regions)
    boundaries = []
    threshold = np.std(gradient) * 1.5

    for i in range(1, len(smoothed) - 1):
        # Local minimum in pLDDT
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            if smoothed[i] < 70:  # Low confidence region
                boundaries.append(resnums[i + window_size // 2])

    return boundaries


def analyze_domains(structure: Dict) -> Dict:
    """
    Analyze domain structure of protein.
    """
    boundaries = detect_domain_boundaries(structure)
    residues = structure["residues"]
    resnums = sorted(residues.keys())

    if not boundaries:
        # Single domain
        domains = [
            {
                "start": resnums[0],
                "end": resnums[-1],
                "length": len(resnums),
                "mean_plddt": np.mean([residues[r]["plddt"] for r in resnums]),
            }
        ]
    else:
        # Multiple domains
        domains = []
        prev_boundary = resnums[0]

        for boundary in boundaries:
            domain_residues = [r for r in resnums if prev_boundary <= r < boundary]
            if domain_residues:
                domains.append(
                    {
                        "start": prev_boundary,
                        "end": boundary - 1,
                        "length": len(domain_residues),
                        "mean_plddt": np.mean([residues[r]["plddt"] for r in domain_residues]),
                    }
                )
            prev_boundary = boundary

        # Last domain
        domain_residues = [r for r in resnums if prev_boundary <= r]
        if domain_residues:
            domains.append(
                {
                    "start": prev_boundary,
                    "end": resnums[-1],
                    "length": len(domain_residues),
                    "mean_plddt": np.mean([residues[r]["plddt"] for r in domain_residues]),
                }
            )

    return {
        "boundaries": boundaries,
        "domains": domains,
        "n_domains": len(domains),
    }


# ============================================================================
# SPATIAL ANALYSIS
# ============================================================================


def compute_spatial_clustering(sites: List[Dict]) -> Dict:
    """
    Analyze spatial clustering of citrullination sites.

    Are immunodominant sites clustered together?
    """
    if len(sites) < 2:
        return {"insufficient_data": True}

    # Extract coordinates
    coords = np.array([[s["x"], s["y"], s["z"]] for s in sites])
    labels = [s["immunodominant"] for s in sites]

    # Compute pairwise distances
    distances = pdist(coords)

    # Distance matrix
    n = len(sites)
    dist_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = distances[idx]
            dist_matrix[j, i] = distances[idx]
            idx += 1

    # Compute mean distances within/between groups
    imm_indices = [i for i, l in enumerate(labels) if l]
    silent_indices = [i for i, l in enumerate(labels) if not l]

    results = {
        "n_sites": n,
        "n_immunodominant": len(imm_indices),
        "n_silent": len(silent_indices),
    }

    # Within-group distances
    if len(imm_indices) > 1:
        imm_dists = [dist_matrix[i, j] for i in imm_indices for j in imm_indices if i < j]
        results["mean_imm_imm_distance"] = np.mean(imm_dists) if imm_dists else 0

    if len(silent_indices) > 1:
        silent_dists = [dist_matrix[i, j] for i in silent_indices for j in silent_indices if i < j]
        results["mean_silent_silent_distance"] = np.mean(silent_dists) if silent_dists else 0

    # Between-group distances
    if imm_indices and silent_indices:
        between_dists = [dist_matrix[i, j] for i in imm_indices for j in silent_indices]
        results["mean_imm_silent_distance"] = np.mean(between_dists) if between_dists else 0

    # Overall clustering
    results["mean_all_distance"] = np.mean(distances)
    results["std_all_distance"] = np.std(distances)

    return results


def compute_burial_depth(structure: Dict, target_resnum: int, radius: float = 8.0) -> float:
    """
    Estimate burial depth based on number of neighbors.

    More neighbors within radius = more buried.
    """
    residues = structure["residues"]

    if target_resnum not in residues:
        return 0.0

    target = residues[target_resnum]
    if "CA" not in target["atoms"]:
        return 0.0

    tx, ty, tz = target["atoms"]["CA"]

    neighbor_count = 0
    for resnum, res in residues.items():
        if resnum == target_resnum:
            continue
        if "CA" not in res["atoms"]:
            continue

        x, y, z = res["atoms"]["CA"]
        dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)

        if dist < radius:
            neighbor_count += 1

    # Normalize: 0 = exposed, 1 = buried
    # Typical max neighbors within 8A is ~25
    burial = min(1.0, neighbor_count / 25.0)

    return burial


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_protein(gene_name: str, pdb_path: Path, sites: List[Dict]) -> Dict:
    """
    Complete structural analysis for one protein.
    """
    print(f"\n  Analyzing {gene_name}...")

    # Parse structure
    structure = parse_pdb_full(pdb_path)
    print(f"    Residues: {structure['n_residues']}")

    # Disorder analysis
    disorder = analyze_disorder_regions(structure)
    print(f"    Disordered regions: {disorder['stats']['n_disordered_regions']}")
    print(f"    % disordered: {disorder['stats']['pct_disordered']:.1f}%")

    # Domain analysis
    domains = analyze_domains(structure)
    print(f"    Domains detected: {domains['n_domains']}")

    # Enhance site data with structural features
    enhanced_sites = []
    for site in sites:
        resnum = site["r_position"]

        # Get disorder class for this position
        disorder_class = "unknown"
        for profile_res in disorder["profile"]:
            if profile_res["resnum"] == resnum:
                disorder_class = profile_res["disorder_class"]
                break

        # Compute burial depth
        burial = compute_burial_depth(structure, resnum)

        # Find which domain this residue belongs to
        domain_idx = -1
        for i, domain in enumerate(domains["domains"]):
            if domain["start"] <= resnum <= domain["end"]:
                domain_idx = i
                break

        # Check if near domain boundary
        near_boundary = any(abs(resnum - b) < 10 for b in domains["boundaries"])

        enhanced_sites.append(
            {
                **site,
                "disorder_class": disorder_class,
                "burial_depth": burial,
                "domain_index": domain_idx,
                "near_domain_boundary": near_boundary,
            }
        )

    # Spatial clustering analysis
    clustering = compute_spatial_clustering(enhanced_sites)

    return {
        "gene_name": gene_name,
        "structure": {
            "n_residues": structure["n_residues"],
            "mean_plddt": disorder["stats"]["mean_plddt"],
        },
        "disorder": disorder["stats"],
        "domains": domains,
        "sites": enhanced_sites,
        "spatial_clustering": clustering,
    }


def generate_visualizations(all_analyses: List[Dict], output_dir: Path):
    """Generate comprehensive structural visualizations."""

    # Collect all sites
    all_sites = []
    for analysis in all_analyses:
        for site in analysis["sites"]:
            site["gene"] = analysis["gene_name"]
            all_sites.append(site)

    if not all_sites:
        print("  No sites to visualize")
        return

    df = pd.DataFrame(all_sites)

    # Figure 1: Disorder vs Immunogenicity
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1a. pLDDT distribution by immunodominance
    ax = axes[0, 0]
    imm_plddt = df[df["immunodominant"]]["plddt"].values
    silent_plddt = df[~df["immunodominant"]]["plddt"].values

    ax.hist(
        imm_plddt,
        bins=10,
        alpha=0.6,
        color="#e53935",
        label=f"Immunodominant (n={len(imm_plddt)})",
        density=True,
    )
    ax.hist(
        silent_plddt,
        bins=10,
        alpha=0.6,
        color="#1e88e5",
        label=f"Silent (n={len(silent_plddt)})",
        density=True,
    )
    ax.axvline(50, color="orange", linestyle="--", lw=2, label="Disorder threshold")
    ax.axvline(70, color="green", linestyle="--", lw=2, label="High confidence")
    ax.set_xlabel("AlphaFold pLDDT Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("pLDDT Distribution by Immunodominance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 1b. Disorder class breakdown
    ax = axes[0, 1]
    disorder_counts = df.groupby(["immunodominant", "disorder_class"]).size().unstack(fill_value=0)
    disorder_counts.plot(kind="bar", ax=ax, color=["#ff9999", "#99ff99", "#9999ff", "#ffff99"])
    ax.set_xlabel("Immunodominant", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Disorder Class Distribution", fontsize=12, fontweight="bold")
    ax.set_xticklabels(["Silent", "Immunodominant"], rotation=0)
    ax.legend(title="Disorder Class", fontsize=9)

    # 1c. Entropy vs pLDDT with disorder regions highlighted
    ax = axes[1, 0]
    colors = ["#e53935" if imm else "#1e88e5" for imm in df["immunodominant"]]
    sizes = [150 if dc in ["disordered", "flexible"] else 80 for dc in df["disorder_class"]]
    markers = ["s" if dc in ["disordered", "flexible"] else "o" for dc in df["disorder_class"]]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row["plddt"],
            row["entropy_change"],
            c=colors[i],
            s=sizes[i],
            alpha=0.7,
            marker=("s" if row["disorder_class"] in ["disordered", "flexible"] else "o"),
            edgecolors="black" if row["immunodominant"] else "none",
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(50, color="orange", linestyle="--", alpha=0.5, lw=2)
    ax.axvline(70, color="green", linestyle="--", alpha=0.5, lw=2)
    ax.set_xlabel("AlphaFold pLDDT Score", fontsize=12)
    ax.set_ylabel("Entropy Change", fontsize=12)
    ax.set_title(
        "Entropy Change vs Structure Confidence\n(squares = disordered/flexible)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e53935",
            markersize=10,
            label="Immunodominant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1e88e5",
            markersize=10,
            label="Silent",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Disordered/Flexible",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # 1d. Burial depth vs entropy
    ax = axes[1, 1]
    ax.scatter(df["burial_depth"], df["entropy_change"], c=colors, alpha=0.7, s=80)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Burial Depth (0=exposed, 1=buried)", fontsize=12)
    ax.set_ylabel("Entropy Change", fontsize=12)
    ax.set_title("Entropy Change vs Burial Depth", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Compute correlation
    if len(df) > 2:
        r, p = stats.pearsonr(df["burial_depth"], df["entropy_change"])
        ax.text(
            0.05,
            0.95,
            f"r={r:.3f}, p={p:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "disorder_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: disorder_analysis.png")

    # Figure 2: Domain and spatial analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2a. Sites near domain boundaries
    ax = axes[0]
    near_boundary = df.groupby(["immunodominant", "near_domain_boundary"]).size().unstack(fill_value=0)
    near_boundary.plot(kind="bar", ax=ax, color=["#66b3ff", "#ff6666"])
    ax.set_xlabel("Immunodominant", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sites Near Domain Boundaries", fontsize=12, fontweight="bold")
    ax.set_xticklabels(["Silent", "Immunodominant"], rotation=0)
    ax.legend(["Not near boundary", "Near boundary"], fontsize=10)

    # 2b. Summary statistics by disorder class
    ax = axes[1]
    summary_data = (
        df.groupby("disorder_class")
        .agg(
            {
                "entropy_change": ["mean", "std", "count"],
                "immunodominant": "mean",
            }
        )
        .round(3)
    )

    # Create text summary
    text = "DISORDER CLASS SUMMARY\n" + "=" * 40 + "\n\n"
    for disorder_class in df["disorder_class"].unique():
        class_data = df[df["disorder_class"] == disorder_class]
        n = len(class_data)
        n_imm = class_data["immunodominant"].sum()
        mean_entropy = class_data["entropy_change"].mean()
        text += f"{disorder_class.upper()}\n"
        text += f"  N sites: {n}\n"
        text += f"  Immunodominant: {n_imm} ({100*n_imm/n:.0f}%)\n"
        text += f"  Mean entropy: {mean_entropy:.4f}\n\n"

    ax.text(
        0.1,
        0.9,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axis("off")
    ax.set_title("Summary by Disorder Class", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "domain_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: domain_analysis.png")

    # Save detailed data
    df.to_csv(output_dir / "deep_structural_analysis.csv", index=False)
    print("  Saved: deep_structural_analysis.csv")


def main():
    print("=" * 80)
    print("DEEP STRUCTURAL ANALYSIS")
    print("Disorder regions, domain boundaries, and spatial patterns")
    print("=" * 80)

    output_dir = get_output_dir()
    input_dir = get_input_dir()
    print(f"\nOutput directory: {output_dir}")
    print(f"Input directory: {input_dir}")

    # Load site data from previous analysis
    print("\n[1] Loading site structural data...")
    site_data_path = input_dir / "site_structural_analysis.csv"

    if not site_data_path.exists():
        print(f"  ERROR: Site data not found at {site_data_path}")
        print("  Please run script 19_alphafold_structure_mapping.py first")
        return

    sites_df = pd.read_csv(site_data_path)
    print(f"  Loaded {len(sites_df)} sites")

    # Group sites by protein
    sites_by_protein = sites_df.groupby("epitope_id").apply(lambda x: x.iloc[0]["epitope_id"].split("_")[0]).to_dict()

    # Get unique proteins
    proteins = set()
    for epitope_id in sites_df["epitope_id"]:
        proteins.add(epitope_id.split("_")[0])

    print(f"  Proteins: {proteins}")

    # Load PDB files and analyze
    print("\n[2] Analyzing protein structures...")
    pdb_dir = input_dir / "pdb_files"

    all_analyses = []

    for gene in proteins:
        # Find PDB file
        pdb_files = list(pdb_dir.glob("AF-*-F1-model_v*.pdb"))

        # Map gene to UniProt ID
        gene_to_uniprot = {
            "VIM": "P08670",
            "FGA": "P02671",
            "FGB": "P02675",
            "ENO1": "P06733",
            "TNC": "P24821",
            "FN1": "P02751",
            "CLU": "P10909",
        }

        if gene not in gene_to_uniprot:
            continue

        uniprot = gene_to_uniprot[gene]
        pdb_path = None

        for pdb_file in pdb_files:
            if uniprot in pdb_file.name:
                pdb_path = pdb_file
                break

        if pdb_path is None or not pdb_path.exists():
            print(f"  Skipping {gene}: PDB not found")
            continue

        # Get sites for this protein
        protein_sites = sites_df[sites_df["epitope_id"].str.startswith(gene)].to_dict("records")

        if not protein_sites:
            continue

        analysis = analyze_protein(gene, pdb_path, protein_sites)
        all_analyses.append(analysis)

    # Generate visualizations
    print("\n[3] Generating visualizations...")
    generate_visualizations(all_analyses, output_dir)

    # Statistical summary
    print("\n[4] Computing statistical summary...")

    all_sites = []
    for analysis in all_analyses:
        all_sites.extend(analysis["sites"])

    if all_sites:
        df = pd.DataFrame(all_sites)

        # Test: Are immunodominant sites more likely to be in disordered regions?
        imm_disordered = df[df["immunodominant"]]["disorder_class"].isin(["disordered", "flexible"]).sum()
        imm_total = df["immunodominant"].sum()
        silent_disordered = df[~df["immunodominant"]]["disorder_class"].isin(["disordered", "flexible"]).sum()
        silent_total = (~df["immunodominant"]).sum()

        # Fisher's exact test
        contingency = [
            [imm_disordered, imm_total - imm_disordered],
            [silent_disordered, silent_total - silent_disordered],
        ]
        odds_ratio, fisher_p = stats.fisher_exact(contingency)

        print("\n  DISORDER VS IMMUNODOMINANCE:")
        print(f"    Immunodominant in disordered regions: {imm_disordered}/{imm_total} ({100*imm_disordered/imm_total:.1f}%)")
        print(f"    Silent in disordered regions: {silent_disordered}/{silent_total} ({100*silent_disordered/silent_total:.1f}%)")
        print(f"    Fisher's exact test: OR={odds_ratio:.2f}, p={fisher_p:.4f}")

        # pLDDT comparison
        imm_plddt = df[df["immunodominant"]]["plddt"].values
        silent_plddt = df[~df["immunodominant"]]["plddt"].values
        t_stat, t_p = stats.ttest_ind(imm_plddt, silent_plddt)

        print("\n  pLDDT COMPARISON:")
        print(f"    Immunodominant mean pLDDT: {np.mean(imm_plddt):.1f} +/- {np.std(imm_plddt):.1f}")
        print(f"    Silent mean pLDDT: {np.mean(silent_plddt):.1f} +/- {np.std(silent_plddt):.1f}")
        print(f"    t-test: t={t_stat:.2f}, p={t_p:.4f}")

        # Burial depth comparison
        imm_burial = df[df["immunodominant"]]["burial_depth"].values
        silent_burial = df[~df["immunodominant"]]["burial_depth"].values
        t_stat_b, t_p_b = stats.ttest_ind(imm_burial, silent_burial)

        print("\n  BURIAL DEPTH COMPARISON:")
        print(f"    Immunodominant mean burial: {np.mean(imm_burial):.3f}")
        print(f"    Silent mean burial: {np.mean(silent_burial):.3f}")
        print(f"    t-test: t={t_stat_b:.2f}, p={t_p_b:.4f}")

    # Save summary
    summary = {
        "n_proteins": len(all_analyses),
        "n_sites": len(all_sites),
        "proteins": [
            {
                "gene": a["gene_name"],
                "n_residues": a["structure"]["n_residues"],
                "mean_plddt": a["structure"]["mean_plddt"],
                "pct_disordered": a["disorder"]["pct_disordered"],
                "n_domains": a["domains"]["n_domains"],
                "n_sites": len(a["sites"]),
            }
            for a in all_analyses
        ],
        "disorder_immunodominance_test": (
            {
                "imm_in_disordered": int(imm_disordered) if all_sites else 0,
                "imm_total": int(imm_total) if all_sites else 0,
                "silent_in_disordered": (int(silent_disordered) if all_sites else 0),
                "silent_total": int(silent_total) if all_sites else 0,
                "odds_ratio": float(odds_ratio) if all_sites else 0,
                "p_value": float(fisher_p) if all_sites else 1,
            }
            if all_sites
            else {}
        ),
    }

    summary_path = output_dir / "deep_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("DEEP STRUCTURAL ANALYSIS COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    return all_analyses


if __name__ == "__main__":
    main()
