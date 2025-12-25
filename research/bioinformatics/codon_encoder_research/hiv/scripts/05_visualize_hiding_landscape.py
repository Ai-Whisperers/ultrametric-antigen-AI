#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Visualization of HIV Hiding Landscape

Creates visualizations of:
1. Protein distance matrix in hiding space
2. Vulnerability zone network
3. Hierarchy level distribution
4. Evolutionary possibility space

Author: AI Whisperers
Date: 2025-12-24
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Configuration
FIGURE_DPI = 150
COLORS = {
    "gag": "#E74C3C",  # Red
    "pol": "#3498DB",  # Blue
    "env": "#2ECC71",  # Green
    "regulatory": "#9B59B6",  # Purple
    "accessory": "#F39C12",  # Orange
}

PROTEIN_GENES = {
    "Gag_MA_p17": "gag",
    "Gag_CA_p24": "gag",
    "Gag_NC_p7": "gag",
    "Pol_PR": "pol",
    "Pol_RT": "pol",
    "Pol_IN": "pol",
    "Env_gp120": "env",
    "Env_gp41": "env",
    "Tat": "regulatory",
    "Rev": "regulatory",
    "Nef": "accessory",
    "Vif": "accessory",
    "Vpr": "accessory",
    "Vpu": "accessory",
}


def load_results() -> Dict:
    """Load hiding landscape results."""
    results_dir = Path(__file__).parent.parent / "results"
    results_file = results_dir / "hiv_hiding_landscape.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Run 04_hiv_hiding_landscape.py first: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def get_protein_color(protein: str) -> str:
    """Get color for protein based on gene."""
    gene = PROTEIN_GENES.get(protein, "accessory")
    return COLORS.get(gene, "#95A5A6")


def create_distance_matrix(results: Dict) -> Tuple[np.ndarray, List[str]]:
    """Create distance matrix from pairwise distances."""
    distances = results["hiding_geometry"]["protein_distances"]

    # Extract protein names
    proteins = set()
    for pair in distances.keys():
        p1, p2 = pair.split("-")
        proteins.add(p1)
        proteins.add(p2)
    proteins = sorted(proteins)

    # Create matrix
    n = len(proteins)
    matrix = np.zeros((n, n))

    for pair, dist in distances.items():
        p1, p2 = pair.split("-")
        i, j = proteins.index(p1), proteins.index(p2)
        matrix[i, j] = dist
        matrix[j, i] = dist

    return matrix, proteins


def plot_distance_heatmap(results: Dict, output_dir: Path):
    """Plot protein distance heatmap."""
    matrix, proteins = create_distance_matrix(results)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="equal")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Poincaré Distance (hiding space)", fontsize=10)

    # Labels
    ax.set_xticks(range(len(proteins)))
    ax.set_yticks(range(len(proteins)))
    ax.set_xticklabels(proteins, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(proteins, fontsize=9)

    # Add distance values
    for i in range(len(proteins)):
        for j in range(len(proteins)):
            if i != j:
                color = "white" if matrix[i, j] > 2.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    ax.set_title(
        "HIV Protein Distance Matrix in Hiding Space\n(3-adic Poincaré Geometry)",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "hiv_hiding_distance_matrix.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hiv_hiding_distance_matrix.png")


def plot_vulnerability_network(results: Dict, output_dir: Path):
    """Plot vulnerability zone network."""
    distances = results["hiding_geometry"]["protein_distances"]
    vulnerabilities = results["evolutionary_predictions"]["vulnerability_zones"]

    # Create graph
    G = nx.Graph()

    # Add nodes
    proteins = set()
    for pair in distances.keys():
        p1, p2 = pair.split("-")
        proteins.add(p1)
        proteins.add(p2)

    for p in proteins:
        G.add_node(p, color=get_protein_color(p))

    # Add edges (vulnerability zones only - distance > 2.0)
    vulnerability_pairs = set()
    for v in vulnerabilities:
        pair = v["proteins"]
        vulnerability_pairs.add(pair)
        p1, p2 = pair.split("-")
        G.add_edge(p1, p2, weight=v["distance"], vulnerability=True)

    # Add close connections too (distance < 1.0)
    for pair, dist in distances.items():
        if dist < 1.0:
            p1, p2 = pair.split("-")
            if not G.has_edge(p1, p2):
                G.add_edge(p1, p2, weight=dist, vulnerability=False)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)

    # Draw edges
    vuln_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("vulnerability", False)]
    close_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("vulnerability", False)]

    # Vulnerability zones (red, dashed)
    if vuln_edges:
        weights = [G[u][v]["weight"] for u, v in vuln_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=vuln_edges,
            edge_color="red",
            style="dashed",
            width=1.5,
            alpha=0.6,
            ax=ax,
        )

    # Close connections (green, solid)
    if close_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=close_edges,
            edge_color="green",
            style="solid",
            width=2,
            alpha=0.8,
            ax=ax,
        )

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["gag"], label="Gag (structural)"),
        mpatches.Patch(color=COLORS["pol"], label="Pol (enzymes)"),
        mpatches.Patch(color=COLORS["env"], label="Env (surface)"),
        mpatches.Patch(color=COLORS["regulatory"], label="Regulatory (Tat/Rev)"),
        mpatches.Patch(color=COLORS["accessory"], label="Accessory (Nef/Vif/Vpr/Vpu)"),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            label="Vulnerability zones (d > 2.0)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="green",
            linestyle="-",
            linewidth=2,
            label="Clustered hiding (d < 1.0)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(
        "HIV Protein Hiding Network\nRed dashed = Vulnerability zones | Green solid = Shared hiding",
        fontsize=12,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "hiv_vulnerability_network.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hiv_vulnerability_network.png")


def plot_hierarchy_distribution(results: Dict, output_dir: Path):
    """Plot mechanism distribution by hierarchy level."""
    by_level = results["summary"]["mechanisms_by_level"]
    by_protein = results["summary"]["mechanisms_by_protein"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By level
    ax1 = axes[0]
    levels = list(by_level.keys())
    counts = list(by_level.values())
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6"]

    bars = ax1.bar(levels, counts, color=colors)
    ax1.set_ylabel("Number of Mechanisms", fontsize=11)
    ax1.set_xlabel("Hierarchy Level", fontsize=11)
    ax1.set_title("Hiding Mechanisms by Hierarchy Level", fontsize=12)

    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # By protein
    ax2 = axes[1]
    proteins = list(by_protein.keys())
    counts = list(by_protein.values())
    colors = [get_protein_color(p) for p in proteins]

    bars = ax2.barh(proteins, counts, color=colors)
    ax2.set_xlabel("Number of Mechanisms", fontsize=11)
    ax2.set_title("Hiding Mechanisms by Protein", fontsize=12)

    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "hiv_hiding_distribution.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hiv_hiding_distribution.png")


def plot_evolutionary_space(results: Dict, output_dir: Path):
    """Plot evolutionary possibility space."""
    geometry = results["hiding_geometry"]["geometry"]
    by_level = results["hiding_geometry"]["by_level"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Poincare disk representation
    ax1 = axes[0]

    # Draw Poincare disk boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax1.fill(np.cos(theta), np.sin(theta), alpha=0.1, color="gray")

    # Plot centroids by level
    level_colors = {
        "protein": "#3498DB",
        "signaling": "#E74C3C",
        "glycan": "#2ECC71",
        "peptide": "#9B59B6",
    }

    for level, data in by_level.items():
        norm = data["centroid_norm"]
        # Place at angle based on level
        angles = {
            "protein": 0,
            "signaling": np.pi / 2,
            "glycan": np.pi,
            "peptide": 3 * np.pi / 2,
        }
        angle = angles.get(level, 0)

        x = norm * np.cos(angle)
        y = norm * np.sin(angle)

        ax1.scatter(
            [x],
            [y],
            s=200,
            c=level_colors[level],
            label=f'{level} (n={data["n_proteins"]})',
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        ax1.annotate(
            level,
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    # Mark overall centroid
    overall_norm = geometry["overall_centroid_norm"]
    ax1.scatter([0], [0], s=100, c="black", marker="x", linewidths=2, zorder=6)
    ax1.annotate(
        "Center\n(flexibility)",
        (0, 0),
        xytext=(-50, -30),
        textcoords="offset points",
        fontsize=8,
        ha="center",
    )

    # Add concentric circles for norm reference
    for r in [0.3, 0.6, 0.9]:
        ax1.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            "k--",
            alpha=0.3,
            linewidth=0.5,
        )
        ax1.text(r, 0.05, f"{r}", fontsize=7, alpha=0.5)

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect("equal")
    ax1.set_title(
        "HIV Hiding in Poincaré Space\n(center = flexibility, boundary = constraint)",
        fontsize=11,
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axis("off")

    # Right: Bar chart of centroid norms
    ax2 = axes[1]

    levels = list(by_level.keys())
    norms = [by_level[l]["centroid_norm"] for l in levels]
    colors = [level_colors[l] for l in levels]

    bars = ax2.bar(levels, norms, color=colors, edgecolor="black", linewidth=1.5)

    # Add reference lines
    ax2.axhline(
        y=0.3,
        color="green",
        linestyle="--",
        alpha=0.7,
        label="Flexible zone (< 0.3)",
    )
    ax2.axhline(
        y=0.7,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Constrained zone (> 0.7)",
    )

    # Add overall centroid line
    ax2.axhline(
        y=geometry["overall_centroid_norm"],
        color="black",
        linestyle="-",
        linewidth=2,
        label=f'Overall centroid ({geometry["overall_centroid_norm"]:.3f})',
    )

    ax2.set_ylabel("Centroid Norm (distance from center)", fontsize=11)
    ax2.set_xlabel("Hierarchy Level", fontsize=11)
    ax2.set_title(
        "Evolutionary Flexibility by Level\n(lower = more flexible)",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.0)

    for bar, norm in zip(bars, norms):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            norm + 0.02,
            f"{norm:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "hiv_evolutionary_space.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hiv_evolutionary_space.png")


def plot_integrase_isolation(results: Dict, output_dir: Path):
    """Highlight integrase isolation - the key finding."""
    distances = results["hiding_geometry"]["protein_distances"]

    # Get all distances involving Pol_IN
    in_distances = {}
    for pair, dist in distances.items():
        if "Pol_IN" in pair:
            other = pair.replace("Pol_IN-", "").replace("-Pol_IN", "")
            in_distances[other] = dist

    fig, ax = plt.subplots(figsize=(12, 6))

    proteins = list(in_distances.keys())
    dists = list(in_distances.values())
    colors = [get_protein_color(p) for p in proteins]

    bars = ax.barh(proteins, dists, color=colors, edgecolor="black", linewidth=1)

    # Add threshold line
    ax.axvline(
        x=2.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Vulnerability threshold (d=2.0)",
    )

    ax.set_xlabel("Poincaré Distance from Pol_IN (Integrase)", fontsize=11)
    ax.set_title(
        "INTEGRASE ISOLATION: Distance from All Other HIV Proteins\n" "(Largest distances = weakest hiding connections)",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    # Mark all as vulnerability zones
    for bar, dist in zip(bars, dists):
        color = "red" if dist > 2.0 else "black"
        ax.text(
            dist + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{dist:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold" if dist > 3.0 else "normal",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "hiv_integrase_isolation.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hiv_integrase_isolation.png")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("HIV HIDING LANDSCAPE VISUALIZATION")
    print("=" * 60)

    # Load results
    print("\n[1] Loading analysis results...")
    results = load_results()
    print(f"  Loaded: {results['metadata']['total_proteins']} proteins, " f"{results['metadata']['total_mechanisms']} mechanisms")

    # Output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n[2] Generating visualizations...")

    print("\n  Distance heatmap...")
    plot_distance_heatmap(results, output_dir)

    print("\n  Vulnerability network...")
    plot_vulnerability_network(results, output_dir)

    print("\n  Hierarchy distribution...")
    plot_hierarchy_distribution(results, output_dir)

    print("\n  Evolutionary space...")
    plot_evolutionary_space(results, output_dir)

    print("\n  Integrase isolation...")
    plot_integrase_isolation(results, output_dir)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print("\n  Generated files:")
    print("    - hiv_hiding_distance_matrix.png")
    print("    - hiv_vulnerability_network.png")
    print("    - hiv_hiding_distribution.png")
    print("    - hiv_evolutionary_space.png")
    print("    - hiv_integrase_isolation.png")


if __name__ == "__main__":
    main()
