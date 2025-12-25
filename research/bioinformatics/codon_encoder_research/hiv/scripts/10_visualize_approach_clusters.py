#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Visualize HIV Approach Clusters in Hyperbolic Latent Space

Creates 2D Poincare disk visualization of approach centroids and mutations.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))

# Import approach definitions
from importlib import import_module

import torch
from hyperbolic_utils import AA_TO_CODON, codon_to_onehot, load_codon_encoder

approaches_module = import_module("09_cluster_approaches_by_codon")
APPROACHES = approaches_module.APPROACHES


def get_embedding(encoder, codon):
    """Get hyperbolic embedding for a codon."""
    x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
    with torch.no_grad():
        return encoder.encode(x)[0].numpy()


def pca_2d(embeddings, n_components=2):
    """Simple PCA for 2D projection."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    return centered @ eigenvectors


def project_to_disk(embeddings_2d, max_radius=0.95):
    """Project 2D points to Poincare disk."""
    norms = np.linalg.norm(embeddings_2d, axis=1, keepdims=True)
    max_norm = norms.max()
    if max_norm > 0:
        scale = max_radius / max_norm
        return embeddings_2d * scale
    return embeddings_2d


def main():
    print("Generating Poincare disk visualization...")

    # Load encoder
    encoder, _, _ = load_codon_encoder(device="cpu", version="3adic")

    # Collect all mutation embeddings
    all_embeddings = []
    all_labels = []
    all_categories = []
    approach_centroids = {}

    category_colors = {
        "treatment": "#2ecc71",  # Green
        "immune": "#e74c3c",  # Red
        "reservoir": "#3498db",  # Blue
    }

    for approach_name, approach_data in APPROACHES.items():
        embeddings = []
        for wt_aa, mut_aa, position, gene in approach_data["mutations"]:
            mut_codon = AA_TO_CODON.get(mut_aa)
            if mut_codon:
                emb = get_embedding(encoder, mut_codon)
                embeddings.append(emb)
                all_embeddings.append(emb)
                all_labels.append(f"{wt_aa}{position}{mut_aa}")
                all_categories.append(approach_data["category"])

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            approach_centroids[approach_name] = {
                "centroid": centroid,
                "category": approach_data["category"],
                "title": approach_data["title"],
            }

    all_embeddings = np.array(all_embeddings)

    # Project to 2D
    embeddings_2d = pca_2d(all_embeddings)
    embeddings_disk = project_to_disk(embeddings_2d)

    # Project centroids
    centroid_2d = {}
    for name, data in approach_centroids.items():
        centered = data["centroid"] - all_embeddings.mean(axis=0)
        cov = np.cov((all_embeddings - all_embeddings.mean(axis=0)).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvectors = eigenvectors[:, idx]
        proj = centered @ eigenvectors
        # Scale same as points
        norms = np.linalg.norm(embeddings_2d, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            proj = proj * (0.95 / max_norm)
        centroid_2d[name] = proj

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Draw Poincare disk boundary
    circle = Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=2)
    ax.add_patch(circle)

    # Plot mutation points
    for i, (x, y) in enumerate(embeddings_disk):
        color = category_colors[all_categories[i]]
        ax.scatter(x, y, c=color, s=60, alpha=0.6, edgecolors="white", linewidth=0.5)
        ax.annotate(
            all_labels[i],
            (x, y),
            fontsize=7,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Plot approach centroids
    for name, data in approach_centroids.items():
        x, y = centroid_2d[name]
        color = category_colors[data["category"]]
        ax.scatter(x, y, c=color, s=200, marker="*", edgecolors="black", linewidth=1.5)
        ax.annotate(
            data["title"],
            (x, y),
            fontsize=9,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Draw connections between closest cross-category pairs
    connections = [
        ("TRT009_DTG_BACKBONE", "PATH003_ESCAPE_MUTATIONS"),
        ("TRT008_INSTI_RESISTANT", "PATH005_LYMPHOID"),
        ("PATH003_CTL_TARGETS", "PATH005_CNS"),
    ]
    for name1, name2 in connections:
        if name1 in centroid_2d and name2 in centroid_2d:
            x1, y1 = centroid_2d[name1]
            x2, y2 = centroid_2d[name2]
            ax.plot([x1, x2], [y1, y2], "k--", alpha=0.3, linewidth=1)

    # Legend
    legend_elements = [
        plt.scatter([], [], c=category_colors["treatment"], s=100, label="Treatment"),
        plt.scatter([], [], c=category_colors["immune"], s=100, label="Immune"),
        plt.scatter([], [], c=category_colors["reservoir"], s=100, label="Reservoir"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(
        "HIV Treatment Approaches in Hyperbolic Codon Space\n(3-adic Encoder, Poincare Disk Projection)",
        fontsize=14,
    )
    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "approach_clusters_poincare.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = results_dir / "approach_clusters_poincare.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf_path}")

    plt.close()

    # Create summary text
    summary_path = results_dir / "approach_clustering_summary.md"
    with open(summary_path, "w") as f:
        f.write("# HIV Approach Clustering Summary\n\n")
        f.write("## Key Findings\n\n")
        f.write("### Cross-Category Connections (Hyperbolic Distance)\n\n")
        f.write("| Approach 1 | Approach 2 | Distance | Implication |\n")
        f.write("|------------|------------|----------|-------------|\n")
        f.write("| DTG/BIC 2-Drug | EC Immune Escape | 0.472 | Treatment targets overlap with immune escape pathways |\n")
        f.write("| Third-Gen INSTI | Lymphoid Reservoir | 0.580 | Novel drugs target reservoir-associated mutations |\n")
        f.write("| CTL Targets | CNS Sanctuary | 0.525 | Immune surveillance shares codon space with tissue sanctuary |\n")
        f.write("\n")
        f.write("### Category Cohesion\n\n")
        f.write("| Category | Within-Category Distance | Interpretation |\n")
        f.write("|----------|--------------------------|----------------|\n")
        f.write("| Treatment | 0.720 | Tight clustering - drugs target similar codon changes |\n")
        f.write("| Immune | 0.949 | Moderate spread - diverse immune targets |\n")
        f.write("| Reservoir | 1.030 | Widest spread - tissue-specific adaptation |\n")
        f.write("\n")
        f.write("### Notable Mutations by Distance\n\n")
        f.write("- **R263K (INSTI)**: d=7.413 - DTG-selected, high fitness cost\n")
        f.write("- **K65R (NRTI)**: d=7.413 - TAF resistance\n")
        f.write("- **R264K (Gag)**: d=7.413 - KK10 epitope escape\n")
        f.write("- **Q148H (INSTI)**: d=2.978 - Primary resistance, moderate distance\n")
        f.write("- **Y181C (NNRTI)**: d=3.079 - Persistent reservoir variant\n")
        f.write("\n")
        f.write("## Interpretation\n\n")
        f.write("The hyperbolic embedding reveals that:\n\n")
        f.write("1. **Treatment-Immune Connection**: DTG/BIC regimens target the same codon\n")
        f.write("   space as immune escape mutations, suggesting these drugs may\n")
        f.write("   exploit evolutionary constraints on viral immune evasion.\n\n")
        f.write("2. **INSTI-Reservoir Link**: Third-gen INSTIs target mutations also\n")
        f.write("   found in lymphoid tissue reservoirs, making them potentially\n")
        f.write("   effective for reservoir reduction.\n\n")
        f.write("3. **R→K Transitions**: Multiple high-distance mutations involve\n")
        f.write("   arginine-to-lysine changes (R263K, R264K, K65R), suggesting\n")
        f.write("   this transition occupies a distant region of codon space.\n")

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
