"""
High-Detail Visualizations for Immunogenicity Analysis

Generates publication-quality 2D and 3D plots for the codon-encoder-3-adic
immunogenicity analysis results.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Load augmented database
import importlib.util

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Import utilities
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, get_results_dir,
                              load_codon_encoder)
from matplotlib.patches import Circle
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

spec = importlib.util.spec_from_file_location("augmented_db", Path(__file__).parent / "08_augmented_epitope_database.py")
augmented_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmented_db)
RA_AUTOANTIGENS_AUGMENTED = augmented_db.RA_AUTOANTIGENS_AUGMENTED

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "immunodominant": "#E63946",  # Red
    "silent": "#457B9D",  # Blue
    "neutral": "#A8DADC",  # Light blue
    "highlight": "#F4A261",  # Orange
}
DPI = 300


def encode_epitope(sequence: str, encoder, device="cpu") -> np.ndarray:
    """Encode epitope sequence to embeddings."""
    embeddings = []
    for aa in sequence:
        codon = AA_TO_CODON.get(aa, "NNN")
        if codon == "NNN":
            continue
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode(onehot).cpu().numpy().squeeze()
        embeddings.append(emb)
    return np.array(embeddings)


def collect_all_embeddings(encoder, device="cpu"):
    """Collect embeddings for all epitopes."""
    data = {
        "embeddings": [],
        "centroids": [],
        "labels": [],
        "epitope_ids": [],
        "proteins": [],
        "sequences": [],
        "acpa": [],
    }

    for protein_id, protein in RA_AUTOANTIGENS_AUGMENTED.items():
        for epitope in protein["epitopes"]:
            emb = encode_epitope(epitope["sequence"], encoder, device)
            if len(emb) == 0:
                continue

            centroid = np.mean(emb, axis=0)
            label = "immunodominant" if epitope["immunodominant"] else "silent"

            data["embeddings"].append(emb)
            data["centroids"].append(centroid)
            data["labels"].append(label)
            data["epitope_ids"].append(epitope["id"])
            data["proteins"].append(protein_id)
            data["sequences"].append(epitope["sequence"])
            data["acpa"].append(epitope.get("acpa_reactivity", 0))

    data["centroids"] = np.array(data["centroids"])
    return data


def plot_poincare_disk_2d(data, encoder, output_path):
    """Plot epitope centroids on 2D Poincaré disk projection."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA to 2D
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(data["centroids"])

    # Normalize to disk
    max_norm = np.max(np.linalg.norm(centroids_2d, axis=1))
    centroids_2d = centroids_2d / (max_norm * 1.1) * 0.95

    # Left plot: By immunodominance
    ax1 = axes[0]
    ax1.set_aspect("equal")

    # Draw Poincaré disk boundary
    circle = Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    ax1.add_patch(circle)

    # Plot points
    for i, (x, y) in enumerate(centroids_2d):
        color = COLORS["immunodominant"] if data["labels"][i] == "immunodominant" else COLORS["silent"]
        ax1.scatter(x, y, c=color, s=100, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel("PC1", fontsize=12)
    ax1.set_ylabel("PC2", fontsize=12)
    ax1.set_title(
        "Epitope Centroids in Poincaré Disk\n(by Immunodominance)",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    imm_patch = mpatches.Patch(
        color=COLORS["immunodominant"],
        label=f'Immunodominant (n={sum(1 for l in data["labels"] if l=="immunodominant")})',
    )
    sil_patch = mpatches.Patch(
        color=COLORS["silent"],
        label=f'Silent (n={sum(1 for l in data["labels"] if l=="silent")})',
    )
    ax1.legend(handles=[imm_patch, sil_patch], loc="upper right", fontsize=10)

    # Right plot: By protein
    ax2 = axes[1]
    ax2.set_aspect("equal")
    circle2 = Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    ax2.add_patch(circle2)

    protein_colors = plt.cm.tab10(np.linspace(0, 1, len(set(data["proteins"]))))
    protein_color_map = {p: protein_colors[i] for i, p in enumerate(sorted(set(data["proteins"])))}

    for i, (x, y) in enumerate(centroids_2d):
        color = protein_color_map[data["proteins"][i]]
        marker = "o" if data["labels"][i] == "immunodominant" else "s"
        ax2.scatter(
            x,
            y,
            c=[color],
            s=100,
            alpha=0.7,
            marker=marker,
            edgecolors="white",
            linewidth=0.5,
        )

    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_xlabel("PC1", fontsize=12)
    ax2.set_ylabel("PC2", fontsize=12)
    ax2.set_title(
        "Epitope Centroids in Poincaré Disk\n(by Protein)",
        fontsize=14,
        fontweight="bold",
    )

    # Protein legend
    handles = [mpatches.Patch(color=protein_color_map[p], label=p) for p in sorted(set(data["proteins"]))]
    ax2.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_3d_embedding_space(data, encoder, output_path):
    """3D visualization of epitope embedding space."""
    fig = plt.figure(figsize=(18, 6))

    # PCA to 3D
    pca = PCA(n_components=3)
    centroids_3d = pca.fit_transform(data["centroids"])

    # Subplot 1: By immunodominance
    ax1 = fig.add_subplot(131, projection="3d")

    imm_mask = np.array([l == "immunodominant" for l in data["labels"]])
    sil_mask = ~imm_mask

    ax1.scatter(
        centroids_3d[imm_mask, 0],
        centroids_3d[imm_mask, 1],
        centroids_3d[imm_mask, 2],
        c=COLORS["immunodominant"],
        s=80,
        alpha=0.7,
        label="Immunodominant",
        edgecolors="white",
    )
    ax1.scatter(
        centroids_3d[sil_mask, 0],
        centroids_3d[sil_mask, 1],
        centroids_3d[sil_mask, 2],
        c=COLORS["silent"],
        s=80,
        alpha=0.7,
        label="Silent",
        edgecolors="white",
    )

    ax1.set_xlabel("PC1", fontsize=10)
    ax1.set_ylabel("PC2", fontsize=10)
    ax1.set_zlabel("PC3", fontsize=10)
    ax1.set_title("By Immunodominance", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: By ACPA reactivity
    ax2 = fig.add_subplot(132, projection="3d")

    acpa = np.array(data["acpa"])
    scatter = ax2.scatter(
        centroids_3d[:, 0],
        centroids_3d[:, 1],
        centroids_3d[:, 2],
        c=acpa,
        cmap="RdYlBu_r",
        s=80,
        alpha=0.7,
        edgecolors="white",
    )

    ax2.set_xlabel("PC1", fontsize=10)
    ax2.set_ylabel("PC2", fontsize=10)
    ax2.set_zlabel("PC3", fontsize=10)
    ax2.set_title("By ACPA Reactivity", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label("ACPA Reactivity", fontsize=10)

    # Subplot 3: By protein
    ax3 = fig.add_subplot(133, projection="3d")

    proteins = sorted(set(data["proteins"]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(proteins)))

    for i, protein in enumerate(proteins):
        mask = np.array([p == protein for p in data["proteins"]])
        ax3.scatter(
            centroids_3d[mask, 0],
            centroids_3d[mask, 1],
            centroids_3d[mask, 2],
            c=[colors[i]],
            s=80,
            alpha=0.7,
            label=protein,
            edgecolors="white",
        )

    ax3.set_xlabel("PC1", fontsize=10)
    ax3.set_ylabel("PC2", fontsize=10)
    ax3.set_zlabel("PC3", fontsize=10)
    ax3.set_title("By Protein", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=7, ncol=2)

    plt.suptitle(
        "3D Epitope Embedding Space (PCA Projection)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_3d_poincare_ball(data, encoder, output_path):
    """3D Poincaré ball visualization."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # PCA to 3D and normalize to ball
    pca = PCA(n_components=3)
    centroids_3d = pca.fit_transform(data["centroids"])
    norms = np.linalg.norm(centroids_3d, axis=1, keepdims=True)
    centroids_3d = centroids_3d / (np.max(norms) * 1.1) * 0.9

    # Draw wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1, linewidth=0.5)

    # Plot points
    imm_mask = np.array([l == "immunodominant" for l in data["labels"]])
    sil_mask = ~imm_mask

    ax.scatter(
        centroids_3d[imm_mask, 0],
        centroids_3d[imm_mask, 1],
        centroids_3d[imm_mask, 2],
        c=COLORS["immunodominant"],
        s=120,
        alpha=0.8,
        label="Immunodominant",
        edgecolors="white",
        linewidth=0.5,
    )
    ax.scatter(
        centroids_3d[sil_mask, 0],
        centroids_3d[sil_mask, 1],
        centroids_3d[sil_mask, 2],
        c=COLORS["silent"],
        s=120,
        alpha=0.8,
        label="Silent",
        edgecolors="white",
        linewidth=0.5,
    )

    # Draw origin
    ax.scatter([0], [0], [0], c="black", s=50, marker="+", linewidth=2)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.set_title(
        "Epitope Centroids in Poincaré Ball\n(3D PCA Projection)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11)

    # Set equal aspect
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_statistical_comparisons(results_path, output_path):
    """Plot statistical comparison results."""
    with open(results_path) as f:
        results = json.load(f)

    comparisons = results["statistical_comparisons"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics = [
        ("embedding_norm", "Embedding Norm"),
        ("cluster_homogeneity", "Cluster Homogeneity"),
        ("mean_neighbor_distance", "Mean Neighbor Distance"),
        ("boundary_potential", "Boundary Potential"),
        ("cit_mean_js_divergence", "JS Divergence (Cit)"),
        ("cit_mean_entropy_change", "Entropy Change (Cit)"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        if metric_key not in comparisons:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
            ax.set_title(metric_name, fontsize=12)
            continue

        comp = comparisons[metric_key]

        # Bar plot
        x = ["Immunodominant", "Silent"]
        means = [comp["immunodominant_mean"], comp["silent_mean"]]
        stds = [comp["immunodominant_std"], comp["silent_std"]]
        colors = [COLORS["immunodominant"], COLORS["silent"]]

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            alpha=0.7,
            capsize=5,
            edgecolor="white",
            linewidth=2,
        )

        # Significance annotation
        p_val = comp["p_value"]
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        max_height = max(means) + max(stds)
        ax.annotate(
            sig_text,
            xy=(0.5, max_height * 1.05),
            fontsize=14,
            ha="center",
            fontweight="bold",
        )

        ax.set_title(
            f'{metric_name}\n(p={p_val:.4f}, d={comp["cohens_d"]:.2f})',
            fontsize=11,
        )
        ax.set_ylabel("Value", fontsize=10)

        # Highlight significant results
        if p_val < 0.05:
            ax.patch.set_facecolor("#ffffcc")
            ax.patch.set_alpha(0.3)

    plt.suptitle(
        "Statistical Comparisons: Immunodominant vs Silent Epitopes",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_entropy_change_detail(results_path, output_path):
    """Detailed visualization of the key entropy change finding."""
    with open(results_path) as f:
        results = json.load(f)

    # Extract entropy data
    imm_entropy = []
    sil_entropy = []

    for epitope in results["epitope_analyses"]:
        if epitope["citrullination"] is None:
            continue
        entropy = epitope["citrullination"]["mean_entropy_change"]
        if epitope["immunodominant"]:
            imm_entropy.append(entropy)
        else:
            sil_entropy.append(entropy)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Box plot
    ax1 = axes[0]
    bp = ax1.boxplot(
        [imm_entropy, sil_entropy],
        labels=["Immunodominant", "Silent"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor(COLORS["immunodominant"])
    bp["boxes"][1].set_facecolor(COLORS["silent"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_ylabel("Entropy Change (ΔS)", fontsize=12)
    ax1.set_title("Entropy Change Distribution", fontsize=12, fontweight="bold")

    # Add statistics
    t_stat, p_val = stats.ttest_ind(imm_entropy, sil_entropy)
    ax1.text(
        0.5,
        0.95,
        f"p = {p_val:.4f}",
        transform=ax1.transAxes,
        fontsize=11,
        ha="center",
        fontweight="bold",
    )

    # Plot 2: Histogram
    ax2 = axes[1]
    bins = np.linspace(-0.3, 0.3, 25)
    ax2.hist(
        imm_entropy,
        bins=bins,
        alpha=0.6,
        color=COLORS["immunodominant"],
        label=f"Immunodominant (n={len(imm_entropy)})",
        edgecolor="white",
    )
    ax2.hist(
        sil_entropy,
        bins=bins,
        alpha=0.6,
        color=COLORS["silent"],
        label=f"Silent (n={len(sil_entropy)})",
        edgecolor="white",
    )
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(
        x=np.mean(imm_entropy),
        color=COLORS["immunodominant"],
        linestyle="-",
        linewidth=2,
    )
    ax2.axvline(
        x=np.mean(sil_entropy),
        color=COLORS["silent"],
        linestyle="-",
        linewidth=2,
    )
    ax2.set_xlabel("Entropy Change (ΔS)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Entropy Change Histograms", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    # Plot 3: Individual points with means
    ax3 = axes[2]
    x_imm = np.random.normal(0, 0.04, len(imm_entropy))
    x_sil = np.random.normal(1, 0.04, len(sil_entropy))

    ax3.scatter(
        x_imm,
        imm_entropy,
        c=COLORS["immunodominant"],
        alpha=0.6,
        s=60,
        edgecolors="white",
    )
    ax3.scatter(
        x_sil,
        sil_entropy,
        c=COLORS["silent"],
        alpha=0.6,
        s=60,
        edgecolors="white",
    )

    # Mean bars
    ax3.hlines(
        np.mean(imm_entropy),
        -0.2,
        0.2,
        colors=COLORS["immunodominant"],
        linewidth=3,
    )
    ax3.hlines(np.mean(sil_entropy), 0.8, 1.2, colors=COLORS["silent"], linewidth=3)

    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Immunodominant", "Silent"])
    ax3.set_ylabel("Entropy Change (ΔS)", fontsize=12)
    ax3.set_title("Individual Epitope Values", fontsize=12, fontweight="bold")
    ax3.set_xlim(-0.5, 1.5)

    # Add annotation
    ax3.annotate(
        f"Mean: +{np.mean(imm_entropy):.3f}",
        xy=(0, np.mean(imm_entropy) + 0.03),
        fontsize=10,
        ha="center",
        color=COLORS["immunodominant"],
        fontweight="bold",
    )
    ax3.annotate(
        f"Mean: {np.mean(sil_entropy):.3f}",
        xy=(1, np.mean(sil_entropy) - 0.03),
        fontsize=10,
        ha="center",
        color=COLORS["silent"],
        fontweight="bold",
    )

    plt.suptitle(
        "KEY FINDING: Entropy Preservation Upon Citrullination\n" "Immunodominant epitopes INCREASE entropy; Silent epitopes DECREASE entropy",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_3d_tsne(data, output_path):
    """t-SNE 3D visualization."""
    fig = plt.figure(figsize=(14, 6))

    # Perform t-SNE
    tsne = TSNE(n_components=3, perplexity=15, random_state=42, max_iter=1000)
    centroids_tsne = tsne.fit_transform(data["centroids"])

    # Subplot 1: By immunodominance
    ax1 = fig.add_subplot(121, projection="3d")

    imm_mask = np.array([l == "immunodominant" for l in data["labels"]])
    sil_mask = ~imm_mask

    ax1.scatter(
        centroids_tsne[imm_mask, 0],
        centroids_tsne[imm_mask, 1],
        centroids_tsne[imm_mask, 2],
        c=COLORS["immunodominant"],
        s=100,
        alpha=0.7,
        label="Immunodominant",
        edgecolors="white",
    )
    ax1.scatter(
        centroids_tsne[sil_mask, 0],
        centroids_tsne[sil_mask, 1],
        centroids_tsne[sil_mask, 2],
        c=COLORS["silent"],
        s=100,
        alpha=0.7,
        label="Silent",
        edgecolors="white",
    )

    ax1.set_xlabel("t-SNE 1", fontsize=10)
    ax1.set_ylabel("t-SNE 2", fontsize=10)
    ax1.set_zlabel("t-SNE 3", fontsize=10)
    ax1.set_title("t-SNE: By Immunodominance", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)

    # Subplot 2: By ACPA
    ax2 = fig.add_subplot(122, projection="3d")

    acpa = np.array(data["acpa"])
    scatter = ax2.scatter(
        centroids_tsne[:, 0],
        centroids_tsne[:, 1],
        centroids_tsne[:, 2],
        c=acpa,
        cmap="RdYlBu_r",
        s=100,
        alpha=0.7,
        edgecolors="white",
    )

    ax2.set_xlabel("t-SNE 1", fontsize=10)
    ax2.set_ylabel("t-SNE 2", fontsize=10)
    ax2.set_zlabel("t-SNE 3", fontsize=10)
    ax2.set_title("t-SNE: By ACPA Reactivity", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label("ACPA Reactivity", fontsize=10)

    plt.suptitle(
        "t-SNE 3D Embedding Visualization",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cluster_distribution_heatmap(data, encoder, output_path):
    """Heatmap of cluster distributions for immunodominant vs silent."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    n_clusters = 21
    device = "cpu"  # Force CPU to avoid device mismatch

    # Collect cluster distributions
    imm_distributions = []
    sil_distributions = []

    for protein_id, protein in RA_AUTOANTIGENS_AUGMENTED.items():
        for epitope in protein["epitopes"]:
            emb = encode_epitope(epitope["sequence"], encoder, device)
            if len(emb) == 0:
                continue

            # Get clusters
            clusters = []
            for aa in epitope["sequence"]:
                codon = AA_TO_CODON.get(aa, "NNN")
                if codon == "NNN":
                    continue
                onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    cluster_id, _ = encoder.get_cluster(onehot)
                clusters.append(cluster_id.item())

            # Distribution
            dist = np.zeros(n_clusters)
            for c in clusters:
                if 0 <= c < n_clusters:
                    dist[c] += 1
            dist /= dist.sum() + 1e-10

            if epitope["immunodominant"]:
                imm_distributions.append(dist)
            else:
                sil_distributions.append(dist)

    imm_distributions = np.array(imm_distributions)
    sil_distributions = np.array(sil_distributions)

    # Heatmaps
    ax1 = axes[0]
    im1 = ax1.imshow(imm_distributions, aspect="auto", cmap="Reds", interpolation="nearest")
    ax1.set_xlabel("Cluster ID", fontsize=12)
    ax1.set_ylabel("Epitope Index", fontsize=12)
    ax1.set_title(
        f"Immunodominant Epitopes (n={len(imm_distributions)})\nCluster Distribution",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im1, ax=ax1, label="Frequency")

    ax2 = axes[1]
    im2 = ax2.imshow(sil_distributions, aspect="auto", cmap="Blues", interpolation="nearest")
    ax2.set_xlabel("Cluster ID", fontsize=12)
    ax2.set_ylabel("Epitope Index", fontsize=12)
    ax2.set_title(
        f"Silent Epitopes (n={len(sil_distributions)})\nCluster Distribution",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im2, ax=ax2, label="Frequency")

    plt.suptitle("Cluster Distribution Patterns", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("IMMUNOGENICITY VISUALIZATIONS")
    print("High-detail 2D and 3D plots")
    print("=" * 80)

    # Setup
    results_dir = get_results_dir(hyperbolic=True)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    print(f"\nVisualizations will be saved to: {viz_dir}")

    # Load encoder (force CPU for visualizations - no GPU acceleration needed)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, mapping, _ = load_codon_encoder(device=device, version="3adic")

    # Collect embeddings
    print("\nCollecting epitope embeddings...")
    data = collect_all_embeddings(encoder, device)
    print(f"  Collected {len(data['centroids'])} epitope centroids")

    # Generate plots
    print("\nGenerating visualizations...")

    print("\n[1/7] 2D Poincaré disk projection...")
    plot_poincare_disk_2d(data, encoder, viz_dir / "poincare_disk_2d.png")

    print("\n[2/7] 3D embedding space (PCA)...")
    plot_3d_embedding_space(data, encoder, viz_dir / "embedding_space_3d.png")

    print("\n[3/7] 3D Poincaré ball...")
    plot_3d_poincare_ball(data, encoder, viz_dir / "poincare_ball_3d.png")

    print("\n[4/7] Statistical comparisons...")
    results_path = results_dir / "immunogenicity_analysis_augmented.json"
    if results_path.exists():
        plot_statistical_comparisons(results_path, viz_dir / "statistical_comparisons.png")

    print("\n[5/7] Entropy change detail...")
    if results_path.exists():
        plot_entropy_change_detail(results_path, viz_dir / "entropy_change_detail.png")

    print("\n[6/7] 3D t-SNE visualization...")
    plot_3d_tsne(data, viz_dir / "tsne_3d.png")

    print("\n[7/7] Cluster distribution heatmap...")
    plot_cluster_distribution_heatmap(data, encoder, viz_dir / "cluster_heatmap.png")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
