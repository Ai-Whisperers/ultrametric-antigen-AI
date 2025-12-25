"""
Distance Matrix Heatmaps - Scientific Visualization
Epitope distance matrices with JS divergence overlays.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import get_loader
from utils.plotting import PALETTE, save_figure, setup_scientific_style

OUTPUT_DIR = Path(__file__).parent


def create_epitope_distance_heatmap():
    """Create heatmap of epitope-to-epitope distances."""
    setup_scientific_style()

    # Load data
    loader = get_loader()
    data = loader.load_goldilocks_data()

    # Extract epitope data
    epitopes = []
    shifts = []
    js_divergences = []
    immunodominant = []

    for shift_data in data.all_shifts:
        epitope_id = shift_data["epitope_id"]
        imm = shift_data["immunodominant"]
        arg_shifts = shift_data["arg_shifts"]

        if arg_shifts:
            mean_shift = np.mean([a["centroid_shift"] for a in arg_shifts])
            mean_js = np.mean([a["js_divergence"] for a in arg_shifts])
            epitopes.append(epitope_id)
            shifts.append(mean_shift)
            js_divergences.append(mean_js)
            immunodominant.append(imm)

    n = len(epitopes)

    # Create synthetic distance matrix based on shift differences
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = abs(shifts[i] - shifts[j])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Distance matrix heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(distance_matrix, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(epitopes, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(epitopes, fontsize=9)
    ax1.set_title(
        "Epitope Distance Matrix\n(Centroid Shift Differences)",
        fontweight="bold",
    )

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("|Δ Centroid Shift|")

    # Mark immunodominant epitopes
    for i, imm in enumerate(immunodominant):
        if imm:
            ax1.text(
                -0.7,
                i,
                "●",
                fontsize=12,
                color=PALETTE["immunodominant"],
                ha="center",
                va="center",
            )

    # Right: JS Divergence vs Centroid Shift
    ax2 = axes[1]
    colors = [PALETTE["immunodominant"] if imm else PALETTE["silent"] for imm in immunodominant]
    sizes = [120 if imm else 80 for imm in immunodominant]

    scatter = ax2.scatter(
        shifts,
        js_divergences,
        c=colors,
        s=sizes,
        edgecolors="white",
        linewidths=1.5,
        alpha=0.8,
    )

    # Add labels
    for i, ep in enumerate(epitopes):
        ax2.annotate(
            ep,
            (shifts[i], js_divergences[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    # Goldilocks zone shading
    ax2.axvspan(0.15, 0.30, color="#FFD700", alpha=0.15, label="Goldilocks Zone")

    ax2.set_xlabel("Centroid Shift")
    ax2.set_ylabel("JS Divergence")
    ax2.set_title("Shift vs Divergence\nby Immunodominance", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PALETTE["immunodominant"], label="Immunodominant"),
        Patch(facecolor=PALETTE["silent"], label="Silent"),
        Patch(facecolor="#FFD700", alpha=0.3, label="Goldilocks Zone"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def create_comparison_heatmap():
    """Create heatmap comparing immunodominant vs silent epitopes."""
    setup_scientific_style()

    # Load data
    loader = get_loader()
    data = loader.load_goldilocks_data()
    comparisons = data.comparisons

    # Prepare comparison data
    metrics = [
        "centroid_shift",
        "relative_shift",
        "js_divergence",
        "entropy_change",
    ]
    metric_labels = [
        "Centroid Shift",
        "Relative Shift",
        "JS Divergence",
        "Entropy Change",
    ]

    imm_means = [comparisons[m]["imm_mean"] for m in metrics]
    imm_stds = [comparisons[m]["imm_std"] for m in metrics]
    sil_means = [comparisons[m]["sil_mean"] for m in metrics]
    sil_stds = [comparisons[m]["sil_std"] for m in metrics]
    p_values = [comparisons[m]["p_value"] for m in metrics]
    effect_sizes = [comparisons[m]["effect_size"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Means comparison bar chart
    ax1 = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        imm_means,
        width,
        yerr=imm_stds,
        capsize=4,
        label="Immunodominant",
        color=PALETTE["immunodominant"],
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        sil_means,
        width,
        yerr=sil_stds,
        capsize=4,
        label="Silent",
        color=PALETTE["silent"],
        alpha=0.8,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax1.set_ylabel("Mean Value")
    ax1.set_title("Metric Comparison: Immunodominant vs Silent", fontweight="bold")
    ax1.legend()
    ax1.axhline(0, color="gray", linewidth=0.5)

    # Top right: P-values
    ax2 = axes[0, 1]
    colors = ["#4CAF50" if p < 0.05 else "#9E9E9E" for p in p_values]
    bars = ax2.barh(
        metric_labels,
        [-np.log10(p) for p in p_values],
        color=colors,
        alpha=0.8,
    )

    ax2.axvline(
        -np.log10(0.05),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="p = 0.05",
    )
    ax2.axvline(
        -np.log10(0.01),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="p = 0.01",
    )
    ax2.set_xlabel("-log₁₀(p-value)")
    ax2.set_title("Statistical Significance", fontweight="bold")
    ax2.legend(loc="lower right")

    # Add actual p-values as text
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        p_text = f"p={p:.4f}" if p >= 0.001 else "p<0.001"
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            p_text,
            va="center",
            fontsize=9,
        )

    # Bottom left: Effect sizes
    ax3 = axes[1, 0]
    colors_es = [("#D32F2F" if abs(es) > 0.8 else "#FF9800" if abs(es) > 0.5 else "#4CAF50") for es in effect_sizes]
    ax3.barh(metric_labels, effect_sizes, color=colors_es, alpha=0.8)
    ax3.axvline(0, color="gray", linewidth=1)
    ax3.axvline(-0.8, color="gray", linestyle=":", alpha=0.5)
    ax3.axvline(0.8, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Cohen's d Effect Size")
    ax3.set_title("Effect Size Magnitude", fontweight="bold")

    # Add interpretation
    ax3.text(
        0.95,
        0.05,
        "Large effect: |d| > 0.8\nMedium: 0.5-0.8\nSmall: < 0.5",
        transform=ax3.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Bottom right: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    for i, m in enumerate(metrics):
        sig = "✓" if p_values[i] < 0.05 else "✗"
        es_label = "Large" if abs(effect_sizes[i]) > 0.8 else "Medium" if abs(effect_sizes[i]) > 0.5 else "Small"
        table_data.append(
            [
                metric_labels[i],
                f"{imm_means[i]:.4f}",
                f"{sil_means[i]:.4f}",
                f"{p_values[i]:.4f}",
                es_label,
                sig,
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=[
            "Metric",
            "Imm. Mean",
            "Silent Mean",
            "p-value",
            "Effect",
            "Sig.",
        ],
        loc="center",
        cellLoc="center",
        colColours=["#E3F2FD"] * 6,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Statistical Summary", fontweight="bold", y=0.95)

    plt.tight_layout()
    return fig


def main():
    fig1 = create_epitope_distance_heatmap()
    save_figure(fig1, OUTPUT_DIR, "epitope_distance_matrix")

    fig2 = create_comparison_heatmap()
    save_figure(fig2, OUTPUT_DIR, "statistical_comparison")

    print(f"Saved heatmaps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
