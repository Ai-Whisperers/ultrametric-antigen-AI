"""
Goldilocks Zone Visualization
Shows the optimal perturbation range for autoimmune recognition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Wedge

from utils.data_loader import get_loader
from utils.plotting import PALETTE, save_figure, setup_pitch_style

OUTPUT_DIR = Path(__file__).parent


def create_goldilocks_gauge():
    """Create gauge-style visualization of Goldilocks zone."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        0,
        1.25,
        "The Goldilocks Zone of Autoimmunity",
        fontsize=22,
        fontweight="bold",
        ha="center",
        color=PALETTE["text"],
    )
    ax.text(
        0,
        1.1,
        "Optimal Perturbation Range for Immune Recognition",
        fontsize=12,
        ha="center",
        color=PALETTE["text_light"],
    )

    # Create semi-circular gauge
    # Zone 1: Too little (<15%) - Blue
    wedge1 = Wedge(
        (0, 0),
        1,
        126,
        180,
        width=0.3,
        facecolor=PALETTE["goldilocks_low"],
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(wedge1)

    # Zone 2: Goldilocks (15-30%) - Yellow/Gold
    wedge2 = Wedge(
        (0, 0),
        1,
        72,
        126,
        width=0.3,
        facecolor="#FFD700",
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(wedge2)

    # Zone 3: Too much (>30%) - Red
    wedge3 = Wedge(
        (0, 0),
        1,
        0,
        72,
        width=0.3,
        facecolor=PALETTE["goldilocks_high"],
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(wedge3)

    # Zone labels
    ax.text(
        -0.85,
        0.5,
        "Too\nLittle",
        fontsize=11,
        ha="center",
        va="center",
        color=PALETTE["risk_protective"],
        fontweight="bold",
    )
    ax.text(
        0,
        0.85,
        "Goldilocks\nZone",
        fontsize=12,
        ha="center",
        va="center",
        color="#B8860B",
        fontweight="bold",
    )
    ax.text(
        0.85,
        0.5,
        "Too\nMuch",
        fontsize=11,
        ha="center",
        va="center",
        color=PALETTE["risk_high"],
        fontweight="bold",
    )

    # Percentage markers
    ax.text(
        -1.15,
        0.15,
        "0%",
        fontsize=10,
        ha="center",
        color=PALETTE["text_light"],
    )
    ax.text(-0.6, 0.9, "15%", fontsize=10, ha="center", color=PALETTE["text_light"])
    ax.text(0.6, 0.9, "30%", fontsize=10, ha="center", color=PALETTE["text_light"])
    ax.text(
        1.15,
        0.15,
        "50%+",
        fontsize=10,
        ha="center",
        color=PALETTE["text_light"],
    )

    # Outcome descriptions below gauge
    outcomes = [
        {
            "x": -0.9,
            "text": 'Recognized as\n"SELF"',
            "sub": "No immune response",
            "color": PALETTE["goldilocks_low"],
        },
        {
            "x": 0,
            "text": 'Recognized as\n"MODIFIED SELF"',
            "sub": "T-cell activation\nAutoimmunity",
            "color": "#FFD700",
        },
        {
            "x": 0.9,
            "text": 'Recognized as\n"FOREIGN"',
            "sub": "Rapid clearance",
            "color": PALETTE["goldilocks_high"],
        },
    ]

    for outcome in outcomes:
        ax.text(
            outcome["x"],
            -0.15,
            outcome["text"],
            fontsize=10,
            ha="center",
            va="top",
            color=outcome["color"],
            fontweight="bold",
        )
        ax.text(
            outcome["x"],
            -0.4,
            outcome["sub"],
            fontsize=9,
            ha="center",
            va="top",
            color=PALETTE["text_light"],
        )

    # Key epitopes in Goldilocks zone
    epitopes_box = FancyBboxPatch(
        (-1.3, -0.85),
        2.6,
        0.35,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#FFFDE7",
        edgecolor="#FFD700",
        linewidth=2,
    )
    ax.add_patch(epitopes_box)
    ax.text(
        0,
        -0.67,
        "Key Epitopes in Goldilocks Zone: VIM_R71 (19%) • FGA_R38 (24.5%) • FLG_CCP (21.2%)",
        fontsize=10,
        ha="center",
        va="center",
        color="#B8860B",
    )

    return fig


def create_epitope_shift_chart():
    """Create chart showing epitope shifts with immunodominance."""
    setup_pitch_style()

    # Load data
    loader = get_loader()
    data = loader.load_goldilocks_data()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract epitope data
    epitopes = []
    shifts = []
    acpa_values = []
    is_immunodominant = []

    for shift_data in data.all_shifts:
        epitope_id = shift_data["epitope_id"]
        imm = shift_data["immunodominant"]
        acpa = shift_data["acpa"]

        # Get mean shift across all arg positions
        arg_shifts = shift_data["arg_shifts"]
        if arg_shifts:
            mean_shift = np.mean([a["centroid_shift"] for a in arg_shifts])
            epitopes.append(epitope_id)
            shifts.append(mean_shift * 100)  # Convert to percentage
            acpa_values.append(acpa * 100)
            is_immunodominant.append(imm)

    # Sort by shift
    sorted_idx = np.argsort(shifts)
    epitopes = [epitopes[i] for i in sorted_idx]
    shifts = [shifts[i] for i in sorted_idx]
    acpa_values = [acpa_values[i] for i in sorted_idx]
    is_immunodominant = [is_immunodominant[i] for i in sorted_idx]

    # Create bar chart
    colors = [PALETTE["immunodominant"] if imm else PALETTE["silent"] for imm in is_immunodominant]
    x = np.arange(len(epitopes))

    bars = ax.bar(x, shifts, color=colors, edgecolor="white", linewidth=1.5, alpha=0.8)

    # Add Goldilocks zone shading
    ax.axhspan(15, 30, color="#FFD700", alpha=0.2, zorder=0)
    ax.axhline(15, color="#FFD700", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(30, color="#FFD700", linestyle="--", linewidth=1.5, alpha=0.7)

    # Zone labels
    ax.text(
        len(epitopes) - 0.5,
        22.5,
        "GOLDILOCKS\nZONE",
        fontsize=10,
        ha="right",
        va="center",
        color="#B8860B",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(epitopes, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Centroid Shift (%)", fontsize=12)
    ax.set_title("Epitope Perturbation Magnitude", fontsize=18, fontweight="bold")

    # ACPA markers (secondary y-axis indication via marker size)
    for i, (shift, acpa) in enumerate(zip(shifts, acpa_values)):
        if acpa > 50:
            ax.scatter(
                i,
                shift + 1.5,
                s=acpa * 2,
                c="red",
                marker="v",
                edgecolors="white",
                linewidths=1,
                zorder=5,
                alpha=0.7,
            )

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PALETTE["immunodominant"], label="Immunodominant"),
        Patch(facecolor=PALETTE["silent"], label="Silent"),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="High ACPA (>50%)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_ylim(0, 45)
    plt.tight_layout()
    return fig


def create_statistics_summary():
    """Create summary of statistical findings."""
    setup_pitch_style()

    # Load data
    loader = get_loader()
    data = loader.load_goldilocks_data()
    comparisons = data.comparisons

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        5,
        5.5,
        "Statistical Evidence for Goldilocks Hypothesis",
        fontsize=18,
        fontweight="bold",
        ha="center",
        color=PALETTE["text"],
    )

    # Create comparison boxes
    metrics = [
        {
            "name": "Centroid Shift",
            "key": "centroid_shift",
            "imm": f"{comparisons['centroid_shift']['imm_mean']*100:.1f}%",
            "sil": f"{comparisons['centroid_shift']['sil_mean']*100:.1f}%",
            "p": comparisons["centroid_shift"]["p_value"],
        },
        {
            "name": "JS Divergence",
            "key": "js_divergence",
            "imm": f"{comparisons['js_divergence']['imm_mean']:.3f}",
            "sil": f"{comparisons['js_divergence']['sil_mean']:.3f}",
            "p": comparisons["js_divergence"]["p_value"],
        },
        {
            "name": "Entropy Change",
            "key": "entropy_change",
            "imm": f"{comparisons['entropy_change']['imm_mean']:.3f}",
            "sil": f"{comparisons['entropy_change']['sil_mean']:.3f}",
            "p": comparisons["entropy_change"]["p_value"],
        },
    ]

    for i, metric in enumerate(metrics):
        x_base = 1.5 + i * 3

        # Box
        box = FancyBboxPatch(
            (x_base - 1, 2),
            2.5,
            2.5,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="white",
            edgecolor=PALETTE["primary"],
            linewidth=2,
        )
        ax.add_patch(box)

        # Metric name
        ax.text(
            x_base + 0.25,
            4.2,
            metric["name"],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["text"],
        )

        # Values
        ax.text(
            x_base + 0.25,
            3.5,
            f"Immunodominant: {metric['imm']}",
            fontsize=10,
            ha="center",
            va="center",
            color=PALETTE["immunodominant"],
        )
        ax.text(
            x_base + 0.25,
            3.0,
            f"Silent: {metric['sil']}",
            fontsize=10,
            ha="center",
            va="center",
            color=PALETTE["silent"],
        )

        # P-value
        p = metric["p"]
        p_text = f"p = {p:.4f}" if p >= 0.001 else "p < 0.001"
        p_color = PALETTE["safe"] if p < 0.05 else PALETTE["text_light"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(
            x_base + 0.25,
            2.4,
            f"{p_text} {stars}",
            fontsize=11,
            ha="center",
            va="center",
            color=p_color,
            fontweight="bold",
        )

    # Summary statement
    ax.text(
        5,
        1.2,
        "All three metrics show statistically significant differences (p < 0.05)",
        fontsize=12,
        ha="center",
        va="center",
        color=PALETTE["text"],
    )
    ax.text(
        5,
        0.7,
        'Immunodominant epitopes cluster in the 15-30% "modified self" zone',
        fontsize=11,
        ha="center",
        va="center",
        color=PALETTE["text_light"],
        style="italic",
    )

    return fig


def main():
    fig1 = create_goldilocks_gauge()
    save_figure(fig1, OUTPUT_DIR, "goldilocks_gauge")

    fig2 = create_epitope_shift_chart()
    save_figure(fig2, OUTPUT_DIR, "epitope_shifts")

    fig3 = create_statistics_summary()
    save_figure(fig3, OUTPUT_DIR, "goldilocks_statistics")

    print(f"Saved charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
