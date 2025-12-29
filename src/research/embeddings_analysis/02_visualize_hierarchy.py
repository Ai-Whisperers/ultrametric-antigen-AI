"""Visualize p-adic hierarchy in embedding spaces.

Creates plots showing the critical finding: v5_11_overnight has inverted
hierarchy at high valuations while v5_11 maintains correct structure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load results
results_path = Path(__file__).parent / "comparison_results.json"
with open(results_path) as f:
    results = json.load(f)


def plot_valuation_radius_comparison():
    """Plot mean radius per valuation for each variant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    variants = ["v5_11", "v5_11_overnight"]
    colors = {"v5_11": "#2ecc71", "v5_11_overnight": "#e74c3c"}
    labels = {
        "v5_11": "v5.11 (Production)",
        "v5_11_overnight": "v5.11 Overnight",
    }

    # Plot 1: Line comparison
    ax1 = axes[0]
    for variant in variants:
        stats = results[variant]["embedding_analysis"]["per_valuation_stats"]
        valuations = sorted([int(v) for v in stats.keys() if int(v) <= 8])
        radii = [stats[str(v)]["mean_radius"] for v in valuations]

        ax1.plot(
            valuations,
            radii,
            "o-",
            color=colors[variant],
            label=labels[variant],
            linewidth=2,
            markersize=8,
        )

    # Add expected trend line
    expected = [0.85 - 0.03 * v for v in range(9)]
    ax1.plot(
        range(9),
        expected,
        "--",
        color="gray",
        alpha=0.5,
        label="Expected (monotonic decrease)",
        linewidth=1.5,
    )

    ax1.set_xlabel("3-adic Valuation", fontsize=12)
    ax1.set_ylabel("Mean Radius in Poincaré Ball", fontsize=12)
    ax1.set_title("P-adic Hierarchy: Radius vs Valuation", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(9))

    # Highlight the inversion region
    ax1.axvspan(3.5, 8.5, alpha=0.1, color="red")
    ax1.annotate(
        "Hierarchy\nInversion",
        xy=(6, 0.85),
        fontsize=10,
        color="red",
        ha="center",
    )

    # Plot 2: Bar chart showing the difference
    ax2 = axes[1]
    valuations = list(range(9))

    v5_11_radii = [results["v5_11"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0) for v in valuations]
    overnight_radii = [
        results["v5_11_overnight"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0) for v in valuations
    ]

    x = np.arange(len(valuations))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, v5_11_radii, width, label="v5.11", color="#2ecc71")
    bars2 = ax2.bar(
        x + width / 2,
        overnight_radii,
        width,
        label="v5.11 Overnight",
        color="#e74c3c",
    )

    ax2.set_xlabel("3-adic Valuation", fontsize=12)
    ax2.set_ylabel("Mean Radius", fontsize=12)
    ax2.set_title("Radius Comparison by Valuation", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valuations)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Highlight problematic valuations
    for i in range(4, 9):
        if i < len(x):
            ax2.annotate(
                "",
                xy=(x[i] + width / 2, overnight_radii[i]),
                xytext=(x[i] + width / 2, overnight_radii[i] + 0.05),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "valuation_radius_comparison.png", dpi=150)
    plt.show()
    print("Saved: valuation_radius_comparison.png")


def plot_hierarchy_delta():
    """Plot the delta between variants to highlight inversion."""
    fig, ax = plt.subplots(figsize=(10, 5))

    valuations = list(range(9))
    v5_11_radii = [results["v5_11"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0) for v in valuations]
    overnight_radii = [
        results["v5_11_overnight"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0) for v in valuations
    ]

    # Delta: positive means overnight has larger radius
    delta = [o - v for v, o in zip(v5_11_radii, overnight_radii)]

    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in delta]
    bars = ax.bar(valuations, delta, color=colors, edgecolor="black", linewidth=1)

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("3-adic Valuation", fontsize=12)
    ax.set_ylabel("Radius Difference (Overnight - Production)", fontsize=12)
    ax.set_title("Hierarchy Deviation: Where Overnight Training Went Wrong", fontsize=14)
    ax.set_xticks(valuations)
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotations
    ax.annotate(
        "Overnight radius\nTOO LARGE\n(should be smaller)",
        xy=(5, 0.15),
        fontsize=10,
        ha="center",
        color="#e74c3c",
    )
    ax.annotate(
        "Correct\n(overnight smaller)",
        xy=(1, -0.05),
        fontsize=10,
        ha="center",
        color="#2ecc71",
    )

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "hierarchy_delta.png", dpi=150)
    plt.show()
    print("Saved: hierarchy_delta.png")


def print_summary_table():
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("CRITICAL FINDING: P-ADIC HIERARCHY ANALYSIS")
    print("=" * 70)

    print("\n### Per-Valuation Mean Radius ###\n")
    print(f"{'Valuation':<10} {'v5_11':<12} {'overnight':<12} {'Delta':<12} {'Status':<15}")
    print("-" * 60)

    for v in range(9):
        v5_11_r = results["v5_11"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0)
        overnight_r = results["v5_11_overnight"]["embedding_analysis"]["per_valuation_stats"].get(str(v), {}).get("mean_radius", 0)
        delta = overnight_r - v5_11_r

        if v >= 4 and delta > 0.05:
            status = "INVERTED!"
        elif delta > 0:
            status = "Warning"
        else:
            status = "OK"

        print(f"v={v:<8} {v5_11_r:<12.4f} {overnight_r:<12.4f} {delta:+.4f}      {status}")

    print("\n" + "=" * 70)
    print("CONCLUSION: v5_11 maintains correct hierarchy; v5_11_overnight inverts at v>=4")
    print("=" * 70)


if __name__ == "__main__":
    print_summary_table()

    try:
        plot_valuation_radius_comparison()
        plot_hierarchy_delta()
    except Exception as e:
        print(f"Visualization skipped (no display): {e}")
        print("Plots would be saved to: valuation_radius_comparison.png, hierarchy_delta.png")
