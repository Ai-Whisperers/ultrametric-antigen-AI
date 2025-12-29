"""
Shared plotting utilities for RA visualizations.
Provides consistent styling, color palettes, and export functions.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Color Palettes
# =============================================================================

PALETTE = {
    # Risk gradient (red spectrum)
    "risk_high": "#D32F2F",
    "risk_moderate": "#FF9800",
    "risk_low": "#4CAF50",
    "risk_protective": "#2196F3",
    # Safety gradient
    "safe": "#4CAF50",
    "partial": "#FF9800",
    "unsafe": "#D32F2F",
    # Pathway colors
    "parasympathetic": "#3F51B5",
    "sympathetic": "#F44336",
    "regeneration": "#4CAF50",
    "gut_barrier": "#9C27B0",
    "inflammation": "#FF5722",
    # Immunodominance
    "immunodominant": "#E91E63",
    "silent": "#9E9E9E",
    # Goldilocks zones
    "goldilocks_low": "#90CAF9",
    "goldilocks_zone": "#FFEB3B",
    "goldilocks_high": "#EF9A9A",
    # General
    "primary": "#1976D2",
    "secondary": "#424242",
    "background": "#FAFAFA",
    "grid": "#E0E0E0",
    "text": "#212121",
    "text_light": "#757575",
}

# HLA allele risk categories
HLA_RISK_COLORS = {
    "high": "#D32F2F",
    "moderate": "#FF9800",
    "neutral": "#9E9E9E",
    "protective": "#2196F3",
}

# Colorblind-safe scientific palettes
VIRIDIS = plt.cm.viridis
PLASMA = plt.cm.plasma


def get_risk_cmap():
    """Returns a diverging colormap from protective (blue) to high risk (red)."""
    colors = ["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#D32F2F"]
    return LinearSegmentedColormap.from_list("risk", colors)


def get_safety_cmap():
    """Returns colormap for safety metrics (green=safe, red=unsafe)."""
    colors = ["#D32F2F", "#FF9800", "#4CAF50"]
    return LinearSegmentedColormap.from_list("safety", colors)


# =============================================================================
# Style Configuration
# =============================================================================


def setup_pitch_style():
    """Configure matplotlib for pitch/presentation visualizations."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PALETTE["background"],
            "figure.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def setup_scientific_style():
    """Configure matplotlib for scientific/publication visualizations."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


# =============================================================================
# Common Figure Templates
# =============================================================================


def create_pitch_figure(figsize=(12, 8)):
    """Create figure with pitch styling."""
    setup_pitch_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_scientific_figure(figsize=(10, 8), nrows=1, ncols=1):
    """Create figure with scientific styling."""
    setup_scientific_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


# =============================================================================
# Export Functions
# =============================================================================


def save_figure(fig, output_dir: Path, name: str, formats=("png", "svg")):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300 if fmt == "png" else None)

    plt.close(fig)
    return output_dir / f"{name}.png"


# =============================================================================
# Common Plot Elements
# =============================================================================


def add_goldilocks_zones(ax, ymin=0, ymax=1, alpha=0.15):
    """Add Goldilocks zone shading to axis."""
    # Too low (<15%)
    ax.axvspan(
        0,
        0.15,
        color=PALETTE["goldilocks_low"],
        alpha=alpha,
        label="Self (no response)",
    )
    # Goldilocks (15-30%)
    ax.axvspan(
        0.15,
        0.30,
        color=PALETTE["goldilocks_zone"],
        alpha=alpha,
        label="Modified self (autoimmunity)",
    )
    # Too high (>30%)
    ax.axvspan(
        0.30,
        1.0,
        color=PALETTE["goldilocks_high"],
        alpha=alpha,
        label="Foreign (cleared)",
    )


def add_significance_annotation(ax, x1, x2, y, p_value, height=0.02):
    """Add significance bracket with p-value."""
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color="black",
        linewidth=1,
    )
    ax.text((x1 + x2) / 2, y + height, stars, ha="center", va="bottom", fontsize=12)


def create_legend_handles(labels_colors: dict):
    """Create legend handles from label-color dict."""
    return [mpatches.Patch(color=color, label=label) for label, color in labels_colors.items()]
