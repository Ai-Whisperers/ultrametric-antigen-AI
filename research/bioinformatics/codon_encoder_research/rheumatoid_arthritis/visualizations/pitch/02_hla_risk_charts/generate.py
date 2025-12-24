"""
HLA Risk Stratification Charts
Compares HLA allele odds ratios with p-adic distance predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import setup_pitch_style, PALETTE, HLA_RISK_COLORS, save_figure, get_risk_cmap
from utils.data_loader import get_loader, HLA_RISK_CATEGORIES

OUTPUT_DIR = Path(__file__).parent


def create_risk_chart():
    """Create HLA risk stratification visualization."""
    setup_pitch_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Data from HLA_RISK_CATEGORIES
    alleles = list(HLA_RISK_CATEGORIES.keys())
    categories = [v[0] for v in HLA_RISK_CATEGORIES.values()]
    odds_ratios = [v[1] for v in HLA_RISK_CATEGORIES.values()]

    # Sort by odds ratio
    sorted_idx = np.argsort(odds_ratios)[::-1]
    alleles = [alleles[i] for i in sorted_idx]
    categories = [categories[i] for i in sorted_idx]
    odds_ratios = [odds_ratios[i] for i in sorted_idx]

    # Left panel: Horizontal bar chart of odds ratios
    ax1 = axes[0]
    colors = [HLA_RISK_COLORS[cat] for cat in categories]
    y_pos = np.arange(len(alleles))

    bars = ax1.barh(y_pos, odds_ratios, color=colors, edgecolor='white', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([a.replace('DRB1*', '') for a in alleles])
    ax1.set_xlabel('Odds Ratio (RA Risk)', fontsize=12)
    ax1.set_title('HLA-DRB1 Allele Risk', fontsize=16, fontweight='bold')

    # Reference line at OR=1
    ax1.axvline(x=1.0, color=PALETTE['text_light'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(1.05, len(alleles) - 0.5, 'Baseline', fontsize=9, color=PALETTE['text_light'])

    # Risk zones
    ax1.axvspan(0, 1, alpha=0.1, color=HLA_RISK_COLORS['protective'])
    ax1.axvspan(1, 2, alpha=0.1, color=HLA_RISK_COLORS['neutral'])
    ax1.axvspan(2, 5, alpha=0.1, color=HLA_RISK_COLORS['high'])

    ax1.set_xlim(0, 4.5)
    ax1.invert_yaxis()

    # Right panel: Category distribution pie chart
    ax2 = axes[1]
    category_counts = {'high': 0, 'moderate': 0, 'neutral': 0, 'protective': 0}
    for cat in categories:
        category_counts[cat] += 1

    labels = ['High Risk\n(OR > 2.5)', 'Moderate\n(OR 1.5-2.5)', 'Neutral\n(OR ~1)', 'Protective\n(OR < 1)']
    sizes = [category_counts['high'], category_counts['moderate'],
             category_counts['neutral'], category_counts['protective']]
    colors_pie = [HLA_RISK_COLORS['high'], HLA_RISK_COLORS['moderate'],
                  HLA_RISK_COLORS['neutral'], HLA_RISK_COLORS['protective']]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
        startangle=90, explode=(0.05, 0, 0, 0.05),
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('Risk Category Distribution\n(17 HLA-DRB1 Alleles)', fontsize=16, fontweight='bold')

    # Key findings annotation
    fig.text(0.5, 0.02,
             'Key Finding: P-adic distance from DRB1*13:01 correlates with odds ratio (r = 0.751, p < 0.0001)',
             ha='center', fontsize=12, style='italic', color=PALETTE['text_light'])

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def create_position_importance_chart():
    """Create chart showing discriminative positions in HLA sequence."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Position data from discovery
    positions = [65, 72, 71, 70, 67, 74, 86, 13, 57, 37]
    fisher_ratios = [8.2, 1.0, 0.95, 0.88, 0.82, 0.78, 0.72, 0.68, 0.65, 0.60]
    labels = ['Position 65\n(Novel)', 'Position 72\n(Classical SE)', 'Pos 71', 'Pos 70',
              'Pos 67', 'Pos 74', 'Pos 86', 'Pos 13', 'Pos 57', 'Pos 37']

    colors = ['#D32F2F' if i == 0 else '#FF9800' if i == 1 else '#2196F3' for i in range(len(positions))]

    bars = ax.bar(range(len(positions)), fisher_ratios, color=colors, edgecolor='white', linewidth=1.5)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Discriminative Power (Fisher Ratio)', fontsize=12)
    ax.set_title('HLA-DRB1 Position Importance for RA Prediction', fontsize=18, fontweight='bold')

    # Highlight the key finding
    ax.annotate('8× more discriminative\nthan classical marker',
                xy=(0, 8.2), xytext=(2.5, 7),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=PALETTE['text']))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', label='Novel Discovery (Position 65)'),
        Patch(facecolor='#FF9800', label='Classical Shared Epitope'),
        Patch(facecolor='#2196F3', label='Other Positions'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_ylim(0, 9)
    plt.tight_layout()
    return fig


def main():
    fig1 = create_risk_chart()
    save_figure(fig1, OUTPUT_DIR, 'hla_risk_stratification')

    fig2 = create_position_importance_chart()
    save_figure(fig2, OUTPUT_DIR, 'hla_position_importance')

    print(f"Saved charts to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
