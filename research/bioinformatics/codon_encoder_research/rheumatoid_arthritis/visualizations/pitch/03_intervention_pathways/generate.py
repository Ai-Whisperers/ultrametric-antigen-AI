"""
Intervention Pathway Diagrams
3-tier therapeutic protocol based on regenerative axis discovery.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

from utils.plotting import setup_pitch_style, PALETTE, save_figure
from utils.data_loader import INTERVENTION_TIERS

OUTPUT_DIR = Path(__file__).parent


def create_intervention_diagram():
    """Create 3-tier intervention pathway visualization."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.5, 'Three-Tier Regenerative Protocol',
            fontsize=24, fontweight='bold', ha='center', color=PALETTE['text'])
    ax.text(8, 8.9, 'Sequential Intervention Based on P-adic Pathway Geometry',
            fontsize=13, ha='center', color=PALETTE['text_light'])

    # Tier colors
    tier_colors = [PALETTE['parasympathetic'], PALETTE['gut_barrier'], PALETTE['regeneration']]

    # Central pathway positions
    tier_y = [6.5, 4.5, 2.5]
    tier_x = 8

    for i, tier in enumerate(INTERVENTION_TIERS):
        y = tier_y[i]
        color = tier_colors[i]

        # Main tier box
        box = FancyBboxPatch(
            (4, y - 0.8), 8, 1.6,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color, edgecolor='white', linewidth=3, alpha=0.9
        )
        ax.add_patch(box)

        # Tier number circle
        circle = Circle((2.5, y), 0.5, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(2.5, y, str(tier['tier']), fontsize=20, fontweight='bold',
                ha='center', va='center', color='white')

        # Tier name
        ax.text(8, y + 0.4, tier['name'], fontsize=16, fontweight='bold',
                ha='center', va='center', color='white')

        # Targets
        targets_text = ' • '.join(tier['targets'])
        ax.text(8, y - 0.3, targets_text, fontsize=11,
                ha='center', va='center', color='white', alpha=0.95)

        # Rationale on the right
        ax.text(13, y, tier['rationale'], fontsize=10,
                ha='left', va='center', color=color, style='italic')

        # Connecting arrows
        if i < len(INTERVENTION_TIERS) - 1:
            arrow = FancyArrowPatch(
                (8, y - 0.85), (8, tier_y[i + 1] + 0.85),
                arrowstyle='->', mutation_scale=20,
                color=PALETTE['text_light'], linewidth=2.5
            )
            ax.add_patch(arrow)

    # Side annotations - Pathway geometry insight
    ax.text(0.5, 6.5, 'Parasympathetic\nCentrality\n(d=0.697)',
            fontsize=9, ha='center', va='center', color=PALETTE['parasympathetic'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=PALETTE['parasympathetic'], alpha=0.8))

    ax.text(0.5, 4.5, 'Inflammatory\nProximity\n(d=0.440)',
            fontsize=9, ha='center', va='center', color=PALETTE['inflammation'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=PALETTE['inflammation'], alpha=0.8))

    ax.text(0.5, 2.5, 'Regeneration\nAccess\nRestored',
            fontsize=9, ha='center', va='center', color=PALETTE['regeneration'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=PALETTE['regeneration'], alpha=0.8))

    # Outcome box
    outcome_box = FancyBboxPatch(
        (5.5, 0.3), 5, 1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#E8F5E9', edgecolor=PALETTE['regeneration'], linewidth=2
    )
    ax.add_patch(outcome_box)
    ax.text(8, 0.8, '✓ Tissue Repair Enabled', fontsize=14, fontweight='bold',
            ha='center', va='center', color=PALETTE['regeneration'])

    # Final arrow
    arrow_final = FancyArrowPatch(
        (8, 1.65), (8, 1.35),
        arrowstyle='->', mutation_scale=18,
        color=PALETTE['regeneration'], linewidth=2
    )
    ax.add_patch(arrow_final)

    return fig


def create_pathway_distances_chart():
    """Create bar chart showing pathway distances."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Data from regenerative axis discovery
    pathways = ['Parasympathetic', 'Sympathetic']
    regen_dist = [0.697, 0.792]
    inflam_dist = [0.440, 0.724]

    x = np.arange(len(pathways))
    width = 0.35

    bars1 = ax.bar(x - width/2, regen_dist, width, label='Distance to Regeneration',
                   color=PALETTE['regeneration'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, inflam_dist, width, label='Distance to Inflammation',
                   color=PALETTE['inflammation'], edgecolor='white', linewidth=1.5)

    ax.set_ylabel('P-adic Distance', fontsize=12)
    ax.set_title('Autonomic Pathway Geometry', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pathways, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Key insight annotation
    ax.annotate('Parasympathetic has\nprivileged access to\nboth regeneration\nand inflammation',
                xy=(0, 0.5), xytext=(-0.4, 0.3),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=PALETTE['parasympathetic']))

    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    return fig


def main():
    fig1 = create_intervention_diagram()
    save_figure(fig1, OUTPUT_DIR, 'intervention_pathway')

    fig2 = create_pathway_distances_chart()
    save_figure(fig2, OUTPUT_DIR, 'pathway_distances')

    print(f"Saved diagrams to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
