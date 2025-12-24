"""
RA Pathophysiology Funnel Visualization
6-stage cascade from genetic susceptibility to regenerative failure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from utils.plotting import setup_pitch_style, PALETTE, save_figure
from utils.data_loader import PATHOPHYSIOLOGY_STAGES

OUTPUT_DIR = Path(__file__).parent


def create_funnel_diagram():
    """Create the 6-stage pathophysiology funnel."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Rheumatoid Arthritis Pathophysiology',
            fontsize=24, fontweight='bold', ha='center', va='top',
            color=PALETTE['text'])
    ax.text(5, 10.8, 'P-adic Geometry Reveals Six-Stage Disease Cascade',
            fontsize=14, ha='center', va='top', color=PALETTE['text_light'])

    # Funnel parameters - widening to narrowing
    stage_colors = [
        '#3F51B5',  # Stage 1 - Genetic (blue)
        '#9C27B0',  # Stage 2 - Environmental (purple)
        '#FF9800',  # Stage 3 - Molecular (orange)
        '#F44336',  # Stage 4 - Immune (red)
        '#D32F2F',  # Stage 5 - Autoimmune (dark red)
        '#424242',  # Stage 6 - Failure (dark gray)
    ]

    # Funnel widths (narrowing down)
    widths = [8, 7, 6, 5, 4, 3.5]
    y_positions = [9.5, 8, 6.5, 5, 3.5, 2]
    box_height = 1.2

    for i, stage in enumerate(PATHOPHYSIOLOGY_STAGES):
        width = widths[i]
        y = y_positions[i]
        color = stage_colors[i]

        # Main box
        x_left = 5 - width / 2
        box = FancyBboxPatch(
            (x_left, y - box_height / 2),
            width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color, edgecolor='white', linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)

        # Stage number
        ax.text(x_left + 0.3, y, f"{stage['stage']}",
                fontsize=20, fontweight='bold', ha='left', va='center',
                color='white')

        # Stage name
        ax.text(5, y + 0.25, stage['name'],
                fontsize=14, fontweight='bold', ha='center', va='center',
                color='white')

        # Description
        ax.text(5, y - 0.25, stage['description'],
                fontsize=10, ha='center', va='center',
                color='white', alpha=0.9)

        # Discovery label on right
        ax.text(x_left + width + 0.2, y, f"→ {stage['discovery']}",
                fontsize=9, ha='left', va='center',
                color=color, style='italic')

        # Arrows between stages
        if i < len(PATHOPHYSIOLOGY_STAGES) - 1:
            arrow = FancyArrowPatch(
                (5, y - box_height / 2 - 0.05),
                (5, y_positions[i + 1] + box_height / 2 + 0.05),
                arrowstyle='->', mutation_scale=15,
                color=PALETTE['text_light'], linewidth=2
            )
            ax.add_patch(arrow)

    # Add intervention points annotation
    ax.annotate('', xy=(1, 8), xytext=(1, 3.5),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=2))
    ax.text(0.5, 5.75, 'Intervention\nWindow', fontsize=10, ha='center',
            va='center', color='#4CAF50', fontweight='bold', rotation=90)

    # Footer
    ax.text(5, 0.5, 'Four discoveries from p-adic embedding analysis unify RA pathogenesis',
            fontsize=11, ha='center', va='center', color=PALETTE['text_light'])

    return fig


def main():
    fig = create_funnel_diagram()
    output_path = save_figure(fig, OUTPUT_DIR, 'pathophysiology_funnel')
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
