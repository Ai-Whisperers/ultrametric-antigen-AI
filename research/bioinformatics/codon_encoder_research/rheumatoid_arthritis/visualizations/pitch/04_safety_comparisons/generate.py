"""
Codon Safety Comparison Visualizations
Shows citrullination safety metrics for synovial proteins.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import setup_pitch_style, PALETTE, save_figure, get_safety_cmap
from utils.data_loader import get_loader

OUTPUT_DIR = Path(__file__).parent


def create_safety_comparison():
    """Create safety comparison bar chart."""
    setup_pitch_style()

    # Load data
    loader = get_loader()
    data = loader.load_codon_optimization_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Extract protein data
    proteins = []
    safety_rates = []
    mean_margins = []
    functions = []

    for key, pdata in data.proteins.items():
        proteins.append(pdata['name'].split(' - ')[0].split('(')[0].strip())
        metrics = pdata['metrics']
        safety_rates.append(metrics['cit_safety_rate'] * 100)
        mean_margins.append(metrics['mean_margin'])
        functions.append(pdata['function'])

    # Sort by safety rate
    sorted_idx = np.argsort(safety_rates)[::-1]
    proteins = [proteins[i] for i in sorted_idx]
    safety_rates = [safety_rates[i] for i in sorted_idx]
    mean_margins = [mean_margins[i] for i in sorted_idx]
    functions = [functions[i] for i in sorted_idx]

    # Left: Safety rate bars
    ax1 = axes[0]
    colors = [PALETTE['safe'] if s == 100 else PALETTE['partial'] if s >= 50 else PALETTE['unsafe']
              for s in safety_rates]

    bars = ax1.barh(range(len(proteins)), safety_rates, color=colors, edgecolor='white', linewidth=2)

    ax1.set_yticks(range(len(proteins)))
    ax1.set_yticklabels([f"{p}\n({f})" for p, f in zip(proteins, functions)], fontsize=10)
    ax1.set_xlabel('Citrullination Safety Rate (%)', fontsize=12)
    ax1.set_title('Codon-Optimized Protein Safety', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.invert_yaxis()

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, safety_rates)):
        label = '100%' if rate == 100 else f'{rate:.0f}%'
        color = 'white' if rate > 50 else PALETTE['text']
        ax1.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2,
                 label, ha='right', va='center', fontsize=12,
                 fontweight='bold', color=color)

    # Safe zone marker
    ax1.axvline(x=100, color=PALETTE['safe'], linestyle='--', linewidth=2, alpha=0.5)

    # Right: Boundary margin comparison
    ax2 = axes[1]
    x = np.arange(len(proteins))
    bars2 = ax2.bar(x, mean_margins, color=[colors[i] for i in range(len(proteins))],
                    edgecolor='white', linewidth=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(proteins, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Mean Boundary Margin', fontsize=12)
    ax2.set_title('Distance from Cluster Boundaries', fontsize=16, fontweight='bold')

    # Add value labels
    for bar, margin in zip(bars2, mean_margins):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{margin:.2f}', ha='center', va='bottom', fontsize=10)

    # Key findings
    fig.text(0.5, 0.02,
             'Codon-optimized PRG4 and Collagen II achieve 100% citrullination safety for regenerative therapy',
             ha='center', fontsize=11, style='italic', color=PALETTE['text_light'])

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def create_safety_summary():
    """Create visual summary of safety findings."""
    setup_pitch_style()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'Regenerative Therapy Safety Profile',
            fontsize=20, fontweight='bold', ha='center', color=PALETTE['text'])

    # Three boxes for key proteins
    box_data = [
        {'name': 'PRG4 (Lubricin)', 'safety': '100%', 'function': 'Joint Lubrication',
         'color': PALETTE['safe'], 'x': 1.5},
        {'name': 'Collagen II', 'safety': '100%', 'function': 'Cartilage Structure',
         'color': PALETTE['safe'], 'x': 5},
        {'name': 'HAS2', 'safety': '67%', 'function': 'Hyaluronic Acid',
         'color': PALETTE['partial'], 'x': 8.5},
    ]

    for box in box_data:
        # Main circle
        circle = plt.Circle((box['x'], 3), 1.2, facecolor=box['color'],
                            edgecolor='white', linewidth=3, alpha=0.9)
        ax.add_patch(circle)

        # Safety percentage
        ax.text(box['x'], 3.2, box['safety'], fontsize=24, fontweight='bold',
                ha='center', va='center', color='white')

        # SAFE/PARTIAL label
        label = 'SAFE' if box['safety'] == '100%' else 'PARTIAL'
        ax.text(box['x'], 2.6, label, fontsize=10, ha='center', va='center',
                color='white', alpha=0.9)

        # Protein name below
        ax.text(box['x'], 1.3, box['name'], fontsize=12, fontweight='bold',
                ha='center', va='center', color=PALETTE['text'])
        ax.text(box['x'], 0.9, box['function'], fontsize=10,
                ha='center', va='center', color=PALETTE['text_light'])

    # Recommendation box
    from matplotlib.patches import FancyBboxPatch
    rec_box = FancyBboxPatch(
        (1, 0.1), 8, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#E8F5E9', edgecolor=PALETTE['safe'], linewidth=2
    )
    ax.add_patch(rec_box)
    ax.text(5, 0.35, '✓ PRG4 and Collagen II recommended for cell therapy applications',
            fontsize=11, ha='center', va='center', color=PALETTE['safe'])

    return fig


def main():
    fig1 = create_safety_comparison()
    save_figure(fig1, OUTPUT_DIR, 'safety_comparison')

    fig2 = create_safety_summary()
    save_figure(fig2, OUTPUT_DIR, 'safety_summary')

    print(f"Saved charts to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
