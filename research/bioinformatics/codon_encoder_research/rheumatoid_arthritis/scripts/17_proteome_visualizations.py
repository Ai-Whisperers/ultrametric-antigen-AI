#!/usr/bin/env python3
"""
Proteome-Wide Visualization Suite

Generate publication-quality figures for the proteome-wide
citrullination analysis.

Output directory: results/proteome_wide/17_visualizations/

Version: 1.0
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Plot settings
DPI = 300
FIGSIZE_SINGLE = (10, 8)
FIGSIZE_DOUBLE = (14, 6)
FIGSIZE_TRIPLE = (18, 6)

# Colors
COLORS = {
    'very_high': '#d32f2f',
    'high': '#f57c00',
    'moderate': '#fbc02d',
    'low': '#4caf50',
    'very_low': '#2196f3',
}

# Output configuration
SCRIPT_NUM = "17"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_visualizations"
PREDICTIONS_SUBDIR = "15_predictions"
ENRICHMENT_SUBDIR = "16_enrichment"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "proteome_wide" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_predictions_dir() -> Path:
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "proteome_wide" / PREDICTIONS_SUBDIR


def get_enrichment_dir() -> Path:
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "proteome_wide" / ENRICHMENT_SUBDIR


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_risk_distribution(predictions: List[Dict], output_dir: Path):
    """Plot distribution of immunogenic probabilities."""
    print("\n[1] Risk distribution histogram...")

    probs = [p['immunogenic_probability'] for p in predictions
             if p.get('immunogenic_probability') is not None]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Histogram
    ax = axes[0]
    ax.hist(probs, bins=50, color='#1976d2', alpha=0.7, edgecolor='white')
    ax.axvline(0.5, color='red', linestyle='--', lw=2, label='Threshold (0.5)')
    ax.axvline(0.75, color='orange', linestyle='--', lw=1.5, label='High risk (0.75)')
    ax.axvline(0.9, color='darkred', linestyle='--', lw=1.5, label='Very high (0.9)')
    ax.set_xlabel('Immunogenic Probability', fontsize=12)
    ax.set_ylabel('Number of Arginine Sites', fontsize=12)
    ax.set_title('Distribution of Predicted Immunogenicity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Risk category pie chart
    ax = axes[1]
    risk_counts = Counter(p['risk_category'] for p in predictions
                         if p.get('risk_category') != 'unknown')

    categories = ['very_high', 'high', 'moderate', 'low', 'very_low']
    counts = [risk_counts.get(c, 0) for c in categories]
    colors = [COLORS[c] for c in categories]
    labels = [f"{c.replace('_', ' ').title()}\n({cnt:,})" for c, cnt in zip(categories, counts)]

    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    ax.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: risk_distribution.png")


def plot_top_proteins(predictions: List[Dict], output_dir: Path):
    """Plot top proteins by immunogenic potential."""
    print("\n[2] Top proteins by risk...")

    # Aggregate by protein
    protein_data = {}
    for p in predictions:
        pid = p.get('protein_id')
        gene = p.get('gene_name', pid)
        prob = p.get('immunogenic_probability')

        if not pid or prob is None:
            continue

        if pid not in protein_data:
            protein_data[pid] = {'gene': gene, 'probs': [], 'high_count': 0}

        protein_data[pid]['probs'].append(prob)
        if p.get('risk_category') in ['very_high', 'high']:
            protein_data[pid]['high_count'] += 1

    # Calculate metrics
    for pid, data in protein_data.items():
        data['max_prob'] = max(data['probs'])
        data['mean_prob'] = np.mean(data['probs'])
        data['n_sites'] = len(data['probs'])

    # Sort by max probability
    sorted_proteins = sorted(protein_data.items(),
                            key=lambda x: x[1]['max_prob'], reverse=True)[:30]

    fig, ax = plt.subplots(figsize=(12, 10))

    genes = [d[1]['gene'][:15] for d in sorted_proteins]
    max_probs = [d[1]['max_prob'] for d in sorted_proteins]
    mean_probs = [d[1]['mean_prob'] for d in sorted_proteins]
    n_sites = [d[1]['n_sites'] for d in sorted_proteins]

    y_pos = np.arange(len(genes))

    # Horizontal bar chart
    bars = ax.barh(y_pos, max_probs, color='#d32f2f', alpha=0.8, label='Max probability')
    ax.barh(y_pos, mean_probs, color='#1976d2', alpha=0.6, label='Mean probability')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Immunogenic Probability', fontsize=12)
    ax.set_title('Top 30 Proteins by Maximum Immunogenic Probability', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(0.75, color='orange', linestyle='--', lw=1, alpha=0.7)
    ax.axvline(0.9, color='darkred', linestyle='--', lw=1, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='x')

    # Add site counts as text
    for i, (prob, n) in enumerate(zip(max_probs, n_sites)):
        ax.text(prob + 0.01, i, f'n={n}', va='center', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_proteins.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top_proteins.png")


def plot_feature_vs_probability(predictions: List[Dict], output_dir: Path):
    """Plot key features vs immunogenic probability."""
    print("\n[3] Feature correlation plots...")

    # Sample for plotting (too many points otherwise)
    sample = predictions[::10] if len(predictions) > 10000 else predictions

    features = ['centroid_shift', 'entropy_change', 'embedding_norm']
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRIPLE)

    for ax, feat in zip(axes, features):
        x = [p.get(feat, 0) for p in sample if p.get('immunogenic_probability') is not None]
        y = [p['immunogenic_probability'] for p in sample if p.get('immunogenic_probability') is not None]

        # Color by risk
        colors = []
        for p in sample:
            if p.get('immunogenic_probability') is None:
                continue
            risk = p.get('risk_category', 'unknown')
            colors.append(COLORS.get(risk, 'gray'))

        ax.scatter(x[:len(colors)], y[:len(colors)], c=colors, alpha=0.3, s=10)
        ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Immunogenic Probability', fontsize=12)
        ax.set_title(f'{feat.replace("_", " ").title()} vs Prediction', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: feature_correlations.png")


def plot_enrichment_dotplot(enrichment_results: Dict, output_dir: Path):
    """Plot enrichment results as dot plot."""
    print("\n[4] Enrichment dot plot...")

    # Collect significant results
    all_sig = []
    for category, results in enrichment_results.items():
        for r in results[:10]:  # Top 10 per category
            if r.get('p_value', 1) < 0.1:  # Relaxed threshold for visualization
                all_sig.append({
                    'term': r['term'][:40],  # Truncate long names
                    'category': category,
                    'fold_enrichment': r.get('fold_enrichment', 1),
                    'p_value': r['p_value'],
                    'n_hit': r.get('n_hit', 0),
                })

    if not all_sig:
        print("  No significant enrichments to plot")
        return

    # Sort by p-value
    all_sig.sort(key=lambda x: x['p_value'])
    all_sig = all_sig[:25]  # Top 25 overall

    fig, ax = plt.subplots(figsize=(12, 10))

    terms = [r['term'] for r in all_sig]
    fold = [r['fold_enrichment'] for r in all_sig]
    pvals = [-np.log10(r['p_value']) for r in all_sig]
    sizes = [r['n_hit'] * 5 for r in all_sig]

    # Color by category
    cat_colors = {
        'go_biological_process': '#4caf50',
        'go_cellular_component': '#2196f3',
        'subcellular_location': '#ff9800',
    }
    colors = [cat_colors.get(r['category'], 'gray') for r in all_sig]

    y_pos = np.arange(len(terms))

    scatter = ax.scatter(fold, y_pos, c=colors, s=sizes, alpha=0.7, edgecolors='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xlabel('Fold Enrichment', fontsize=12)
    ax.set_title('GO Term Enrichment in High-Risk Proteins', fontsize=14, fontweight='bold')
    ax.axvline(1, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Legend for categories
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                              markersize=10, label=cat.replace('_', ' ').title())
                      for cat, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'enrichment_dotplot.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: enrichment_dotplot.png")


def plot_position_analysis(predictions: List[Dict], output_dir: Path):
    """Analyze position of R within proteins."""
    print("\n[5] Position analysis...")

    # Get high vs low risk sites
    high_risk = [p for p in predictions if p.get('risk_category') in ['very_high', 'high']]
    low_risk = [p for p in predictions if p.get('risk_category') in ['very_low', 'low']]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Normalized position distribution
    ax = axes[0]
    for data, label, color in [(high_risk, 'High Risk', '#d32f2f'),
                                (low_risk, 'Low Risk', '#2196f3')]:
        # Calculate normalized position if not present
        positions = []
        for p in data:
            r_pos = p.get('r_position', 0)
            prot_len = p.get('protein_length', 1)
            if prot_len > 0:
                positions.append(r_pos / prot_len)

        if positions:
            ax.hist(positions, bins=20, alpha=0.5, label=label, color=color, density=True)

    ax.set_xlabel('Normalized Position in Protein (0=N-term, 1=C-term)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Position of Arginine Sites', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Probability vs protein length
    ax = axes[1]
    lengths = [p.get('protein_length', 0) for p in predictions
               if p.get('immunogenic_probability') is not None]
    probs = [p['immunogenic_probability'] for p in predictions
             if p.get('immunogenic_probability') is not None]

    # Bin by length
    bins = [0, 200, 400, 600, 800, 1000, 2000, 5000]
    bin_probs = []
    bin_labels = []

    for i in range(len(bins) - 1):
        mask = [(bins[i] <= l < bins[i+1]) for l in lengths]
        bin_vals = [p for p, m in zip(probs, mask) if m]
        if bin_vals:
            bin_probs.append(bin_vals)
            bin_labels.append(f"{bins[i]}-{bins[i+1]}")

    if bin_probs:
        bp = ax.boxplot(bin_probs, labels=bin_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#1976d2')
            patch.set_alpha(0.6)

    ax.set_xlabel('Protein Length (AA)', fontsize=12)
    ax.set_ylabel('Immunogenic Probability', fontsize=12)
    ax.set_title('Probability by Protein Length', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'position_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: position_analysis.png")


def plot_summary_dashboard(predictions: List[Dict], output_dir: Path):
    """Create a summary dashboard."""
    print("\n[6] Summary dashboard...")

    fig = plt.figure(figsize=(16, 12))

    # Statistics text
    valid_preds = [p for p in predictions if p.get('immunogenic_probability') is not None]
    probs = [p['immunogenic_probability'] for p in valid_preds]
    risk_counts = Counter(p['risk_category'] for p in valid_preds)
    n_proteins = len(set(p['protein_id'] for p in valid_preds))
    high_risk_proteins = len(set(p['protein_id'] for p in valid_preds
                                 if p.get('risk_category') in ['very_high', 'high']))

    # Top left: Key metrics
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    metrics_text = f"""
    PROTEOME-WIDE CITRULLINATION ANALYSIS
    =====================================

    Total Arginine Sites: {len(valid_preds):,}
    Total Proteins: {n_proteins:,}

    Risk Distribution:
    • Very High (>90%): {risk_counts.get('very_high', 0):,}
    • High (75-90%): {risk_counts.get('high', 0):,}
    • Moderate (50-75%): {risk_counts.get('moderate', 0):,}
    • Low (25-50%): {risk_counts.get('low', 0):,}
    • Very Low (<25%): {risk_counts.get('very_low', 0):,}

    Proteins with High-Risk Sites: {high_risk_proteins:,}

    Mean Probability: {np.mean(probs):.3f}
    Median Probability: {np.median(probs):.3f}
    """
    ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    # Top middle: Probability histogram
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(probs, bins=30, color='#1976d2', alpha=0.7, edgecolor='white')
    ax2.axvline(0.5, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Immunogenic Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability Distribution')
    ax2.grid(True, alpha=0.3)

    # Top right: Risk pie chart
    ax3 = fig.add_subplot(2, 3, 3)
    categories = ['very_high', 'high', 'moderate', 'low', 'very_low']
    counts = [risk_counts.get(c, 0) for c in categories]
    colors = [COLORS[c] for c in categories]
    ax3.pie(counts, labels=[c.replace('_', ' ').title() for c in categories],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Risk Categories')

    # Bottom: Top 15 proteins
    ax4 = fig.add_subplot(2, 1, 2)
    protein_max = {}
    for p in valid_preds:
        pid = p['protein_id']
        gene = p.get('gene_name', pid)
        prob = p['immunogenic_probability']
        if pid not in protein_max or prob > protein_max[pid][1]:
            protein_max[pid] = (gene, prob)

    top_15 = sorted(protein_max.items(), key=lambda x: x[1][1], reverse=True)[:15]
    genes = [d[1][0][:12] for d in top_15]
    probs_top = [d[1][1] for d in top_15]

    bars = ax4.barh(range(len(genes)), probs_top, color='#d32f2f', alpha=0.8)
    ax4.set_yticks(range(len(genes)))
    ax4.set_yticklabels(genes)
    ax4.invert_yaxis()
    ax4.set_xlabel('Maximum Immunogenic Probability')
    ax4.set_title('Top 15 Proteins by Immunogenic Risk')
    ax4.axvline(0.75, color='orange', linestyle='--', lw=1)
    ax4.axvline(0.9, color='darkred', linestyle='--', lw=1)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_dashboard.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PROTEOME-WIDE VISUALIZATIONS")
    print("Publication-quality figures for citrullination analysis")
    print("=" * 80)

    # Setup directories
    output_dir = get_output_dir()
    predictions_dir = get_predictions_dir()
    enrichment_dir = get_enrichment_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load predictions
    print("\n[0] Loading data...")
    pred_path = predictions_dir / "predictions_full.json"

    if not pred_path.exists():
        print(f"  ERROR: Predictions not found: {pred_path}")
        print("  Please run scripts 12-15 first")
        return

    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    print(f"  Loaded {len(predictions):,} predictions")

    # Load enrichment results if available
    enrichment_results = {}
    enrich_path = enrichment_dir / "enrichment_results.json"
    if enrich_path.exists():
        with open(enrich_path, 'r') as f:
            enrichment_results = json.load(f)
        print(f"  Loaded enrichment results")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_risk_distribution(predictions, output_dir)
    plot_top_proteins(predictions, output_dir)
    plot_feature_vs_probability(predictions, output_dir)
    plot_position_analysis(predictions, output_dir)
    plot_summary_dashboard(predictions, output_dir)

    if enrichment_results:
        plot_enrichment_dotplot(enrichment_results, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
