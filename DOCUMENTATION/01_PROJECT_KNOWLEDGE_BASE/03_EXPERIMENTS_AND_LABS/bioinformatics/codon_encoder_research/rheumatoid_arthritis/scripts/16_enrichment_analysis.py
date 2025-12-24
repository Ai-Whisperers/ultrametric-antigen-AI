#!/usr/bin/env python3
"""
Enrichment Analysis for High-Risk Citrullination Sites

Analyze which GO terms, pathways, and cellular locations are enriched
among high-risk arginine sites compared to the background proteome.

Output directory: results/proteome_wide/16_enrichment/

Version: 1.0
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter, defaultdict
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Enrichment parameters
MIN_TERM_SIZE = 10  # Minimum genes per GO term
MAX_TERM_SIZE = 500  # Maximum genes per GO term
FDR_THRESHOLD = 0.05  # Benjamini-Hochberg threshold

# Output configuration
SCRIPT_NUM = "16"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_enrichment"
INPUT_SUBDIR = "15_predictions"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "proteome_wide" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_input_dir() -> Path:
    """Get input directory (from previous script)."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "proteome_wide" / INPUT_SUBDIR


# ============================================================================
# GO TERM PARSING
# ============================================================================

def parse_go_terms(go_string: str) -> List[str]:
    """Parse GO terms from UniProt annotation string."""
    if not go_string or pd.isna(go_string):
        return []

    terms = []
    # GO terms are typically formatted as "term1 [GO:xxxxx]; term2 [GO:xxxxx]"
    parts = go_string.replace(']', '').split('[GO:')

    for part in parts[1:]:  # Skip first empty part
        go_id = part.split(';')[0].split()[0].strip()
        if go_id:
            terms.append(f"GO:{go_id}")

    # Also capture term names
    for part in go_string.split(';'):
        part = part.strip()
        if part and '[GO:' not in part:
            terms.append(part)

    return list(set(terms))


def parse_subcellular_location(loc_string: str) -> List[str]:
    """Parse subcellular locations from UniProt annotation."""
    if not loc_string or pd.isna(loc_string):
        return []

    locations = []

    # Common locations to extract
    keywords = [
        'Nucleus', 'Cytoplasm', 'Membrane', 'Mitochondrion', 'Endoplasmic reticulum',
        'Golgi', 'Lysosome', 'Peroxisome', 'Extracellular', 'Secreted',
        'Cell membrane', 'Plasma membrane', 'Cell junction', 'Cytoskeleton'
    ]

    loc_lower = loc_string.lower()
    for kw in keywords:
        if kw.lower() in loc_lower:
            locations.append(kw)

    return locations


# ============================================================================
# ENRICHMENT ANALYSIS
# ============================================================================

def fisher_exact_test(n_hit_in_query: int, n_query: int,
                      n_hit_in_background: int, n_background: int) -> tuple:
    """
    Perform Fisher's exact test for enrichment.

    Returns (odds_ratio, p_value)
    """
    # Contingency table:
    #                 Has term    No term
    # Query set:      a           b
    # Background:     c           d

    a = n_hit_in_query
    b = n_query - n_hit_in_query
    c = n_hit_in_background - n_hit_in_query
    d = n_background - n_query - c

    # Ensure non-negative
    c = max(0, c)
    d = max(0, d)

    contingency = [[a, b], [c, d]]

    try:
        odds_ratio, p_value = stats.fisher_exact(contingency, alternative='greater')
    except Exception:
        odds_ratio, p_value = 1.0, 1.0

    return odds_ratio, p_value


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[i] = p * n / (i + 1)

    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)

    # Reorder to original indices
    result = np.zeros(n)
    result[sorted_indices] = adjusted

    return result.tolist()


def run_enrichment(high_risk_proteins: Set[str],
                   all_proteins: Set[str],
                   protein_to_terms: Dict[str, List[str]],
                   term_name: str = "GO") -> List[Dict]:
    """
    Run enrichment analysis for a set of terms.

    Args:
        high_risk_proteins: Set of protein IDs with high-risk R sites
        all_proteins: Set of all protein IDs
        protein_to_terms: Dict mapping protein ID to list of terms
        term_name: Name for the term type (e.g., "GO_BP", "Location")

    Returns:
        List of enrichment results
    """
    # Build term to proteins mapping
    term_to_proteins = defaultdict(set)
    for protein, terms in protein_to_terms.items():
        for term in terms:
            term_to_proteins[term].add(protein)

    # Filter by size
    valid_terms = {
        term: proteins for term, proteins in term_to_proteins.items()
        if MIN_TERM_SIZE <= len(proteins) <= MAX_TERM_SIZE
    }

    results = []
    n_query = len(high_risk_proteins)
    n_background = len(all_proteins)

    for term, term_proteins in valid_terms.items():
        n_hit_in_query = len(high_risk_proteins & term_proteins)
        n_hit_in_background = len(term_proteins)

        if n_hit_in_query < 2:  # Require at least 2 hits
            continue

        odds_ratio, p_value = fisher_exact_test(
            n_hit_in_query, n_query,
            n_hit_in_background, n_background
        )

        results.append({
            'term': term,
            'term_type': term_name,
            'n_hit': n_hit_in_query,
            'n_query': n_query,
            'n_term': n_hit_in_background,
            'n_background': n_background,
            'fold_enrichment': (n_hit_in_query / n_query) / (n_hit_in_background / n_background)
                               if n_hit_in_background > 0 else 0,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
        })

    # Apply FDR correction
    if results:
        p_values = [r['p_value'] for r in results]
        fdr_values = benjamini_hochberg(p_values)
        for r, fdr in zip(results, fdr_values):
            r['fdr'] = fdr
            r['significant'] = fdr < FDR_THRESHOLD

    # Sort by p-value
    results.sort(key=lambda x: x['p_value'])

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_enrichments(predictions: List[Dict], output_dir: Path) -> Dict:
    """Run all enrichment analyses."""
    print("\n[1] Preparing data for enrichment analysis...")

    # Build protein-level data
    protein_data = defaultdict(lambda: {
        'go_bp': set(), 'go_cc': set(), 'go_mf': set(),
        'locations': set(), 'max_prob': 0, 'has_high_risk': False
    })

    for pred in predictions:
        pid = pred.get('protein_id')
        if not pid:
            continue

        prob = pred.get('immunogenic_probability', 0)
        if prob is None:
            continue

        protein_data[pid]['max_prob'] = max(protein_data[pid]['max_prob'], prob)

        if pred.get('risk_category') in ['very_high', 'high']:
            protein_data[pid]['has_high_risk'] = True

        # Parse GO terms
        for term in parse_go_terms(pred.get('go_bp', '')):
            protein_data[pid]['go_bp'].add(term)
        for term in parse_go_terms(pred.get('go_cc', '')):
            protein_data[pid]['go_cc'].add(term)

        # Parse locations
        for loc in parse_subcellular_location(pred.get('subcellular_location', '')):
            protein_data[pid]['locations'].add(loc)

    all_proteins = set(protein_data.keys())
    high_risk_proteins = {pid for pid, data in protein_data.items() if data['has_high_risk']}

    print(f"  Total proteins: {len(all_proteins):,}")
    print(f"  High-risk proteins: {len(high_risk_proteins):,}")

    # Run enrichments
    all_results = {}

    # GO Biological Process
    print("\n[2] GO Biological Process enrichment...")
    protein_to_bp = {pid: list(data['go_bp']) for pid, data in protein_data.items()}
    bp_results = run_enrichment(high_risk_proteins, all_proteins, protein_to_bp, "GO_BP")
    all_results['go_biological_process'] = bp_results
    sig_bp = sum(1 for r in bp_results if r.get('significant'))
    print(f"  Tested: {len(bp_results)}, Significant (FDR<0.05): {sig_bp}")

    # GO Cellular Component
    print("\n[3] GO Cellular Component enrichment...")
    protein_to_cc = {pid: list(data['go_cc']) for pid, data in protein_data.items()}
    cc_results = run_enrichment(high_risk_proteins, all_proteins, protein_to_cc, "GO_CC")
    all_results['go_cellular_component'] = cc_results
    sig_cc = sum(1 for r in cc_results if r.get('significant'))
    print(f"  Tested: {len(cc_results)}, Significant (FDR<0.05): {sig_cc}")

    # Subcellular Location
    print("\n[4] Subcellular location enrichment...")
    protein_to_loc = {pid: list(data['locations']) for pid, data in protein_data.items()}
    loc_results = run_enrichment(high_risk_proteins, all_proteins, protein_to_loc, "Location")
    all_results['subcellular_location'] = loc_results
    sig_loc = sum(1 for r in loc_results if r.get('significant'))
    print(f"  Tested: {len(loc_results)}, Significant (FDR<0.05): {sig_loc}")

    return all_results


def save_enrichment_results(results: Dict, output_dir: Path):
    """Save enrichment results."""
    print("\n[5] Saving enrichment results...")

    # Save full results as JSON
    json_path = output_dir / "enrichment_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save significant results as CSV
    all_significant = []
    for category, cat_results in results.items():
        for r in cat_results:
            if r.get('significant'):
                r['category'] = category
                all_significant.append(r)

    if all_significant:
        sig_df = pd.DataFrame(all_significant)
        sig_df = sig_df.sort_values('p_value')
        sig_path = output_dir / "significant_enrichments.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"  Saved: {sig_path} ({len(sig_df)} significant terms)")

    # Save top results per category
    for category, cat_results in results.items():
        if cat_results:
            cat_df = pd.DataFrame(cat_results[:50])  # Top 50
            cat_path = output_dir / f"top_{category}.csv"
            cat_df.to_csv(cat_path, index=False)
            print(f"  Saved: {cat_path}")


def generate_summary(results: Dict, output_dir: Path):
    """Generate summary statistics."""
    print("\n[6] Generating summary...")

    summary = {
        'total_terms_tested': sum(len(r) for r in results.values()),
        'significant_terms': sum(
            sum(1 for t in cat_results if t.get('significant'))
            for cat_results in results.values()
        ),
        'by_category': {}
    }

    for category, cat_results in results.items():
        sig = [r for r in cat_results if r.get('significant')]
        summary['by_category'][category] = {
            'tested': len(cat_results),
            'significant': len(sig),
            'top_terms': [r['term'] for r in sig[:5]]
        }

    summary_path = output_dir / "enrichment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Print summary
    print("\n  Enrichment Summary:")
    print(f"    Total terms tested: {summary['total_terms_tested']}")
    print(f"    Significant (FDR<0.05): {summary['significant_terms']}")

    for category, cat_summary in summary['by_category'].items():
        print(f"\n    {category}:")
        print(f"      Tested: {cat_summary['tested']}, Significant: {cat_summary['significant']}")
        if cat_summary['top_terms']:
            print(f"      Top terms: {', '.join(cat_summary['top_terms'][:3])}")

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ENRICHMENT ANALYSIS")
    print("GO term and pathway enrichment for high-risk citrullination sites")
    print("=" * 80)

    # Setup directories
    input_dir = get_input_dir()
    output_dir = get_output_dir()
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load predictions
    print("\n[0] Loading predictions...")
    pred_path = input_dir / "predictions_full.json"

    if not pred_path.exists():
        print(f"  ERROR: Predictions file not found: {pred_path}")
        print("  Please run script 15_predict_immunogenicity.py first")
        return

    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    print(f"  Loaded {len(predictions):,} predictions")

    # Run enrichment analysis
    results = analyze_enrichments(predictions, output_dir)

    # Save results
    save_enrichment_results(results, output_dir)

    # Generate summary
    summary = generate_summary(results, output_dir)

    print("\n" + "=" * 80)
    print("ENRICHMENT ANALYSIS COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")

    return results


if __name__ == '__main__':
    main()
