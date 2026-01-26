# Rojas Package: JSON Schema Catalog

**Doc-Type:** Schema Analysis · Version 1.0 · 2026-01-05 · AI Whisperers

---

## Overview

27 JSON files identified across 4 directories. This document catalogs schemas, identifies overlap, and proposes consolidation.

---

## File Inventory

### data/ (6 files) - Source Data

| File | Type | Size | Purpose |
|------|------|------|---------|
| `cache/denv4_sequences.json` | dict[accession→sequence] | 270 entries | Raw genome sequences |
| `cache/denv4_metadata.json` | dict[accession→metadata] | 270 entries | Per-genome metadata |
| `dengue_metadata.json` | dict[serotype→stats] | 4 entries | Serotype-level stats |
| `dengue_strains.json` | dict | Large | Strain data |
| `denv4/accession_order.json` | list[accession] | 270 items | Ordered accession list |
| `refseq_genomes.json` | dict[serotype→sequence] | 4 entries | Reference genomes |

### results/ (11 files) - Analysis Outputs

| File | Keys | Indexed By |
|------|------|------------|
| `padic_integration/padic_integration_results.json` | timestamp, parameters, region_analysis, genome_scan_summary, top_primer_candidates, genome_scan | position (36 windows) |
| `phylogenetic/phylogenetic_analysis_results.json` | timestamp, pipeline_steps | pipeline step |
| `phylogenetic/per_clade_conservation.json` | Clade_A..E | clade name |
| `phylogenetic/subclade_analysis_results.json` | timestamp, parameters, optimal_k, k_analysis, subclades, primer_suitable_clades, primer_candidates, coverage | subclade name |
| `phylogenetic/degenerate_primer_results.json` | timestamp, n_sequences, candidate_windows, designed_primers, validation, multiplex_cocktail | window position |
| `phylogenetic/clade_representatives.json` | Clade_A..E→accession | clade name |
| `phylogenetic/geographic_distribution.json` | clade→country→count | clade, country |
| `phylogenetic/temporal_distribution.json` | clade→year_stats | clade |
| `tiered_detection/tiered_detection_results.json` | timestamp, tier1, tier2, tier3, validation | tier |
| `pan_arbovirus_primers/library_summary.json` | targets, statistics, cross_reactivity_matrix, design_parameters | virus |
| `dengue_forecast.json` | - | - |

### results/conjecture/ (4 files) - Hypothesis Testing

| File | Keys | Common Schema |
|------|------|---------------|
| `synonymous_conjecture_results.json` | timestamp, n_sequences, predictions, correlations, gene_analysis, **position_data**, **synonymous_data**, summary | Yes |
| `revised_conjecture_results.json` | timestamp, n_sequences, predictions, correlations, **gene_stats**, summary | Yes |
| `codon_bias_conjecture_results.json` | timestamp, n_sequences, predictions, correlations, **gene_stats**, summary | Yes |
| `codon_pair_conjecture_results.json` | timestamp, n_sequences, predictions, correlations, **gene_stats**, summary, cpb_table_size, **primer_recommendations** | Yes |

### validation/ (7 files) - Validation Results

| File | Keys |
|------|------|
| `cdc_recovery_report.json` | timestamp, primers_tested, recovery_rate, gene_corrections |
| `dengue_strain_variation_report.json` | timestamp, serotypes_analyzed, entropy_comparison |
| `population_hypothesis_results.json` | timestamp, hypothesis, overall_result, tests_falsified, results, entropy_data |
| `alternative_hypotheses_results.json` | timestamp, hypotheses_tested, falsified, supported, results |
| `padic_conservation_results.json` | timestamp, n_primers, overall_entropy_radius_correlation, primer_analyses |
| `ns5_deep_analysis_results.json` | timestamp, proteins_analyzed, domain_variability, motif_variability, fidelity_residues |
| `evolutionary_divergence_results.json` | timestamp, within_serotype_divergence, denv4_within_identity |

---

## Overlap Analysis

### Position-Based Data (36 positions, 300bp step)

**DUPLICATED across 5 files:**

| File | Field | Data Per Position |
|------|-------|-------------------|
| `padic_integration_results.json` | genome_scan | position, variance, n_sequences |
| `synonymous_conjecture_results.json` | position_data | position, hyp_variance, pos1/2/3_entropy, wobble_ratio |
| `synonymous_conjecture_results.json` | synonymous_data | position, hyp_variance, synonymous, non_synonymous, syn_ratio |
| `codon_bias_conjecture_results.json` | (computed internally) | mean_codon_entropy, mean_consistency |
| `codon_pair_conjecture_results.json` | (computed internally) | mean_cpb, pair_entropy, consistency |
| `revised_conjecture_results.json` | (computed internally) | mean_delta_hydro, mean_euclidean |

**Common key:** `position` (0, 300, 600, ..., 10500)

### Gene-Level Stats (4 genes: E, NS1, NS3, NS5)

**DUPLICATED across 4 files:**

| File | Fields per Gene |
|------|-----------------|
| `synonymous_conjecture` | gene_analysis: mean_syn_ratio, mean_hyp_var, n_windows |
| `codon_bias_conjecture` | gene_stats: mean_hyp_var, mean_codon_entropy, mean_consistency |
| `codon_pair_conjecture` | gene_stats: mean_hyp_var, mean_cpb, mean_pair_entropy, mean_consistency, n_windows |
| `revised_conjecture` | gene_stats: mean_delta_hydro, mean_euclidean, mean_hyp_var, n_windows |

**Observation:** `mean_hyp_var` duplicated in ALL 4 files.

### Clade-Level Data (5 clades: A, B, C, D, E)

**Scattered across 4 files:**

| File | Clade Data |
|------|------------|
| `phylogenetic_analysis_results.json` | pipeline_steps[clustering]: sizes, identities |
| `per_clade_conservation.json` | n_sequences, mean_entropy, conserved_windows |
| `subclade_analysis_results.json` | subclades: size, mean_entropy, has_primers |
| `clade_representatives.json` | accession, country, year |

### Correlation Results (Spearman tests)

**Scattered across 5 files:**

| File | Correlations Tested |
|------|---------------------|
| `synonymous_conjecture` | hyp_var_vs_syn_ratio |
| `codon_bias_conjecture` | hyp_var_vs_codon_entropy, hyp_var_vs_within_aa_diversity, hyp_var_vs_consistency |
| `codon_pair_conjecture` | hyp_var_vs_mean_cpb, hyp_var_vs_pair_entropy, hyp_var_vs_consistency |
| `revised_conjecture` | hyp_var_vs_hydrophobicity, hyp_var_vs_euclidean |
| `padic_conservation_results` | entropy_radius_correlation, entropy_valuation_correlation |

---

## Redundancy Summary

| Data Type | Occurrences | Files |
|-----------|-------------|-------|
| Position-level hyp_var | 5x | padic, syn, bias, pair, revised |
| Gene-level hyp_var | 4x | syn, bias, pair, revised |
| Clade sizes | 3x | phylo, per_clade, subclade |
| Clade entropy | 2x | per_clade, subclade |
| Correlation p-values | 10x | scattered |

---

## Proposed ML-Ready Consolidation

### 1. `ml_ready/positions.json` - Per-Position Features

```json
{
  "_metadata": {
    "version": "1.0",
    "created": "2026-01-05",
    "source_files": ["padic_integration", "synonymous_conjecture", ...],
    "n_positions": 36,
    "window_size": 75,
    "step_size": 300
  },
  "positions": [
    {
      "position": 0,
      "gene": "5UTR",
      "hyp_variance": 0.0252,
      "shannon_entropy": {"pos1": 1.41, "pos2": 1.38, "pos3": 1.41, "mean": 1.40},
      "synonymous": {"count": 123, "ratio": 0.039},
      "codon_bias": {"entropy": 3.23, "consistency": 0.33},
      "codon_pair": {"mean_cpb": 0.42, "entropy": 3.80, "consistency": 0.52},
      "aa_properties": {"delta_hydro": 3.29, "euclidean": 77.7}
    }
  ]
}
```

### 2. `ml_ready/clades.json` - Per-Clade Features

```json
{
  "_metadata": {
    "version": "1.0",
    "source_files": ["phylogenetic_analysis", "per_clade_conservation", "subclade_analysis"]
  },
  "clades": {
    "Clade_E": {
      "size": 211,
      "within_identity": 0.89,
      "mean_entropy": 1.58,
      "year_range": [1976, 2023],
      "has_conserved_windows": false,
      "subclades": ["E.1.3.2.1", "E.1.3.2.3", "E.2.2.2", "E.3.2"],
      "representative_accession": "EU854299"
    }
  }
}
```

### 3. `ml_ready/correlations.json` - All Statistical Tests

```json
{
  "_metadata": {
    "version": "1.0",
    "source_files": ["all conjecture files"]
  },
  "correlations": [
    {
      "test_name": "hyp_var_vs_syn_ratio",
      "source": "synonymous_conjecture",
      "spearman_rho": 0.03,
      "p_value": 0.86,
      "confirmed": false,
      "interpretation": "No correlation"
    }
  ]
}
```

### 4. `ml_ready/primer_candidates.json` - Unified Primer Data

```json
{
  "_metadata": {
    "version": "1.0",
    "source_files": ["padic_integration", "tiered_detection", "subclade_analysis"]
  },
  "candidates": [
    {
      "position": 2400,
      "gene": "E",
      "source": "padic_integration",
      "hyp_variance": 0.0183,
      "rank_by_hyp_var": 1,
      "sequence": null,
      "validated": false
    }
  ]
}
```

---

## Consolidation Rules

1. **No duplication:** Each data point appears in exactly one ML-ready file
2. **Source tracking:** `_metadata.source_files` lists origin files
3. **Version control:** `_metadata.version` for schema changes
4. **Null for missing:** Use `null` not omission for missing fields
5. **Consistent keys:** All files use snake_case

---

*Schema analysis complete: 2026-01-05*
