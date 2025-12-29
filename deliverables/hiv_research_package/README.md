# HIV Research Package

**Project:** Ternary VAE Bioinformatics
**Status:** Comprehensive Clinical & Research Analysis

---

## Overview

This package contains the core outputs of our **HIV Ternary VAE Platform**, which uses p-adic geometry and protein language models (ESM-2) to analyze over 200,000 HIV sequences.

## Contents

### 1. Documentation

- `docs/COMPLETE_PLATFORM_ANALYSIS.md`: The definitive guide to our findings, models, and clinical applications. **Start here.**

### 2. Analysis Scripts

These scripts reproduce our key findings:

- `scripts/run_complete_analysis.py`: Main entry point for the analysis pipeline.
- `scripts/02_hiv_drug_resistance.py`: Analysis of resistance patterns across 23 drugs.
- `scripts/07_validate_all_conjectures.py`: Mathematical proof of our 7 key biological conjectures (including the "Integrase Vulnerability").
- `scripts/analyze_stanford_resistance.py`: Interface with the Stanford HIVdb data.

## Key Findings

1. **Integrase Vulnerability**: The Pol_IN protein is the most geometrically isolated, making it a prime target for novel drugs with low resistance potential.
2. **Hiding Hierarchy**: HIV uses a 5-level strategy to hide its codon usage, which we have mapped completely.
3. **Vaccine Targets**: We have identified **328 resistance-free vaccine targets** that are geometrically constrained.

## Usage

To run the analysis:

```bash
cd scripts
python run_complete_analysis.py
```

## Citation

If you use this data, please cite the **Ternary VAE Project**.
