# Reproducibility Suite for P-adic DDG Benchmarks

**Doc-Type:** Reproducibility Guide · Version 1.0 · Updated 2026-01-03

---

## Purpose

This folder contains everything needed to independently verify the benchmark claims made in our documentation. Dr. Colbes (or any reviewer) can reproduce our results by running the scripts below.

---

## Quick Start

```bash
cd deliverables/partners/jose_colbes/reproducibility

# Step 1: Download S669 benchmark dataset
python download_s669.py

# Step 2: Run p-adic validation
python validate_padic_s669.py

# Step 3: Generate comparison report
python generate_benchmark_report.py

# View results
cat results/BENCHMARK_REPORT.md
```

---

## Directory Structure

```
reproducibility/
├── README.md                    # This file
├── download_s669.py             # Download S669 benchmark dataset
├── validate_padic_s669.py       # Run p-adic predictions on S669
├── generate_benchmark_report.py # Create markdown comparison report
├── data/
│   ├── s669.csv                 # Downloaded S669 dataset
│   └── s669.json                # Dataset metadata
└── results/
    ├── s669_validation_results.json  # Validation metrics
    ├── s669_predictions.json         # Per-mutation predictions
    └── BENCHMARK_REPORT.md           # Generated report
```

---

## What Each Script Does

### download_s669.py

Downloads the S669 benchmark dataset from the Bologna DDGEmb portal.

**S669 Details:**
- 669 single-point mutations
- 94 proteins with <25% sequence homology to training sets
- Experimental DDG values in kcal/mol
- Standard benchmark for fair method comparison

**Sources:**
- Primary: https://ddgemb.biocomp.unibo.it/datasets/
- Paper: https://academic.oup.com/bib/article/23/2/bbab555/6502552

### validate_padic_s669.py

Runs three p-adic prediction models on the S669 dataset:

1. **padic_radial** - Uses radial position in hyperbolic space
2. **padic_weighted** - Property-weighted p-adic distance
3. **padic_geodesic** - Hyperbolic geodesic distance

Computes:
- Pearson correlation (linear relationship)
- Spearman correlation (rank ordering)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

### generate_benchmark_report.py

Creates a formatted markdown report comparing our results against literature:

- Rosetta ddg_monomer: r = 0.69
- Rosetta cartesian_ddg: 59.1% accuracy
- FoldX: r = 0.48-0.69
- ELASPIC-2: r = 0.50 (2024)
- State-of-art 2024: r = 0.53-0.56

---

## Verifying Literature Claims

All literature benchmarks cited in our documentation can be verified:

| Claim | Source | How to Verify |
|-------|--------|---------------|
| Rosetta r = 0.69 | Official docs | [ddg-monomer docs](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/ddg-monomer) |
| FoldX r = 0.48 | ACS Omega 2020 | [Paper](https://pubs.acs.org/doi/10.1021/acsomega.9b04105) |
| S669 benchmark | Pancotti 2022 | [Paper](https://academic.oup.com/bib/article/23/2/bbab555/6502552) |
| ProThermDB | NAR 2021 | [Database](https://academic.oup.com/nar/article/49/D1/D420/5983626) |
| ProtDDG-Bench | Public resource | [Website](https://protddg-bench.github.io/s2648/) |

---

## Expected Results

Based on our internal validation (N=65), expected S669 performance:

| Model | Expected Spearman r | Status |
|-------|---------------------|--------|
| padic_radial | 0.40-0.50 | Competitive |
| padic_weighted | 0.42-0.52 | Best p-adic |
| padic_geodesic | 0.30-0.40 | Baseline |

**Note:** S669 is a stringent benchmark. Correlations are typically lower than on training-similar datasets.

---

## Troubleshooting

### Dataset download fails

If the Bologna portal is unavailable, the script creates a fallback dataset from literature values. For the full S669, download manually from the paper's supplementary materials.

### scipy not installed

The validation script will use numpy fallback for correlations. Install scipy for p-values:

```bash
pip install scipy
```

### Results differ from expected

Small variations (±0.05 in correlation) are expected due to:
- Different random seeds
- Floating point precision
- Subset selection if data is partial

---

## Contact

For questions about reproducing these results:
- See main documentation: `../docs/BENCHMARK_COMPARISON.md`
- Project repo: https://github.com/Ai-Whisperers/ternary-vaes

---

*Part of the Ternary VAE Bioinformatics Partnership - Jose Colbes Deliverables*
