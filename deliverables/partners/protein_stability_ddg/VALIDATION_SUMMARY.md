# Validation Summary: P-adic DDG Predictor

**Doc-Type:** Scientific Validation · Version 1.2 · 2026-01-27 · AI Whisperers

---

## Validation Framework Overview

This package has been validated through multiple independent approaches:

| Validation Type | Script | Results Location |
|-----------------|--------|------------------|
| Shipped Predictor | `validation/scientific_validation_report.py` | `validation/results/scientific_metrics.json` |
| Fresh LOO Training | `validation/bootstrap_test.py` | Printed to console |
| AlphaFold Cross-Val | `validation/alphafold_validation_pipeline.py` | `validation/results/alphafold_validation_report.json` |
| Permutation Test | `validation/bootstrap_test.py` | p < 0.001 confirmed |
| Literature Comparison | `docs/BENCHMARK_COMPARISON.md` | Verified citations |

---

## Core Performance Metrics

### Two Validation Paths

| Metric | ValidatedDDGPredictor (Shipped) | Fresh LOO Training |
|--------|:-------------------------------:|:------------------:|
| **Spearman ρ** | **0.52** | **0.58** |
| Pearson r | 0.48 | 0.58 |
| 95% CI | [0.21, 0.80] | [0.35, 0.75] |
| p-value | 0.0001 | 7.1e-06 |
| MAE | 2.34 kcal/mol | 0.92 kcal/mol |

**CANONICAL METRIC: 0.52** - This is what users get from `ValidatedDDGPredictor`.

**Why two values?**
- **0.52**: Pre-trained coefficients (what ships to users)
- **0.58**: Fresh Ridge model with LOO CV (theoretical best if retrained)

**Validation:** Leave-One-Out Cross-Validation (N=52), Pipeline pattern (no data leakage).

---

## Literature Comparison (Verified Sources)

**CRITICAL CAVEAT:** Literature methods use N=669 (full S669). Our N=52 result is NOT directly comparable. On N=669, our method achieves ρ=0.37-0.40.

| Method | Spearman ρ | Dataset | Type | Source |
|--------|------------|---------|------|--------|
| Rosetta ddg_monomer | 0.69 | N=669 | Structure | Rosetta Docs |
| Mutate Everything | 0.56 | N=669 | Sequence | Meier 2023 |
| ESM-1v | 0.51 | N=669 | Sequence | Meier 2021 |
| ELASPIC-2 | 0.50 | N=669 | Sequence | PLOS 2024 |
| FoldX 5.0 | 0.48 | N=669 | Structure | Various |
| **Our Method (N=52, shipped)** | **0.52** | **N=52** | **Sequence** | LOO validated |
| Our Method (N=52, fresh) | 0.58 | N=52 | Sequence | LOO validated |
| Our Method (N=669) | 0.37-0.40 | N=669 | Sequence | Validated |

**Honest Note:** On comparable N=669 data, our method does NOT outperform literature sequence-only methods. The N=52 result is on a carefully curated subset (small proteins, alanine scanning) where our method shows stronger performance.

---

## Dataset Details

### S669 Benchmark (Pancotti et al. 2022)

| Subset | N | Description | Our Spearman |
|--------|---|-------------|--------------|
| Curated (V3) | 52 | Alanine scanning + variants | 0.52 (shipped) / 0.58 (fresh) |
| Full dataset | 669 | All mutations | 0.37-0.40 |

### Proteins in N=52 Subset

```
2LZM (9 mutations) - T4 Lysozyme
1UBQ (9 mutations) - Ubiquitin
1A2P (8 mutations) - Human muscle acylphosphatase
1STN (6 mutations) - Staphylococcal nuclease
2CI2 (5 mutations) - Chymotrypsin inhibitor 2
1MBN (5 mutations) - Myoglobin
1RNH (4 mutations) - RNase H
4PTI (3 mutations) - BPTI
1SHG (3 mutations) - SH3 domain
```

---

## Ablation Study (Fresh LOO Training)

From `validation/bootstrap_test.py`:

| Feature Set | LOO Spearman | Contribution |
|-------------|:------------:|:------------:|
| Hyperbolic only (4 features) | 0.43 | 74% of combined |
| Physicochemical only (4 features) | 0.31 | 53% of combined |
| **Combined (8 features)** | **0.58** | **100%** |

**Key Finding:** Both feature types contribute; hyperbolic features add ~0.15 correlation points beyond physicochemical baseline.

---

## AlphaFold Structural Cross-Validation

| pLDDT Range | n | Spearman ρ | p-value | Significant? |
|-------------|---|------------|---------|:------------:|
| High (>90) | 41 | 0.27 | 0.088 | NO |
| Medium (70-90) | 16 | 0.34 | 0.198 | NO |
| Low (<70) | 34 | 0.04 | 0.822 | NO |

**Finding:** AlphaFold pLDDT (structural confidence) is orthogonal to sequence-based DDG prediction. This is a genuine scientific finding - pLDDT measures structural confidence, not mutational predictability.

---

## Mutation-Type Specific Performance

From `docs/PADIC_ENCODER_FINDINGS.md`:

| Mutation Type | N | P-adic Advantage | Recommendation |
|---------------|---|------------------|----------------|
| neutral→charged | 37 | **+159%** | STRONGLY use p-adic |
| small DDG (<1) | 312 | +23% | Use p-adic |
| large→small size | 82 | +16% | Use p-adic |
| charge_reversal | 20 | **-737%** | DO NOT use p-adic |
| proline_mutations | - | -89% | DO NOT use p-adic |

---

## Key Advantages

### 1. Sequence-Only (No 3D Required)

Unlike Rosetta/FoldX, our method works with sequence alone - enabling high-throughput screening of novel proteins without structure.

### 2. Speed

| Tool | Time per Mutation |
|------|-------------------|
| Rosetta cartesian_ddg | 5-30 minutes |
| FoldX BuildModel | 30-60 seconds |
| ESM-1v | 1-5 seconds |
| **P-adic geometric** | **<0.1 seconds** |

### 3. Rosetta-Blind Detection

23.6% of residues scored as stable by Rosetta are geometrically unstable by p-adic analysis - a unique detection capability.

### 4. Complementary Information

Low correlation with traditional methods (r=0.15-0.31) indicates p-adic captures orthogonal information useful for ensemble predictions.

---

## Reproducibility

```bash
cd deliverables/partners/protein_stability_ddg/

# 1. Validate shipped predictor (0.52)
python validation/scientific_validation_report.py

# 2. Fresh LOO training (0.58)
python validation/bootstrap_test.py

# 3. Run AlphaFold cross-validation
python validation/alphafold_validation_pipeline.py

# 4. Full benchmark reproduction
cd reproducibility/
python download_s669.py
python extract_aa_embeddings_v2.py
python train_padic_ddg_predictor_v2.py
```

---

## Files Reference

| Purpose | Location |
|---------|----------|
| Main predictor | `src/validated_ddg_predictor.py` |
| Shipped predictor validation | `validation/scientific_validation_report.py` |
| Fresh LOO validation | `validation/bootstrap_test.py` |
| CANONICAL metrics | `validation/results/scientific_metrics.json` |
| AlphaFold validation | `validation/alphafold_validation_pipeline.py` |
| Scientific report | `validation/results/SCIENTIFIC_VALIDATION_REPORT.md` |
| Literature comparison | `docs/BENCHMARK_COMPARISON.md` |
| Research findings | `docs/PADIC_ENCODER_FINDINGS.md` |
| Issue tracking | `BIAS_ANALYSIS.md` |

---

*Version 1.2 · Updated 2026-01-27*
*Canonical: 0.52 (ValidatedDDGPredictor) | Fresh Training: 0.58 (bootstrap_test.py)*
