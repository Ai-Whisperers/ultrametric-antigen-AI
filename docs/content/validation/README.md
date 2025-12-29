# Validation

> **Benchmarks, test results, and validation methodology.**

---

## Overview

Validation follows a multi-level approach:

1. **Unit tests**: Component correctness
2. **Integration tests**: Pipeline functionality
3. **Scientific validation**: Biological accuracy
4. **Clinical validation**: Real-world utility

---

## Test Suite

### Summary

| Category | Tests | Pass Rate |
|:---------|:------|:----------|
| Unit tests | 198 | 97.5% |
| Integration tests | 33 | 97.0% |
| **Total** | **231** | **97.4%** |

### By Module

| Module | Tests | Status |
|:-------|:------|:-------|
| BaseVAE | 33 | Pass |
| Epistasis | 32 | Pass |
| Structure-Aware | 35 | Pass |
| Epistasis Loss | 29 | Pass |
| Uncertainty | 21 | Pass |
| Transfer Learning | 30 | Pass |
| AlphaFold Encoder | 18 | Pass |
| Full Pipeline | 33 | Pass |

---

## Scientific Benchmarks

### HIV Drug Resistance

| Metric | Value | Standard |
|:-------|:------|:---------|
| Mean Spearman | 0.890 | >0.8 (clinical) |
| Best drug (3TC) | 0.981 | - |
| Worst drug | 0.780 | - |
| Known mutations | 65-70% F1 | - |

### Cross-Disease Transfer

| Source → Target | Improvement |
|:----------------|:------------|
| HIV → HBV | +15% |
| HIV → TB | +8% |
| Flu → COVID | +12% |

### Geometry Validation

| Test | Result |
|:-----|:-------|
| P-adic vs Hamming | r = 0.8339 |
| Hyperbolic embedding | Validated |
| Tree structure | Captured |

---

## Validation Methodology

### Drug Resistance

1. **Data**: Stanford HIVDB (200K+ sequences)
2. **Split**: 80/10/10 train/val/test
3. **Metric**: Spearman correlation with phenotype
4. **Benchmark**: Compare to HIVDB algorithm

### Uncertainty Calibration

1. **Method**: Temperature scaling
2. **Metric**: Expected Calibration Error (ECE)
3. **Target**: ECE < 0.05

### Transfer Learning

1. **Method**: Pre-train on HIV, fine-tune on target
2. **Metric**: Improvement over baseline
3. **Target**: >10% improvement on low-data diseases

---

## Running Validation

```bash
# Run full validation suite
python scripts/experiments/run_full_validation.py

# Run specific benchmark
pytest tests/integration/test_full_pipeline.py -v

# Generate validation report
python scripts/eval/generate_validation_report.py
```

---

## Reports

| Report | Location |
|:-------|:---------|
| Latest validation | `results/validation/` |
| HIV benchmarks | `results/clinical_applications/` |
| Research discoveries | `results/research_discoveries/` |

---

_Last updated: 2025-12-28_
