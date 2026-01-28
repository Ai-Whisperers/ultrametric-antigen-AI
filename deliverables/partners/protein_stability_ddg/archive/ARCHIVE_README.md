# Archive - Historical DDG Prediction Attempts

**Created:** 2026-01-27
**Purpose:** Preserve historical training attempts and internal documents not for partner delivery

---

## v1_v2_attempts/

Early training attempts with different model configurations. These are **superseded** by the current `ValidatedDDGPredictor` which uses `TrainableCodonEncoder`.

| File | Description | Why Archived |
|------|-------------|--------------|
| `padic_ddg_trained.json` | V1 model parameters | Superseded by TrainableCodonEncoder |
| `padic_ddg_trained.results.json` | V1 validation results | Metrics not reproducible with current code |
| `padic_ddg_v2_trained.json` | V2 model parameters | Superseded by TrainableCodonEncoder |
| `padic_ddg_v2_trained.results.json` | V2 validation results | Metrics not reproducible with current code |

---

## internal_docs/

Documents intended for internal use only, not for partner delivery.

| File | Description | Why Archived |
|------|-------------|--------------|
| `DISRUPTION_POTENTIAL.md` | Internal sales/positioning document | Contains inflated metrics (0.585, 0.605); not scientifically rigorous |

---

## Current Canonical Results

The authoritative validation results are in:
- `validation/results/scientific_metrics.json` - **Spearman 0.521** (N=52)
- `validation/results/SCIENTIFIC_VALIDATION_REPORT.md` - Full validation report
- `reproducibility/results/full_analysis_results.json` - N=669 results (Spearman 0.31-0.40)

---

## Metric History

| Version | Spearman (N=52) | Notes |
|---------|:---------------:|-------|
| V1 (padic_ddg_trained) | ~0.30 | Baseline p-adic only |
| V2 (padic_ddg_v2_trained) | ~0.45 | Added physicochemical features |
| V3 (TrainableCodonEncoder) | 0.58 | Fresh LOO training (bootstrap_test.py) |
| **Current (ValidatedDDGPredictor)** | **0.52** | Shipped predictor with pre-trained coefficients |

---

*These files are preserved for historical reference and reproducibility audits.*
