# Experiment Comparison: Baseline vs Improvements

**Date**: December 28, 2024

## Summary

After implementing and testing all quick wins, the **baseline model remains the best approach** for most drugs. Targeted improvements help specific problematic drugs.

---

## Results Comparison

### NRTI Drugs

| Drug | Baseline | + TAM | + Interactions | Change |
|------|----------|-------|----------------|--------|
| 3TC | **+0.978** | +0.904 | - | -0.074 (worse) |
| ABC | **+0.911** | +0.906 | - | -0.005 |
| AZT | **+0.894** | +0.887 | - | -0.007 |
| D4T | **+0.872** | +0.851 | - | -0.021 |
| DDI | **+0.865** | +0.798 | - | -0.067 (worse) |
| TDF | +0.773 | **+0.793** | +0.793 | **+0.020** |

**Conclusion**: TAM encoding only helps TDF. For other NRTI drugs, baseline is better.

### NNRTI Drugs

| Drug | Baseline | + Position Weights | + Interactions | Change |
|------|----------|-------------------|----------------|--------|
| NVP | **+0.957** | +0.905 | - | -0.052 (worse) |
| EFV | **+0.915** | +0.896 | - | -0.019 |
| DOR | **+0.814** | +0.732 | - | -0.082 (worse) |
| ETR | +0.799 | - | **+0.830** | **+0.031** |
| RPV | +0.588 | - | **+0.603** | **+0.015** |

**Conclusion**: Interactions help ETR and RPV. Position weights hurt other drugs.

### INI Drugs

| Drug | Baseline | + Ensemble (5) | Change |
|------|----------|---------------|--------|
| EVG | **+0.972** | - | - |
| RAL | **+0.943** | - | - |
| BIC | **+0.791** | - | - |
| DTG | **+0.756** | +0.680 | -0.076 (worse) |

**Conclusion**: Ensemble didn't help DTG. Baseline is better.

### PI Drugs

| Drug | Baseline | Multi-Task | Change |
|------|----------|------------|--------|
| All | **+0.92 avg** | 0.00 (bug) | - |

**Conclusion**: Multi-task training has implementation bugs.

---

## What Works

### 1. Baseline (Best Overall)
- **Architecture**: BatchNorm + ReLU + Dropout
- **Loss**: Reconstruction + KL + Ranking + Contrastive
- **Average**: +0.88 correlation

### 2. Drug-Specific Interactions (For Problem Drugs)
- **RPV**: E138K + M184V/I interaction → +0.015 improvement
- **ETR**: Y181C + K101E interaction → +0.031 improvement
- **TDF**: K65R + TAM interactions → +0.020 improvement

### 3. What Doesn't Work
- **TAM encoding**: Hurts 3TC significantly (-0.074)
- **Position weights**: Adds instability
- **Ensemble**: Doesn't help small datasets
- **Multi-task**: Implementation bugs

---

## Recommended Strategy

### Tier 1: Use Baseline (18 drugs)
All drugs with baseline correlation > 0.80:
- PI: All 8 drugs
- NRTI: ABC, AZT, D4T, DDI, 3TC
- NNRTI: NVP, EFV, DOR
- INI: EVG, RAL, BIC

### Tier 2: Baseline + Interactions (3 drugs)
Drugs that benefit from interaction features:
- **TDF**: Add K65R + TAM interactions
- **ETR**: Add Y181C + K101E interactions
- **RPV**: Add E138K + M184V/I interactions

### Tier 3: Needs More Data (2 drugs)
Drugs that don't improve with current methods:
- **DTG**: +0.756 (limited samples)
- **CAB**: Failed (64 samples)

---

## Implementation

### Final Training Script Usage

```bash
# Most drugs: Use baseline
python scripts/experiments/run_on_real_data.py

# Problem drugs: Add interactions
python scripts/experiments/run_improved_training.py \
    --drug RPV --use-interactions

python scripts/experiments/run_improved_training.py \
    --drug ETR --use-interactions

python scripts/experiments/run_improved_training.py \
    --drug TDF --use-tam --use-interactions
```

---

## Final Results Summary

| Drug Class | Baseline Avg | Best Strategy |
|------------|-------------|---------------|
| PI | +0.92 | Baseline |
| NRTI | +0.88 | Baseline (TDF: +TAM+Interactions) |
| NNRTI | +0.81 | Baseline (RPV, ETR: +Interactions) |
| INI | +0.87 | Baseline |
| **Overall** | **+0.88** | Mixed strategy |

---

## Bugs to Fix

1. **Multi-task training**: Zero correlations (loss not propagating)
2. **Enhanced training**: NaN losses (LayerNorm instability)
3. **TAM encoding**: Hurts simple drugs like 3TC

---

## Next Steps

1. **Production**: Deploy baseline model for 18/23 drugs
2. **Targeted**: Use interactions for TDF, ETR, RPV
3. **Research**: Investigate why TAM hurts 3TC
4. **Data**: Collect more samples for DTG, CAB
5. **Validation**: External validation against Stanford HIVdb
