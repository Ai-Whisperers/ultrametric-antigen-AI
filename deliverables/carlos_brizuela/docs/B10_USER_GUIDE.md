# B10: Synthesis Optimization - User Guide

**Tool:** `B10_synthesis_optimization.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The Synthesis Optimization tool balances antimicrobial activity with practical synthesis considerations. Many highly active peptides are difficult or expensive to synthesize. This tool finds the sweet spot.

### Key Trade-offs Addressed
- **Activity vs. Difficulty:** Complex peptides may be more active but harder to make
- **Cost vs. Performance:** Expensive amino acids vs. standard ones
- **Purity vs. Yield:** Difficult sequences = lower synthesis yields

---

## Quick Start

### Demo Mode
```bash
python scripts/B10_synthesis_optimization.py
```

### Custom Weights
```bash
python scripts/B10_synthesis_optimization.py \
    --activity_weight 0.7 \
    --difficulty_weight 0.3 \
    --max_cost 50.0 \
    --output_dir results/synthesis_balanced/
```

---

## Synthesis Difficulty Metrics

### Aggregation Propensity

Hydrophobic stretches cause aggregation during synthesis:

| Pattern | Risk | Penalty |
|---------|------|---------|
| 3+ consecutive hydrophobic (I, L, V, F, W) | High | +3.0 |
| Beta-sheet forming sequences | Medium | +2.0 |
| C-terminal hydrophobic | Low | +1.0 |

### Racemization Risk

Certain sequences promote racemization at the alpha-carbon:

| Pattern | Risk | Penalty |
|---------|------|---------|
| His-Xxx (any) | High | +2.0 |
| Asp-Gly | High | +2.5 |
| C-terminal Cys | Medium | +1.5 |

### Difficult Couplings

Steric hindrance slows coupling:

| Sequence | Difficulty | Expected Coupling Efficiency |
|----------|------------|------------------------------|
| Ile-Ile | Very High | <80% |
| Val-Val | High | 85% |
| Aib-Xxx | Very High | <75% |
| Pro-Pro | Medium | 90% |

---

## Understanding the Output

### Results CSV

```csv
rank,sequence,activity_score,synthesis_difficulty,coupling_efficiency,estimated_cost
1,HRGTGKRTIKKLAVAGKFGA,0.908,14.79,50.9%,$36.50
2,GKRSLALGKRVLNCGARKGN,0.882,14.62,51.5%,$36.50
3,YAGGKKGVKSAYARFINKPL,0.926,16.04,46.8%,$36.00
```

### Metric Interpretation

| Metric | Optimal Range | Notes |
|--------|---------------|-------|
| `activity_score` | 0.8 - 1.0 | Higher = more active |
| `synthesis_difficulty` | < 15 | Lower = easier to synthesize |
| `coupling_efficiency` | > 60% | Higher = better yields |
| `estimated_cost` | < $40/mg | For 25 nmol scale |

---

## Cost Estimation

### Base Costs (per amino acid)

| Category | Examples | Cost Factor |
|----------|----------|-------------|
| Standard | A, G, L, V, I | 1.0x |
| Moderate | K, R, E, D, S, T | 1.2x |
| Expensive | W, M, C, H | 2.0x |
| Very Expensive | Fmoc-Cys(Trt), Fmoc-Arg(Pbf) | 3.0x |

### Additional Costs

| Factor | Cost Increase |
|--------|---------------|
| HPLC purification | +$50-100 |
| Disulfide formation | +$100-200 |
| Cyclization | +$150-300 |
| N-terminal modifications | +$30-50 |

---

## Optimization Objectives

| Objective | Goal | Description |
|-----------|------|-------------|
| Maximize Activity | Higher predicted antimicrobial effect | |
| Minimize Difficulty | Lower aggregation, racemization risk | |
| Minimize Cost | Cheaper synthesis | |

### Pareto Front Interpretation

The output gives multiple solutions representing trade-offs:

```
                     High Activity
                          ▲
                          │     ● Candidate 1 (hard to synthesize)
                          │
                          │  ● Candidate 2 (balanced)
                          │
                          │       ● Candidate 3 (easy to synthesize)
                          └────────────────────────────────────►
                                            Low Difficulty
```

---

## Practical Guidelines

### For Academic Labs (Limited Budget)

```bash
python scripts/B10_synthesis_optimization.py \
    --difficulty_weight 0.6 \
    --cost_weight 0.3 \
    --activity_weight 0.1 \
    --max_difficulty 12
```

### For Industry (Performance Priority)

```bash
python scripts/B10_synthesis_optimization.py \
    --activity_weight 0.7 \
    --difficulty_weight 0.2 \
    --cost_weight 0.1 \
    --min_activity 0.9
```

---

## Avoiding Problematic Sequences

### Sequences to Avoid

| Pattern | Problem | Alternative |
|---------|---------|-------------|
| VVVV | Aggregation | VLVL or VAVA |
| DG | Aspartimide formation | DN or EG |
| HX (at any position) | Racemization | Move H to N-terminus |
| NG | Deamidation | NQ or QG |
| DP | Peptide bond lability | EP or DA |

### Recommended Modifications

| Issue | Solution |
|-------|----------|
| Aggregation | Add pseudoproline dipeptides |
| Low solubility | Add Lys at C-terminus |
| Oxidation-sensitive | Replace Met with Nle |

---

## Integration with Synthesis Vendors

### GenScript Quote Format

Export peptides in vendor-ready format:

```python
# After running optimization
import pandas as pd

df = pd.read_csv("results/synthesis_optimized/candidates.csv")

# Format for GenScript bulk order
genscript_format = df[["sequence"]].copy()
genscript_format["scale"] = "5mg"
genscript_format["purity"] = ">95%"
genscript_format["modifications"] = "N-term acetylation, C-term amidation"

genscript_format.to_csv("genscript_order.csv", index=False)
```

---

## Troubleshooting

### Issue: All peptides have high difficulty scores

**Cause:** Activity optimization favors complex sequences
**Solution:** Increase difficulty weight, add explicit constraints

### Issue: Very low coupling efficiency predictions

**Cause:** Multiple difficult couplings in sequence
**Solution:** Consider pseudoproline insertions or backbone modifications

### Issue: Costs much higher than expected

**Cause:** Multiple expensive amino acids
**Solution:** Use --max_cost flag, or replace expensive residues (W->F, C->S)

---

## Validation

### Comparing Predicted vs. Actual Synthesis

After receiving synthesized peptides:

1. **Compare purity** (HPLC) with predicted coupling efficiency
2. **Check mass** (MS) for deletions/modifications
3. **Update model** with empirical feedback

---

*Part of the Ternary VAE Bioinformatics Partnership*
