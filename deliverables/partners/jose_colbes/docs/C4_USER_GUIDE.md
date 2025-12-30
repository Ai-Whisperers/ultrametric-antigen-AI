# C4: Mutation Effect Predictor - User Guide

**Tool:** `C4_mutation_effect_predictor.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The Mutation Effect Predictor estimates the change in protein stability (DDG) caused by single amino acid mutations. It uses p-adic geometric features combined with physicochemical properties.

### Key Metrics
- **DDG:** Change in Gibbs free energy (kcal/mol)
  - Positive = destabilizing
  - Negative = stabilizing
  - Near zero = neutral

---

## Quick Start

### Demo Mode (Random Mutations)
```bash
python scripts/C4_mutation_effect_predictor.py
```

### Specific Mutations
```bash
python scripts/C4_mutation_effect_predictor.py \
    --mutations "G45A,D156K,V78I,K103N" \
    --output_dir results/my_mutations/
```

### From Mutation File
```bash
python scripts/C4_mutation_effect_predictor.py \
    --mutation_file mutations.txt \
    --output_dir results/batch_predictions/
```

Where `mutations.txt` contains one mutation per line:
```
G45A
D156K
V78I
K103N
```

---

## Mutation Notation

Standard format: `<WT_AA><Position><Mut_AA>`

| Example | Meaning |
|---------|---------|
| G45A | Glycine at position 45 -> Alanine |
| D156K | Aspartate at position 156 -> Lysine |
| K103N | Lysine at position 103 -> Asparagine |

---

## Classification Thresholds

| DDG Range | Classification | Color Code |
|-----------|---------------|------------|
| < -1.0 | Stabilizing | Green |
| -1.0 to +1.0 | Neutral | Yellow |
| +1.0 to +3.0 | Mildly destabilizing | Orange |
| > +3.0 | Severely destabilizing | Red |

---

## Feature Contributions

The predictor uses weighted features:

| Feature | Weight | Description |
|---------|--------|-------------|
| Delta Volume | 0.015 | (MutVol - WTVol) in A^3 |
| Delta Hydrophobicity | 0.5 | Kyte-Doolittle scale |
| Delta Charge | 1.5 | Net charge change |
| Delta Geometric | 1.2 | P-adic stability change |
| Context Factor | 1.0-2.0 | Core vs. surface |

### Base Formula
```
DDG = w_vol * dVol + w_hydro * dHydro + w_charge * dCharge + w_geom * dGeom
DDG *= context_factor
```

---

## Understanding Output

### Per-Mutation Results

```json
{
  "mutation": "D156K",
  "position": 156,
  "wt_aa": "D",
  "mut_aa": "K",
  "predicted_ddg": 12.19,
  "classification": "destabilizing",
  "confidence": 0.44,
  "delta_volume": 57.5,
  "delta_hydrophobicity": -0.6,
  "delta_charge": 2,
  "delta_geometric": 7.235
}
```

### Confidence Interpretation

| Confidence | Meaning |
|------------|---------|
| > 0.9 | High confidence (conservative mutation) |
| 0.7-0.9 | Good confidence |
| 0.5-0.7 | Moderate confidence |
| < 0.5 | Low confidence (extreme mutation) |

Confidence decreases for:
- Large physicochemical changes
- Unusual mutation types (e.g., charge reversals)
- Positions with poor structural data

---

## Common Mutation Patterns

### Typically Destabilizing (DDG > +2)

| Pattern | Example | Reason |
|---------|---------|--------|
| Charge reversal | D->K, E->R | Electrostatic clash |
| Large-to-core | G->W, A->F | Steric clash |
| Proline insertion | X->P | Backbone strain |
| Cysteine disruption | C->X (in disulfide) | Broken bond |

### Typically Neutral (DDG ~ 0)

| Pattern | Example | Reason |
|---------|---------|--------|
| Conservative | I->L, V->I | Similar properties |
| Surface hydrophilic | K->R, D->E | Charge maintained |
| Aromatic swap | F->Y, W->F | Similar size/properties |

### Potentially Stabilizing (DDG < -1)

| Pattern | Example | Reason |
|---------|---------|--------|
| Buried polar->hydrophobic | K->I (if buried) | Reduced desolvation |
| Remove cavity | G->A, A->V | Fill void |
| Improve packing | Small->optimal size | Better contacts |

---

## Batch Analysis

### Systematic Mutagenesis
```bash
# Generate all single mutants at position 156
python scripts/C4_mutation_effect_predictor.py \
    --position 156 \
    --all_substitutions \
    --output_dir results/position_156_scan/
```

### Mutation Landscape
```python
import pandas as pd
import seaborn as sns

# Load results
df = pd.read_csv('mutation_effects.csv')

# Create heatmap
pivot = df.pivot(index='wt_aa', columns='mut_aa', values='predicted_ddg')
sns.heatmap(pivot, cmap='RdYlGn_r', center=0)
```

---

## Integration with Structure

### With PDB File
```bash
python scripts/C4_mutation_effect_predictor.py \
    --pdb 1CRN.pdb \
    --mutations "L5A,K7E,S31N" \
    --output_dir results/1crn_mutations/
```

The structural context improves predictions by:
- Identifying buried vs. surface positions
- Detecting secondary structure context
- Finding neighboring interactions

---

## Validation

### Against ProTherm Database
```python
# Expected correlation with experimental DDG
# Target: Spearman r > 0.6
from scipy.stats import spearmanr
r, p = spearmanr(predicted_ddg, experimental_ddg)
```

### Against Deep Mutational Scanning
Use mega-scale DMS datasets for comprehensive validation.

---

## Troubleshooting

### Issue: All predictions near zero

**Cause:** Context factor not applied or features not calculated
**Solution:** Ensure structure information is provided

### Issue: Extremely high DDG (>20)

**Cause:** Multiple feature contributions compounding
**Solution:** Check for unusual mutations, verify input format

### Issue: Low confidence for all mutations

**Cause:** Demo mode uses mock context
**Solution:** Provide real PDB structure

---

## Best Practices

1. **Provide structure context** when available
2. **Group similar mutations** for comparison
3. **Validate against known data** before new predictions
4. **Consider confidence scores** when prioritizing experiments

---

*Part of the Ternary VAE Bioinformatics Partnership*
