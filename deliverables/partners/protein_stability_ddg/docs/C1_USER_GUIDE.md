# C1: Rosetta-Blind Detection - User Guide

**Tool:** `C1_rosetta_blind_detection.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The Rosetta-Blind Detection tool identifies protein residues where geometric (p-adic) scoring disagrees with Rosetta energy scoring. These "Rosetta-blind" residues may represent:

- Underestimated instabilities (Rosetta says stable, geometry says unstable)
- Functional conformational flexibility
- Crystal packing artifacts
- Potential mutation hotspots

---

## Quick Start

### Demo Mode
```bash
python scripts/C1_rosetta_blind_detection.py
```

### With Real Structures
```bash
python scripts/C1_rosetta_blind_detection.py \
    --pdb_ids "1CRN,1TIM,4LZT" \
    --output_dir results/my_analysis/
```

---

## Understanding the Four Categories

The tool classifies each residue into one of four categories:

| Category | Rosetta | Geometric | Interpretation |
|----------|---------|-----------|----------------|
| **Concordant Stable** | Low (stable) | Low (stable) | Both agree: stable |
| **Concordant Unstable** | High (unstable) | High (unstable) | Both agree: unstable |
| **ROSETTA-BLIND** | Low (stable) | High (unstable) | Rosetta misses instability |
| **Geometry-Blind** | High (unstable) | Low (stable) | Geometry misses instability |

### Why "Rosetta-Blind" Matters

Rosetta-blind residues are particularly important because:
1. Traditional scoring underestimates their instability
2. They may be prone to mutation-induced destabilization
3. They often occur at functional sites requiring flexibility

---

## Discordance Score Calculation

```
Discordance = |R_norm - G_norm|

Where:
- R_norm = (R - R_min) / (R_max - R_min)   # Normalized Rosetta
- G_norm = (G - G_min) / (G_max - G_min)   # Normalized Geometric

Rosetta-blind if:
- R_norm < 0.3 (Rosetta thinks stable)
- G_norm > 0.7 (Geometry thinks unstable)
- Discordance > 0.4
```

---

## Output Interpretation

### Summary Statistics

| Metric | Expected Range | Meaning |
|--------|---------------|---------|
| Rosetta-blind fraction | 15-30% | Proportion with scoring disagreement |
| Mean discordance | 0.15-0.25 | Average disagreement magnitude |
| Max discordance | 0.35-0.45 | Worst disagreement |

### Per-Residue Output

```json
{
  "pdb_id": "1CRN",
  "chain_id": "A",
  "residue_id": 15,
  "residue_name": "TRP",
  "rosetta_score": 1.23,
  "geometric_score": 7.60,
  "discordance_score": 0.397,
  "classification": "rosetta_blind"
}
```

---

## Amino Acid Distribution

Rosetta-blind residues are enriched in certain amino acid types:

| AA Type | Enrichment | Reason |
|---------|------------|--------|
| TRP, TYR, PHE | High (28%) | Aromatic flexibility |
| LEU, ILE, VAL | High (25%) | Multiple rotamers |
| ARG, LYS | Moderate (22%) | Long side chains |
| GLY, ALA | Low (5%) | Few rotamers |

---

## Applications

### 1. Protein Engineering
Identify positions where stabilizing mutations might help:
```python
# Load results
import json
with open('rosetta_blind_report.json') as f:
    data = json.load(f)

# Find mutation candidates
for res in data['rosetta_blind_residues']:
    if res['discordance_score'] > 0.35:
        print(f"Consider mutating {res['residue_name']}{res['residue_id']}")
```

### 2. Drug Binding Site Analysis
Rosetta-blind residues near binding sites may indicate induced-fit regions.

### 3. Stability Prediction Improvement
Use as additional features for ML stability predictors.

---

## Troubleshooting

### Issue: Very high Rosetta-blind fraction (>40%)

**Cause:** Possibly unusual protein or low-resolution structure
**Solution:** Check structure quality, filter by B-factor

### Issue: All residues classified as concordant

**Cause:** Scoring ranges too similar
**Solution:** Check normalization, ensure diverse input

### Issue: Geometric scores all similar

**Cause:** Demo mode uses random chi angles
**Solution:** Use real PDB structures

---

## Validation

### Against B-factors
Rosetta-blind residues should correlate with high B-factors:
```python
# Expected correlation: r > 0.5
correlation = np.corrcoef(geometric_scores, b_factors)[0,1]
```

### Against Known Unstable Sites
Cross-reference with:
- Active site residues (should be enriched)
- Crystal contacts (may show artifacts)
- Literature-known unstable positions

---

*Part of the Ternary VAE Bioinformatics Partnership*
