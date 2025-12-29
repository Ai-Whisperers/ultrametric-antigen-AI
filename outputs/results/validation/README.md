# Structural Validation Results

> **AlphaFold3 predictions and benchmark validations**

**Last Updated:** December 29, 2025

---

## Overview

This directory contains structural validation data, primarily AlphaFold3 predictions used to validate the sentinel glycan hypothesis and other geometric findings.

---

## Directory Structure

```
validation/
├── README.md                    # This file
│
├── alphafold_predictions/
│   ├── VALIDATION_RESULTS.md    # Summary of AF3 validation
│   ├── terms_of_use.md          # AlphaFold licensing
│   └── [structure JSONs]        # AF3 prediction files (LFS)
│
└── benchmarks/
    └── RESOLUTION_COMPARISON.md # Method comparison results
```

---

## AlphaFold3 Validation

### Sentinel Glycan Hypothesis

**Hypothesis:** Glycosylation sites in the "Goldilocks Zone" (15-30% centroid shift) shield epitopes but aren't structurally essential.

**Validation Method:**
1. Generate AF3 predictions for BG505 gp120 wild-type
2. Generate predictions with single/multiple glycan removals
3. Compare structural stability (ipTM) and disorder

### Key Results

**Correlation:** r = -0.89 between Goldilocks score and ipTM stability

| Variant | ipTM | Goldilocks | Disorder % |
|---------|------|------------|------------|
| Wild-type | 0.89 | - | 5% |
| ΔN429 | 0.34 | 28% | 95% |
| ΔN58 | 0.67 | 18% | 42% |
| ΔN276 | 0.72 | 12% | 28% |
| ΔN332 | 0.81 | 8% | 15% |

**Interpretation:**
- Sites with Goldilocks scores 15-30% show dramatic destabilization
- Low Goldilocks sites (< 15%) remain stable without glycan
- High Goldilocks sites (> 30%) are structurally integral

---

## File Formats

### AlphaFold3 JSON Predictions

**Note:** These files are tracked with Git LFS due to size.

```json
{
  "name": "BG505_gp120_delta_N429",
  "sequences": [...],
  "structure": {
    "chains": [...],
    "atom_positions": [...],
    "plddt": [...],
    "pae": [...]
  },
  "confidence_metrics": {
    "ipTM": 0.34,
    "pTM": 0.42,
    "ranking_confidence": 0.38
  }
}
```

### Benchmark Results

```markdown
| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| P-adic | 0.87 | 0.85 | 0.89 | 0.87 |
| Baseline | 0.72 | 0.70 | 0.74 | 0.72 |
```

---

## Usage

### Loading AF3 Predictions

```python
import json

# Load prediction (after git lfs pull)
with open("outputs/results/validation/alphafold_predictions/BG505_delta_N429.json") as f:
    prediction = json.load(f)

# Get confidence metrics
iptm = prediction["confidence_metrics"]["ipTM"]
print(f"ipTM: {iptm}")

# Analyze disorder
plddt = prediction["structure"]["plddt"]
disorder_fraction = sum(1 for p in plddt if p < 50) / len(plddt)
```

### Pulling LFS Files

```bash
# Pull all LFS files
git lfs pull

# Check LFS status
git lfs ls-files | grep alphafold
```

---

## Benchmarks

### Resolution Comparison

**File:** `benchmarks/RESOLUTION_COMPARISON.md`

Compares p-adic predictions against:
- Experimental structures (X-ray, Cryo-EM)
- Other computational methods
- Literature baselines

---

## Terms of Use

AlphaFold predictions are subject to licensing terms. See `alphafold_predictions/terms_of_use.md` for details.

Key points:
- Academic use: Generally permitted with attribution
- Commercial use: May require license
- Redistribution: Check current terms

---

## Generating New Predictions

To generate new AlphaFold3 predictions:

```python
# See research/bioinformatics/alphafold3/ for prediction scripts
# Requires:
# - AlphaFold3 installation or API access
# - Sequence files in FASTA format
# - Structure templates (optional)
```

---

## Related Documents

- `../research/` - Hypothesis generation
- `../clinical/` - Clinical implications
- `../../reports/` - Full project reports
- `research/bioinformatics/alphafold3/` - Prediction scripts
