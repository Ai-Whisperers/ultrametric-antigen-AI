# Partner Collaboration Results

> **Validation results from collaborative research projects**

**Last Updated:** December 29, 2025

---

## Overview

This directory contains outputs from partner collaborations that validated the p-adic/hyperbolic framework across different biological domains.

---

## Directory Structure

```
partners/
├── README.md                    # This file
├── VALIDATION_SUMMARY.md        # Summary of all validations
│
├── brizuela/                    # Antimicrobial peptides
│   └── [AMP optimization results]
│
├── colbes/                      # Rotamer stability
│   └── rotamer_stability.json
│
└── rojas/                       # Arbovirus forecasting
    └── dengue_forecast.json
```

---

## Partner Projects

### 1. Carlos Brizuela - Antimicrobial Peptides

**Focus:** NSGA-II optimization of AMPs in VAE latent space

**Method:**
- Multi-objective optimization: MIC vs. hemolysis
- Latent space navigation for novel peptide design
- Pareto front identification

**Results:**
- Generated Pareto-optimal peptide candidates
- Improved MIC without increasing hemolysis
- Status: Experimental validation pending

**Files:**
- `brizuela/` - Optimization results and candidate sequences

### 2. Dr. José Colbes - Rotamer Stability

**Focus:** P-adic metric for side-chain conformation prediction

**Method:**
- 3-adic distance between rotamer states
- Stability scoring based on geometric clustering
- Validation against crystallographic data

**Results:**
- **87% accuracy** in predicting stable rotamers
- Outperforms traditional potential-based methods for certain residue types

**Files:**
- `colbes/rotamer_stability.json` - Prediction results

**JSON Schema:**
```json
{
  "residue": "PHE",
  "position": 42,
  "predicted_rotamer": "t80",
  "confidence": 0.87,
  "padic_score": 0.92,
  "crystal_rotamer": "t80",
  "correct": true
}
```

### 3. Alejandra Rojas - Arbovirus Forecasting

**Focus:** Hyperbolic trajectory embedding for dengue outbreak prediction

**Method:**
- Serotype embeddings in Poincaré ball
- Temporal trajectory analysis
- 14-day forecast horizon

**Results:**
- **82% accuracy** in outbreak prediction
- Successfully predicted 2024 Paraguay outbreak timing

**Files:**
- `rojas/dengue_forecast.json` - Forecast data

**JSON Schema:**
```json
{
  "region": "Asuncion",
  "forecast_date": "2025-01-15",
  "horizon_days": 14,
  "outbreak_probability": 0.78,
  "confidence_interval": [0.65, 0.89],
  "dominant_serotype": "DENV-2"
}
```

---

## Summary Table

| Partner | Domain | Accuracy | Status |
|---------|--------|----------|--------|
| Brizuela | AMPs | N/A | Experimental validation pending |
| Colbes | Rotamers | 87% | Published methodology |
| Rojas | Dengue | 82% | Active surveillance |

---

## Usage

### Loading Partner Results

```python
import json

# Load rotamer predictions
with open("outputs/results/partners/colbes/rotamer_stability.json") as f:
    rotamers = json.load(f)

# Calculate accuracy
correct = sum(1 for r in rotamers if r["correct"])
accuracy = correct / len(rotamers)
print(f"Rotamer accuracy: {accuracy:.1%}")

# Load dengue forecasts
with open("outputs/results/partners/rojas/dengue_forecast.json") as f:
    forecasts = json.load(f)

# Get high-probability outbreaks
high_risk = [f for f in forecasts if f["outbreak_probability"] > 0.7]
```

---

## Collaboration Guidelines

### Adding New Partner Results

1. Create subdirectory: `partners/<partner_name>/`
2. Include JSON results with documented schema
3. Update `VALIDATION_SUMMARY.md`
4. Update this README

### Data Sharing

- Partner data is shared under research collaboration agreements
- Attribution required for any publications
- Contact partners before external use

---

## Related Documents

- `VALIDATION_SUMMARY.md` - Comprehensive validation report
- `../research/` - Underlying methodology
- `../validation/` - AlphaFold3 structural validation
