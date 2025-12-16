# HIV Analysis using P-Adic Geometry

**Doc-Type:** Project Index · Version 1.0 · Updated 2025-12-16

---

## Overview

This directory contains analysis of HIV-1 sequences using p-adic geometric methods derived from the Ternary VAE v1.1.0 model. The analysis examines:

1. **CTL Escape Mutations** - How immune escape variants travel in p-adic space
2. **Drug Resistance** - Geometric organization of resistance mutations by drug class
3. **Fitness Prediction** - P-adic distance as predictor of mutation fitness cost

---

## Key Findings Summary

| Discovery | Metric | Significance |
|-----------|--------|--------------|
| **Distance-Fitness Correlation** | r = 0.24 | Larger p-adic jumps cost more fitness |
| **INSTI Constraint** | d = 4.30 | Integrase active site most constrained |
| **NNRTI Flexibility** | d = 3.59 | Allosteric pocket most permissive |
| **HLA-B27 Protection** | d = 4.40 | Escape requires high fitness cost |

---

## Directory Structure

```
HIV_analysis/
├── README.md                      # This file
├── scripts/
│   ├── 01_hiv_escape_analysis.py  # CTL escape mutation analysis
│   └── 02_hiv_drug_resistance.py  # Drug resistance patterns
├── discoveries/
│   └── DISCOVERY_HIV_PADIC_RESISTANCE.md
├── results/
│   ├── hiv_escape_analysis.png
│   ├── hiv_escape_results.json
│   ├── hiv_drug_resistance.png
│   └── hiv_drug_resistance_results.json
└── data/
    └── [Uses codon_encoder from RA_analysis]
```

---

## Scripts

### 01. HIV Escape Analysis
**File:** `scripts/01_hiv_escape_analysis.py`

Analysis of 6 CTL epitopes with 9 escape variants.

**Key Results:**
- 100% boundary crossing (expected: AA changes)
- Average p-adic distance: 3.93
- HLA-B27 restricted R264K: d = 4.40 (high fitness cost)
- Distance-efficacy correlation: r = 0.29

**Run:**
```bash
python scripts/01_hiv_escape_analysis.py
```

---

### 02. Drug Resistance Analysis
**File:** `scripts/02_hiv_drug_resistance.py`

Analysis of 18 drug resistance mutations across 4 drug classes.

**Key Results:**

| Drug Class | Mean Distance | Interpretation |
|------------|---------------|----------------|
| INSTI | 4.30 ± 0.55 | Most constrained |
| NRTI | 4.06 ± 0.41 | High constraint |
| NNRTI | 3.59 ± 0.03 | Most flexible |
| PI | 3.52 ± 0.34 | Moderate flexibility |

**Run:**
```bash
python scripts/02_hiv_drug_resistance.py
```

---

## Epitopes Analyzed

### CTL Epitopes

| Epitope | Protein | HLA | Wild-Type | Key Escape |
|---------|---------|-----|-----------|------------|
| SL9 | Gag p17 | A*02:01 | SLYNTVATL | Y79F (d=3.68) |
| KK10 | Gag p24 | B*27:05 | KRWIILGLNK | R264K (d=4.40) |
| TW10 | Gag p24 | B*57:01 | TSTLQEQIGW | T242N (d=4.18) |
| FL8 | Nef | A*24:02 | FLKEKGGL | K94R (d=4.41) |
| IV9 | RT | A*02:01 | ILKEPVHGV | V181I (d=3.89) |
| RL9 | Env | B*08:01 | RLRDLLLIW | D314N (d=3.53) |

### Drug Resistance Mutations

**Highest Distance (Most Constrained):**
1. Y143R (INSTI): d = 5.08 - Raltegravir escape
2. K65R (NRTI): d = 4.41 - Tenofovir resistance
3. K70R (NRTI): d = 4.41 - TAM pathway
4. R263K (INSTI): d = 4.40 - Dolutegravir resistance
5. N155H (INSTI): d = 4.19 - Major INSTI mutation

---

## Biological Insights

### Why INSTIs Have Highest Distances

```
Integrase Active Site:
  - DDE catalytic triad (D64, D116, E152)
  - Metal coordination essential
  - Mutations must preserve chemistry
  - Result: Large p-adic jumps required
```

### Why NNRTIs Have Lowest Distances

```
NNRTI Binding Pocket:
  - Allosteric (not active site)
  - More tolerant of substitutions
  - K103N: high resistance, minimal fitness cost
  - Result: Small p-adic jumps sufficient
```

---

## Therapeutic Implications

### 1. Drug Design
Target regions requiring large p-adic distance for escape:
- Integrase active site (current INSTIs)
- Conserved RT residues beyond current targets

### 2. Combination Therapy
Optimal combinations span multiple p-adic regions:
```
High-Distance Combo: INSTI + NRTI
  Total escape distance: ~8.3

Low-Distance Combo: NNRTI + PI
  Total escape distance: ~7.1 (easier escape)
```

### 3. Vaccine Design
Target epitopes where:
- Escape requires d > 4.0
- HLA-B27 and B*57 restricted epitopes are geometric barriers

---

## Comparison to RA Analysis

| Feature | HIV | RA |
|---------|-----|-----|
| Primary Question | Escape fitness cost | Boundary immunogenicity |
| Boundary Crossing | 100% (AA change) | 14% (PTM) |
| Key Finding | Distance = fitness | Boundary = autoimmunity |
| Sentinel Events | High-d escapes | FGA_R38, FLG_R30 |

---

## Dependencies

```python
torch           # Neural network
numpy           # Numerical computation
scipy           # Statistical tests
matplotlib      # Visualization
```

Uses codon encoder from `../RA_analysis/data/` or `../results/`.

---

## Running the Full Analysis

```bash
# From riemann_hypothesis_sandbox directory
cd HIV_analysis/scripts

# Run in sequence
python 01_hiv_escape_analysis.py
python 02_hiv_drug_resistance.py
```

---

## Future Directions

1. **Expand dataset** - Include all Stanford HIVDB mutations
2. **Compensatory analysis** - How do fitness-restoring mutations affect total distance?
3. **Predictive model** - ML classifier using p-adic features
4. **Clinical correlation** - Validate with patient outcome data

---

## Connection to Main Project

| Component | HIV Application |
|-----------|-----------------|
| VAE v1.1.0 | Defines p-adic cluster structure |
| Codon Encoder | Maps mutations to embedding space |
| 21 Clusters | Match amino acid groups |
| Wobble Pattern | Explains synonymous tolerance |

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-16 | 1.0 | Initial implementation |

---

**Status:** Complete analysis pipeline, ready for dataset expansion
