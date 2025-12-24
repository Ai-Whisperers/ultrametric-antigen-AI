# Detailed Predictions

**Access Level**: Partner
**Format**: Numerical outputs from proprietary computational platform

---

## 1. Integrase Vulnerability Analysis

### Protein Isolation Scores

| Protein | Mean Distance | Max Distance | Vulnerability Zones |
|:--------|-------------:|-------------:|--------------------:|
| Pol_IN (Integrase) | 3.24 | 4.27 | 13 |
| Tat | 2.89 | 3.92 | 12 |
| Rev | 2.76 | 3.65 | 10 |
| Vif | 2.71 | 3.58 | 10 |
| Gag_NC | 2.45 | 3.21 | 9 |

### LEDGF Interface Targets

| Mutation | Reveal Score | Mechanism |
|:---------|-------------:|:----------|
| E166K | 34.93 | Salt bridge reversal |
| K175E | 34.93 | Charge reversal |
| W131A | 33.03 | Aromatic cap removal |

---

## 2. Sentinel Glycan Predictions

### Goldilocks Zone Sites (15-30% Perturbation)

| Site | Position | Region | Perturbation | Score | bnAb Relevance |
|:-----|:--------:|:-------|-------------:|------:|:---------------|
| N58 | 57 | V1 | 22.4% | 1.193 | V1/V2 shield |
| N429 | 428 | C5 | 22.6% | 1.189 | Structural |
| N103 | 102 | V2 | 23.7% | 1.036 | V1/V2 bnAbs |
| N204 | 203 | V3 | 25.1% | 0.855 | V3 supersite |
| N107 | 106 | V2 | 17.0% | 0.462 | V1/V2 bnAbs |
| N271 | 270 | C3 | 28.4% | 0.417 | Core glycan |
| N265 | 264 | C3 | 29.1% | 0.324 | Core glycan |

### AlphaFold3 Correlation

| Variant | Our Score | AF3 pLDDT | Disorder % |
|:--------|----------:|----------:|-----------:|
| Wild-type | 0.00 | 78.3 | 0% |
| N58Q | 1.19 | 73.2 | 75% |
| N429Q | 1.19 | 71.1 | 100% |
| N103Q | 1.04 | 75.8 | 67% |
| Combined | 2.50+ | 68.4 | 85% |

**Correlation**: r = -0.89 (p < 0.01)

---

## 3. CTL Escape Barrier Predictions

### Epitope-Specific Barriers

| Epitope | HLA | Wild-Type | Escape Mutation | Barrier Score | Fitness Cost |
|:--------|:----|:----------|:----------------|-------------:|:-------------|
| Gag p17 77-85 | A*02:01 | SLYNTVATL | Y79F | 3.68 | Low |
| Gag p17 77-85 | A*02:01 | SLYNTVATL | T84I | 4.38 | Moderate |
| Gag p24 263-272 | B*27:05 | KRWIILGLNK | R264K | 4.40 | High |
| Gag p24 263-272 | B*27:05 | KRWIILGLNK | L268M | 3.28 | Low |
| Gag p24 240-249 | B*57:01 | TSTLQEQIGW | T242N | 4.18 | Moderate |
| Gag p24 240-249 | B*57:01 | TSTLQEQIGW | G248A | 3.60 | Low |
| Nef 90-97 | A*24:02 | FLKEKGGL | K94R | 4.40 | Low |
| RT 179-187 | A*02:01 | ILKEPVHGV | V181I | 3.89 | Low |
| Env 311-319 | B*08:01 | RLRDLLLIW | D314N | 3.53 | High |

### Elite Controller Signature

Mean barrier score for B*27/B*57 epitopes: **4.29**
Mean barrier score for other HLA: **3.72**
Difference: **15% higher barriers for elite controller alleles**

---

## 4. Drug Resistance Predictions

### By Drug Class

| Class | Mean Barrier | Key Mutations |
|:------|-------------:|:--------------|
| NRTI | 4.06 | M184V (4.00), K65R (4.40), K70R (4.40) |
| INSTI | 4.30 | Y143R (5.08), N155H (4.19), R263K (4.40) |
| NNRTI | 3.59 | K103N (3.55), Y181C (3.62), G190A (3.60) |
| PI | 3.52 | D30N (3.53), M46I (2.93), I84V (3.89) |

### Individual Mutations

| Mutation | Barrier | Resistance Level | Fitness Impact |
|:---------|--------:|:-----------------|:---------------|
| M184V | 4.00 | High | Moderate decrease |
| K65R | 4.40 | Moderate | Moderate decrease |
| K103N | 3.55 | High | Minimal |
| Y181C | 3.62 | High | Minimal |
| Y143R | 5.08 | High | Moderate decrease |
| Q148H | 3.54 | High | High decrease |
| M46I | 2.93 | Moderate | Moderate decrease |
| I84V | 3.89 | High | High decrease |

---

## 5. Protein Proximity Map

### Closest Pairs (Shared Evasion Architecture)

| Pair | Proximity Score | Functional Connection |
|:-----|---------------:|:----------------------|
| NC-Vif | 0.565 | RNA protection |
| CA-PR | 0.714 | Structural processing |
| CA-gp120 | 0.760 | Surface-core link |
| gp120-Nef | 0.961 | Immune evasion triad |

### Implication
Single intervention targeting NC-Vif proximity can disrupt both proteins' evasion simultaneously.

---

## 6. Hierarchy Analysis

### Evasion Centroids by Level

| Level | Centroid Norm | Constraint | Unexplored Space |
|:------|-------------:|:-----------|:-----------------|
| Protein | 0.144 | Low | 85.6% |
| Glycan | 0.237 | Moderate | 76.3% |
| Signaling | 0.262 | High | 73.8% |
| Peptide | 0.303 | Highest | 69.7% |
| **Overall** | **0.161** | - | **83.9%** |

### Therapeutic Implication
Peptide-level (CTL) therapies exploit highest constraint.
HIV has limited escape options at peptide level.

---

## 7. 49-Gap Therapeutic Map (Summary)

### Distribution

| Severity | Count | Distance Range | Example |
|:---------|------:|:---------------|:--------|
| Severe | 6 | > 3.5 | IN-Tat (4.27) |
| Moderate | 22 | 2.5 - 3.5 | Multiple pairs |
| Mild | 21 | < 2.5 | NC-Vif (0.57) |

### Protein Gap Counts

| Protein | Gaps | Priority |
|:--------|-----:|:---------|
| Pol_IN | 13 | Highest |
| Tat | 12 | High |
| Rev | 10 | High |
| Vif | 10 | High |
| Gag_NC | 9 | Moderate |

Full 49-gap matrix available in [DATA/vulnerability_zones.json](./DATA/vulnerability_zones.json).

---

## Interpretation Guide

### Barrier Scores
- **< 3.0**: Low barrier, easy escape
- **3.0 - 4.0**: Moderate barrier
- **> 4.0**: High barrier, difficult escape

### Perturbation Scores
- **< 0.5**: Low impact modification
- **0.5 - 1.0**: Moderate impact
- **> 1.0**: High impact (Goldilocks zone)

### Proximity Scores
- **< 1.0**: Shared evasion architecture
- **1.0 - 2.5**: Related evasion
- **> 2.5**: Independent evasion

---

*Scores derived from proprietary geometric analysis. Methodology available under Tier 2 partnership.*
