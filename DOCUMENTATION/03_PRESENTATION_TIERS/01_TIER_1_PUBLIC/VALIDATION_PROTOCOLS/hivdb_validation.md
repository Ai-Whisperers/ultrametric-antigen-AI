# Stanford HIVdb Drug Resistance Validation Protocol

**Purpose**: Validate drug class constraint predictions
**Difficulty**: Low (public database queries)
**Time**: ~30 minutes

---

## Overview

Our platform predicts that different antiretroviral drug classes impose different evolutionary constraints on HIV. This can be validated using the Stanford HIV Drug Resistance Database (HIVdb).

---

## Materials Required

1. Internet access
2. Stanford HIVdb: https://hivdb.stanford.edu/
3. Our drug resistance predictions (see [../PREDICTIONS.md](../PREDICTIONS.md))

---

## Our Predictions

| Drug Class | Constraint Level | Prediction |
|:-----------|----------------:|:-----------|
| NRTI | 4.06 | Highest barrier, few single-mutation escapes |
| INSTI | 4.30 | High barrier, explains dolutegravir durability |
| NNRTI | 3.59 | Moderate, single mutations confer resistance |
| PI | 3.52 | Lower, requires multiple mutations |

---

## Protocol

### Step 1: Access Stanford HIVdb

1. Navigate to https://hivdb.stanford.edu/
2. Select "Resistance" → "Drug Summaries"

### Step 2: Compare Resistance Patterns by Class

For each drug class, note:
1. Number of major resistance mutations
2. Number of accessory mutations required
3. Single vs multi-mutation resistance

### Step 3: Expected Patterns

#### NRTI (Highest constraint = 4.06)
- M184V: Single mutation, but causes significant fitness cost
- K65R: High resistance but rare (high barrier)
- TAMs: Require accumulation of multiple mutations
- **Prediction**: Few high-level single mutations

#### INSTI (High constraint = 4.30)
- Dolutegravir: Rarely fails in treatment-naive
- R263K: Causes high fitness cost
- **Prediction**: Highest barrier, least resistance emergence

#### NNRTI (Moderate constraint = 3.59)
- K103N: Single mutation → high-level resistance
- Y181C: Single mutation → resistance
- **Prediction**: Easy single-mutation escape

#### PI (Lower constraint = 3.52)
- Major + accessory mutations required
- Multiple pathways to resistance
- **Prediction**: More flexible escape routes

### Step 4: Quantify Resistance Complexity

| Class | Single-Mutation Resistance | Multi-Mutation Resistance | Our Prediction |
|:------|:--------------------------:|:-------------------------:|:--------------:|
| NRTI | Few (M184V) | Many (TAMs) | Highest |
| INSTI | Very few | R263K+others | High |
| NNRTI | Many (K103N, Y181C) | Fewer | Moderate |
| PI | Few | Many | Lower |

---

## Clinical Validation

### Dolutegravir (DTG) Durability

If our INSTI constraint score (4.30) is correct:
- DTG should have lowest resistance emergence
- Query HIVdb for DTG failure rates in treatment-naive
- Compare with EFV (NNRTI) failure rates

### NNRTI Single-Mutation Escape

If our NNRTI constraint score (3.59) is correct:
- K103N should be common in EFV failures
- Query HIVdb for K103N prevalence
- Compare with M184V (NRTI) prevalence

---

## Interpretation

### Strong validation:
- Drug class durability correlates with our constraint scores
- INSTI > NRTI > PI > NNRTI in clinical durability

### Moderate validation:
- General trend matches predictions
- Some class-specific exceptions

### Weak validation:
- Consider drug-specific effects beyond class
- Combination therapy confounds single-drug predictions

---

## Data Recording Template

```
Drug class: ___________
Representative drug: ___________
Major mutations: ___________
Accessory mutations: ___________
Single-mutation resistance: Yes/No
Multi-mutation required: Yes/No
Our constraint score: ___________
HIVdb durability ranking: ___________
```

---

## Additional Resources

- HIVdb Mutation Scoring: https://hivdb.stanford.edu/hivdb/by-mutations/
- Drug Resistance Tutorial: https://hivdb.stanford.edu/pages/resistance-tutorial/

---

*For full drug resistance data, see [../DATA/drug_resistance.json](../DATA/drug_resistance.json)*
