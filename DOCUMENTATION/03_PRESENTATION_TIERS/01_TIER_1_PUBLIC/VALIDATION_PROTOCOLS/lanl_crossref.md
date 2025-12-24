# Los Alamos HIV Database Cross-Reference Protocol

**Purpose**: Validate escape barrier predictions against clinical data
**Difficulty**: Low (public database queries)
**Time**: ~1 hour

---

## Overview

Our platform assigns "escape barrier" scores to HIV epitopes. Higher scores predict that immune escape mutations at that epitope will be rare (due to high fitness cost).

The Los Alamos HIV Sequence Database contains real-world mutation frequencies that can validate these predictions.

---

## Materials Required

1. Internet access
2. Los Alamos HIV Database: https://www.hiv.lanl.gov/
3. Our epitope predictions (see [../PREDICTIONS.md](../PREDICTIONS.md))

---

## Protocol

### Step 1: Access LANL Database

1. Navigate to https://www.hiv.lanl.gov/
2. Select "Search Interface" → "Sequence Search"
3. Filter for HIV-1 subtype B (or relevant subtype)

### Step 2: Query CTL Epitopes

For each epitope in our predictions:

| Epitope | Position | HLA | Our Barrier Score |
|:--------|:---------|:----|------------------:|
| SLYNTVATL | Gag 77-85 | A*02 | 3.68 |
| KRWIILGLNK | Gag 263-272 | B*27 | 4.40 |
| TSTLQEQIGW | Gag 240-249 | B*57 | 4.18 |
| FLKEKGGL | Nef 90-97 | A*24 | 4.40 |

### Step 3: Record Mutation Frequencies

For each epitope position:
1. Query amino acid frequency at each position
2. Note the wild-type vs mutant ratio
3. Calculate mutation frequency

### Step 4: Correlate with Predictions

| Epitope | Our Score | Mutation Frequency | Expected Correlation |
|:--------|----------:|-------------------:|:---------------------|
| High barrier (>4.0) | 4.40 | Low (< 5%) | Inverse |
| Moderate (3.5-4.0) | 3.68 | Moderate (5-15%) | Inverse |
| Low (< 3.5) | 3.28 | Higher (> 15%) | Inverse |

---

## Expected Results

### Elite Controller Epitopes (B*27, B*57)

These epitopes should show:
- **Low mutation frequency** (< 5% at anchor positions)
- **High fitness cost mutations** when escape does occur
- **Our barrier scores > 4.0**

### Non-Elite Epitopes

These epitopes should show:
- **Higher mutation frequency** (5-20%)
- **Lower fitness cost mutations**
- **Our barrier scores 3.0-4.0**

---

## Specific Queries

### Gag p24 KK10 (B*27)

Position 264 (Arg→Lys escape):
1. Search Gag p24 sequences
2. Check amino acid at position 264
3. Calculate R/K ratio
4. Expected: R264 conserved in >95% of sequences

### Gag p24 TW10 (B*57)

Position 242 (Thr→Asn escape):
1. Search Gag p24 sequences
2. Check amino acid at position 242
3. Calculate T/N ratio
4. Expected: T242 conserved in >90% of sequences

---

## Interpretation

### Strong validation:
- Inverse correlation (r < -0.7) between barrier score and mutation frequency
- Elite controller epitopes show lowest mutation frequencies

### Moderate validation:
- Trend in expected direction but with exceptions
- May indicate strain-specific effects

### Weak/no validation:
- Consider that database may include drug-experienced patients
- Barrier reflects fitness cost, not absolute conservation

---

## Confounding Factors

1. **Drug experience**: Some mutations may be selected by therapy
2. **Geographic sampling**: LANL has subtype B bias
3. **Time of sampling**: Recent vs historical sequences
4. **Transmission bottleneck**: Some mutations may be transient

---

## Data Recording Template

```
Epitope: ___________
Position: ___________
Wild-type AA: ___________
Mutant AA: ___________
Wild-type frequency: ___________
Mutant frequency: ___________
Our barrier score: ___________
Notes: ___________
```

---

*For full epitope list, see [../DATA/escape_predictions.json](../DATA/escape_predictions.json)*
