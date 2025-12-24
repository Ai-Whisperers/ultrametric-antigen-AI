# AlphaFold3 Validation Protocol

**Purpose**: Independent verification of sentinel glycan predictions
**Difficulty**: Low (requires only AlphaFold3 access)
**Time**: ~2 hours per variant

---

## Overview

Our platform predicts that specific glycan sites in HIV gp120 are in a "Goldilocks zone" - their removal causes optimal structural perturbation to expose bnAb epitopes without destabilizing the protein.

AlphaFold3 pLDDT scores provide independent validation of this prediction.

---

## Materials Required

1. AlphaFold3 server access (https://alphafoldserver.com/)
2. FASTA sequences (provided in [../SEQUENCES/](../SEQUENCES/))
3. Spreadsheet for recording results

---

## Protocol

### Step 1: Prepare Sequences

Use the BG505 SOSIP gp120 sequences provided:
- `bg505_wt.fasta` - Wild-type reference
- `bg505_n58q.fasta` - N58 removed (N→Q substitution)
- `bg505_n429q.fasta` - N429 removed
- `bg505_n103q.fasta` - N103 removed
- `bg505_all_goldilocks.fasta` - All 7 Goldilocks sites removed

### Step 2: Submit to AlphaFold3

For each sequence:
1. Navigate to AlphaFold3 server
2. Paste FASTA sequence
3. Select "Protein structure prediction"
4. Submit job

### Step 3: Record Results

For each prediction, record:
- **pLDDT score** (overall confidence)
- **Per-residue pLDDT** at modified positions
- **Disorder prediction** percentage

### Step 4: Calculate Correlation

Compare your results with our predictions:

| Variant | Our Perturbation Score | Expected pLDDT | Expected Disorder |
|:--------|----------------------:|--------------:|------------------:|
| Wild-type | 0.00 | ~78 | ~0% |
| N58Q | 1.19 | ~73 | ~75% |
| N429Q | 1.19 | ~71 | ~100% |
| N103Q | 1.04 | ~76 | ~67% |
| Combined | 2.50+ | ~68 | ~85% |

Calculate Pearson correlation between:
- Our perturbation scores
- Your measured pLDDT scores

Expected: r < -0.85 (strong inverse correlation)

---

## Interpretation

### If correlation confirmed (r < -0.85):
Our geometric predictions successfully identify structurally sensitive sites.

### If correlation moderate (-0.5 > r > -0.85):
Partial validation; some sites may have strain-specific effects.

### If no correlation (r > -0.5):
Predictions may be strain-specific or require different validation approach.

---

## Controls

### Positive control
N332 removal (known to destabilize V3 epitopes) - should show high disorder.

### Negative control
N-terminal modifications - should show minimal structural impact.

---

## Troubleshooting

| Issue | Solution |
|:------|:---------|
| pLDDT higher than expected | Check sequence alignment, ensure correct positions |
| Inconsistent results | Run 5 models per variant, take median |
| Server timeout | Break into smaller sequence fragments |

---

## Expected Results

If our predictions are correct:

1. **N429Q shows maximum disorder** (100%)
2. **Goldilocks sites show more disorder than non-Goldilocks**
3. **Combined removal shows synergistic destabilization**
4. **Strong inverse correlation (r ~ -0.89)**

---

## Data Recording Template

```
Variant: ___________
Date submitted: ___________
pLDDT (overall): ___________
pLDDT (modified residue): ___________
Disorder %: ___________
Notes: ___________
```

---

*Sequences provided in [../SEQUENCES/](../SEQUENCES/). Questions: See [../../pitch_tier2/CONTACT.md](../../pitch_tier2/CONTACT.md)*
