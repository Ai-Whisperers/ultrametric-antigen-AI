# AlphaFold3 Input Sequences

**Purpose**: Ready-to-submit FASTA sequences for validating sentinel glycan predictions.

---

## Quick Start

1. Go to https://alphafoldserver.com/
2. Upload the FASTA file of your choice
3. Submit structure prediction job
4. Compare pLDDT scores with our predictions

---

## Files

| File | Description | Expected pLDDT |
|:-----|:------------|---------------:|
| `bg505_wt.fasta` | Wild-type BG505 gp120 | ~78 |
| `bg505_n58q.fasta` | N58 glycan removed | ~73 |
| `bg505_n429q.fasta` | N429 glycan removed | ~71 |
| `bg505_n103q.fasta` | N103 glycan removed | ~76 |
| `bg505_n204q.fasta` | N204 glycan removed | TBD |
| `bg505_all_goldilocks.fasta` | All 7 Goldilocks sites removed | ~68 |

---

## Sequence Details

**Base sequence**: BG505 SOSIP.664 gp120
**Length**: 476 amino acids
**Modifications**: N→Q substitutions at glycosylation sites

The N→Q substitution removes the glycan attachment site while minimizing structural perturbation (both are polar, similar size).

---

## Expected Validation Results

If our predictions are correct:

1. **N429Q** will show **maximum disorder** (pLDDT ~71, 100% disorder)
2. **Goldilocks sites** will show **more disorder** than non-Goldilocks sites
3. **Combined removal** will show **synergistic destabilization**
4. **Correlation r < -0.85** between our scores and pLDDT

---

## Notes

- AlphaFold3 may require glycan annotations for accurate modeling
- For best results, run 5 models per variant and take median pLDDT
- Focus on per-residue pLDDT at the mutation site and surrounding epitopes

---

*For interpretation guidance, see [../VALIDATION_PROTOCOLS/alphafold3_validation.md](../VALIDATION_PROTOCOLS/alphafold3_validation.md)*
