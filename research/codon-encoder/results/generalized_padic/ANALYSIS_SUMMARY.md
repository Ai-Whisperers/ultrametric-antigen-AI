# P-adic Generalization Analysis - Summary

**Date:** 2026-01-04
**Version:** v5.12.4 Ternary VAE

## Experiment Overview

Tested whether p-adic structure on amino acid indices (0-19) captures biologically meaningful relationships for DDG prediction.

## Results: Prime Comparison

| Prime | Spearman | Pearson | MAE |
|-------|----------|---------|-----|
| **p=2** | **0.3476** | 0.3236 | 1.160 |
| p=5 | 0.3472 | 0.3118 | 1.164 |
| p=3 | 0.3323 | 0.3256 | 1.155 |
| p=11 | 0.3284 | 0.3148 | 1.165 |
| p=7 | 0.3146 | 0.2946 | 1.165 |

**Best:** p=2 with Spearman=0.3476

## Comparison with Codon-Based Approach

| Approach | Spearman | Pearson |
|----------|----------|---------|
| AA-index p-adic (best) | 0.35 | 0.32 |
| **Codon-based v2** | **0.81** | **0.80** |

**Gap:** 2.3x worse for AA-index approach

## Key Insight

**P-adic structure on amino acid indices does NOT capture biological relationships.**

The codon-based approach works because:
1. Codons have natural 4^3 = 64 structure (4 bases × 3 positions)
2. Wobble position (3rd base) allows synonymous codons
3. P-adic distance captures evolutionary relationships:
   - Codons differing in 3rd position → p-adically close
   - Codons differing in 1st position → p-adically distant
4. Amino acids encoded by similar codons share evolutionary history

## Recommendation

For DDG prediction:
- Use codon-based p-adic structure (v2 approach)
- Aggregate codon embeddings per amino acid
- Don't apply p-adic structure directly to AA indices

For segment-based encoding:
- Apply to codon sequences, not AA sequences
- Use GeneralizedPadicEncoder with codon vocab (64 codons)
- Segment size: 10-20 codons with 50% overlap

## Files Created

- `train_generalized_padic_encoder.py` - Training script
- `prime_comparison.json` - Comparison results
- `aa_encoder_p*.pt` - Trained models per prime
- `ANALYSIS_SUMMARY.md` - This summary

## Next Steps

1. Create segment-based codon encoder (not AA encoder)
2. Test on long peptide DDG prediction
3. Integrate with v5.12.4 TernaryVAE embeddings
