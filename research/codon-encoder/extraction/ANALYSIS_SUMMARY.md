# Codon Embedding Analysis Summary

**Doc-Type:** Technical Analysis · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Executive Summary

Analysis of the v5.12.3 TernaryVAE embeddings for DDG prediction reveals a fundamental mismatch: the VAE is trained on 3-adic ternary operations (3^9=19683), while codons have 4-adic structure (4^3=64). The mod 3 mapping loses critical information.

---

## Key Findings

### 1. VAE Hierarchy is Correct

The v5.12.3 encoder_B shows proper p-adic hierarchy:

| Valuation | Mean Mu Norm | Count |
|-----------|--------------|-------|
| v=0 | 7.22 | 13,122 |
| v=1 | 6.13 | 4,374 |
| v=2 | 5.44 | 1,458 |
| v=9 | 2.86 | 1 |

**Spearman correlation (valuation vs mu_norm): -0.633**

This confirms the VAE has learned the intended p-adic radial structure.

### 2. DDG Prediction Results (Honest LOO Evaluation)

| Method | LOO Spearman | LOO Pearson | Notes |
|--------|--------------|-------------|-------|
| P-adic distance only | **0.30** | 0.47 | Best sequence-only |
| VAE embeddings | 0.12 | 0.41 | Information loss in mapping |
| Original v2 (train) | 0.80 | 0.80 | Overfitting (CV R²=0.15) |
| Rosetta (structure) | 0.69 | - | Uses 3D structure |

### 3. The Mapping Problem

Codon → Ternary mapping using mod 3 loses critical information:

```
Codon bases: A=0, C=1, G=2, T=3
Ternary digits: mod 3 → T becomes 0

Result: ATG and AAG map to same ternary!
```

This explains why VAE embeddings (0.12) perform worse than simple p-adic (0.30).

---

## Recommendations

### Short-term: Use P-adic Distance

For immediate DDG prediction, use the `compute_padic_distance_between_codons` function from `src.encoders.codon_encoder`. This preserves the hierarchical structure of the genetic code.

**Expected performance: LOO Spearman ~0.30**

### Medium-term: Train Codon-Specific Encoder

Design a new encoder specifically for the 64 codons:

1. **Input**: 12-dim one-hot (4 bases × 3 positions)
2. **Output**: Hyperbolic embedding on Poincaré ball
3. **Loss**: Preserve codon p-adic structure + amino acid properties

This would avoid the information loss from 4→3 mapping.

### Long-term: Multi-Modal Integration

Combine:
- P-adic structure (genetic code hierarchy)
- Amino acid properties (physicochemical)
- ESM embeddings (contextual, if available)
- Position information (structural context)

---

## Files Created

| File | Purpose |
|------|---------|
| `src/encoders/hyperbolic_codon_encoder.py` | Hyperbolic encoder design |
| `research/codon-encoder/extraction/extract_hyperbolic_embeddings.py` | VAE extraction script |
| `research/codon-encoder/training/ddg_hyperbolic_training.py` | P-adic DDG training |
| `research/codon-encoder/training/ddg_vae_embeddings.py` | VAE DDG training |

---

## Technical Details

### Codon Embeddings Extracted

Saved to: `research/codon-encoder/extraction/results/codon_embeddings_v5_12_3.json`

Contains:
- 64 codon embeddings (16-dim tangent vectors)
- Mu norms (hierarchy proxy)
- Amino acid mappings

### Overfitting Detection

The original v2 training showed:
- Training Pearson: 0.80
- CV R² mean: 0.15 ± 0.47

**Overfitting ratio: 5.3x**

LOO evaluation gives honest baseline of Spearman ~0.30.

---

## Conclusion

The TernaryVAE learned proper p-adic hierarchy but cannot be directly applied to codons without information loss. For DDG prediction, simple p-adic codon distances outperform VAE embeddings until a codon-specific encoder is trained.
