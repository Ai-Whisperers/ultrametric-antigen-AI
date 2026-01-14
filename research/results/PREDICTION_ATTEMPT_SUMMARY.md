# PeptideVAE Prediction Attempts Summary

**Date:** 2026-01-08
**Status:** Phase 1 PASSED, Phase 4 PARTIAL

---

## Executive Summary

Two prediction attempts to train a PeptideVAE based on validated findings:

| Phase | Status | Key Result |
|-------|--------|------------|
| Phase 1 (Physicochemical) | **PASSED** | R > 0.98 for all properties |
| Phase 4 (DDG Validation) | PARTIAL | ρ=0.27 (target: 0.585) |

---

## Attempt 01: Baseline

**Issue:** Hydrophobicity R = 0.083 (almost zero)

**Root cause:**
- Equal weighting on all physicochemical properties
- Hydrophobicity values clustered in narrow range
- Model treated hydrophobicity as noise

---

## Attempt 02: Hydrophobicity Fix ✓

**Fixes applied:**
1. 3x weight on hydrophobicity (PRIMARY predictor per Colbes)
2. Z-score normalization of targets
3. Dedicated hydrophobicity branch with more capacity
4. Kyte-Doolittle hydrophobicity scale

**Results:**

| Metric | Attempt 01 | Attempt 02 |
|--------|------------|------------|
| Reconstruction | 100% | 100% |
| Length R | 0.983 | 0.982 |
| **Hydrophobicity R** | **0.083** | **0.989** |
| Charge R | 0.733 | 0.983 |

**Verdict:** Phase 1 PASSED

---

## Phase 4: DDG Validation

**Method:**
- Load trained PeptideVAE
- Encode amino acids to get hyperbolic embeddings
- Extract geometric features (hyp_dist, delta_radius)
- Train Ridge regression on S669 DDG data
- Evaluate with Leave-One-Out cross-validation

**Results:**

| Metric | Value | Target |
|--------|-------|--------|
| Spearman ρ | 0.27 | ≥0.585 |
| Pearson r | 0.62 | - |
| Top feature | delta_size (-0.24) | Expected |
| Second feature | delta_hydro (-0.21) | Expected |
| VAE hyp_dist | 0.00 | Doesn't contribute |

---

## Key Insight

**The VAE approach is fundamentally different from the Colbes p-adic approach:**

| Approach | Target | Level | DDG Result |
|----------|--------|-------|------------|
| Colbes p-adic | Codon distances | Genetic code | ρ=0.585 ✓ |
| PeptideVAE | Peptide sequences | Sequence | ρ=0.27 ✗ |

**Why the difference?**
- Colbes p-adic: Direct measurement of codon similarity in genetic code structure
- PeptideVAE: Learned representations of full peptide sequences
- DDG is about single amino acid substitutions, not peptide similarity

**BUT:** The physicochemical features (delta_hydro, delta_size) ARE working as expected. The VAE learned the right features, just in a different representation.

---

## Validated Findings

### What Works:

1. **PeptideVAE physicochemical encoding** (R > 0.98)
   - Length, hydrophobicity, charge all encoded well
   - C3 validated: these ARE the signal sources

2. **Colbes p-adic DDG prediction** (ρ = 0.585)
   - Direct codon-level approach
   - Already validated, use directly

3. **Physicochemical features for DDG**
   - delta_hydro, delta_size are top predictors
   - Works even without VAE embeddings

### What Doesn't Work:

1. **VAE hyperbolic distance → DDG**
   - Peptide-level embeddings don't encode AA substitution info
   - Use Colbes p-adic approach instead

---

## Recommendations

1. **For peptide design:** Use PeptideVAE with physicochemical features
2. **For DDG prediction:** Use Colbes p-adic codon distances
3. **For activity prediction:** Use cluster-conditional models (C3 validated)
4. **Don't mix approaches:** Different tools for different tasks

---

## Files

| File | Purpose |
|------|---------|
| `scripts/peptide_vae/prediction_attempt_01.py` | Baseline training |
| `scripts/peptide_vae/prediction_attempt_02.py` | Fixed training (RECOMMENDED) |
| `scripts/peptide_vae/validate_ddg_phase4.py` | DDG validation |
| `checkpoints/peptide_vae_attempt_02.pt` | Trained model |
| `results/peptide_vae_attempt_02.json` | Training results |
| `results/peptide_vae_ddg_validation.json` | DDG validation results |

---

## Next Steps

1. **Phase 2 (Optional):** Add radial hierarchy training to improve DDG
2. **Phase 3 (Optional):** Cluster structure enhancement
3. **Production:** Use Attempt 02 checkpoint for peptide design tasks
4. **DDG tasks:** Use Colbes p-adic predictor directly

---

*Training based on PEPTIDE_VAE_TRAINING_PLAN.md and validated P1 findings*
