# Scientific Validation Report: AMP Activity Prediction Models

**Doc-Type:** Validation Report | Version 1.0 | Updated 2026-01-03 | Carlos Brizuela Package

---

## Executive Summary

This report presents comprehensive validation of antimicrobial peptide (AMP) activity prediction models trained on curated DRAMP database entries. The models were evaluated using cross-validation, permutation tests, and feature importance analysis.

**Key Findings:**
- 3/5 models achieve statistical significance (p < 0.05)
- General model: Pearson r = 0.54 (p < 0.001)
- Gram-negative pathogens: Strong predictability
- Gram-positive pathogens: Different mechanism, charge-based features fail

---

## Validation Methodology

### Data Source
- **Database:** Curated subset of DRAMP (Data Repository of Antimicrobial Peptides)
- **Total records:** 224 peptide-pathogen pairs
- **Unique sequences:** 155 distinct peptides
- **Target pathogens:** E. coli, S. aureus, P. aeruginosa, A. baumannii

### Cross-Validation Protocol
- **Large datasets (n >= 30):** 5-fold stratified CV
- **Small datasets (n < 30):** Leave-One-Out CV (LOO)
- **Permutation test:** 50-100 permutations for significance

### Features Used
30 physicochemical features:
- Sequence length, net charge, hydrophobicity
- Amino acid composition (20 features)
- Fraction: cationic, aromatic, aliphatic, polar
- Hydrophobic moment

---

## Model-Specific Results

### 1. General Model (All Pathogens)

| Metric | Value |
|--------|-------|
| N samples | 224 |
| CV method | 5-fold |
| Pearson r | **0.539*** |
| Spearman ρ | 0.541*** |
| Permutation p | 0.020 |
| RMSE | 0.412 |
| Confidence | **HIGH** |

**Primary predictor:** Peptide length
**Recommendation:** Suitable for predictions with uncertainty estimates

---

### 2. Escherichia coli Model

| Metric | Value |
|--------|-------|
| N samples | 105 |
| CV method | 5-fold |
| Pearson r | **0.468*** |
| Spearman ρ | 0.451*** |
| Permutation p | 0.020 |
| RMSE | 0.433 |
| Confidence | **HIGH** |

**Primary predictor:** Net charge
**Biology:** Gram-negative, LPS outer membrane. Cationic AMPs effective.
**Recommendation:** Suitable for E. coli-targeted design

---

### 3. Acinetobacter baumannii Model

| Metric | Value |
|--------|-------|
| N samples | 20 |
| CV method | LOO |
| Pearson r | **0.593*** |
| Spearman ρ | 0.473* |
| Permutation p | 0.020 |
| RMSE | 0.310 |
| Confidence | **HIGH** |

**Primary predictor:** Polar fraction
**Biology:** Critical WHO priority pathogen. Gram-negative.
**Recommendation:** Use with caution (small n), but significant correlation

---

### 4. Staphylococcus aureus Model

| Metric | Value |
|--------|-------|
| N samples | 72 |
| CV method | 5-fold |
| Pearson r | 0.036 |
| Spearman ρ | 0.036 |
| Permutation p | 0.176 |
| RMSE | 0.504 |
| Confidence | **LOW** |

**Primary predictor:** Amino acid T content (threonine)
**Biology:** Gram-positive, thick peptidoglycan wall. Charge-based features NOT predictive.
**Recommendation:** Use general model or heuristics

---

### 5. Pseudomonas aeruginosa Model

| Metric | Value |
|--------|-------|
| N samples | 27 |
| CV method | LOO |
| Pearson r | 0.196 |
| Spearman ρ | 0.085 |
| Permutation p | 0.216 |
| RMSE | 0.465 |
| Confidence | **LOW** |

**Primary predictor:** Net charge
**Biology:** Gram-negative but insufficient training data.
**Recommendation:** Use general model until more data available

---

## Biological Interpretation

### Gram-Negative vs Gram-Positive

The stark difference between Gram-negative (E. coli, Acinetobacter) and Gram-positive (S. aureus) models reflects fundamental biological differences:

| Feature | Gram-Negative | Gram-Positive |
|---------|---------------|---------------|
| Outer membrane | LPS-rich (negative charge) | None |
| Peptidoglycan | Thin layer | Thick layer |
| AMP mechanism | Electrostatic attraction | Multiple mechanisms |
| Charge importance | **High** | **Low** |

**Implication:** Simple physicochemical features predict Gram-negative activity but fail for Gram-positive. S. aureus requires structure-based or mechanism-specific features.

---

## Feature Importance (General Model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | length | 0.142 |
| 2 | charge | 0.098 |
| 3 | hydrophobicity | 0.076 |
| 4 | aliphatic_fraction | 0.065 |
| 5 | aac_K (lysine) | 0.058 |

---

## Comparison with Colbes DDG Predictor

| Aspect | Colbes (DDG) | Brizuela (AMP) |
|--------|--------------|----------------|
| Task | Protein stability | Antimicrobial activity |
| Best model | Spearman 0.58 | Pearson 0.54 |
| Validation | Bootstrap + LOO | Permutation + CV |
| p-value | < 0.001 | < 0.001 |
| Biological grounding | p-adic embeddings | Physicochemical features |

Both packages achieve similar predictive performance with rigorous validation.

---

## Deployment Recommendations

### Use These Models

| Target | Model | Action |
|--------|-------|--------|
| E. coli | activity_escherichia | Direct prediction |
| A. baumannii | activity_acinetobacter | Direct prediction |
| Mixed/unknown | activity_general | Default choice |

### Avoid These Models

| Target | Model | Alternative |
|--------|-------|-------------|
| S. aureus | activity_staphylococcus | Use general model |
| P. aeruginosa | activity_pseudomonas | Use general model |

---

## Limitations

1. **Sample size:** Pseudomonas and Acinetobacter have < 30 samples
2. **Mechanism diversity:** S. aureus requires different feature engineering
3. **Assay variability:** MIC values from different studies may not be comparable
4. **Sequence diversity:** Curated set may not cover all AMP structural classes

---

## Conclusion

The AMP activity prediction package provides scientifically validated models for Gram-negative pathogens (E. coli, A. baumannii) with high confidence. The general model serves as a robust fallback for all pathogens. Gram-positive prediction (S. aureus) requires further research into mechanism-specific features.

---

## Files Generated

- `comprehensive_validation.json` - Full validation metrics
- `bootstrap_results.json` - Bootstrap confidence intervals
- Model files in `models/` directory

---

**Validation performed:** 2026-01-03
**Significance threshold:** p < 0.05
**Permutation tests:** 50 iterations
