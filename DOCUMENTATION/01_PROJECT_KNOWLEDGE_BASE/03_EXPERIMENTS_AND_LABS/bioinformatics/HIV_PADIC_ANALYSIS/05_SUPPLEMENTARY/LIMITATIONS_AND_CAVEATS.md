# Limitations and Caveats

## Critical Considerations for Interpreting Results

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Overview

Every scientific analysis has limitations. This document transparently describes the constraints, assumptions, and potential biases in our HIV p-adic hyperbolic analysis. Understanding these limitations is essential for proper interpretation and future improvements.

---

## 1. Data Limitations

### 1.1 Subtype Bias

**Issue:** Predominant representation of HIV-1 subtype B

| Dataset | Subtype B | Other Subtypes |
|---------|-----------|----------------|
| Stanford HIVDB | ~85% | ~15% |
| CATNAP | ~42% | ~58% |
| V3 Coreceptor | ~60% | ~40% |

**Impact:**
- Findings may not generalize to subtypes C, A, D, CRFs
- Subtype-specific mutations may be underrepresented
- Geographic bias toward North America/Europe

**Mitigation:**
- Results should be validated in non-B subtypes before clinical application
- Subtype C is particularly important (>50% global infections)

**Recommendation:** Future work should include subtype-stratified analyses.

---

### 1.2 Temporal Bias

**Issue:** Cross-sectional data, no longitudinal tracking

**Impact:**
- Cannot observe mutation trajectories over time
- Cannot measure actual escape rates
- Selection pressures inferred, not directly measured

**Example:**
We calculate "escape velocity" from population-level data, but this is a proxy for actual within-patient escape dynamics.

**Recommendation:** Integrate with longitudinal cohort data when available.

---

### 1.3 Treatment History Unknown

**Issue:** Stanford HIVDB does not consistently record prior treatments

**Impact:**
- Cannot distinguish transmitted vs. acquired resistance
- Mutation order cannot be determined
- Compensatory mutation context may be incomplete

**Mitigation:**
- Analyses focus on mutation presence, not emergence order
- Interpretations framed as associations, not causation

---

### 1.4 CATNAP Selection Bias

**Issue:** CATNAP contains curated research antibodies, not representative of natural immunity

**Impact:**
- Antibody breadth/potency reflects research-selected bnAbs
- Natural antibody responses may differ substantially
- Vaccine-elicited responses underrepresented

**Mitigation:**
- bnAb findings apply to therapeutic antibody development
- Vaccine implications require additional validation

---

## 2. Methodological Limitations

### 2.1 Codon Approximation

**Issue:** We use representative codons for each amino acid

```python
# Example: All M (Methionine) encoded as ATG
# But synonymous variation exists for other amino acids
'L': 'CTT'  # But could be TTA, TTG, CTC, CTA, CTG
```

**Impact:**
- Synonymous variation within amino acid classes is lost
- Cannot distinguish synonymous vs. non-synonymous at codon level
- May miss signals in synonymous codon usage

**Magnitude:**
Affects ~40% of possible mutations (synonymous changes ignored)

**Recommendation:** Future versions should use actual codon sequences where available.

---

### 2.2 Position Mapping Limitations

**Issue:** HXB2 reference-based mapping

**Impact:**
- Insertions/deletions relative to HXB2 may be misaligned
- Some sequences with large indels excluded
- Variable loop lengths not fully captured

**Affected Regions:**
- V1/V2 loops (high indel frequency)
- V4/V5 loops (moderate indel frequency)
- Generally stable in PR, RT, IN

---

### 2.3 Embedding Dimensionality

**Issue:** 16-dimensional Poincaré ball is arbitrary choice

**Impact:**
- May under- or over-represent codon relationships
- Optimal dimensionality unknown
- Different dimensions may suit different analyses

**Validation:**
We validated that 16D captures conservation (r=0.67), but this doesn't prove optimality.

**Recommendation:** Sensitivity analysis across different dimensionalities.

---

### 2.4 Linear Correlations

**Issue:** Many analyses use Pearson correlation (assumes linearity)

**Impact:**
- Non-linear relationships may be underestimated
- Outliers can distort correlation estimates

**Mitigation:**
- Spearman correlations reported alongside Pearson
- Visual inspection of scatter plots
- Non-linear models (Random Forest) used for prediction

---

## 3. Statistical Limitations

### 3.1 Multiple Testing

**Issue:** Thousands of tests performed

| Analysis | Tests | Correction |
|----------|-------|------------|
| Mutation-distance | 3,647 | Bonferroni |
| Position-tropism | 35 | Bonferroni |
| Epitope comparisons | 2,115 | FDR |

**Impact:**
- Even with correction, some false positives expected
- Bonferroni is very conservative (may miss true effects)
- FDR controls proportion, not absolute number

**Recommendation:** Key findings should be independently validated.

---

### 3.2 Effect Size vs. Sample Size

**Issue:** Large sample sizes produce significant p-values for small effects

**Example:**
```
Tropism centroid distance: 0.0222
p-value: 0.992 (not significant)

BUT: If sample size were 10x larger, this small effect
might become "significant" despite being biologically trivial.
```

**Mitigation:**
- Effect sizes reported alongside p-values
- Practical significance discussed separately from statistical significance

---

### 3.3 Correlation ≠ Causation

**Issue:** All findings are correlational

**Examples:**
- Distance correlates with resistance, but doesn't prove geometric cause
- Centrality correlates with breadth, but other factors may explain both
- Trade-off scores predict outcomes, but mechanism unproven

**Recommendation:** Experimental validation needed for causal claims.

---

## 4. Biological Limitations

### 4.1 Protein Structure Not Integrated

**Issue:** Analysis is sequence-based, ignoring 3D structure

**Impact:**
- Surface accessibility not considered
- Protein-protein interfaces not weighted
- Allosteric effects not captured

**Example:**
Two mutations with same geometric distance may have very different structural impacts (surface vs. buried, binding site vs. distal).

**Recommendation:** Future work should integrate structural data (AlphaFold, experimental structures).

---

### 4.2 Epistasis Ignored

**Issue:** Mutations analyzed independently

**Impact:**
- Synergistic/antagonistic combinations not detected
- Higher-order interactions missed
- Fitness landscape topology oversimplified

**Example:**
M184V + K65R together have different effect than sum of individual effects.

**Mitigation:**
- Co-occurrence patterns could be added
- Network-based analysis of mutation combinations

---

### 4.3 Fitness Costs Not Measured

**Issue:** Replication capacity data not integrated

**Impact:**
- "Constraint" inferred from conservation, not directly measured
- Escape velocity assumes equal fitness for all mutations
- May overestimate constraint at truly deleterious positions

**Recommendation:** Integrate with experimental fitness data where available.

---

### 4.4 Host Factors Excluded

**Issue:** Analysis focuses on virus, not host

**Excluded Factors:**
- HLA frequencies in population
- Innate immunity (APOBEC, tetherin, SAMHD1)
- Host genetics beyond HLA
- Microbiome effects
- Co-infections

**Impact:**
- Vaccine targets ranked by HLA count, but HLA frequencies vary by population
- Some "protective" HLAs may be rare in high-burden regions

---

## 5. Interpretation Caveats

### 5.1 Vaccine Targets

**Caveat:** High-scoring epitopes are candidates, not proven targets

**What We Show:**
- Epitopes with broad HLA restriction
- Low geometric escape velocity
- No drug resistance overlap

**What We Don't Show:**
- Actual immunogenicity
- Processing/presentation efficiency
- Memory response durability
- Protection in challenge studies

**Recommendation:** Top targets require immunological validation.

---

### 5.2 Tropism Prediction

**Caveat:** 85% accuracy means 15% error rate

**Clinical Implications:**
- False R5 prediction → Maraviroc failure
- False X4 prediction → Unnecessary drug exclusion

**Comparison:**
Our method is comparable to established tools, but all methods have error margins.

**Recommendation:** Combine with phenotypic testing for clinical decisions.

---

### 5.3 Resistance Prediction

**Caveat:** Novel mutations are extrapolated, not directly measured

**What We Can Do:**
- Predict likely resistance based on geometric similarity to known mutations

**What We Cannot Do:**
- Prove resistance without phenotypic testing
- Account for novel mechanisms

---

## 6. Technical Caveats

### 6.1 Reproducibility Window

**Issue:** External databases update over time

- Stanford HIVDB adds new sequences
- CATNAP adds new antibodies/viruses
- Nomenclature may change

**Mitigation:**
- Analysis dated (December 2025)
- Specific versions should be archived

---

### 6.2 Software Dependencies

**Issue:** Package versions affect numerical results

```
numpy==1.24.3 may give slightly different results than numpy==1.26.0
```

**Mitigation:**
- Exact versions recorded in requirements.txt
- Key results robust to minor version changes

---

## 7. What This Analysis Cannot Do

| Cannot Do | Why | Alternative |
|-----------|-----|-------------|
| Prove causation | Observational data | Experimental studies |
| Replace phenotypic testing | Predictions have error | Clinical labs |
| Design vaccines directly | Immunogenicity untested | Vaccine trials |
| Treat patients | Not clinical tool | Physician judgment |
| Guarantee predictions | Biology is complex | Probability ranges |

---

## 8. Recommendations for Users

### For Researchers:

1. Validate key findings before building on them
2. Consider subtype when applying to non-B viruses
3. Use effect sizes, not just p-values
4. Read NOVELTY_ASSESSMENT.md for prior work

### For Clinicians:

1. This is research, not a clinical tool
2. Tropism predictions require confirmation
3. Vaccine targets require immunological validation
4. Consult IAS-USA guidelines for resistance interpretation

### For Reviewers:

1. Novel findings require independent validation
2. Confirmations strengthen method validity
3. Limitations are explicitly stated
4. Reproducibility guide provided

---

## 9. Future Improvements

| Limitation | Proposed Solution | Priority |
|------------|-------------------|----------|
| Subtype bias | Multi-subtype validation | High |
| Codon approximation | Use nucleotide sequences | High |
| No structure | Integrate AlphaFold | Medium |
| No epistasis | Mutation network analysis | Medium |
| Cross-sectional | Longitudinal cohorts | Medium |
| No host factors | HLA population frequencies | Low |

---

## 10. Summary

This analysis provides valuable insights but should be interpreted with awareness of:

1. **Data biases** (subtype B predominant, curated databases)
2. **Methodological approximations** (representative codons, reference mapping)
3. **Statistical constraints** (correlation not causation, multiple testing)
4. **Biological simplifications** (no structure, no epistasis, no host factors)

**The findings are scientifically rigorous within these constraints**, and the framework provides a foundation for future refinement.

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
