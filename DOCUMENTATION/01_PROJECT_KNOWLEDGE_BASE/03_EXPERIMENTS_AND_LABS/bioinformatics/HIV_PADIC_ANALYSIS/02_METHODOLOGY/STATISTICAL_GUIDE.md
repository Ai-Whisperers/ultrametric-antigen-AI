# Statistical Interpretation Guide

## Understanding the Statistical Methods and Results

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Table of Contents

1. [Key Statistical Concepts](#1-key-statistical-concepts)
2. [Interpretation of P-values](#2-interpretation-of-p-values)
3. [Effect Size Metrics](#3-effect-size-metrics)
4. [Multiple Testing Correction](#4-multiple-testing-correction)
5. [Machine Learning Metrics](#5-machine-learning-metrics)
6. [Common Pitfalls](#6-common-pitfalls)
7. [How to Read Our Results](#7-how-to-read-our-results)

---

## 1. Key Statistical Concepts

### 1.1 Null Hypothesis Testing

**Framework:**
- H₀ (Null): No effect or relationship exists
- H₁ (Alternative): Effect or relationship exists
- p-value: Probability of observing data as extreme if H₀ is true

**Example from our analysis:**

```
Test: Is hyperbolic distance correlated with drug resistance?
H₀: r = 0 (no correlation)
H₁: r ≠ 0 (correlation exists)
Result: r = 0.34, p < 10^-50
Interpretation: Strong evidence against H₀; correlation exists
```

### 1.2 Confidence Intervals

**Definition:** Range of values likely to contain the true parameter

**Example:**

```
Correlation: r = 0.34
95% CI: [0.31, 0.37]

Interpretation:
- Point estimate is 0.34
- We are 95% confident true correlation is between 0.31 and 0.37
- CI does not include 0, confirming significance
```

### 1.3 Statistical vs. Practical Significance

**Important distinction:**

| Statistical Significance | Practical Significance |
|-------------------------|----------------------|
| p < 0.05 | Effect size large enough to matter |
| Rejects H₀ | Has real-world implications |
| Can be achieved with large N | Independent of sample size |

**Our approach:** We report both p-values AND effect sizes to assess both types of significance.

---

## 2. Interpretation of P-values

### 2.1 P-value Thresholds Used

| p-value | Interpretation | Symbol |
|---------|----------------|--------|
| p < 0.001 | Highly significant | *** |
| 0.001 ≤ p < 0.01 | Very significant | ** |
| 0.01 ≤ p < 0.05 | Significant | * |
| 0.05 ≤ p < 0.10 | Marginally significant | . |
| p ≥ 0.10 | Not significant | ns |

### 2.2 Very Small P-values

Our analyses often produce extremely small p-values (e.g., p < 10^-50). This reflects:
1. Large sample sizes (189,879 neutralization records)
2. Strong biological effects
3. Well-powered statistical tests

**How to interpret:**

```
p < 10^-50 means:
- Effect is definitely not due to chance
- But doesn't tell us how large the effect is
- Always check effect size alongside p-value
```

### 2.3 Non-significant Results

When p > 0.05:
- Does NOT prove no effect exists
- May reflect low power or small effect
- Consider confidence intervals

**Example from tropism analysis:**

```
CCR5 vs CXCR4 centroid distance: 0.0222
p-value: 0.992

Interpretation:
- Centroids are NOT significantly different
- Tropism distinction requires position-specific analysis
- Global embedding doesn't capture tropism
```

---

## 3. Effect Size Metrics

### 3.1 Correlation Coefficients

**Pearson's r (continuous variables):**

| |r| | Interpretation |
|-----|----------------|
| 0.00-0.10 | Negligible |
| 0.10-0.30 | Weak |
| 0.30-0.50 | Moderate |
| 0.50-0.70 | Strong |
| 0.70-1.00 | Very strong |

**Our key correlations:**

| Analysis | r | Interpretation |
|----------|---|----------------|
| Resistance-distance | 0.34-0.41 | Moderate |
| Conservation-radius | 0.67 | Strong |
| Breadth-centrality | 0.68 | Strong |

### 3.2 Cohen's d (Group Comparisons)

**Formula:**
```
d = (μ₁ - μ₂) / σ_pooled
```

**Interpretation:**

| |d| | Interpretation |
|-----|----------------|
| 0.00-0.20 | Small |
| 0.20-0.50 | Small-Medium |
| 0.50-0.80 | Medium |
| 0.80-1.20 | Large |
| > 1.20 | Very Large |

**Our key effect sizes:**

| Analysis | d | Interpretation |
|----------|---|----------------|
| Primary vs accessory mutations | 1.24 | Very Large |
| Tropism radial separation | 0.08 | Small (expected) |
| TDF resistance distance | 1.62 | Very Large |

### 3.3 Eta-squared (ANOVA)

**Formula:**
```
η² = SS_between / SS_total
```

**Interpretation:**

| η² | Interpretation |
|----|----------------|
| 0.01-0.06 | Small |
| 0.06-0.14 | Medium |
| > 0.14 | Large |

**Our key η² values:**

| Analysis | η² | Interpretation |
|----------|----|--------------  |
| Protein escape velocity | 0.18 | Large |
| Epitope class potency | 0.24 | Large |

### 3.4 AUC-ROC (Classification)

**Area Under ROC Curve:**

| AUC | Interpretation |
|-----|----------------|
| 0.50-0.60 | No discrimination |
| 0.60-0.70 | Poor |
| 0.70-0.80 | Acceptable |
| 0.80-0.90 | Excellent |
| 0.90-1.00 | Outstanding |

**Our classifier performance:**

| Model | AUC | Interpretation |
|-------|-----|----------------|
| Tropism prediction | 0.86 | Excellent |
| Primary/accessory | 0.87 | Excellent |
| Neutralization prediction | 0.86 | Excellent |

---

## 4. Multiple Testing Correction

### 4.1 The Problem

When performing many tests, some will be significant by chance:
- 100 tests at α = 0.05 → expect 5 false positives
- 3,647 mutations tested → expect 182 false positives

### 4.2 Bonferroni Correction

**Most conservative approach:**

```
α_adjusted = α / m

Where:
- α = desired significance level (0.05)
- m = number of tests

For 3,647 mutations:
α_adjusted = 0.05 / 3647 = 1.37 × 10^-5
```

### 4.3 Benjamini-Hochberg FDR

**Less conservative, controls false discovery rate:**

```
1. Rank p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
2. Find largest k where p(k) ≤ (k/m) × q
3. Reject all H_i for i ≤ k

Where q = desired FDR (typically 0.05 or 0.10)
```

### 4.4 Our Approach

| Analysis | Correction Method | Threshold |
|----------|-------------------|-----------|
| Drug resistance | Bonferroni | 1.37 × 10^-5 |
| CTL epitopes | Bonferroni | 2.36 × 10^-5 |
| Neutralization | BH FDR | q = 0.05 |
| Position analysis | Bonferroni | Per-analysis |

---

## 5. Machine Learning Metrics

### 5.1 Classification Metrics

**Confusion Matrix:**

```
                 Predicted
              |  Pos  |  Neg  |
    Actual Pos|  TP   |  FN   |
    Actual Neg|  FP   |  TN   |
```

**Derived Metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Positive predictive value |
| Recall (Sensitivity) | TP/(TP+FN) | True positive rate |
| Specificity | TN/(TN+FP) | True negative rate |
| F1 Score | 2×(Prec×Rec)/(Prec+Rec) | Harmonic mean |

### 5.2 Cross-Validation

**5-Fold Stratified CV:**

```
Data split into 5 equal folds
For each fold:
  - Train on 4 folds (80%)
  - Test on 1 fold (20%)
  - Record performance
Report mean ± std across folds
```

**Interpreting CV Results:**

| CV Mean | CV Std | Interpretation |
|---------|--------|----------------|
| High | Low | Stable, reliable model |
| High | High | Variable, may overfit |
| Low | Low | Consistently poor |
| Low | High | Unstable, needs work |

**Our CV results:**

```
Tropism classifier:
- CV Mean: 0.859
- CV Std: 0.013
- 95% CI: [0.846, 0.872]

Interpretation: Stable, reliable performance
```

### 5.3 ROC and AUC

**Receiver Operating Characteristic:**
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Sensitivity)
- AUC = Area under the curve

**Interpretation:**
- AUC = 0.5: Random classifier
- AUC = 1.0: Perfect classifier
- Higher is better

---

## 6. Common Pitfalls

### 6.1 P-value Misinterpretation

**Wrong:** "p = 0.03 means 97% probability the effect is real"

**Correct:** "p = 0.03 means 3% probability of seeing this data if no effect exists"

### 6.2 Confusing Significance with Effect Size

**Scenario:**
```
Study 1: n = 100, r = 0.40, p = 0.001
Study 2: n = 10,000, r = 0.10, p = 0.001

Both significant, but Study 1 has larger effect!
```

### 6.3 Ignoring Multiple Testing

**Scenario:**
```
Test 1000 positions for association
Find 50 with p < 0.05
Expected by chance: 50

Conclusion: Need multiple testing correction!
```

### 6.4 Overfitting in Machine Learning

**Signs of overfitting:**
- Training accuracy >> Test accuracy
- High variance in CV
- Model too complex for data

**Our mitigation:**
- Cross-validation
- Simple models (logistic regression, random forest)
- Feature selection (PCA)

---

## 7. How to Read Our Results

### 7.1 Standard Results Table Format

```
| Analysis | Metric | Value | 95% CI | p-value | Effect Size | Interpretation |
|----------|--------|-------|--------|---------|-------------|----------------|
| Example  | r      | 0.34  | [0.31, 0.37] | <10^-50 | Moderate | Significant correlation |
```

**Reading order:**
1. Check p-value (is it significant after correction?)
2. Check effect size (is it practically meaningful?)
3. Check confidence interval (is it narrow?)
4. Read interpretation

### 7.2 Example: Drug Resistance Correlation

```
Drug Class: NRTI
Correlation (r): 0.41
95% CI: [0.38, 0.44]
p-value: <10^-78
Effect Size: Moderate
n: 21,456

Step-by-step interpretation:

1. p < 10^-78: Extremely significant, well below
   Bonferroni threshold (1.37 × 10^-5)

2. r = 0.41: Moderate correlation
   - Hyperbolic distance explains ~17% of fold-change variance (r²)
   - Meaningful biological relationship

3. CI [0.38, 0.44]: Narrow interval
   - Precise estimate
   - Entire interval indicates moderate effect

4. n = 21,456: Large sample
   - Well-powered analysis
   - Confirms generalizability

Conclusion: Hyperbolic distance moderately predicts NRTI
resistance. The geometric framework captures real biological
signal in drug resistance evolution.
```

### 7.3 Example: Tropism Separation

```
Metric: Centroid Distance (CCR5 vs CXCR4)
Value: 0.0222
p-value: 0.992
Effect Size (d): 0.08 (Small)

Step-by-step interpretation:

1. p = 0.992: Not significant
   - Cannot reject null hypothesis
   - Centroids not distinguishable

2. d = 0.08: Very small effect
   - Groups heavily overlap
   - Global embedding insufficient

3. Biological meaning:
   - Tropism not determined by overall sequence character
   - Position-specific analysis needed
   - V3 loop diversity is similar between tropisms

Conclusion: This "negative" result is informative—it tells us
that position-specific geometric features, not global
embeddings, determine tropism.
```

### 7.4 Reporting Standards Used

All analyses follow these standards:

1. **Sample sizes always reported**
2. **Exact p-values when possible** (not just "p < 0.05")
3. **Effect sizes alongside significance**
4. **Confidence intervals for key estimates**
5. **Multiple testing correction applied**
6. **Cross-validation for ML models**

---

## Appendix A: Statistical Formulas

### Pearson Correlation

```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

### Cohen's d

```
d = (μ₁ - μ₂) / √[(s₁² + s₂²) / 2]
```

### Mann-Whitney U

```
U = n₁n₂ + n₁(n₁+1)/2 - R₁

Where R₁ = sum of ranks for group 1
```

### Kruskal-Wallis H

```
H = (12 / N(N+1)) × Σ(Ri²/ni) - 3(N+1)

Where Ri = sum of ranks in group i
```

### AUC-ROC

```
AUC = Σ(TPR × ΔFPR) over all thresholds
```

---

## Appendix B: Software and Packages

All analyses performed with:

| Package | Version | Use |
|---------|---------|-----|
| scipy | 1.11.0 | Statistical tests |
| statsmodels | 0.14.0 | Advanced statistics |
| scikit-learn | 1.3.0 | Machine learning |
| numpy | 1.24.0 | Numerical operations |
| pandas | 2.0.0 | Data manipulation |

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
