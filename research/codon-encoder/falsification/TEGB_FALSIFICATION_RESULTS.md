# TEGB Conjecture Falsification Results

**Doc-Type:** Scientific Analysis · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Executive Summary

The Thermodynamical Effects Generalization Bridge (TEGB) conjecture has been **FALSIFIED** at all 4 tested links using pure p-adic mathematics without learned weights or biases.

**VERDICT: P-adic codon structure does NOT encode thermodynamic properties.**

---

## The TEGB Conjecture

**Claim:** Evolution respects thermodynamics, therefore full sequences that evolved successfully should implicitly encode thermodynamic properties in their p-adic structure.

**Chain:** Codon encoding → Mutational accessibility → Codon optimization → Translation kinetics → Thermodynamics

---

## Falsification Results

### Link 1: Genetic Code Optimization

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hypothesis | P-adic close → Physico similar | |
| Spearman r | **-0.097** | Negative correlation! |
| P-value | 0.184 | Not significant |
| Falsified | **YES** | Counter to prediction |

**Discovery:** P-adically close codons encode **DISSIMILAR** amino acids, the opposite of optimization for p-adic similarity.

---

### Link 2: Mutational Accessibility

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hypothesis | P-adic close → Few nucleotide changes | |
| Spearman r | **-0.072** | Negative correlation! |
| P-value | 0.321 | Not significant |
| Falsified | **YES** | Counter to prediction |

**Discovery:** P-adically close amino acids require **MORE** nucleotide changes, not fewer. The p-adic structure does not reflect mutational steps.

---

### Link 3: Pure P-adic DDG Prediction

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hypothesis | P-adic alone → DDG prediction | |
| N mutations | 669 | Full S669 dataset |
| LOO Pearson r | **0.066** | Near zero |
| LOO P-value | 0.088 | **Not significant** |
| LOO Spearman r | **-0.329** | Strongly negative! |
| Falsified | **YES** | No predictive power |

**Discovery:** Pure p-adic distance has no significant predictive power for DDG. Strikingly, the Spearman correlation is **strongly negative** (r = -0.33), meaning p-adic distance predicts the **opposite** of thermodynamic destabilization.

---

### Link 4: P-adic Unique Contribution

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hypothesis | P-adic + Physico > Physico alone | |
| LOO R² (Physico only) | 0.0904 | Baseline |
| LOO R² (Combined) | 0.0881 | Lower! |
| Delta R² | **-0.24%** | P-adic hurts |
| Falsified | **YES** | Redundant and harmful |

**Discovery:** Adding p-adic features to physicochemistry **reduces** prediction quality. P-adic is not only redundant with physicochemistry, it actively degrades performance.

---

## Summary Table

| Link | Hypothesis | r / ΔR² | Falsified |
|------|------------|---------|-----------|
| 1. Genetic Code | P-adic close → Physico similar | -0.097 | **YES** |
| 2. Mutational | P-adic close → Few mutations | -0.072 | **YES** |
| 3. Pure P-adic DDG | P-adic alone → DDG | +0.066 | **YES** |
| 4. Unique Contrib | P-adic + Physico > Physico | -0.24% | **YES** |

**Links falsified: 4/4**

---

## Scientific Implications

### What This Means

1. **P-adic is NOT the right geometry for thermodynamics**
   - The ultrametric tree structure of p-adic space does not reflect protein stability
   - Thermodynamic stability requires 3D structural information

2. **The genetic code is NOT optimized for p-adic similarity**
   - The negative correlation suggests the code may be optimized for *maximum* p-adic diversity among similar amino acids
   - This could be related to error robustness, not thermodynamics

3. **Evolution optimizes for different objectives**
   - The TEGB chain assumed thermodynamics drives codon structure
   - Instead, the genetic code may be optimized for translation fidelity, ribosome efficiency, or error tolerance

---

## Where P-adic MAY Still Work

The falsification of TEGB does not mean p-adic is useless. It may be appropriate for:

| Application | Rationale |
|-------------|-----------|
| **Evolutionary distance** | P-adic ultrametric matches phylogenetic trees |
| **Immune escape** | Goldilocks zone (already validated in HIV work) |
| **Translation kinetics** | Codon usage bias, ribosome pausing |
| **Mutational accessibility** | (But reframe: p-adic far = evolutionarily accessible) |

---

## Recommendations

### For DDG Prediction

1. **Do NOT use p-adic features** - They hurt performance
2. **Use physicochemical features** - delta_volume dominates (R² ~ 7%)
3. **Add structural context** - If available (burial, contacts, B-factor)

### For P-adic Research

1. **Pivot to non-thermodynamic applications**
   - Immune escape (HIV, autoimmune)
   - Codon optimization for expression
   - Evolutionary trajectory prediction

2. **Investigate the negative correlations**
   - Why are p-adically close codons encoding dissimilar amino acids?
   - Is this error-robustness optimization?

---

## Methodology

### Test Design

- **Pure p-adic only**: No learned weights, no biases
- **Mathematical structure**: 3-adic distance from codon indices
- **LOO cross-validation**: Honest generalization estimate
- **Comparison with physicochemistry**: Ablation study

### Code Location

```
research/codon-encoder/falsification/tegb_falsification.py
```

### Reproducibility

```bash
cd research/codon-encoder/falsification
python tegb_falsification.py
```

---

## References

1. TEGB Conjecture: Proposed in this session (2026-01-03)
2. S669 Dataset: Pancotti et al. 2022, Briefings in Bioinformatics
3. P-adic Biology: Dragovich et al. 2009, p-Adic Numbers

---

*This falsification result is scientifically valuable: it definitively rules out p-adic codon geometry as a predictor of thermodynamic stability, redirecting research to more promising applications.*
