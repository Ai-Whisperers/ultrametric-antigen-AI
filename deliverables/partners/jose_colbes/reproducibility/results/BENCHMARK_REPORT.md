# P-adic DDG Prediction Benchmark Report

**Generated:** 2026-01-03
**Dataset:** S669 Benchmark (Pancotti et al. 2022)
**N Mutations:** 669 (full dataset)

---

## Executive Summary

| Version | Dataset | Spearman r (Train) | Spearman r (CV) | CV R² | Assessment |
|---------|---------|-------------------|-----------------|-------|------------|
| V1 (Heuristic) | 52 fallback | 0.53 | N/A | N/A | Overfitted |
| V1.5 (VAE) | 52 fallback | 0.58 | N/A | N/A | Overfitted |
| V2 (Codon) | 52 fallback | 0.81 | 0.15 | 0.15 | **Overfitted** |
| **V2 (Full S669)** | 669 | 0.34 | **0.31** | 0.08 | **Honest baseline** |

**CRITICAL FINDING:** Results on 52 mutations were severely overfitted. The full 669-mutation dataset reveals true generalization performance.

---

## Full S669 Results (V2 - Proper Codon Distances)

### Training Metrics
| Metric | Value |
|--------|-------|
| N Mutations | 669 |
| Spearman r (train) | 0.345 |
| Pearson r (train) | 0.343 |
| MAE | 1.145 |
| RMSE | 1.536 |

### Cross-Validation Metrics (5-Fold)
| Metric | Value |
|--------|-------|
| CV R² | 0.078 ± 0.012 |
| CV Spearman | 0.312 |
| CV Pearson | 0.308 |
| CV MAE | 1.160 |
| CV RMSE | 1.557 |

### Learned Feature Weights
| Feature | Weight | Interpretation |
|---------|--------|----------------|
| padic_min | -0.67 | Closer codons = more destabilizing (unexpected) |
| padic_mean | 0.82 | Mean distance contribution |
| **delta_volume** | **0.67** | **Size change dominates prediction** |
| delta_hydro | 0.11 | Hydrophobicity minor role |
| delta_charge | 0.07 | Charge change minimal |
| delta_polarity | -0.24 | Polarity contribution |
| degeneracy_ratio | 0.16 | Codon degeneracy minimal |

---

## Feature Contribution Analysis

### Ablation Study
| Removed Feature | CV R² | Δ R² | Interpretation |
|-----------------|-------|------|----------------|
| padic_min | 0.076 | +0.002 | P-adic min hurts slightly |
| padic_mean | 0.075 | +0.003 | P-adic mean hurts slightly |
| **delta_volume** | **0.019** | **+0.058** | **Volume carries ~75% of signal** |
| delta_hydro | 0.079 | -0.001 | Neutral |
| delta_charge | 0.079 | -0.001 | Neutral |
| delta_polarity | 0.078 | 0.000 | Neutral |
| degeneracy_ratio | 0.077 | +0.001 | Neutral |

### Feature-Only Models
| Features | CV R² | Conclusion |
|----------|-------|------------|
| P-adic only | **-0.024** | Worse than random! |
| Physicochemical only | 0.076 | All predictive signal |
| All features | 0.078 | P-adic adds ~0.2% |

**CONCLUSION:** P-adic codon distances provide no unique predictive signal for DDG prediction. The physicochemical property `delta_volume` dominates prediction.

---

## Comparison with Published Tools

| Tool | Method | N | Spearman r | Pearson r | MAE |
|------|--------|---|------------|-----------|-----|
| MAESTRO | ML | 669 | 0.463 | 0.497 | 1.062 |
| ACDC-NN | Deep Learning | 669 | 0.454 | 0.460 | 1.045 |
| DDGun3D | ML + Structure | 667 | 0.427 | 0.432 | 1.109 |
| DDGun | ML | 642 | 0.427 | 0.404 | 1.266 |
| PremPS | ML | 666 | 0.416 | 0.405 | 1.092 |
| DUET | ML | 669 | 0.416 | 0.413 | 1.096 |
| PopMusic | Physics | 666 | 0.415 | 0.415 | 1.088 |
| ThermoNet | Deep Learning | 669 | 0.373 | 0.391 | 1.174 |
| mCSM | ML | 669 | 0.363 | 0.359 | 1.130 |
| **P-adic V2** | **Codon geometry** | **669** | **0.345** | **0.343** | **1.145** |
| FoldX | Physics | 667 | 0.283 | 0.214 | 1.568 |

**Assessment:** P-adic V2 ranks above FoldX but below all ML-based tools.

---

## Error Analysis

### By DDG Magnitude
| DDG Range | Count | MAE | Spearman | Note |
|-----------|-------|-----|----------|------|
| [-10, -2) | 151 | 1.90 | 0.16 | Highly destabilizing - poor |
| [-2, -1) | 147 | 0.64 | -0.06 | Destabilizing - moderate |
| [-1, 0) | 201 | 0.52 | 0.14 | Mild destab - best |
| [0, 1) | 114 | 1.11 | 0.07 | Neutral - moderate |
| [1, 2) | 34 | 2.07 | 0.32 | Stabilizing - poor |
| [2, 10) | 22 | 3.76 | 0.13 | Highly stabilizing - very poor |

**Pattern:** Model performs best on mild destabilizing mutations, struggles with extreme values.

### By P-adic Distance
| P-adic Dist | Count | MAE |
|-------------|-------|-----|
| 0.11 (close) | 27 | 0.94 |
| 0.33 (medium) | 223 | 1.20 |
| 1.00 (far) | 419 | 1.13 |

**Note:** P-adic distance shows no clear correlation with error magnitude.

---

## Triviality Assessment

### Criteria Evaluation
| Criterion | Pass | Value |
|-----------|------|-------|
| CV R² > 0 (learns something) | ✓ | 0.078 |
| P-adic adds to physicochemical | ✗ | +0.002 |
| CV R² > 0.05 (generalizes) | ✓ | 0.078 |
| CV Spearman > 0.3 | ✓ | 0.312 |

**Assessment: PARTIALLY MEANINGFUL**
- Model generalizes (not memorizing)
- But P-adic features contribute nothing unique
- delta_volume (amino acid size change) does all the work

---

## Why P-adic Features Failed

### Hypothesis 1: Information Already Captured
P-adic codon distances may correlate with physicochemical properties (amino acids with similar codons often have similar properties due to genetic code optimization).

### Hypothesis 2: Wrong Level of Abstraction
DDG depends on:
- 3D structural context (neighboring residues, burial)
- Specific molecular interactions (H-bonds, hydrophobic packing)
- Entropic effects (conformational flexibility)

P-adic codon geometry captures none of these.

### Hypothesis 3: Codon-Level Irrelevant for Protein Stability
The genetic code's structure relates to translation fidelity and evolutionary optimization, not protein thermodynamics.

---

## Potential Improvements

### 1. Add Structural Context (if available)
```python
# Features that would help:
- Relative solvent accessibility (RSA)
- Secondary structure state (helix/sheet/coil)
- Contact number (number of neighboring atoms)
- B-factor (flexibility)
```

### 2. Use P-adic for Different Task
P-adic geometry might be useful for:
- Evolutionary distance estimation
- Mutational accessibility (one mutation away)
- Synonymous mutation effects
- Codon usage bias

### 3. Sequence Context Features
```python
# Neighboring amino acids matter
- Window of ±3 residues around mutation
- Local hydrophobicity profile
- Conserved positions from MSA
```

### 4. Non-Linear Models
```python
# Linear model may miss interactions
- Random Forest / Gradient Boosting
- Neural network with p-adic embedding layer
- Attention mechanism over codon structure
```

---

## Honest Conclusions

1. **P-adic codon geometry does NOT predict protein stability (DDG)**
   - CV R² contribution: ~0.2% (negligible)
   - P-adic features alone: worse than random

2. **The "V2 Spearman = 0.81" was overfitting**
   - 52 mutations is too small for 7 features
   - Full 669 mutations shows true performance: 0.31

3. **Simple physicochemistry dominates**
   - `delta_volume` alone explains ~75% of model signal
   - This is well-known in the field (Grantham distance, etc.)

4. **P-adic approach may be useful elsewhere**
   - Evolutionary analysis
   - Codon optimization
   - Translation kinetics
   - NOT thermodynamic stability

---

## Reproducibility

### Download Full S669
```bash
cd deliverables/partners/jose_colbes/reproducibility
python download_s669.py  # Downloads from Zenodo
```

### Run Full Analysis
```bash
python analyze_padic_ddg_full.py
```

### Check Results
```bash
cat results/full_analysis_results.json
```

---

## References

1. S669 Dataset: Pancotti et al. 2022, Briefings in Bioinformatics
2. MAESTRO: Laimer et al. 2015, Bioinformatics
3. FoldX: Schymkowitz et al. 2005, Nucleic Acids Research
4. P-adic Biology: Dragovich et al. 2009, p-Adic Numbers

---

*Generated by the Ternary VAE Bioinformatics Partnership*
*Honest assessment: P-adic geometry is not the right tool for DDG prediction*
