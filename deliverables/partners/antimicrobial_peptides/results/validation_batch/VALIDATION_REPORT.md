# Validation Batch Report - Brizuela AMP Package

**Doc-Type:** Validation Report | Version 1.0 | 2026-01-06 | AI Whisperers

---

## Validation Configuration

| Parameter | Value |
|-----------|-------|
| Date | 2026-01-06 |
| Generations | 40 |
| Population | 80 |
| Random Seed | 123 |
| Model | PeptideVAE (Spearman r=0.74) |
| Purpose | Validate 4 key findings from initial run |

---

## Finding 1: MIC Convergence Across Pathogens

### Hypothesis
MIC predictions converge to ~0.80 ug/mL regardless of target pathogen.

### Validation Results

| Pathogen | WHO Priority | Gram | Best MIC | Median MIC | N |
|----------|--------------|------|----------|------------|---|
| A_baumannii | Critical | Negative | **0.79** | 0.83 | 80 |
| S_aureus | High | Positive | **0.79** | 0.81 | 80 |
| P_aeruginosa | Critical | Negative | **0.79** | 0.83 | 80 |
| Enterobacteriaceae | Critical | Negative | **0.79** | 0.84 | 80 |
| H_pylori | High | Negative | **0.79** | 0.82 | 79 |

**Statistical Summary:**
- Best MIC range: 0.79 - 0.79 ug/mL
- Standard deviation: 0.0022
- **FINDING CONFIRMED**: All pathogens converge to identical MIC

### Root Cause Analysis

The PeptideVAE model was trained on **general antimicrobial activity** from the DRAMP database, not pathogen-specific activity. The model learns:
1. Sequence → log10(MIC) mapping
2. General AMP features (charge, hydrophobicity, length)
3. No pathogen-specific membrane composition differences

### Impact Assessment

| Severity | Impact |
|----------|--------|
| HIGH | Cannot differentiate pathogen-specific activity |
| HIGH | All optimization runs converge to same solution space |
| MEDIUM | DRAMP activity models exist but not integrated into NSGA-II |

### Recommended Improvements

1. **Immediate**: Integrate DRAMP pathogen-specific models into B1 objective function
   ```python
   # Current: Uses only PeptideVAE MIC prediction
   mic = self.predictor.predict(sequence).predicted_mic

   # Improved: Ensemble with DRAMP models
   mic_vae = self.predictor.predict(sequence).predicted_mic
   mic_dramp = self.dramp_models[pathogen].predict(features)
   mic = 0.5 * mic_vae + 0.5 * mic_dramp
   ```

2. **Short-term**: Train pathogen-specific PeptideVAE heads
   - Requires pathogen-labeled training data
   - Fine-tune last layers per pathogen

3. **Long-term**: Incorporate membrane composition features
   - Gram+/- differentiation
   - Lipopolysaccharide content
   - Membrane fluidity parameters

---

## Finding 2: Gut Selectivity Achievable

### Hypothesis
Gut microbiome selectivity index > 1.0 is achievable.

### Validation Results

| Metric | Value |
|--------|-------|
| Best SI | 1.40 |
| Median SI | 0.44 |
| N candidates | 60 |
| SI > 1.0 | 8 (13.3%) |
| SI > 0.5 | 34 (56.7%) |

**Top Selective Candidates:**

| Rank | Sequence | SI | MIC |
|------|----------|-----|-----|
| 1 | CVKVKTTFKVVKTVTVKVVKFKTTV... | 1.40 | 3.37 |
| 2 | CVKVFFKVVKTVTFKVKFKTTV... | 1.40 | 3.36 |
| 3 | RVKKFFKVVKTVFKFKVRTFVR... | 0.93 | 3.50 |

**FINDING CONFIRMED**: Gut selectivity SI > 1.0 achievable

### Biological Interpretation

Gut microbiome has high taxonomic diversity:
- Pathogens: C. difficile (Firmicutes), E. coli (Proteobacteria), Salmonella (Proteobacteria)
- Commensals: Lactobacillus (Firmicutes), Bifidobacterium (Actinobacteria), Bacteroides (Bacteroidetes)

The phylogenetic distance allows for selective targeting.

### Recommended Actions

1. **Prioritize gut context** for wet-lab validation
2. Focus on SI > 1.0 candidates (8 peptides)
3. Test against actual gut microbiome panels

---

## Finding 3: Skin Selectivity Challenging

### Hypothesis
Skin microbiome selectivity is capped below 1.0.

### Validation Results

| Metric | Value |
|--------|-------|
| Best SI | 0.77 |
| Median SI | 0.40 |
| N candidates | 69 |
| SI > 0.5 | 30 (43.5%) |
| SI > 0.7 | 6 (8.7%) |

**FINDING CONFIRMED**: Max skin SI = 0.77 < 1.0

### Biological Interpretation

Skin microbiome has low phylogenetic diversity for target organisms:
- Pathogens: S. aureus, MRSA, P. acnes pathogenic
- Commensals: S. epidermidis, C. acnes, Corynebacterium

**Critical insight**: S. aureus and S. epidermidis are both Staphylococcus species with nearly identical membrane compositions. Any AMP that kills S. aureus will likely harm S. epidermidis.

### Recommended Improvements

1. **Accept biological limitation** - SI < 1.0 may be inherent
2. **Reframe objective** - Optimize for minimal commensal harm rather than selectivity
3. **Alternative approach** - Target S. aureus virulence factors instead of membrane

---

## Finding 4: Synthesis Difficulty 100% DIFFICULT

### Hypothesis
All synthesis-optimized candidates receive DIFFICULT grade.

### Validation Results

| Grade | Count | Percentage |
|-------|-------|------------|
| EXCELLENT | 0 | 0% |
| GOOD | 0 | 0% |
| MODERATE | 0 | 0% |
| CHALLENGING | 0 | 0% |
| **DIFFICULT** | **50** | **100%** |

**FINDING CONFIRMED**: All candidates DIFFICULT grade

### Root Cause Analysis

Two potential causes:

**Cause A: Synthesis heuristics too strict**
- Current difficulty thresholds may be pessimistic
- Real-world synthesis success rates may be higher

**Cause B: NSGA-II prioritizes MIC over synthesis**
- Multi-objective optimization converges to MIC-optimal region
- Synthesis-friendly sequences have poor MIC predictions

### Evidence Analysis

Examining top candidates:
```
Rank  Sequence         Difficulty  Coupling  Cost
1     DQSNGQSDNIDNINE  1.000       0.498     $29
7     DSQSASSNGAQ      1.000       0.700     $18
13    DQSSNGAQ         1.000       0.751     $14
```

Coupling efficiency ranges 50-75%, costs $14-$29, all difficulty=1.0

**Conclusion**: The difficulty metric is not differentiating - all candidates cluster at 1.0

### Recommended Improvements

1. **Immediate**: Audit `predict_synthesis_difficulty()` function
   - Check threshold calibration
   - Verify aggregation propensity calculation

2. **Short-term**: Add synthesis-friendly seed sequences
   ```python
   # Current seeds are optimized for activity
   seeds = ["KLAKLAKKLAKLAK", "KLWKKLKKALK", ...]

   # Add synthesis-friendly seeds
   seeds += ["AAKAAKAAK", "GKGKGKGK", ...]  # Simple, easy to synthesize
   ```

3. **Long-term**: Reweight NSGA-II objectives
   ```python
   # Current: All objectives equally weighted (-1.0, -1.0, -1.0, -1.0)
   # Proposed: Weight synthesis difficulty higher
   weights = (-1.0, -2.0, -1.0, -1.0)  # Double difficulty weight
   ```

---

## Summary of Validated Findings

| # | Finding | Status | Severity | Improvement Priority |
|---|---------|--------|----------|---------------------|
| 1 | MIC convergence across pathogens | **CONFIRMED** | HIGH | P1 |
| 2 | Gut selectivity > 1.0 achievable | **CONFIRMED** | LOW | Working |
| 3 | Skin selectivity capped at 0.77 | **CONFIRMED** | MEDIUM | P3 |
| 4 | Synthesis 100% DIFFICULT | **CONFIRMED** | HIGH | P2 |

---

## Prioritized Improvement Roadmap

### P1: Fix MIC Convergence (Highest Impact)

**Effort**: 2-3 days
**Impact**: Unlocks pathogen-specific optimization

```python
# File: scripts/B1_pathogen_specific_design.py

class PathogenObjectives(BaseObjectives):
    def __init__(self, pathogen: str, predictor, dramp_model=None):
        self.dramp_model = dramp_model  # NEW: Add DRAMP integration

    def evaluate(self, sequence: str) -> tuple:
        mic_vae = self.predictor.predict(sequence).predicted_mic
        if self.dramp_model is not None:
            features = compute_peptide_features(sequence)
            mic_dramp = self.dramp_model.predict([features])[0]
            mic = 0.6 * mic_vae + 0.4 * mic_dramp  # Ensemble
        else:
            mic = mic_vae
        return mic, toxicity, stability
```

### P2: Fix Synthesis Difficulty (High Impact)

**Effort**: 1-2 days
**Impact**: Enables practical peptide synthesis

1. Audit `predict_synthesis_difficulty()` thresholds
2. Add synthesis-friendly seed sequences
3. Test with different objective weights

### P3: Document Skin Selectivity Limitation (Low Effort)

**Effort**: 0.5 days
**Impact**: Sets realistic expectations

1. Add warning to B8 for skin context
2. Document biological limitation in README
3. Suggest alternative approaches (virulence targeting)

---

## Files Generated

### Result Files (Individual Pipeline Outputs)

| File | Contents |
|------|----------|
| A_baumannii_results.json | 80 candidates |
| S_aureus_results.json | 80 candidates |
| P_aeruginosa_results.json | 80 candidates |
| Enterobacteriaceae_results.json | 80 candidates |
| H_pylori_results.json | 79 candidates |
| microbiome_safe_gut_results.json | 60 candidates |
| microbiome_safe_skin_results.json | 69 candidates |
| synthesis_optimized_results.json | 50 candidates |

### Validation Summary Files (Structured Analysis)

| File | Contents |
|------|----------|
| **validation_summary.json** | Executive summary with key metrics and insights |
| **validation_findings.json** | Complete findings analysis with evidence and root causes |
| **validation_statistics.json** | Detailed pipeline statistics and performance metrics |
| **prioritized_improvements.json** | Technical implementation roadmap with code examples |
| **VALIDATION_REPORT.md** | Human-readable comprehensive analysis report |

**Total**: 578 candidates across 8 runs, 4 validated findings, 3 prioritized improvements

---

## Conclusion

All 4 key findings from the initial production run have been **validated** with a larger dataset (40 generations, 80 population). The validation confirms:

1. **MIC convergence is a real issue** requiring DRAMP model integration
2. **Gut selectivity works** and should be prioritized for wet-lab validation
3. **Skin selectivity has biological limitations** that must be documented
4. **Synthesis difficulty scoring needs recalibration**

Next steps: Implement P1 (DRAMP integration) and P2 (synthesis recalibration) before next production run.
