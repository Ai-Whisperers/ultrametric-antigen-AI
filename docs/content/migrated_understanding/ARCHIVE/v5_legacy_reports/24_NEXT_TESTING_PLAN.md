# Next Testing Plan

## Overview

Based on comprehensive analysis of 60+ experiments, this plan outlines the next steps to:
1. Fix known issues
2. Test untested modules
3. Expand to real biological data
4. Validate with multiple organisms

---

## Phase 1: Bug Fixes & Technical Debt (1-2 days)

### 1.1 Fix K-FAC Optimizer
**Priority: HIGH**

The natural gradient optimizer has a singular matrix bug:

```
File: src/information/fisher_geometry.py:470
Error: torch.linalg.inv: singular matrix
```

**Tasks:**
- [ ] Add regularization to Fisher matrix inversion
- [ ] Implement fallback to standard gradient
- [ ] Test with unified pipeline

**Expected Benefit:** 2-3x faster convergence

### 1.2 Fix Persistence Vectorizer API
**Priority: MEDIUM**

```
File: src/topology/persistent_homology.py
Issue: API mismatch (n_bins vs resolution parameter)
```

**Tasks:**
- [ ] Standardize constructor parameters
- [ ] Add input validation
- [ ] Update integration guide

---

## Phase 2: Untested Module Integration (3-5 days)

### 2.1 Persistent Homology Integration
**Priority: HIGH**

Currently not in main pipeline. Expected to add topological features.

**Tests to run:**
- [ ] Standalone topological feature extraction on HIV data
- [ ] Integration with ranking loss
- [ ] Compare RipsFiltration vs PAdicFiltration
- [ ] Measure impact on correlation

**Hypothesis:** P-adic filtration will capture hierarchical structure better than Rips.

### 2.2 Statistical Physics Module
**Priority: MEDIUM**

Spin glass landscape modeling not tested.

**Tests to run:**
- [ ] Model HIV fitness landscape as spin glass
- [ ] Extract ultrametric tree from landscape
- [ ] Compare with known phylogenetic trees
- [ ] Test replica exchange sampling

**Hypothesis:** Spin glass model will identify evolutionary basins.

### 2.3 Meta-Learning Module
**Priority: HIGH**

Critical for multi-organism generalization.

**Tests to run:**
- [ ] MAML training on HIV subtypes as tasks
- [ ] Few-shot adaptation to HBV (5-10 examples)
- [ ] P-adic task sampling vs random sampling
- [ ] Cross-organism transfer learning

**Hypothesis:** P-adic task sampling will improve generalization.

### 2.4 Hyperbolic GNN
**Priority: MEDIUM**

Full GNN not tested, only projection.

**Tests to run:**
- [ ] Build sequence similarity graph
- [ ] Train HyboWaveNet on graph
- [ ] Compare with non-graph VAE
- [ ] Test on protein interaction networks

---

## Phase 3: Real Biological Data (5-7 days)

### 3.1 HIV Dataset Validation
**Priority: HIGH**

Test on actual HIV sequences with known phenotypes.

**Data sources:**
- Stanford HIV Drug Resistance Database
- Los Alamos HIV Sequence Database
- LANL drug resistance annotations

**Tests to run:**
- [ ] Load real HIV pol sequences
- [ ] Train with ranking loss
- [ ] Predict drug resistance (LAM, EFV, etc.)
- [ ] Compare correlation with published models
- [ ] Cross-validation (5-fold)

**Success metric:** Correlation > 0.7 on held-out data

### 3.2 Multi-Organism Expansion
**Priority: HIGH**

Expand beyond HIV.

**Organisms to add:**

| Organism | Data Source | Phenotype |
|----------|-------------|-----------|
| HBV | HBVdb | Drug resistance |
| HCV | euHCVdb | Treatment response |
| Influenza | GISAID | Antigenic drift |
| SARS-CoV-2 | GISAID | Immune escape |
| TB | PATRIC | Drug resistance |

**Tests to run:**
- [ ] Implement loaders for each organism
- [ ] Train organism-specific models
- [ ] Test cross-organism transfer
- [ ] Meta-learning across organisms

### 3.3 Structural Validation
**Priority: MEDIUM**

Validate latent space structure.

**Tests to run:**
- [ ] Compare latent clustering with phylogenetic trees
- [ ] Measure ultrametric tree correlation
- [ ] Visualize hyperbolic embeddings
- [ ] Check genotype separation

---

## Phase 4: Advanced Experiments (7-10 days)

### 4.1 Optimal Hyperparameter Search
**Priority: MEDIUM**

Grid search over key parameters.

**Parameters to search:**
```python
search_space = {
    "ranking_weight": [0.1, 0.3, 0.5, 0.7],
    "padic_weight": [0.3, 0.5, 0.7],
    "latent_dim": [8, 16, 32, 64],
    "hyperbolic_curvature": [0.5, 1.0, 2.0],
    "padic_prime": [2, 3, 5, 7],
}
```

### 4.2 Curriculum Training Deep Dive
**Priority: MEDIUM**

Optimize phase durations and transitions.

**Tests to run:**
- [ ] Different phase ratios (10/20/30/40 vs 25/25/25/25)
- [ ] Smooth vs hard transitions
- [ ] Loss-based phase switching
- [ ] Compare with standard training on real data

### 4.3 Ensemble Methods
**Priority: LOW**

Combine multiple models.

**Tests to run:**
- [ ] Ensemble of ranking-only models
- [ ] Ensemble of different architectures
- [ ] Stacking with meta-model
- [ ] Uncertainty quantification

### 4.4 Interpretability Analysis
**Priority: MEDIUM**

Understand what the model learns.

**Tests to run:**
- [ ] Latent dimension importance
- [ ] Attention visualization (if applicable)
- [ ] Mutation importance mapping
- [ ] Drug resistance pathway analysis

---

## Phase 5: Publication Preparation (5-7 days)

### 5.1 Benchmark Comparisons
**Priority: HIGH**

Compare with existing methods.

**Baselines to compare:**
- [ ] Standard VAE
- [ ] CNN-based predictors
- [ ] Random Forest on sequence features
- [ ] Existing drug resistance predictors
- [ ] ESM/ProtTrans embeddings

### 5.2 Ablation Study Report
**Priority: HIGH**

Formal ablation for publication.

**Experiments:**
- [ ] Full 32-configuration study on real data
- [ ] Statistical significance tests
- [ ] Error bars and confidence intervals
- [ ] Cross-validation results

### 5.3 Visualization Suite
**Priority: MEDIUM**

Create publication-quality figures.

**Figures to create:**
- [ ] Latent space UMAP/t-SNE by phenotype
- [ ] Hyperbolic embedding visualization
- [ ] Module synergy heatmap
- [ ] Training curves comparison
- [ ] Phylogenetic tree comparison

---

## Immediate Action Items (This Week)

### Day 1-2: Technical Fixes
```
[ ] Fix K-FAC optimizer singular matrix bug
[ ] Standardize PersistenceVectorizer API
[ ] Run unified pipeline end-to-end
```

### Day 3-4: Real Data Integration
```
[ ] Download Stanford HIV resistance data
[ ] Implement HIV loader with real sequences
[ ] Test ranking loss on real HIV data
[ ] Measure correlation with known resistance
```

### Day 5-7: Meta-Learning Setup
```
[ ] Create tasks from HIV subtypes
[ ] Implement P-adic task sampler
[ ] Run MAML training
[ ] Test few-shot adaptation
```

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | K-FAC working | No errors |
| Phase 2 | All modules integrated | 8/8 modules |
| Phase 3 | Real data correlation | > 0.7 |
| Phase 4 | Optimal config found | +5% over current |
| Phase 5 | Publication ready | Benchmarks complete |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| K-FAC unfixable | Low | Medium | Use Adam (already works) |
| Real data correlation low | Medium | High | Tune hyperparameters |
| Meta-learning fails | Medium | Medium | Use transfer learning |
| Compute constraints | Low | Medium | Use smaller batches |

---

## Resource Requirements

| Resource | Amount | Purpose |
|----------|--------|---------|
| GPU hours | ~100 | Training experiments |
| Storage | ~10GB | Sequence databases |
| Time | 3-4 weeks | Full plan execution |

---

## Quick Wins (Can Do Today)

1. **Run ranking-only on larger synthetic data** - verify scalability
2. **Visualize current latent space** - understand structure
3. **Download HIV resistance data** - prepare for real validation
4. **Document current best config** - freeze for comparison

