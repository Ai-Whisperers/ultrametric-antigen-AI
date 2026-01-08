# PeptideVAE Training Plan: Evidence-Based Architecture

**Doc-Type:** Training Specification | Version 1.0 | 2026-01-07 | AI Whisperers

---

## Executive Summary

This plan is derived from **validated scientific findings**, not speculation:

| Validated Finding | Source | Training Implication |
|-------------------|--------|---------------------|
| C3 cluster signal survives | P1 falsification test | Train on (length, charge, hydro) clusters |
| Peptide-only generalizes r=0.88-0.94 | C5 hold-out test | NO pathogen metadata in model |
| Hydrophobicity is primary predictor | Colbes DDG (0.633 importance) | Hydro-aware loss weighting |
| hyp_dist correlates with DDG | Colbes ρ=0.585 | Radial hierarchy matters |
| Pathogen metadata HURTS | C5 avg -0.109 improvement | Explicitly exclude pathogen labels |

---

## Architecture Requirements

### MUST Have (Validated):

1. **Hyperbolic Latent Space**
   - Use `poincare_distance()` for all radial computations
   - NOT `.norm()` (V5.12.2 audit showed this breaks hierarchy)
   - Radial position encodes activity/stability

2. **Physicochemical Feature Encoding**
   - Latent space MUST encode (length, charge, hydrophobicity)
   - These are the C3-validated signal sources
   - Auxiliary prediction heads ensure these are learned

3. **Cluster-Aware Structure**
   - Natural emergence of C3 clusters in latent space
   - NOT supervised by pathogen labels
   - Supervised by physicochemical regime membership

4. **DDG Prediction Capability**
   - Must achieve ρ ≥ 0.585 (Colbes benchmark)
   - Features: hyp_dist, delta_radius, delta_hydro, delta_charge

### MUST NOT Have (Falsified):

1. **NO Pathogen Metadata Input**
   - C5 FALSIFIED: metadata hurts generalization
   - Removes any pathogen ID, Gram type, membrane features

2. **NO Pathogen-Conditional Objectives**
   - No separate heads per pathogen
   - No pathogen-specific loss terms
   - No pathogen routing

3. **NO Failure Mode Classification**
   - C2 FALSIFIED: no signal at clinical thresholds
   - Don't waste capacity on this

---

## Training Phases

### Phase 1: Coverage + Physicochemical Grounding

**Goal:** Accurate sequence reconstruction with physicochemical awareness

**Loss Function:**
```python
L_phase1 = L_reconstruction + β * L_KL + λ_phys * L_physicochemical

# Where L_physicochemical forces latent space to encode C3 features:
L_physicochemical = MSE(pred_length, true_length)
                  + MSE(pred_charge, true_charge)
                  + MSE(pred_hydro, true_hydro)
```

**Architecture:**
```
Encoder: Sequence → z_latent (16D hyperbolic)
         + Aux Head: z → (length, charge, hydro) prediction

Decoder: z_latent → Reconstructed Sequence
```

**Validation Criteria:**
- Sequence reconstruction accuracy ≥ 95%
- Physicochemical prediction R² ≥ 0.9
- Latent (length, charge, hydro) correlation ≥ 0.8

**Duration:** ~2000 epochs or until convergence

---

### Phase 2: Radial Hierarchy from Activity

**Goal:** Radial position encodes peptide activity/stability

**Key Insight from Colbes:**
- hyp_dist coefficient: 0.35 (positive → larger distance = more destabilizing?)
- delta_radius coefficient: 0.28
- This suggests: radial position encodes stability direction

**Loss Function:**
```python
L_phase2 = L_phase1 + λ_hier * L_hierarchy

# Hierarchy: activity should correlate with radial position
# Use MIC as proxy (lower MIC = more active = different radius)
L_hierarchy = -correlation(poincare_radius(z), activity_score)
```

**Target:** Spearman correlation between radius and activity ≥ |0.5|

**Important:** Direction doesn't matter (can be positive or negative correlation), but consistency matters.

---

### Phase 3: Cluster Structure Enhancement

**Goal:** Latent space naturally separates C3-validated clusters

**Method:** Contrastive learning on physicochemical regime clusters

```python
# C3-validated cluster definitions (from P1 tests):
# Cluster 1: Short (13 AA), low charge → SIGNAL
# Cluster 3: Short (13 AA), very hydrophobic → SIGNAL
# Cluster 4: Short (14 AA), moderate → SIGNAL
# Cluster 0, 2: Long/medium peptides → NO SIGNAL

def assign_cluster(peptide):
    features = (len(peptide), net_charge(peptide), hydrophobicity(peptide))
    return kmeans_5.predict(features)

L_phase3 = L_phase2 + λ_cluster * L_contrastive_cluster
```

**Validation Criteria:**
- Silhouette score on 5-cluster assignment ≥ 0.4
- Clusters 1, 3, 4 show internal MIC variance < 0.3
- Inter-cluster MIC variance > 0.5

---

### Phase 4: DDG Validation Head

**Goal:** Prove embedding quality via DDG prediction benchmark

**Architecture:**
```
Frozen Encoder → z_wt, z_mut
                    ↓
Geometric Features:
  - hyp_dist(z_wt, z_mut)
  - delta_radius = radius(z_mut) - radius(z_wt)
  - diff_norm = ||z_mut - z_wt||  (for reference)
  - cos_sim = cosine(z_wt, z_mut)
                    ↓
Physicochemical Features:
  - delta_hydro
  - delta_charge
  - delta_size
                    ↓
Ridge Regression (α=100) → DDG prediction
```

**Success Criterion:**
- Spearman ρ ≥ 0.585 on S669 benchmark (Colbes validated)
- Leave-One-Out cross-validation
- Bootstrap 95% CI must exclude zero

---

## Data Sources

### Training Data:

| Source | N | Use |
|--------|---|-----|
| DRAMP | ~5000 | Activity annotations |
| APD3 | ~3300 | Sequence diversity |
| DBAASP | ~15000 | MIC measurements |
| S669 | 52 | DDG validation |

### Important: Data Cleaning

1. **Remove pathogen labels** from training
2. **Keep only:** sequence, length, charge, hydrophobicity, MIC (if available)
3. **Quality filter:** Remove sequences with missing physicochemical properties

---

## Hyperparameters

### Phase 1:
- `latent_dim`: 16
- `hidden_dim`: 128
- `learning_rate`: 1e-3
- `batch_size`: 64
- `λ_phys`: 1.0

### Phase 2:
- `learning_rate`: 1e-4 (reduced)
- `λ_hier`: 5.0 (from homeostatic_rich)
- Freeze decoder, train encoder only

### Phase 3:
- `learning_rate`: 5e-5 (further reduced)
- `λ_cluster`: 2.0
- Temperature for contrastive: 0.1

### Phase 4:
- Encoder FROZEN
- Ridge α=100 (from Colbes)
- StandardScaler on features

---

## Validation Checkpoints

### After Phase 1:
```python
# Test reconstruction
assert accuracy >= 0.95

# Test physicochemical encoding
for prop in ['length', 'charge', 'hydro']:
    r = correlation(predict_prop(z), true_prop)
    assert r >= 0.8, f"{prop} not encoded: r={r}"
```

### After Phase 2:
```python
# Test radial hierarchy
r = spearmanr(poincare_radius(z), activity)
assert abs(r) >= 0.5, f"No radial hierarchy: r={r}"
```

### After Phase 3:
```python
# Test cluster structure
labels = kmeans_5.predict(physicochemical_features)
silhouette = silhouette_score(z, labels)
assert silhouette >= 0.4, f"Poor clustering: s={silhouette}"
```

### After Phase 4:
```python
# DDG benchmark (CRITICAL - this is the ground truth)
ddg_pred = ddg_head(geometric_features(z_wt, z_mut))
rho = spearmanr(ddg_pred, ddg_true)
assert rho >= 0.585, f"DDG failed: ρ={rho} < 0.585"
```

---

## Anti-Patterns to Avoid

### 1. Pathogen Conditioning (FALSIFIED by C5)
```python
# WRONG - this was V1's mistake
z = encoder(sequence, pathogen_id)  # NO!
loss = pathogen_specific_loss(z, pathogen)  # NO!
```

### 2. Euclidean Norm on Hyperbolic Space (V5.12.2 Audit)
```python
# WRONG
radius = torch.norm(z, dim=-1)

# RIGHT
radius = poincare_distance(z, origin, c=curvature)
```

### 3. Threshold Tuning (R1 Violation)
```python
# WRONG - adjusting thresholds to find signal
for thresh in np.linspace(0, 2, 100):
    if separation_at_thresh(thresh) > 0.5:
        best_thresh = thresh  # NO! This is optimization

# RIGHT - use a priori clinical thresholds
CLINICAL_THRESHOLDS = [0.0, 0.5, 1.0]  # Fixed before looking at data
```

### 4. Looking at Test Set During Development (R2 Violation)
```python
# WRONG
model.fit(train + test)  # Data leakage
score = model.evaluate(test)  # Optimistic

# RIGHT
model.fit(train_subset)
score = model.evaluate(held_out)  # True generalization
```

---

## Expected Outcomes

### If Training Succeeds:

1. **DDG prediction ρ ≥ 0.585** (validates Colbes findings)
2. **MIC prediction generalizes** across held-out pathogens (R2 compliant)
3. **Cluster structure emerges** from physicochemical properties
4. **Radial hierarchy** encodes activity/stability

### If Training Fails:

1. **DDG < 0.585:** Embedding doesn't capture stability information
   - Debug: Check hyp_dist correlation, verify poincare_distance usage

2. **MIC doesn't generalize:** Overfitting to training pathogens
   - Debug: Check for inadvertent pathogen leakage

3. **No cluster structure:** Physicochemical features not encoded
   - Debug: Check auxiliary head losses, increase λ_phys

---

## Relation to Partner Packages

### Colbes (DDG Prediction):
- **Direct validation:** Phase 4 must reproduce ρ=0.585
- **Feature alignment:** hyp_dist, delta_radius are required features

### Brizuela (AMP Design):
- **Update needed:** Remove pathogen-specific B1 claims (C5 falsified)
- **Keep:** B8 (microbiome-safe), B10 (synthesis) are peptide-property based

### Rojas (Arbovirus):
- **Compatible:** Trajectory forecasting uses p-adic embeddings
- **Extension:** Same encoder architecture can work for viral sequences

### HIV:
- **Compatible:** TDR screening is rule-based (WHO SDRMs)
- **Extension:** Could use embeddings for resistance prediction

---

## Success Criteria Summary

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| Sequence reconstruction | ≥ 95% | Standard VAE |
| Physicochemical encoding | R² ≥ 0.9 | C3 validation |
| Radial hierarchy | \|ρ\| ≥ 0.5 | Phase 2 target |
| Cluster silhouette | ≥ 0.4 | Phase 3 target |
| **DDG prediction** | **ρ ≥ 0.585** | **Colbes benchmark (CRITICAL)** |
| MIC generalization | r ≥ 0.8 on held-out | C5 R2 constraint |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-07 | 1.0 | Initial evidence-based plan from P1 investigation |

---

*This plan is derived from validated scientific findings, not speculation.*
*Key sources: P1_CONJECTURE_TESTS.md, Colbes package validation, V5.12.2 audit*
