# Detailed Next Steps Plan

## Executive Summary

Based on all experiments and findings, this document provides a prioritized roadmap for the next phase of development. The plan is organized into:

1. **Immediate Priorities** (1-2 weeks) - Fix critical issues
2. **Short-Term Goals** (2-4 weeks) - Improve non-PI performance
3. **Medium-Term Goals** (1-2 months) - Validation and benchmarking
4. **Long-Term Vision** (2-6 months) - Publication and expansion

---

## Phase 1: Immediate Priorities (1-2 weeks)

### 1.1 Integrate TAM Encoding into Training Pipeline

**Problem**: TAM-aware encoder is implemented but not integrated into the training loop.

**Tasks**:
```
[ ] Modify run_on_real_data.py to use TAMAwareEncoder for NRTI drugs
[ ] Create combined encoding: one-hot + TAM features
[ ] Test on all 6 NRTI drugs
[ ] Compare: one-hot only vs TAM-enhanced
[ ] Document improvement (target: +0.15 correlation)
```

**Implementation**:
```python
# In prepare_data() for NRTI:
from src.encoding.tam_aware_encoder import TAMAwareEncoder

if drug_class in ["nrti", "nnrti"]:
    encoder = TAMAwareEncoder(position_cols)
    X = encoder.encode_dataframe(df_valid)  # One-hot + TAM features
else:
    X = encode_amino_acids(df_valid, position_cols)  # Standard
```

**Expected Outcome**: NRTI correlation from +0.07 to +0.15-0.20

---

### 1.2 Fix Transformer Numerical Stability

**Problem**: Transformer fails (NaN) on long RT sequences (560 positions).

**Tasks**:
```
[ ] Add gradient clipping (max_norm=1.0)
[ ] Use mixed precision (fp16) for memory efficiency
[ ] Implement sparse attention (local window + global tokens)
[ ] Add layer normalization before attention
[ ] Test on RT sequences incrementally (100, 200, 300, 560 positions)
```

**Implementation**:
```python
# Fixes to apply:
1. torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
2. scaler = torch.cuda.amp.GradScaler()  # Mixed precision
3. Use nn.TransformerEncoderLayer with pre_norm=True
4. Reduce d_model for RT (256 → 128)
5. Add warmup learning rate schedule
```

**Expected Outcome**: Transformer works on RT without NaN

---

### 1.3 End-to-End MAML Evaluation

**Problem**: MAML implemented but not evaluated on real few-shot scenario.

**Tasks**:
```
[ ] Create few-shot evaluation protocol:
    - Meta-train on 6 PI drugs
    - Meta-test on 2 held-out PI drugs (TPV, DRV)
    - Vary support set size: 5, 10, 20, 50 samples
[ ] Compare MAML vs fine-tuning baseline
[ ] Test on INI drugs (genuinely low-data)
[ ] Document few-shot learning curves
```

**Evaluation Protocol**:
```python
# Few-shot evaluation
support_sizes = [5, 10, 20, 50]
results = {}

for n_support in support_sizes:
    # Sample n_support examples from new drug
    adapted = trainer.adapt_to_new_drug(support_x[:n_support], support_y[:n_support])

    # Evaluate on remaining data
    corr = evaluate(adapted, test_x, test_y)
    results[n_support] = corr
```

**Expected Outcome**: MAML outperforms fine-tuning at n < 20 samples

---

## Phase 2: Short-Term Goals (2-4 weeks)

### 2.1 Multi-Task Training for PI Drugs

**Problem**: Currently training separate models per drug; may miss shared patterns.

**Tasks**:
```
[ ] Train MultiTaskVAE on all 8 PI drugs simultaneously
[ ] Compare to single-task baselines
[ ] Analyze shared latent space (do drugs cluster?)
[ ] Test cross-drug transfer (train on 7, test on 1)
[ ] Implement GradNorm for balanced multi-task learning
```

**Experiment Design**:
```
Experiment 1: Full multi-task (all 8 PI drugs)
Experiment 2: Leave-one-out cross-validation
Experiment 3: GradNorm vs uniform weighting
Experiment 4: Cross-drug attention variant
```

**Expected Outcome**: +0.01-0.02 improvement, better generalization

---

### 2.2 TAM-Specific Loss Function for NRTIs

**Problem**: Standard ranking loss may not capture TAM pathway interactions.

**Tasks**:
```
[ ] Design TAM-aware loss:
    - Penalize ignoring known TAM positions
    - Weight mutations by clinical importance
    - Add pathway consistency term
[ ] Implement in compute_loss()
[ ] Test on NRTI drugs
[ ] Compare to standard ranking loss
```

**Proposed Loss**:
```python
def tam_aware_loss(out, x, y, tam_features):
    # Standard ranking loss
    rank_loss = ranking_loss(out, y)

    # TAM consistency: latent should reflect TAM patterns
    tam_pred = tam_predictor(out["z"])
    tam_loss = F.binary_cross_entropy(tam_pred, tam_features)

    # Position importance: weight key positions higher
    key_positions = [41, 67, 70, 184, 215, 219]
    position_weights = create_position_weights(key_positions)
    weighted_recon = (position_weights * (x - out["x_recon"])**2).mean()

    return rank_loss + 0.1 * tam_loss + weighted_recon
```

**Expected Outcome**: NRTI from +0.07 to +0.25

---

### 2.3 Cross-Resistance Modeling

**Problem**: NRTI drugs have complex cross-resistance (M184V affects AZT differently than 3TC).

**Tasks**:
```
[ ] Build cross-resistance graph from HIVDB data
[ ] Implement graph neural network over drug relationships
[ ] Predict drug pair correlations
[ ] Use predicted correlations to inform multi-task weights
```

**Cross-Resistance Matrix** (to implement):
```
       AZT   3TC   TDF   ABC   DDI   D4T
AZT    1.0   -0.2  -0.2  0.6   0.5   0.9
3TC   -0.2   1.0   0.3   0.4   0.2   0.1
TDF   -0.2   0.3   1.0   0.6   0.5   0.3
ABC    0.6   0.4   0.6   1.0   0.6   0.5
DDI    0.5   0.2   0.5   0.6   1.0   0.6
D4T    0.9   0.1   0.3   0.5   0.6   1.0
```

---

### 2.4 Attention-Based Position Analysis

**Problem**: Which positions does the model actually use?

**Tasks**:
```
[ ] Extract attention weights from gene-specific VAE (RT)
[ ] Visualize attention heatmaps per drug
[ ] Compare to known resistance positions
[ ] Quantify attention-mutation correlation
[ ] Use insights to improve TAM encoding
```

**Analysis Pipeline**:
```python
# For each drug:
1. Load trained model
2. Run forward pass, capture attention weights
3. Average attention across test set
4. Identify top-attended positions
5. Compare to Stanford HIVDB mutation list
6. Report precision/recall of position discovery
```

**Expected Outcome**: Model discovers 70%+ of known mutations

---

## Phase 3: Medium-Term Goals (1-2 months)

### 3.1 External Dataset Validation

**Problem**: All results on Stanford HIVDB - may be overfitting to database quirks.

**Datasets to Acquire**:
| Database | Source | Drug Classes | Samples |
|----------|--------|--------------|---------|
| Los Alamos | LANL | All | ~50,000 |
| UK HIV DR | UK consortium | PI, NRTI | ~10,000 |
| IHDB | Italy | All | ~5,000 |
| TCE | EuResist | All | ~20,000 |

**Tasks**:
```
[ ] Obtain data access for each database
[ ] Write data loaders for each format
[ ] Harmonize drug names and resistance scales
[ ] Run best PI model on external data
[ ] Report cross-database correlation
[ ] Identify database-specific biases
```

**Expected Outcome**: External validation confirms +0.85 correlation for PI

---

### 3.2 Temporal Validation Study

**Problem**: Random splits may leak information; need realistic clinical simulation.

**Tasks**:
```
[ ] Parse IsolateDate from Stanford HIVDB
[ ] Split: train on pre-2018, test on 2018+
[ ] Compare temporal vs random performance
[ ] Analyze: does model degrade over time?
[ ] Test: does fine-tuning on recent data help?
```

**Temporal Splits**:
```
Split 1: Train 2000-2015, Test 2016-2023
Split 2: Train 2000-2017, Test 2018-2023
Split 3: Train 2000-2019, Test 2020-2023
```

**Expected Outcome**: Temporal performance within 0.05 of random (good generalization)

---

### 3.3 Published Method Comparison

**Problem**: No head-to-head comparison with state-of-the-art.

**Methods to Compare**:
| Method | Type | Reference |
|--------|------|-----------|
| HIVdb Algorithm | Rule-based | Stanford (default) |
| geno2pheno | SVM | Max Planck |
| DeepDTA | CNN | Ozturk 2018 |
| ESM-2 | Transformer | Meta AI 2023 |

**Tasks**:
```
[ ] Implement/obtain each baseline
[ ] Run on identical train/test splits
[ ] Report: correlation, RMSE, ranking metrics
[ ] Statistical significance tests
[ ] Create benchmark table for paper
```

**Expected Outcome**: Our method competitive or better than all baselines

---

### 3.4 Uncertainty Calibration Study

**Problem**: Uncertainty estimates not fully validated.

**Tasks**:
```
[ ] Run MC Dropout on all drugs
[ ] Compute calibration curves (expected vs observed coverage)
[ ] Apply calibration correction if needed
[ ] Test on out-of-distribution samples
[ ] Document uncertainty quality metrics
```

**Calibration Metrics**:
```
- Coverage at 50%, 80%, 90%, 95%, 99% CI
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Negative Log-Likelihood
- Sharpness (average CI width)
```

**Expected Outcome**: Well-calibrated uncertainties (ECE < 0.05)

---

## Phase 4: Long-Term Vision (2-6 months)

### 4.1 Publication Preparation

**Target Venue**: Bioinformatics, PLOS Computational Biology, or Nature Methods

**Paper Structure**:
```
1. Introduction
   - HIV drug resistance problem
   - Limitations of current methods

2. Methods
   - P-adic VAE architecture
   - Ranking loss formulation
   - Gene-specific adaptations

3. Results
   - Ablation study (32 configurations)
   - Real HIV data (23 drugs, 4 classes)
   - External validation
   - Comparison to baselines

4. Discussion
   - Why ranking loss works
   - PI vs non-PI differences
   - Limitations and future work
```

**Tasks**:
```
[ ] Write manuscript draft
[ ] Create publication-quality figures
[ ] Run final experiments for paper
[ ] Internal review and revision
[ ] Submit to journal
```

---

### 4.2 Multi-Organism Expansion

**Problem**: Only tested on HIV.

**Target Organisms**:
| Organism | Gene | Drug Class | Data Source |
|----------|------|------------|-------------|
| HBV | Polymerase | Antivirals | HBVdb |
| HCV | NS5B | DAAs | HCVdb |
| Influenza | Neuraminidase | NAIs | IRD |
| M. tuberculosis | Various | Anti-TB | TBDB |

**Tasks**:
```
[ ] Acquire data for each organism
[ ] Adapt encoding for different gene structures
[ ] Test if p-adic ranking transfers
[ ] Report cross-organism generalization
```

---

### 4.3 Clinical Decision Support Tool

**Goal**: Deploy model for clinical use.

**Components**:
```
1. Web Interface
   - Input: raw sequence or FASTA
   - Output: resistance predictions + uncertainty

2. API Endpoint
   - REST API for integration with LIMS
   - Batch processing support

3. Validation
   - Prospective clinical validation
   - IRB approval for clinical study
```

**Tasks**:
```
[ ] Build FastAPI backend
[ ] Create Streamlit/Gradio frontend
[ ] Deploy on cloud (AWS/GCP)
[ ] Partner with clinical lab for validation
[ ] Regulatory considerations (CE marking?)
```

---

### 4.4 Interpretability Dashboard

**Goal**: Help clinicians understand predictions.

**Features**:
```
1. Mutation Heatmap
   - Show which positions drive prediction
   - Highlight known vs novel mutations

2. Resistance Pathway Visualization
   - Show TAM pathways affected
   - Display cross-resistance predictions

3. Confidence Explanation
   - Why is uncertainty high/low?
   - Similar training examples

4. Treatment Recommendations
   - Suggest alternative drugs
   - Rank by predicted efficacy
```

---

## Priority Matrix

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| TAM encoding integration | High | Low | **P0** |
| Transformer stability fix | Medium | Medium | P1 |
| MAML evaluation | Medium | Low | P1 |
| Multi-task PI training | Medium | Medium | P2 |
| TAM-specific loss | High | High | P2 |
| External validation | High | High | P2 |
| Temporal validation | Medium | Low | P2 |
| Published comparison | High | Medium | P3 |
| Multi-organism | Medium | High | P3 |
| Clinical tool | High | Very High | P4 |

---

## Milestones and Timeline

### Week 1-2
- [ ] TAM encoding integrated and tested
- [ ] Transformer numerical stability fixed
- [ ] MAML few-shot evaluation complete

### Week 3-4
- [ ] Multi-task PI training complete
- [ ] TAM-specific loss implemented and tested
- [ ] NRTI correlation improved to +0.20

### Month 2
- [ ] External validation initiated (at least one database)
- [ ] Temporal validation study complete
- [ ] Baseline comparison started

### Month 3
- [ ] External validation complete
- [ ] Baseline comparison complete
- [ ] Paper draft started

### Month 4-6
- [ ] Paper submitted
- [ ] Clinical tool prototype
- [ ] Multi-organism exploration

---

## Success Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| PI correlation | +0.92 | +0.93 | Week 4 |
| NRTI correlation | +0.07 | +0.25 | Week 4 |
| NNRTI correlation | +0.19 | +0.35 | Month 2 |
| INI correlation | +0.14 | +0.30 | Month 2 |
| External validation | N/A | +0.85 | Month 3 |
| Paper status | N/A | Submitted | Month 4 |

---

## Resource Requirements

### Compute
- GPU: Continue using current CUDA setup
- Extended training: May need cloud GPU for large experiments

### Data
- External databases: Require access requests
- Clinical validation: Requires institutional partnership

### Time
- Estimated total: 4-6 months to publication
- Critical path: External validation → Paper

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| External data unavailable | Medium | High | Start access requests now |
| NRTI never improves | Medium | Medium | Document as limitation |
| Transformer remains unstable | Low | Low | Use VAE (already good) |
| Baseline beats us | Low | High | Focus on unique contributions |

---

## Conclusion

The immediate priority is **TAM encoding integration** - this has highest potential impact with lowest effort. If successful, NRTI correlation should improve significantly.

The medium-term focus should be **external validation** - this is essential for publication credibility.

The long-term vision is a **clinical decision support tool** - but this requires the foundational work to be solid first.

**Recommended first action**: Modify `run_on_real_data.py` to use TAM encoding for NRTI drugs and run experiments.
