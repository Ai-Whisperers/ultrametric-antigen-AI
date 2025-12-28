# Detailed Next Steps Plan

**Generated**: 2025-12-28
**Current Status**: Infrastructure complete, synthetic benchmarks passing, real data integration needed

---

## Executive Summary

### What's Done ✅
| Component | Status | Notes |
|-----------|--------|-------|
| 11 Disease Analyzers | ✅ Complete | HIV, SARS-CoV-2, TB, Influenza, HCV, HBV, Malaria, MRSA, Candida, RSV, Cancer |
| Cross-Disease Benchmark Script | ✅ Complete | `scripts/experiments/run_cross_disease.py` |
| Physics Validation Script | ✅ Complete | `scripts/experiments/run_physics_validation.py` |
| Unit Tests | ✅ 39 tests passing | `tests/unit/diseases/test_disease_analyzers.py` |
| Reproducibility Package | ✅ Started | environment.yml, 4 Jupyter notebooks |
| REST API | ✅ Enhanced | Multi-disease endpoints |
| Experiment Framework | ✅ Complete | `reproducibility/run_all_experiments.py` |

### Critical Gaps ❌
| Gap | Impact | Priority |
|-----|--------|----------|
| **Real clinical data** | Can't validate actual performance | 🔴 CRITICAL |
| Synthetic data quality | 4/9 diseases show 0 correlation | 🟠 HIGH |
| Full VAE model training | Only Ridge baseline tested | 🟠 HIGH |
| Test coverage | Currently ~48% | 🟡 MEDIUM |
| Publication figures with real data | Can't publish synthetic results | 🟠 HIGH |

---

## Phase 1: Data Acquisition (Priority: CRITICAL)

### 1.1 HIV Data (Baseline - Already Available)

**Source**: Stanford HIVDB (https://hivdb.stanford.edu/)

**Tasks**:
| Task | File to Create | Data |
|------|----------------|------|
| Download HIVDB dataset | `scripts/ingest/download_hivdb.py` | ~50,000 sequences |
| Parse mutation-resistance pairs | `scripts/ingest/parse_hivdb.py` | RT, PR, IN genes |
| Create train/val/test splits | `data/hiv/splits/` | 80/10/10 |
| Validate against published results | `notebooks/hiv_baseline_validation.ipynb` | Target: ρ=0.89 |

**Expected Output**:
```
data/hiv/
├── raw/
│   └── hivdb_mutations.csv
├── processed/
│   ├── rt_resistance.csv      # Reverse Transcriptase
│   ├── pr_resistance.csv      # Protease
│   └── in_resistance.csv      # Integrase
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 1.2 SARS-CoV-2 Data

**Sources**:
- GISAID (https://gisaid.org/) - Spike sequences
- CoVDB (https://covdb.stanford.edu/) - Drug resistance
- Bloom Lab DMS (https://jbloomlab.github.io/SARS2-RBD-escape-calc/) - Escape mutations

**Tasks**:
| Task | File to Create | Data |
|------|----------------|------|
| GISAID account setup | Manual | Requires institution affiliation |
| Download spike sequences | `scripts/ingest/download_gisaid_spike.py` | VOC variants |
| Paxlovid resistance data | `scripts/ingest/download_covdb.py` | nsp5 (Mpro) mutations |
| Parse escape mutations | `scripts/ingest/parse_bloom_dms.py` | RBD antibody escape |

**Expected Output**:
```
data/sars_cov_2/
├── raw/
│   ├── gisaid_spike_sequences.fasta
│   └── covdb_resistance.json
├── processed/
│   ├── paxlovid_resistance.csv
│   ├── antibody_escape.csv
│   └── variant_labels.csv
└── splits/
```

### 1.3 Tuberculosis Data

**Sources**:
- WHO TB Mutation Catalogue (https://www.who.int/publications/i/item/9789240082410)
- CRyPTIC Consortium (https://www.crypticproject.org/)

**Tasks**:
| Task | File to Create | Data |
|------|----------------|------|
| Download WHO catalogue | `scripts/ingest/download_who_tb.py` | 38,000 isolates |
| Parse per-drug resistance | `scripts/ingest/parse_tb_resistance.py` | 13 drugs |
| Map to gene sequences | `scripts/ingest/tb_gene_mapping.py` | rpoB, katG, etc. |

**Expected Output**:
```
data/tuberculosis/
├── raw/
│   └── who_tb_catalogue.xlsx
├── processed/
│   ├── rifampicin_resistance.csv
│   ├── isoniazid_resistance.csv
│   ├── fluoroquinolone_resistance.csv
│   └── bedaquiline_resistance.csv
└── splits/
```

### 1.4 Influenza Data

**Sources**:
- GISAID EpiFlu (https://gisaid.org/)
- NCBI Influenza Virus Resource (https://www.ncbi.nlm.nih.gov/genomes/FLU/)

**Tasks**:
| Task | File to Create | Data |
|------|----------------|------|
| Download HA/NA sequences | `scripts/ingest/download_influenza.py` | H1N1, H3N2 |
| Parse drug resistance | `scripts/ingest/parse_flu_resistance.py` | Oseltamivir, Baloxavir |
| Compile vaccine strain history | `scripts/ingest/flu_vaccine_history.py` | WHO recommendations |

**Expected Output**:
```
data/influenza/
├── raw/
│   ├── h3n2_ha_sequences.fasta
│   └── h1n1_na_sequences.fasta
├── processed/
│   ├── oseltamivir_resistance.csv
│   ├── baloxavir_resistance.csv
│   └── vaccine_strains.csv
└── splits/
```

### 1.5 Other Diseases (Lower Priority)

| Disease | Source | Difficulty |
|---------|--------|------------|
| HCV | HCVdb, Geno2pheno | Medium |
| HBV | HBVdb | Medium |
| Malaria | WWARN, MalariaGEN | High (genomic data) |
| MRSA | CARD, PATRIC | Medium |
| Candida | CDC AR Lab Network | High (limited) |
| RSV | NCBI Virus | Medium |
| Cancer | PharmGKB, COSMIC | Medium |

---

## Phase 2: Fix Synthetic Data Generators

### 2.1 Problem Analysis

Current synthetic data for 4 diseases produces random targets:
```python
# BAD: Random targets (no signal)
y = np.random.rand(n_samples)

# GOOD: Targets correlated with features
weights = np.random.randn(n_features) * 0.1
y = sigmoid(np.dot(X, weights) + np.random.normal(0, 0.1, n_samples))
```

### 2.2 Files to Fix

| File | Current Issue | Fix |
|------|---------------|-----|
| `src/diseases/influenza_analyzer.py` | Random fitness scores | Correlate with HA mutations |
| `src/diseases/hbv_analyzer.py` | Random resistance | Use mutation weights |
| `src/diseases/mrsa_analyzer.py` | Random MIC values | Based on known resistance genes |
| `src/diseases/candida_analyzer.py` | Random resistance | Based on ERG11 mutations |

### 2.3 Implementation

Create: `src/diseases/utils/synthetic_data.py`
```python
def generate_correlated_targets(
    X: np.ndarray,
    signal_strength: float = 0.5,
    noise_level: float = 0.2,
    seed: int = 42
) -> np.ndarray:
    """Generate targets that correlate with features."""
    np.random.seed(seed)
    n_samples, n_features = X.shape

    # Select subset of features as "causal"
    n_causal = max(1, n_features // 10)
    causal_idx = np.random.choice(n_features, n_causal, replace=False)

    # Generate weights for causal features
    weights = np.zeros(n_features)
    weights[causal_idx] = np.random.randn(n_causal) * signal_strength

    # Compute targets
    y = np.dot(X, weights)
    y += np.random.normal(0, noise_level, n_samples)
    y = (y - y.min()) / (y.max() - y.min())  # Normalize to [0, 1]

    return y
```

---

## Phase 3: Real Data Benchmarking

### 3.1 HIV Baseline Validation

**Goal**: Reproduce published HIV results (ρ = 0.89)

**Script**: `scripts/experiments/run_hiv_baseline.py`

```python
# Pseudocode
1. Load HIVDB data
2. Apply p-adic encoding
3. Train VAE model (not just Ridge!)
4. 5-fold cross-validation
5. Compute Spearman correlation per drug
6. Compare with Stanford HIVDB published results
```

**Expected Results**:
| Drug | Published | Our Model | Status |
|------|-----------|-----------|--------|
| 3TC | 0.92 | TBD | |
| AZT | 0.88 | TBD | |
| EFV | 0.91 | TBD | |
| ... | ... | ... | |
| **Average** | **0.89** | **Target: 0.85+** | |

### 3.2 Cross-Disease Benchmark with Real Data

**Script**: `scripts/experiments/run_cross_disease_real.py`

```python
# Pseudocode
1. For each disease with real data:
   a. Load processed data
   b. Apply disease-specific p-adic encoding
   c. Train model (with optional transfer learning from HIV)
   d. Evaluate with cross-validation
2. Generate comparison table
3. Create publication figures
```

### 3.3 Transfer Learning Experiments

**Goal**: Show HIV pre-training improves other diseases

**Script**: `scripts/experiments/run_transfer_learning.py`

```python
# Experiment Design
for target_disease in [SARS_CoV_2, TB, Influenza, HCV]:
    # Baseline: train from scratch
    model_scratch = train_vae(target_data)
    score_scratch = evaluate(model_scratch)

    # Transfer: pre-train on HIV, fine-tune on target
    model_pretrained = load_hiv_model()
    model_transfer = fine_tune(model_pretrained, target_data)
    score_transfer = evaluate(model_transfer)

    # Compute improvement
    improvement = (score_transfer - score_scratch) / score_scratch * 100
```

---

## Phase 4: Full VAE Model Training

### 4.1 Current Gap

The benchmark currently uses **Ridge regression** as a baseline:
```python
# Current (scripts/experiments/run_cross_disease.py)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

Need to use the **full VAE model**:
```python
# Required
from src.models.ternary_vae import TernaryVAE
model = TernaryVAE(input_dim=n_features, latent_dim=32)
model.fit(X_train, y_train, epochs=100)
```

### 4.2 Tasks

| Task | File | Notes |
|------|------|-------|
| Verify VAE training works | `tests/unit/models/test_ternary_vae.py` | Add training tests |
| Add resistance prediction head | `src/models/ternary_vae.py` | Regression output |
| Create training script | `scripts/training/train_vae.py` | Configurable |
| Add hyperparameter tuning | `scripts/training/tune_hyperparams.py` | Optuna integration |

### 4.3 Model Architecture

```
Input (n_features)
    ↓
Encoder (FC: 512 → 256 → 128)
    ↓
Hyperbolic Projection (Poincaré ball, dim=32)
    ↓
μ, log_σ (latent distribution)
    ↓
Reparameterization (z = μ + σ * ε)
    ↓
Decoder (FC: 128 → 256 → 512 → n_features)
    ↓
Reconstruction + Resistance Predictor (FC: 32 → 1)
```

---

## Phase 5: Physics Validation with Real Data

### 5.1 ΔΔG Correlation

**Goal**: Show p-adic distance correlates with protein stability changes

**Data Sources**:
- ProTherm (protein stability database)
- FireProtDB (stability upon mutation)
- Existing HIV structure data (`research/.../hiv/data/structures/`)

**Script**: `scripts/experiments/run_ddg_validation.py`

**Expected Output**:
- Correlation: ρ > 0.7 between p-adic distance and ΔΔG
- Universal across proteins (HIV, SARS-CoV-2, TB)

### 5.2 Mass Invariant Validation

**Goal**: Confirm p-adic encoding preserves amino acid mass relationships

**Script**: `scripts/experiments/run_mass_invariant.py`

```python
# Pseudocode
1. For each amino acid substitution:
   a. Compute mass change (Δm)
   b. Compute p-adic distance
   c. Record correlation
2. Plot: mass change vs p-adic distance
3. Compute Pearson correlation (target: r > 0.9)
```

### 5.3 6-Level Hierarchy Validation

**Goal**: Show p-adic captures structure at all scales

| Level | Property | Validation |
|-------|----------|------------|
| 1. Atomic | Bond lengths | Compare with X-ray structures |
| 2. Residue | Side chain orientation | Compare with rotamer libraries |
| 3. Secondary | Helix/sheet propensity | Compare with DSSP |
| 4. Tertiary | Domain organization | Compare with CATH |
| 5. Quaternary | Complex formation | Compare with PDB biological assemblies |
| 6. Evolutionary | Conservation | Compare with ConSurf |

---

## Phase 6: Improve Test Coverage

### 6.1 Current Status

```
Current coverage: ~48%
Target coverage: 70%+
```

### 6.2 Missing Tests

| Module | Current | Needed |
|--------|---------|--------|
| `src/models/ternary_vae.py` | 0% | Training, inference, loss |
| `src/losses/` | 0% | All loss functions |
| `src/encoders/padic/` | 30% | Edge cases, large sequences |
| `src/api/` | 0% | All endpoints |
| `src/diseases/` | 80% | Integration tests |

### 6.3 Test Files to Create

```
tests/
├── unit/
│   ├── models/
│   │   ├── test_ternary_vae.py      # NEW
│   │   ├── test_hyperbolic.py       # NEW
│   │   └── test_encoder_decoder.py  # NEW
│   ├── losses/
│   │   ├── test_vae_loss.py         # NEW
│   │   └── test_padic_loss.py       # NEW
│   ├── encoders/
│   │   ├── test_padic_encoder.py    # EXPAND
│   │   └── test_codon_encoder.py    # EXPAND
│   └── api/
│       └── test_drug_resistance_api.py  # NEW
├── integration/
│   ├── test_end_to_end_pipeline.py  # NEW
│   └── test_cross_disease_flow.py   # NEW
└── conftest.py                       # Shared fixtures
```

---

## Phase 7: Publication Preparation

### 7.1 Required Figures (with Real Data)

| Figure | Content | Notebook |
|--------|---------|----------|
| 1 | Cross-disease performance bar chart | `Figure1_CrossDisease.ipynb` |
| 2 | P-adic physics hierarchy | `Figure2_Physics.ipynb` |
| 3 | Architecture + transfer learning | `Figure3_Architecture.ipynb` |
| 4 | Mass invariant + clinical | `Figure4_MassCorrelation.ipynb` |
| 5 | ΔΔG correlation scatter | `Figure5_DDG.ipynb` (NEW) |
| 6 | ROC curves per disease | `Figure6_ROC.ipynb` (NEW) |

### 7.2 Tables

| Table | Content |
|-------|---------|
| 1 | Dataset statistics (samples, features, drugs) |
| 2 | Per-drug Spearman correlations |
| 3 | Transfer learning improvements |
| 4 | Comparison with published methods |

### 7.3 Supplementary Materials

```
supplementary/
├── S1_full_results.xlsx          # All experimental results
├── S2_hyperparameters.md         # Model configurations
├── S3_data_processing.md         # Data pipeline details
├── S4_statistical_tests.md       # Significance tests
└── S5_computational_resources.md # Hardware, runtime
```

---

## Phase 8: API Testing and Demo

### 8.1 API Test Suite

Create: `tests/unit/api/test_drug_resistance_api.py`

```python
def test_list_diseases():
    response = client.get("/diseases")
    assert response.status_code == 200
    assert len(response.json()) == 11

def test_predict_hiv_resistance():
    response = client.post("/predict/hiv", json={"sequence": "ATGCGT..."})
    assert response.status_code == 200
    assert "resistance_score" in response.json()

def test_invalid_disease():
    response = client.post("/predict/invalid_disease", json={})
    assert response.status_code == 404
```

### 8.2 Interactive Demo Notebook

Create: `notebooks/demo/interactive_resistance_prediction.ipynb`

```python
# Cell 1: Start API server
!uvicorn src.api.drug_resistance_api:app --port 8000 &

# Cell 2: Interactive prediction
import ipywidgets as widgets

disease_dropdown = widgets.Dropdown(options=['hiv', 'sars_cov_2', 'tb', ...])
sequence_input = widgets.Textarea(placeholder="Enter sequence...")
predict_button = widgets.Button(description="Predict")

# Cell 3: Visualization
# Show resistance scores with confidence intervals
```

---

## Prioritized Task List

### Week 1: Data Foundation
- [ ] Download HIVDB dataset
- [ ] Parse HIV mutation-resistance data
- [ ] Create data loading utilities
- [ ] Run HIV baseline with real data

### Week 2: Fix Synthetic + More Data
- [ ] Fix synthetic data generators (4 diseases)
- [ ] Download SARS-CoV-2 data (GISAID, CoVDB)
- [ ] Download TB data (WHO catalogue)
- [ ] Re-run cross-disease benchmark

### Week 3: VAE Training
- [ ] Add VAE training tests
- [ ] Create training script
- [ ] Train HIV model with real data
- [ ] Validate: match ρ=0.89 target

### Week 4: Transfer Learning
- [ ] Implement transfer learning pipeline
- [ ] Run experiments: HIV → SARS-CoV-2
- [ ] Run experiments: HIV → TB
- [ ] Document improvements

### Week 5: Physics Validation
- [ ] Run ΔΔG correlation analysis
- [ ] Validate mass invariant
- [ ] 6-level hierarchy analysis
- [ ] Create physics figures

### Week 6: Tests + Polish
- [ ] Add missing unit tests (target: 70%)
- [ ] API testing
- [ ] Fix any bugs discovered
- [ ] Code cleanup

### Week 7-8: Publication
- [ ] Generate all figures with real data
- [ ] Write results tables
- [ ] Create supplementary materials
- [ ] Final reproducibility check

---

## Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| HIV Spearman | ≥ 0.85 | Cross-validation on HIVDB |
| SARS-CoV-2 Spearman | ≥ 0.75 | Cross-validation on CoVDB |
| TB Spearman | ≥ 0.75 | Cross-validation on WHO data |
| Transfer improvement | ≥ 20% | vs training from scratch |
| ΔΔG correlation | ≥ 0.70 | Pearson r |
| Mass invariant | ≥ 0.85 | Pearson r |
| Test coverage | ≥ 70% | pytest-cov |
| All 11 diseases | Spearman > 0 | No zero-correlation diseases |

---

## File Summary

### To Create
```
scripts/ingest/
├── download_hivdb.py
├── download_gisaid_spike.py
├── download_who_tb.py
├── download_influenza.py
└── parse_*.py (various)

scripts/training/
├── train_vae.py
├── tune_hyperparams.py
└── run_transfer_learning.py

tests/unit/
├── models/test_ternary_vae.py
├── losses/test_*.py
└── api/test_drug_resistance_api.py

notebooks/
├── Figure5_DDG.ipynb
├── Figure6_ROC.ipynb
└── demo/interactive_resistance_prediction.ipynb
```

### To Modify
```
src/diseases/influenza_analyzer.py  # Fix synthetic data
src/diseases/hbv_analyzer.py        # Fix synthetic data
src/diseases/mrsa_analyzer.py       # Fix synthetic data
src/diseases/candida_analyzer.py    # Fix synthetic data
scripts/experiments/run_cross_disease.py  # Add VAE model option
```

---

## Quick Start Commands

```bash
# 1. Set up environment
conda env create -f reproducibility/environment.yml
conda activate ternary-vae

# 2. Run current tests
pytest tests/ -v

# 3. Run synthetic benchmark (current state)
python scripts/experiments/run_cross_disease.py

# 4. Start API server
uvicorn src.api.drug_resistance_api:app --reload

# 5. Run reproducibility check
python reproducibility/run_all_experiments.py
```

---

*Last updated: 2025-12-28*
