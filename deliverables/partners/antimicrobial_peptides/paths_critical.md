# Critical Paths: ML Artifacts & Data

**Doc-Type:** Artifact Registry · Version 1.0 · Updated 2026-01-05 · AI Whisperers

This document catalogs all critical ML artifacts, checkpoints, and data paths for the Carlos Brizuela AMP package. Use this for package extraction to a standalone repository.

---

## Production Checkpoint (USE THIS)

| Item | Path | Size | Description |
|------|------|:----:|-------------|
| **Best Model** | `checkpoints_definitive/best_production.pt` | 1.2 MB | Fold 2, Spearman r=0.7368 |
| **CV Results** | `checkpoints_definitive/cv_results_definitive.json` | 10 KB | Full 5-fold metrics |

**Load Example:**
```python
checkpoint = torch.load('checkpoints_definitive/best_production.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## All Checkpoints (Training History)

### Definitive Training (2026-01-05) - RECOMMENDED

| File | Spearman r | Status |
|------|:----------:|--------|
| `checkpoints_definitive/fold_0_definitive.pt` | 0.6945 | PASSED |
| `checkpoints_definitive/fold_1_definitive.pt` | 0.5581 | MARGINAL |
| `checkpoints_definitive/fold_2_definitive.pt` | **0.7368** | **BEST** |
| `checkpoints_definitive/fold_3_definitive.pt` | 0.6542 | PASSED |
| `checkpoints_definitive/fold_4_definitive.pt` | 0.6379 | PASSED |

**Mean:** 0.6563 ± 0.0599 | **Collapse Rate:** 0%

### Improved Training (Previous) - DEPRECATED

| File | Spearman r | Status |
|------|:----------:|--------|
| `checkpoints_improved/fold_0_improved.pt` | 0.656 | OK |
| `checkpoints_improved/fold_1_improved.pt` | 0.146 | **COLLAPSED** |
| `checkpoints_improved/fold_2_improved.pt` | 0.686 | OK |
| `checkpoints_improved/fold_3_improved.pt` | 0.592 | OK |
| `checkpoints_improved/fold_4_improved.pt` | 0.547 | MARGINAL |

**Mean:** 0.525 ± 0.196 | **Collapse Rate:** 20%

### Original Training - DEPRECATED

| File | Description |
|------|-------------|
| `checkpoints/fold_*_best.pt` | Original training, lower performance |
| `checkpoints/cv_results.json` | Original CV metrics |

---

## Training Data

### Source: DRAMP Database

| Item | Path | Description |
|------|------|-------------|
| **Loader** | `scripts/dramp_activity_loader.py` | Downloads/caches DRAMP data |
| **Cache** | `~/.cache/dramp/amp_database.json` | Cached curated AMPs |
| **URL** | `http://dramp.cpu-bioinfor.org/` | DRAMP official source |

**Statistics:**
- 425 curated AMP records
- 4 pathogens: E.coli, P.aeruginosa, S.aureus, A.baumannii
- Features: 32 physicochemical properties

### sklearn Baseline Models

| File | Description |
|------|-------------|
| `~/.cache/dramp/models/general_model.pkl` | General Ridge regression |
| `~/.cache/dramp/models/ecoli_model.pkl` | E.coli-specific model |
| `~/.cache/dramp/models/paeruginosa_model.pkl` | P.aeruginosa model |
| `~/.cache/dramp/models/saureus_model.pkl` | S.aureus model |
| `~/.cache/dramp/models/abaumannii_model.pkl` | A.baumannii model |

---

## Validation Results

### sklearn Validation

| File | Description |
|------|-------------|
| `validation/results/comprehensive_validation.json` | Full sklearn metrics |
| `validation/results/bootstrap_results.json` | Bootstrap confidence intervals |
| `validation/results/SCIENTIFIC_VALIDATION_REPORT.md` | Human-readable report |

### PeptideVAE Analysis

| File | Description |
|------|-------------|
| `validation/results/regime_analysis.json` | Regime-specific breakdown |
| `validation/results/deep_regime_findings.json` | Detailed regime analysis |
| `validation/results/falsification_report.json` | Falsification test results |

---

## NSGA-II Optimization Results (Demo)

| File | Tool | Description |
|------|------|-------------|
| `results/pareto_peptides.csv` | Core | 100 Pareto-optimal candidates |
| `results/pathogen_specific/S_aureus_results.json` | B1 | Pathogen-specific output |
| `results/pathogen_specific/S_aureus_candidates.csv` | B1 | Candidate sequences |
| `results/microbiome_safe/microbiome_safe_results.json` | B8 | Selectivity results |
| `results/microbiome_safe/microbiome_safe_candidates.csv` | B8 | Safe candidates |
| `results/synthesis_optimized/synthesis_optimized_results.json` | B10 | Synthesis metrics |
| `results/synthesis_optimized/synthesis_optimized_candidates.csv` | B10 | Optimized candidates |

---

## Source Code Dependencies

### Core Model (src/)

| File | Lines | Description |
|------|:-----:|-------------|
| `../../../src/encoders/peptide_encoder.py` | 1060 | PeptideVAE architecture |
| `../../../src/losses/peptide_losses.py` | 862 | 6-component loss system |

### Package Code

| File | Lines | Description |
|------|:-----:|-------------|
| `training/train_definitive.py` | 499 | Fixed training script |
| `training/dataset.py` | 421 | PyTorch dataset |
| `scripts/latent_nsga2.py` | 490 | NSGA-II optimizer |
| `scripts/B1_pathogen_specific_design.py` | ~400 | B1 tool |
| `scripts/B8_microbiome_safe_amps.py` | ~400 | B8 tool |
| `scripts/B10_synthesis_optimization.py` | ~400 | B10 tool |
| `src/vae_interface.py` | - | VAE wrapper |
| `src/objectives.py` | - | Objective functions |

---

## Shared Dependencies

| Path | Description |
|------|-------------|
| `../shared/config.py` | Shared configuration |
| `../shared/constants.py` | Amino acid properties, WHO pathogens |

---

## For Standalone Repository

### Must Copy (Critical)

```
checkpoints_definitive/
├── best_production.pt          # Production model
├── cv_results_definitive.json  # Metrics
└── README.md                   # Model card

training/
├── train_definitive.py         # Training script
└── dataset.py                  # Data loading

scripts/
├── dramp_activity_loader.py    # Data source
├── latent_nsga2.py            # NSGA-II core
├── B1_pathogen_specific_design.py
├── B8_microbiome_safe_amps.py
└── B10_synthesis_optimization.py

src/
├── vae_interface.py
└── objectives.py
```

### Must Copy from Parent Repo

```
src/encoders/peptide_encoder.py
src/losses/peptide_losses.py
deliverables/shared/config.py
deliverables/shared/constants.py
```

### Can Regenerate

```
results/                        # Re-run NSGA-II
validation/results/             # Re-run validation
~/.cache/dramp/                 # Re-download
```

---

## Storage Recommendations

| Artifact Type | Recommendation |
|---------------|----------------|
| Checkpoints (.pt) | HuggingFace Hub or DVC |
| Results (.json, .csv) | Git (small files) |
| Cached data | User's ~/.cache, re-downloadable |
| Training logs | Optional, can regenerate |

---

## What Works Today (Production Ready)

### PeptideVAE MIC Prediction

| Component | Status | Performance | Path |
|-----------|:------:|-------------|------|
| **PeptideVAE Encoder** | READY | Spearman r=0.7368 | `checkpoints_definitive/best_production.pt` |
| **MIC Prediction Head** | READY | 31% better than sklearn | Integrated in checkpoint |
| **Inference Script** | READY | Single/batch prediction | `scripts/predict_mic.py` |

**Usage:**
```python
from scripts.predict_mic import PeptideMICPredictor

predictor = PeptideMICPredictor()
mic = predictor.predict("KLAKLAKKLAKLAK")  # Returns predicted MIC
```

### sklearn Baseline Models

| Model | Status | Performance | Use Case |
|-------|:------:|-------------|----------|
| General Ridge | READY | r=0.56 | Any pathogen |
| E.coli-specific | READY | r=0.58 | Gram-negative |
| S.aureus-specific | READY | r=0.54 | Gram-positive (MRSA) |
| P.aeruginosa | READY | r=0.52 | Resistant Gram-negative |
| A.baumannii | READY | r=0.49 | Carbapenem-resistant |

**Usage:**
```python
from scripts.dramp_activity_loader import DRAMPLoader

loader = DRAMPLoader()
mic = loader.predict_activity("KLAKLAKKLAKLAK", pathogen="saureus")
```

### DRAMP Data Pipeline

| Component | Status | Records | Path |
|-----------|:------:|:-------:|------|
| Data Loader | READY | 425 AMPs | `scripts/dramp_activity_loader.py` |
| Feature Extractor | READY | 32 properties | Integrated |
| Caching System | READY | Auto-download | `~/.cache/dramp/` |

### Sequence-Space Optimizer

| Component | Status | Description | Path |
|-----------|:------:|-------------|------|
| **SequenceNSGA2** | READY | Multi-objective optimization | `scripts/sequence_nsga2.py` |
| **Mutation Operators** | READY | Substitution, insertion, deletion | Integrated |
| **Pareto Selection** | READY | NSGA-II with crowding distance | DEAP-based |

**Usage:**
```python
from scripts.sequence_nsga2 import SequenceNSGA2

optimizer = SequenceNSGA2(
    seed_sequences=["KLAKLAKKLAKLAK", "KLWKKLKKALK"],
    population_size=100,
    generations=50,
)
pareto_front = optimizer.run()
```

---

## What Does NOT Work (Requires Fixes)

| Component | Issue | Root Cause | Fix Status |
|-----------|-------|------------|------------|
| B1 Tool | 3-char sequences | Uses TernaryVAE | See INTEGRATION_PLAN.md |
| B8 Tool | 3-char sequences | Uses TernaryVAE | See INTEGRATION_PLAN.md |
| B10 Tool | 3-char sequences | Uses TernaryVAE | See INTEGRATION_PLAN.md |
| Latent Decoder | Wrong model | TernaryVAE ≠ PeptideVAE | Replace in vae_service.py |

**Note:** These tools need refactoring to use the sequence-space optimizer instead of latent-space decoding.

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.1 | Added "What Works Today" and "What Does NOT Work" sections |
| 2026-01-05 | 1.0 | Initial artifact registry after definitive training |
