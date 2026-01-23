# Full Extraction Audit - Deliverables Folder

**Doc-Type:** Extraction Planning · Version 1.0 · Updated 2026-01-23 · AI Whisperers

---

## Purpose

This document provides a comprehensive audit for extracting the `deliverables/` folder (or individual partner packages) to an independent repository. It includes:
- Complete file inventory with sizes and line counts
- Full transitive dependency chains
- Checkpoint file manifest
- Import statement locations with line numbers
- Checklist format for tracking extraction progress

**Policy:** This document should be updated whenever dependencies change.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Source Files to Copy** | 14 files |
| **Total Lines of Code** | ~7,500 LOC |
| **Total Checkpoint Size** | ~9.5 MB |
| **External pip Dependencies** | 4 packages |
| **Estimated Extraction Effort** | High (2-3 days) |

### Quick Decision Matrix

| Extraction Scenario | Effort | Code Duplication | Maintenance Burden |
|---------------------|--------|------------------|-------------------|
| Keep in main repo | None | None | None |
| Extract shared/ only | Low | ~2,700 LOC | Low |
| Extract partner packages (partial) | Medium | ~4,700 LOC | Medium |
| Full extraction with ML | High | ~7,500 LOC | High |

---

## 1. Complete Dependency Tree

### 1.1 Visual Dependency Graph

```
deliverables/
├── shared/                          [2,719 LOC - SELF-CONTAINED]
│   ├── config.py (153)
│   ├── constants.py (167)
│   ├── hemolysis_predictor.py (399)
│   ├── logging_utils.py (304)
│   ├── peptide_utils.py (319)
│   ├── primer_design.py (557)
│   ├── uncertainty.py (372)
│   └── vae_service.py (348)        → src.models.TernaryVAEV5_11_PartialFreeze
│                                    → checkpoints/homeostatic_rich/best.pt
│
├── partners/jose_colbes/            [~70% SELF-CONTAINED]
│   ├── core/ (LOCAL)               ✓ padic_math.py, constants.py
│   ├── src/validated_ddg_predictor.py
│   │   └── EXTERNAL: TrainableCodonEncoder
│   │       └── src.biology.codons
│   │       └── src.geometry
│   │       └── src.encoders.codon_encoder.AA_PROPERTIES
│   │   └── EXTERNAL: poincare_distance
│   └── validation/bootstrap_test.py
│       └── EXTERNAL: TrainableCodonEncoder, poincare_distance
│
├── partners/alejandra_rojas/        [~85% SELF-CONTAINED]
│   ├── src/ (LOCAL)                ✓ padic_math.py, codons.py, constants.py
│   ├── scripts/denv4_*.py
│   │   └── EXTERNAL: TrainableCodonEncoder
│   │   └── EXTERNAL: poincare_distance
│   │   └── EXTERNAL: src.biology.codons
│   └── src/geometry.py             → src.geometry (optional fallback)
│
├── partners/carlos_brizuela/        [~60% SELF-CONTAINED]
│   ├── src/ (LOCAL)                ✓ constants.py, peptide_utils.py, uncertainty.py
│   ├── scripts/predict_mic.py
│   │   └── EXTERNAL: PeptideVAE
│   │       └── src.encoders.padic_amino_acid_encoder
│   │       └── src.geometry
│   │       └── src.models.hyperbolic_projection
│   ├── training/train_*.py
│   │   └── EXTERNAL: PeptideVAE
│   │   └── EXTERNAL: PeptideLossManager
│   │       └── src.geometry.poincare_distance
│   │       └── src.losses.base
│   └── checkpoints_definitive/     ✓ LOCAL (6 files, 7.0 MB)
│
└── partners/hiv_research_package/   [~90% SELF-CONTAINED]
    └── EXTERNAL: shared.config, shared.constants
```

### 1.2 Transitive Dependency Chain (Full)

```
TrainableCodonEncoder (586 LOC)
├── src.biology.codons (250 LOC)           [LEAF]
├── src.geometry (413 LOC total)
│   ├── poincare.py (356 LOC)              [geoopt]
│   └── __init__.py (57 LOC)               [LEAF]
└── src.encoders.codon_encoder (451 LOC)
    ├── src.biology.codons                 [ALREADY COUNTED]
    ├── src.core.padic_math (489 LOC)      [LEAF]
    └── src.geometry                       [ALREADY COUNTED]

PeptideVAE (1,059 LOC)
├── src.encoders.padic_amino_acid_encoder (832 LOC)
│   └── src.core.padic_math                [ALREADY COUNTED]
├── src.geometry                           [ALREADY COUNTED]
└── src.models.hyperbolic_projection (327 LOC)
    └── src.geometry.ManifoldParameter     [ALREADY COUNTED]

PeptideLossManager (862 LOC)
├── src.geometry.poincare_distance         [ALREADY COUNTED]
└── src.losses.base (216 LOC)              [LEAF]
```

---

## 2. Complete File Inventory

### 2.1 Main Project Source Files Required

| File | Lines | Size | Direct Dependencies | Extraction Status |
|------|-------|------|---------------------|:-----------------:|
| **TIER 1: Core Encoders** |||||
| `src/encoders/trainable_codon_encoder.py` | 586 | 22K | biology.codons, geometry, codon_encoder | [ ] |
| `src/encoders/peptide_encoder.py` | 1,059 | 38K | padic_aa_encoder, geometry, hyperbolic_projection | [ ] |
| `src/encoders/codon_encoder.py` | 451 | 16K | biology.codons, core.padic_math, geometry | [ ] |
| `src/encoders/padic_amino_acid_encoder.py` | 832 | 30K | core.padic_math | [ ] |
| **TIER 2: Geometry/Math** |||||
| `src/geometry/__init__.py` | 57 | 2K | poincare.py | [ ] |
| `src/geometry/poincare.py` | 356 | 13K | geoopt (external) | [ ] |
| `src/core/padic_math.py` | 489 | 17K | None (pure Python) | [ ] |
| `src/biology/codons.py` | 250 | 9K | None (pure Python) | [ ] |
| **TIER 3: Models/Losses** |||||
| `src/models/hyperbolic_projection.py` | 327 | 12K | geometry.ManifoldParameter | [ ] |
| `src/losses/peptide_losses.py` | 862 | 31K | geometry.poincare_distance, losses.base | [ ] |
| `src/losses/base.py` | 216 | 8K | None (pure Python) | [ ] |
| **TIER 4: TernaryVAE (for vae_service)** |||||
| `src/models/ternary_vae.py` | ~1,500 | ~55K | geometry, core, improved_components | [ ] |
| `src/models/improved_components.py` | ~400 | ~15K | None | [ ] |
| **TOTAL** | **~7,385** | **~268K** |||

### 2.2 Shared Infrastructure (Within Deliverables)

| File | Lines | Dependencies | Extraction Status |
|------|-------|--------------|:-----------------:|
| `shared/__init__.py` | 100 | Internal only | [x] MOVES WITH |
| `shared/config.py` | 153 | None | [x] MOVES WITH |
| `shared/constants.py` | 167 | None | [x] MOVES WITH |
| `shared/hemolysis_predictor.py` | 399 | sklearn | [x] MOVES WITH |
| `shared/logging_utils.py` | 304 | None | [x] MOVES WITH |
| `shared/peptide_utils.py` | 319 | numpy | [x] MOVES WITH |
| `shared/primer_design.py` | 557 | biopython | [x] MOVES WITH |
| `shared/uncertainty.py` | 372 | sklearn | [x] MOVES WITH |
| `shared/vae_service.py` | 348 | **TernaryVAEV5_11** | [ ] NEEDS MAIN |
| **TOTAL** | **2,719** |||

### 2.3 Checkpoint Files Required

| Checkpoint | Size | Used By | Required For |
|------------|------|---------|--------------|
| `checkpoints/homeostatic_rich/best.pt` | 421 KB | vae_service | TernaryVAE inference |
| `checkpoints/v5_11_homeostasis/best.pt` | 845 KB | vae_service (fallback) | TernaryVAE inference |
| `checkpoints/peptide_vae_v1/best_production.pt` | 1.2 MB | predict_mic.py | PeptideVAE inference |
| `research/codon-encoder/training/results/trained_codon_encoder.pt` | 51 KB | jose_colbes DDG | TrainableCodonEncoder |
| **TOTAL** | **~2.5 MB** |||

### 2.4 Partner-Local Checkpoints (Already Self-Contained)

| Package | Path | Size | Status |
|---------|------|------|:------:|
| carlos_brizuela | `checkpoints_definitive/best_production.pt` | 1.2 MB | [x] LOCAL |
| carlos_brizuela | `checkpoints_definitive/fold_*_definitive.pt` (5) | 5.8 MB | [x] LOCAL |
| jose_colbes | `models/*.pkl` | ~200 KB | [x] LOCAL |
| **TOTAL LOCAL** | | **~7.2 MB** ||

---

## 3. Import Statement Index

### 3.1 All External `from src.*` Imports

#### Jose Colbes Package

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `src/validated_ddg_predictor.py` | 58 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `src/validated_ddg_predictor.py` | 59 | `from src.geometry import poincare_distance` | [ ] |
| `validation/bootstrap_test.py` | 19 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `validation/bootstrap_test.py` | 20 | `from src.geometry import poincare_distance` | [ ] |
| `reproducibility/extract_aa_embeddings_v2.py` | 31 | `from src.biology.codons import CODON_TO_INDEX, ...` | [ ] |
| `reproducibility/extract_aa_embeddings_v2.py` | 37 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `reproducibility/train_padic_ddg_predictor_v2.py` | 43 | `from src.biology.codons import ...` | [ ] |
| `reproducibility/analyze_padic_ddg_full.py` | 43 | `from src.biology.codons import ...` | [ ] |
| `reproducibility/archive/extract_aa_embeddings.py` | 102 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |
| `reproducibility/archive/extract_aa_embeddings.py` | 204-205 | `from src.core import TERNARY`, `from src.data.generation import ...` | [ ] |
| `reproducibility/archive/extract_embeddings_simple.py` | 154 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |

#### Alejandra Rojas Package

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `scripts/denv4_padic_integration.py` | 50 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `scripts/denv4_padic_integration.py` | 51 | `from src.geometry import poincare_distance` | [ ] |
| `scripts/denv4_padic_integration.py` | 52 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_synonymous_conjecture.py` | 48-50 | Same as above | [ ] |
| `scripts/denv4_codon_bias_conjecture.py` | 57 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_revised_conjecture.py` | 55 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_codon_pair_conjecture.py` | 49 | `from src.biology.codons import ...` | [ ] |
| `src/geometry.py` | 31 | `from src.geometry import (exp_map_zero, log_map_zero, ...)` | [ ] OPTIONAL |
| `research/clade_classification/train_clade_classifier.py` | 49-50 | `TrainableCodonEncoder`, `src.biology.codons` | [ ] |
| `research/functional_convergence/find_convergence_points.py` | varies | `TrainableCodonEncoder` | [ ] |
| `validation/test_padic_conservation_correlation.py` | 44 | `TrainableCodonEncoder`, `poincare_distance` | [ ] |

#### Carlos Brizuela Package

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `scripts/predict_mic.py` | 80 | `from src.encoders.peptide_encoder import PeptideVAE` | [ ] |
| `training/train_definitive.py` | 41 | `from src.encoders.peptide_encoder import PeptideVAE` | [ ] |
| `training/train_definitive.py` | 42 | `from src.losses.peptide_losses import PeptideLossManager, CurriculumSchedule` | [ ] |
| `training/train_peptide_encoder.py` | 46-47 | Same as above | [ ] |
| `training/train_improved.py` | 39-40 | Same as above | [ ] |
| `verify_paths.py` | varies | `PeptideVAE`, `PeptideLossManager` | [ ] |
| `training/dataset.py` | 14 | `from deliverables.shared.peptide_utils import ...` | [ ] |

#### Shared Infrastructure

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `shared/vae_service.py` | 97 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |

### 3.2 All `from deliverables.*` Absolute Imports

| File | Line | Import Statement | Fix Required |
|------|------|------------------|:------------:|
| `partners/alejandra_rojas/tests/test_notebook_integration.py` | 248,256,268 | `from deliverables.shared.primer_design import ...` | Convert to relative |
| `partners/jose_colbes/validation/scientific_validation_report.py` | 193 | `from deliverables.shared...` | Convert to relative |
| `partners/jose_colbes/validation/alphafold_validation_pipeline.py` | 392 | `from deliverables.shared...` | Convert to relative |
| `partners/carlos_brizuela/training/dataset.py` | 14 | `from deliverables.shared.peptide_utils import ...` | Convert to relative |

---

## 4. External pip Dependencies

### 4.1 Required for Full Extraction

| Package | Version | Required By | Installation |
|---------|---------|-------------|--------------|
| `geoopt` | >=0.5.0 | src.geometry.poincare | `pip install geoopt` |
| `torch` | >=2.0.0 | All ML components | See PyTorch install guide |
| `numpy` | >=1.20.0 | All packages | `pip install numpy` |
| `scipy` | >=1.7.0 | Statistics, optimization | `pip install scipy` |
| `scikit-learn` | >=1.0.0 | ML predictors | `pip install scikit-learn` |
| `deap` | >=1.4.0 | carlos_brizuela NSGA-II | `pip install deap` |
| `biopython` | >=1.80 | alejandra_rojas primers | `pip install biopython` |
| `joblib` | >=1.0.0 | Model serialization | `pip install joblib` |

### 4.2 Optional Dependencies

| Package | Required For | Impact if Missing |
|---------|--------------|-------------------|
| `matplotlib` | Visualization | No plots generated |
| `seaborn` | Visualization | No styled plots |
| `requests` | Data downloading | Manual download required |
| `tensorboard` | Training monitoring | No live metrics |

---

## 5. Extraction Scenarios

### Scenario A: Minimal (Keep ML in Main Repo)

**Goal:** Partner packages work for basic operations, ML prediction requires main repo.

**What to Extract:**
- [x] `deliverables/shared/` (except vae_service.py ML parts)
- [x] Partner `core/` or `src/` local modules (already done)
- [x] Partner `requirements.txt` files (already done)

**What Stays:**
- [ ] All `src.*` imports remain as external dependencies
- [ ] ML inference requires main project installation

**Effort:** Already complete (current state)

### Scenario B: Partial (Copy Core Encoders)

**Goal:** ML prediction works for DDG and primers, training stays in main repo.

**Files to Copy (~4,700 LOC):**
- [ ] `src/encoders/trainable_codon_encoder.py` (586)
- [ ] `src/encoders/codon_encoder.py` (451)
- [ ] `src/geometry/` (413)
- [ ] `src/biology/codons.py` (250)
- [ ] `src/core/padic_math.py` (489)
- [ ] Checkpoints: `trained_codon_encoder.pt` (51 KB)

**Updates Required:**
- [ ] Update all `from src.*` imports to local paths
- [ ] Add `geoopt` to requirements
- [ ] Restructure as `deliverables/lib/encoders/`, etc.

**Effort:** Medium (1-2 days)

### Scenario C: Full (Complete Independence)

**Goal:** All features work without main repo.

**Additional Files (~2,800 LOC):**
- [ ] `src/encoders/peptide_encoder.py` (1,059)
- [ ] `src/encoders/padic_amino_acid_encoder.py` (832)
- [ ] `src/models/hyperbolic_projection.py` (327)
- [ ] `src/losses/peptide_losses.py` (862)
- [ ] `src/losses/base.py` (216)
- [ ] `src/models/ternary_vae.py` (~1,500) - for vae_service
- [ ] `src/models/improved_components.py` (~400)
- [ ] Additional checkpoints (~2.5 MB)

**Total:** ~7,500 LOC + ~9.5 MB checkpoints

**Effort:** High (2-3 days)

---

## 6. Extraction Checklist

### Phase 1: Preparation
- [ ] Create target repository structure
- [ ] Set up CI/CD for new repo
- [ ] Create comprehensive requirements.txt

### Phase 2: Shared Infrastructure
- [ ] Copy `deliverables/shared/` to new repo
- [ ] Update internal imports to relative
- [ ] Test shared module imports
- [ ] Verify vae_service.py placeholder/stub

### Phase 3: Partner Packages
- [ ] Copy jose_colbes with local core/
- [ ] Copy alejandra_rojas with local src/
- [ ] Copy carlos_brizuela with local src/ and checkpoints
- [ ] Copy hiv_research_package

### Phase 4: ML Components (if Scenario B/C)
- [ ] Create `lib/` directory structure
- [ ] Copy encoder files
- [ ] Copy geometry files
- [ ] Copy biology files
- [ ] Copy loss files (if full extraction)
- [ ] Copy model files (if full extraction)

### Phase 5: Checkpoint Migration
- [ ] Copy required checkpoints
- [ ] Update checkpoint path references
- [ ] Test model loading

### Phase 6: Import Updates
- [ ] Update all `from src.*` to `from lib.*`
- [ ] Update all `from deliverables.*` to relative
- [ ] Run import validation script

### Phase 7: Testing
- [ ] Run all partner package tests
- [ ] Verify ML inference (if extracted)
- [ ] Verify training (if full extraction)

### Phase 8: Documentation
- [ ] Update README files
- [ ] Document external dependencies
- [ ] Create installation guide

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Circular imports after restructure | HIGH | Careful dependency ordering |
| Checkpoint version mismatch | MEDIUM | Include checkpoint metadata |
| geoopt API changes | LOW | Pin version in requirements |
| Training reproducibility | MEDIUM | Document exact versions |
| Missing transitive dependencies | MEDIUM | Comprehensive testing |

---

## 8. Recommended Approach

**Short-term:** Keep deliverables in main repo. Current partial self-containment (70-90%) is sufficient for most use cases.

**Medium-term (if extraction needed):**
1. Start with Scenario B (partial extraction)
2. Focus on inference capabilities only
3. Keep training in main repo
4. Create `pip install ternary-vae-lite` package

**Long-term:**
1. Publish core components as separate pip packages:
   - `ternary-vae-geometry` (poincare, hyperbolic)
   - `ternary-vae-encoders` (codon, peptide)
   - `ternary-vae-bio` (biology utilities)
2. Deliverables `pip install` these packages
3. Zero code duplication

---

## Update Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-23 | 1.0 | Initial comprehensive audit |

---

*Audit performed: 2026-01-23*
*Auditor: Claude Code*
