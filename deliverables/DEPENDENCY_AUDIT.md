# Deliverables Dependency Audit

**Doc-Type:** Technical Audit · Version 1.0 · Updated 2026-01-23 · AI Whisperers

---

## Executive Summary

This audit documents all external dependencies that would need to be addressed to move the `deliverables/` folder (or individual partner packages) to an independent repository.

### Dependency Categories

| Category | Count | Severity | Notes |
|----------|-------|----------|-------|
| **Main Project ML Models** | 8 modules | HIGH | Core neural network components |
| **Main Project Utilities** | 4 modules | MEDIUM | Biology, geometry, data utilities |
| **Shared Infrastructure** | 6 modules | LOW | Can be bundled with deliverables |
| **Checkpoint Files** | 3 paths | MEDIUM | Binary model weights |
| **Research Scripts** | 11 files | LOW | Reference only, not runtime |

---

## 1. Main Project ML Model Dependencies (HIGH SEVERITY)

These are core machine learning components from `src/` that would need to be either:
- Copied into deliverables (code duplication)
- Published as a separate pip package
- Kept as external git submodule dependency

### 1.1 TernaryVAE Models

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.models.TernaryVAEV5_11_PartialFreeze` | vae_service, vae_integration_demo | `from src.models import TernaryVAEV5_11_PartialFreeze` |

**Files affected:**
- `shared/vae_service.py:97`
- `scripts/vae_integration_demo.py:118`
- `partners/jose_colbes/reproducibility/archive/extract_aa_embeddings.py:102`
- `partners/jose_colbes/reproducibility/archive/extract_embeddings_simple.py:154`

### 1.2 Codon/Peptide Encoders

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.encoders.trainable_codon_encoder.TrainableCodonEncoder` | jose_colbes, alejandra_rojas | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` |
| `src.encoders.codon_encoder` | jose_colbes reproducibility | `from src.encoders.codon_encoder import ...` |
| `src.encoders.peptide_encoder.PeptideVAE` | carlos_brizuela | `from src.encoders.peptide_encoder import PeptideVAE` |

**Files affected:**
- `partners/jose_colbes/src/validated_ddg_predictor.py:58`
- `partners/jose_colbes/validation/bootstrap_test.py:19`
- `partners/jose_colbes/reproducibility/extract_aa_embeddings_v2.py:37`
- `partners/alejandra_rojas/scripts/denv4_padic_integration.py:50`
- `partners/alejandra_rojas/scripts/denv4_synonymous_conjecture.py:48`
- `partners/alejandra_rojas/research/clade_classification/train_clade_classifier.py:49`
- `partners/carlos_brizuela/scripts/predict_mic.py:80`
- `partners/carlos_brizuela/training/train_definitive.py:41`
- `partners/carlos_brizuela/training/train_peptide_encoder.py:46`
- `partners/carlos_brizuela/training/train_improved.py:39`

### 1.3 Loss Functions

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.losses.peptide_losses.PeptideLossManager` | carlos_brizuela training | `from src.losses.peptide_losses import PeptideLossManager` |
| `src.losses.peptide_losses.CurriculumSchedule` | carlos_brizuela training | `from src.losses.peptide_losses import CurriculumSchedule` |

**Files affected:**
- `partners/carlos_brizuela/training/train_definitive.py:42`
- `partners/carlos_brizuela/training/train_peptide_encoder.py:47`
- `partners/carlos_brizuela/training/train_improved.py:40`

---

## 2. Main Project Utility Dependencies (MEDIUM SEVERITY)

These are utility modules that provide biology/geometry functions.

### 2.1 Geometry Module

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.geometry.poincare_distance` | jose_colbes, alejandra_rojas | `from src.geometry import poincare_distance` |
| `src.geometry` (full module) | alejandra_rojas geometry fallback | `from src.geometry import (exp_map_zero, log_map_zero, ...)` |

**Files affected:**
- `partners/jose_colbes/src/validated_ddg_predictor.py:59`
- `partners/jose_colbes/validation/bootstrap_test.py:20`
- `partners/alejandra_rojas/scripts/denv4_padic_integration.py:51`
- `partners/alejandra_rojas/scripts/denv4_synonymous_conjecture.py:49`
- `partners/alejandra_rojas/src/geometry.py:31`
- `partners/alejandra_rojas/validation/test_padic_conservation_correlation.py:44`

### 2.2 Biology Module

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.biology.codons` | jose_colbes, alejandra_rojas | `from src.biology.codons import CODON_TO_INDEX, GENETIC_CODE, ...` |

**Files affected:**
- `partners/jose_colbes/reproducibility/extract_aa_embeddings_v2.py:31`
- `partners/jose_colbes/reproducibility/train_padic_ddg_predictor_v2.py:43`
- `partners/jose_colbes/reproducibility/analyze_padic_ddg_full.py:43`
- `partners/alejandra_rojas/scripts/denv4_padic_integration.py:52`
- `partners/alejandra_rojas/scripts/denv4_synonymous_conjecture.py:50`
- `partners/alejandra_rojas/scripts/denv4_codon_bias_conjecture.py:57`
- `partners/alejandra_rojas/scripts/denv4_revised_conjecture.py:55`
- `partners/alejandra_rojas/scripts/denv4_codon_pair_conjecture.py:49`
- `partners/alejandra_rojas/research/clade_classification/train_clade_classifier.py:50`

### 2.3 Core Module

| Module | Used By | Import Statement |
|--------|---------|------------------|
| `src.core.TERNARY` | jose_colbes archive | `from src.core import TERNARY` |
| `src.data.generation` | jose_colbes archive | `from src.data.generation import generate_all_ternary_operations` |

**Files affected:**
- `partners/jose_colbes/reproducibility/archive/extract_aa_embeddings.py:204-205`

---

## 3. Shared Infrastructure Dependencies (LOW SEVERITY)

These modules are within `deliverables/shared/` and would move with the deliverables folder.

### 3.1 Configuration

| Module | Used By | Files Count |
|--------|---------|-------------|
| `shared.config` | Most scripts | 15+ files |
| `shared.constants` | Most scripts | 10+ files |

### 3.2 Services

| Module | Used By | Files Count |
|--------|---------|-------------|
| `shared.vae_service` | Integration tests, demos | 20+ files |
| `shared.peptide_utils` | Partner scripts, tests | 15+ files |
| `shared.hemolysis_predictor` | Tests, demos | 5+ files |
| `shared.primer_design` | alejandra_rojas tests | 3+ files |

**Note:** These are self-contained within deliverables and would move together.

---

## 4. Checkpoint File Dependencies (MEDIUM SEVERITY)

### 4.1 Main Project Checkpoints

| Path | Used By | Size |
|------|---------|------|
| `checkpoints/homeostatic_rich/best.pt` | vae_service, validated_ddg | ~2 MB |
| `checkpoints/v5_11_homeostasis/best.pt` | vae_service fallback | ~2 MB |
| `checkpoints/pretrained_final.pt` | vae_service fallback | ~2 MB |

### 4.2 Research Checkpoints

| Path | Used By | Size |
|------|---------|------|
| `research/codon-encoder/training/results/trained_codon_encoder.pt` | jose_colbes DDG predictor | ~1 MB |

### 4.3 Partner-Specific Checkpoints (Self-Contained)

| Path | Package | Notes |
|------|---------|-------|
| `partners/jose_colbes/models/` | jose_colbes | Self-contained |
| `partners/carlos_brizuela/checkpoints_definitive/` | carlos_brizuela | Self-contained |
| `partners/carlos_brizuela/models/` | carlos_brizuela | Self-contained |

---

## 5. Cross-Package Dependencies

### 5.1 deliverables.* Absolute Imports

These use absolute imports that assume the deliverables folder is within the main project:

**Files with `from deliverables.*` imports:**
- `partners/alejandra_rojas/tests/test_notebook_integration.py:248,256,268`
- `partners/jose_colbes/validation/scientific_validation_report.py:193`
- `partners/jose_colbes/validation/alphafold_validation_pipeline.py:392`
- `partners/carlos_brizuela/training/dataset.py:14`

**Recommendation:** Convert to relative imports (already partially done).

---

## 6. Partner Package Dependency Matrix

### Jose Colbes (Protein Stability)

| Dependency Type | Status | Action Required |
|-----------------|--------|-----------------|
| P-adic math | ✅ LOCAL | None - uses `core/padic_math.py` |
| AA constants | ✅ LOCAL | None - uses `core/constants.py` |
| TrainableCodonEncoder | ⚠️ EXTERNAL | Needs main project or copy |
| poincare_distance | ⚠️ EXTERNAL | Needs main project or copy |
| src.biology.codons | ⚠️ EXTERNAL | Needs main project or copy |
| Checkpoint files | ⚠️ EXTERNAL | Needs trained_codon_encoder.pt |

**Self-Containment Level:** ~70% (basic operations work, ML prediction requires main project)

### Alejandra Rojas (Arbovirus Surveillance)

| Dependency Type | Status | Action Required |
|-----------------|--------|-----------------|
| P-adic math | ✅ LOCAL | None - uses `src/padic_math.py` |
| Codons | ✅ LOCAL | None - uses `src/codons.py` |
| Constants | ✅ LOCAL | None - uses `src/constants.py` |
| TrainableCodonEncoder | ⚠️ EXTERNAL | Research scripts only |
| poincare_distance | ⚠️ EXTERNAL | Research scripts only |
| src.biology.codons | ⚠️ EXTERNAL | Research scripts only |
| src.geometry | ⚠️ EXTERNAL | Optional fallback in src/geometry.py |

**Self-Containment Level:** ~85% (primer design works, research integration requires main project)

### Carlos Brizuela (AMP Optimization)

| Dependency Type | Status | Action Required |
|-----------------|--------|-----------------|
| AA constants | ✅ LOCAL | None - uses `src/constants.py` |
| Peptide utils | ✅ LOCAL | None - uses `src/peptide_utils.py` |
| Uncertainty | ✅ LOCAL | None - uses `src/uncertainty.py` |
| PeptideVAE | ⚠️ EXTERNAL | Needed for MIC prediction |
| PeptideLossManager | ⚠️ EXTERNAL | Training scripts only |
| shared.vae_service | ⚠️ SHARED | Tests only |

**Self-Containment Level:** ~60% (NSGA-II works, VAE prediction requires main project)

### HIV Research Package

| Dependency Type | Status | Action Required |
|-----------------|--------|-----------------|
| Stanford HIVdb API | ✅ SELF-CONTAINED | External web API |
| shared.config | ⚠️ SHARED | Used by stanford_hivdb_client.py |
| shared.constants | ⚠️ SHARED | HIV drug classes, SDRM lists |

**Self-Containment Level:** ~90% (Stanford API integration, minimal internal deps)

---

## 7. Recommendations for Full Independence

### Option A: Minimal Extraction (Recommended)

Keep deliverables as part of main repo but ensure each partner package works standalone:

1. **Already Done:**
   - Local `core/` or `src/` modules for each package
   - Local `requirements.txt` files

2. **Still Needed:**
   - Copy `shared/` into each partner package OR keep as common dependency
   - Document which features require main project

### Option B: Full Separation

To move deliverables to completely independent repository:

1. **Required Copies:**
   - `src/encoders/trainable_codon_encoder.py` (~500 LOC)
   - `src/encoders/peptide_encoder.py` (~800 LOC)
   - `src/losses/peptide_losses.py` (~400 LOC)
   - `src/geometry/` module (~600 LOC)
   - `src/biology/codons.py` (~200 LOC)
   - Checkpoint files (~10 MB total)

2. **Estimated Duplication:** ~2,500 lines of code + 10MB checkpoints

3. **Maintenance Burden:** HIGH - need to sync updates

### Option C: Package Extraction

Publish core components as pip packages:

1. `ternary-vae-core` - TernaryVAE models and geometry
2. `ternary-vae-encoders` - Codon and peptide encoders
3. `ternary-vae-bio` - Biology utilities

Deliverables would then `pip install` these packages.

---

## 8. Files Requiring Attention by Package

### 8.1 Jose Colbes - Files with External Dependencies

```
src/validated_ddg_predictor.py           # TrainableCodonEncoder, poincare_distance
validation/bootstrap_test.py             # TrainableCodonEncoder, poincare_distance
reproducibility/extract_aa_embeddings_v2.py  # src.biology.codons, codon_encoder
reproducibility/train_padic_ddg_predictor_v2.py  # src.biology.codons, codon_encoder
reproducibility/analyze_padic_ddg_full.py  # src.biology.codons, codon_encoder
reproducibility/archive/extract_aa_embeddings.py  # TernaryVAE, TERNARY
reproducibility/archive/extract_embeddings_simple.py  # TernaryVAE
```

### 8.2 Alejandra Rojas - Files with External Dependencies

```
scripts/denv4_padic_integration.py       # TrainableCodonEncoder, poincare_distance, codons
scripts/denv4_synonymous_conjecture.py   # TrainableCodonEncoder, poincare_distance, codons
scripts/denv4_codon_bias_conjecture.py   # src.biology.codons
scripts/denv4_revised_conjecture.py      # src.biology.codons
scripts/denv4_codon_pair_conjecture.py   # src.biology.codons
src/geometry.py                          # src.geometry (optional)
research/clade_classification/train_clade_classifier.py  # TrainableCodonEncoder
research/functional_convergence/find_convergence_points.py  # TrainableCodonEncoder
validation/test_padic_conservation_correlation.py  # TrainableCodonEncoder, poincare_distance
tests/test_notebook_integration.py       # shared.primer_design
```

### 8.3 Carlos Brizuela - Files with External Dependencies

```
scripts/predict_mic.py                   # PeptideVAE
training/train_definitive.py             # PeptideVAE, PeptideLossManager
training/train_peptide_encoder.py        # PeptideVAE, PeptideLossManager
training/train_improved.py               # PeptideVAE, PeptideLossManager
verify_paths.py                          # PeptideVAE, PeptideLossManager
tests/integration_test.py                # shared.vae_service, shared.peptide_utils
```

### 8.4 HIV Package - Files with External Dependencies

```
scripts/stanford_hivdb_client.py         # shared.config, shared.constants
```

---

## 9. Verification Commands

### Check for remaining external imports:

```bash
# Find all src.* imports
grep -r "from src\." deliverables/ --include="*.py" | grep -v ".pyc"

# Find all shared.* imports
grep -r "from shared\." deliverables/ --include="*.py" | grep -v ".pyc"

# Find absolute deliverables.* imports
grep -r "from deliverables\." deliverables/ --include="*.py" | grep -v ".pyc"
```

### Test package independence:

```bash
# Test each package in isolation
cd deliverables/partners/jose_colbes && python -c "from core import padic_valuation; print('OK')"
cd deliverables/partners/alejandra_rojas && python -c "from src import padic_valuation; print('OK')"
cd deliverables/partners/carlos_brizuela && python -c "from src import AMINO_ACIDS; print('OK')"
```

---

## 10. Conclusion

The deliverables folder has **partial independence**:

| Component | Independence Level | Notes |
|-----------|-------------------|-------|
| Basic operations | ✅ 100% | Math, constants, utilities |
| Partner scripts | ⚠️ 70-85% | Most work standalone |
| ML prediction | ❌ 0% | Requires main project encoders |
| Training scripts | ❌ 0% | Requires main project losses |
| Research scripts | ❌ 0% | Deep integration with main project |

**Recommendation:** Keep deliverables as part of main repository. Document which features work standalone vs which require the full project. The current self-containment work ensures basic functionality is portable.

---

*Audit performed: 2026-01-23*
*Auditor: Claude Code*
