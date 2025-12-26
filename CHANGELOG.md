# Changelog

All notable changes to the Ternary VAE Bioinformatics project.

---

## [Unreleased]

### Reorganized (2025-12-26)
- **HIV Documentation** - Moved from `research/.../hiv/documentation/` to `DOCUMENTATION/.../HIV_PADIC_ANALYSIS/`
  - 27 documents organized into 6 sections (Overview, Findings, Methodology, Reference, Implementation, Supplementary)
  - Added README files for each section
- **Scripts Organization** - Created subdirectories for loose scripts
  - `scripts/hiv/` - HIV analysis pipelines
  - `scripts/clinical/` - Clinical decision support tools
  - `scripts/literature/` - Literature implementation scripts
  - `scripts/maintenance/` - Codebase maintenance utilities
- **Results Organization** - Reorganized into categories
  - `results/clinical/` - Clinical outputs
  - `results/research/` - Research outputs
  - `results/training/` - Training outputs
  - `results/validation/` - Validation outputs
  - `results/literature/` - Literature implementations
- **Data Consolidation**
  - Consolidated `src/training/data.py` into `src/data/stratified.py` (canonical location)
  - Moved `research/datasets/` to `data/research/`
  - Added `data/README.md` documentation
- **Module Clarification**
  - Added documentation distinguishing `src/optimizers/` (algorithms) from `src/optimization/` (workflows)
  - Clarified `src/research/` as path utilities bridge
- **Documentation Consolidation**
  - Moved `LITERATURE_REVIEW_1000_PAPERS.md` to `DOCUMENTATION/.../03_REFERENCE/`
  - Moved `IMPLEMENTATION_SUMMARY.md` to `DOCUMENTATION/02_PROJECT_MANAGEMENT/`

### Added (2025-12-26)
- **Clinical Decision Support System** (`scripts/clinical_applications.py`)
  - Top vaccine candidate identification (TPQDLNTML, priority: 0.970)
  - MDR risk screening (2,489 high-risk sequences, 34.8%)
  - Host-directed therapy targets (247 Tat-interacting proteins)
- **Research Discoveries Pipeline** (`scripts/research_discoveries.py`)
  - 387 vaccine targets ranked by evolutionary stability
  - P-adic geometry validation (Spearman r = 0.8339 vs Hamming)
  - 1,032 MDR-enriched mutations identified
  - 19 HIV proteins targeting druggable human proteins
- **Centralized Biology Module** (`src/biology/`)
  - Single Source of Truth for genetic code, amino acid properties
  - `amino_acids.py`: AMINO_ACID_PROPERTIES, hydrophobicity, charge
  - `codons.py`: GENETIC_CODE, CODON_TO_INDEX, conversion utilities
- **Immunology Analysis Module** (`src/analysis/immunology/`)
  - Epitope encoding and sequence analysis
  - HLA genetic risk computation
  - P-adic valuation and Goldilocks zone detection
  - Shared dataclasses: EpitopeAnalysisResult, HLAAlleleRisk
- **Training Infrastructure**
  - `src/training/data.py`: TernaryDataset, StratifiedBatchSampler
  - HyperbolicVAETrainer with Riemannian gradient updates
  - Environment validation and pre-training checks
- **Comprehensive Dataset Analysis** (`scripts/analyze_all_datasets.py`)
- **HIV Codon VAE Training** (`scripts/train_codon_vae_hiv.py`)
- **Test Coverage**
  - `tests/unit/training/test_hyperbolic_trainer.py`
  - `tests/unit/training/test_monitor.py`
  - `tests/unit/visualization/test_visualization_core.py`

### Added (Previous)
- Hierarchical PTM mapping with 14-level ultrametric tree validation
- Rigorous disruption prediction framework
- Triple PTM combinatorics analysis
- RA research documentation with dual-trigger mechanism

---

## [5.11] - 2025-12-14

### Added
- Unified hyperbolic geometry architecture
- Frozen v5.5 encoder as coverage base
- Three-Body system with position-dependent control
- Unified PAdicGeodesicLoss

### Changed
- StateNet gradient flow now fully differentiable
- Replaced competing losses with unified approach

---

## [5.10] - 2025-12-12

### Added
- Hyperbolic priors (Poincare geometry)
- StateNet with curvature awareness
- Pure hyperbolic training mode
- Config validation and environment checks

### Fixed
- Monitor injection for consistent logging
- Training observability improvements

---

## [5.6] - 2025-12-10

### Added
- TensorBoard integration (local, IP-safe)
- TorchInductor compilation (1.4-2x speedup)
- Weight histograms logging

### Changed
- Renamed v5.5 files to v5.6
- Version bump to 5.6.0

---

## [5.5-srp] - 2025-11-24

### Added
- SRP refactoring complete
- Modular architecture: `src/training/`, `src/losses/`, `src/data/`, `src/artifacts/`
- Comprehensive documentation (4,200+ lines)
- Artifact lifecycle management

### Changed
- Model reduced from 632 to 499 lines (-21%)
- Trainer streamlined from 398 to 350 lines (-12%)

---

## [5.5] - 2025-10-29

### Fixed
- Categorical sampling bug (expectation vs sampling)
- Benchmark script (required checkpoint)
- Test suite (meaningful assertions)
- Coverage metrics (hash-validated 86% vs inflated 99%)

### Added
- Checkpoint certification with SHA256
- JSON benchmark output with traceability

---

## [5.5-initial] - 2025-10-24

### Added
- Dual-VAE architecture (VAE-A/VAE-B)
- StateNet meta-controller
- Phase-scheduled training
- Beta-warmup and free bits

### Achieved
- 100% holdout accuracy
- 86% hash-validated coverage
- 16/16 active latent dimensions

---

*For detailed release notes, see `DOCUMENTATION/.../ARCHIVE/release_notes_archive.md`*
