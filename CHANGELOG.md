# Changelog

All notable changes to the Ternary VAE Bioinformatics project.

---

## [5.12.4] - 2026-01-03

### Added
- **TernaryVAE v5.12.4 Training** - Improved architecture with FrozenEncoder from v5.5
  - Coverage: 100%, Hierarchy_B: -0.82, Q: 1.96
  - Checkpoint: `sandbox-training/checkpoints/v5_12_4/best_Q.pt`

- **Amino Acid Embedding Extraction** - 16-dim hyperbolic embeddings
  - Script: `extract_embeddings_simple.py`
  - All 20 amino acids from 19,683 ternary operations
  - Radius range: [0.80, 1.56] in Poincaré ball

- **Improved DDG Predictor** - Real `poincare_distance()` implementation
  - Script: `train_padic_ddg_predictor.py`
  - Trained on S669 benchmark (52 mutations)

### Results

| Metric | Heuristic (V1) | Real Embeddings (V2) | Improvement |
|--------|----------------|----------------------|-------------|
| Spearman r | 0.53 | 0.58 | +9.4% |
| Pearson r | 0.22 | 0.79 | +259% |
| MAE | 1.91 | 0.73 | -62% |

---

## [5.12.3] - 2026-01-03

### Changed
- **Dev Environment Unification** - Consolidated all tool configs to `pyproject.toml`
  - Deleted duplicate configs: `ruff.toml`, `mypy.ini`, `pytest.ini`, `.coveragerc`, `.flake8`
  - Single source of truth for ruff, mypy, pytest, coverage settings
  - Updated `requirements.txt` to Python 3.10+ with synced dependencies
  - Added `.playwright-mcp/` to `.gitignore`

### Documentation
- **Root Cleanup** - Organized markdown files
  - Moved audit reports to `docs/audits/`
  - Created `docs/audits/v5.12.2-hyperbolic/` for hyperbolic audit files
  - Root now has only `README.md` and `CHANGELOG.md`

---

## [5.12.2] - 2025-12-30

### Fixed
- **Hyperbolic Audit Complete** - All core files fixed
  - 258 `.norm()` calls reviewed, all verified correct or intentionally Euclidean
  - Fixed 39 research scripts with incorrect Euclidean norm on hyperbolic embeddings
  - Proper pattern: `poincare_distance(z_hyp, origin, c=curvature)` instead of `torch.norm(z_hyp)`

### Deprecated
- `src/core/geometry_utils.py` - Use `src.geometry` instead (geoopt-backed)

### Documentation
- Created comprehensive audit reports in `docs/audits/v5.12.2-hyperbolic/`

---

## [5.12.1] - 2025-12-29

### Added
- **Hyperbolic Audit Initiative** - Identified Euclidean/hyperbolic geometry misuse
  - AST scanner script: `scripts/audit_hyperbolic_norms.py`
  - Initial audit of 278 `.norm()` calls across codebase

### Fixed
- HIGH priority: `src/api/cli/train.py`, `src/encoders/holographic_encoder.py`, `src/losses/consequence_predictor.py`
- MEDIUM priority: `src/analysis/crispr/embedder.py`, `src/analysis/evolution.py`, `src/geometry/holographic_poincare.py`

---

## [5.12.0] - 2025-12-27

### Added - Advanced Modules (Production Ready)

Major release: All 10 modules from `src/_future/` are now production-ready with comprehensive test coverage.

#### Migrated Modules (8 modules, ~5,500 lines)
- **`src/graphs/`** - Hyperbolic Graph Neural Networks (88 tests)
  - `HyboWaveNet`: Combined wavelet + hyperbolic GNN architecture
  - `HyperbolicGraphConv`: Graph convolution in Poincare ball
  - `PoincareOperations`, `LorentzOperations`: Hyperbolic math primitives

- **`src/topology/`** - Persistent Homology & TDA (126 tests)
  - `ProteinTopologyEncoder`: Topological features from protein structure
  - `RipsFiltration`, `PAdicFiltration`: Filtration computations
  - `PersistenceVectorizer`: Convert persistence diagrams to vectors

- **`src/information/`** - Fisher Information Geometry (93 tests)
  - `NaturalGradientOptimizer`: Second-order optimization
  - `KFACOptimizer`: Kronecker-factored approximate curvature
  - `FisherInformationEstimator`: Fisher matrix computation

- **`src/contrastive/`** - P-adic Contrastive Learning (97 tests)
  - `PAdicContrastiveLoss`: P-adic distance-based contrastive loss
  - `SimCLREncoder`, `MomentumContrastEncoder`: Self-supervised encoders
  - `PAdicPositiveSampler`: Hierarchical positive sampling

- **`src/physics/`** - Statistical Physics (72 tests)
  - `SpinGlassLandscape`: Spin glass energy computation
  - `ReplicaExchange`: Parallel tempering optimizer
  - `UltrametricTreeExtractor`: Ultrametric structure from overlaps

- **`src/tropical/`** - Tropical Geometry (58 tests)
  - `TropicalSemiring`: Min-plus algebra operations
  - `TropicalNNAnalyzer`: Neural network tropical analysis
  - `TropicalPhylogeneticTree`: Phylogenetic tree construction

- **`src/categorical/`** - Category Theory (61 tests)
  - `CategoricalNetwork`: Compositional neural architecture
  - `ParametricLens`, `ResidualOptic`: Bidirectional transformations
  - `NaturalTransformation`: Functorial mappings

- **`src/meta/`** - Meta-Learning (44 tests)
  - `MAML`: Model-Agnostic Meta-Learning
  - `Reptile`: First-order meta-learning
  - `PAdicTaskSampler`: Hierarchical task sampling

#### New Implementations (2 modules, ~2,500 lines)
- **`src/equivariant/`** - SO(3)/SE(3)-Equivariant Networks (91 tests)
  - `SphericalHarmonics`: Spherical harmonic basis functions
  - `SO3Layer`, `SO3GNN`: Rotation-equivariant layers
  - `SE3Transformer`, `EGNN`: Translation+rotation equivariant
  - `CodonSymmetryLayer`: Biological symmetry (wobble, synonymy)
  - `CodonTransformer`: Full transformer respecting codon structure

- **`src/diffusion/`** - Discrete Diffusion Models (56 tests)
  - `NoiseScheduler`: Linear, cosine, sigmoid, exponential schedules
  - `DiscreteNoiseScheduler`: Absorbing state diffusion (D3PM)
  - `CodonDiffusion`: Discrete diffusion for codon sequences
  - `TransformerDenoiser`: Transformer-based denoising network
  - `StructureConditionedGen`: Inverse folding at codon level
  - `MultiObjectiveDesigner`: Multi-objective sequence optimization

### Changed
- Updated `src/__init__.py` with exports for all 10 new modules
- Updated `src/README.md` with Tier 4: ADVANCED module documentation
- Removed obsolete `src/_future/` directory

### Test Coverage
- **786 new tests** for advanced modules
- **2,462 total tests** passing
- All modules have 80%+ coverage

---

## [Unreleased]

### Research Archive Organization (2025-12-26)
- **Research Implementation Status** - New master index created
  - `RESEARCH_IMPLEMENTATION_STATUS.md` - Tracks 12 implemented, 7 validated, 8 proposed
  - Clear separation of IMPLEMENTED vs PROPOSED research
  - Cross-references between proposals and `src/` code locations
- **Research Master Index** - Consolidated navigation
  - `RESEARCH_MASTER_INDEX.md` - Single entry point for all research docs
  - 98+ paper summaries indexed by topic
  - Links to RESEARCH_LIBRARY, RESEARCH_PROPOSALS, and results
- **RESEARCH_PROPOSALS Cleanup** - Updated README with accurate status
  - Corrected implementation labels for numbered proposals (01-08)
  - Added implementation status indicators [IMPLEMENTED] / [PROPOSED]

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
