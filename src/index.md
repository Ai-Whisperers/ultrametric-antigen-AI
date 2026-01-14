# Source Code Separation Index & Checklist

**Generated:** 2026-01-14
**Total Python Files:** 630
**Purpose:** Index and checklist for creating pure mathematical ML pipeline

---

## Mission Statement

**Goal:** Extract pure mathematical ML pipeline that is completely agnostic to bioinformatics implementations. The resulting codebase should contain only mathematical foundations and general-purpose infrastructure, enabling reuse for any domain requiring advanced geometric and algebraic machine learning.

**Success Criteria:**
- ✅ Zero biological domain dependencies
- ✅ Pure mathematical foundations (p-adic theory, hyperbolic geometry, category theory)
- ✅ General ML infrastructure (training, optimization, monitoring)
- ✅ Self-contained and domain-agnostic

---

## Separation Checklist

### ❌ **REMOVE - Bioinformatics & Domain-Specific Files**

#### 🧬 **Biological Domain Directories (Remove Entirely)**
- [ ] `biology/` - Biological foundations and data structures
- [ ] `clinical/` - Clinical applications and medical interfaces
- [ ] `diseases/` - Disease modeling and analysis
- [ ] `research/bioinformatics/` - All biological research scripts and analysis

#### 🧬 **Biological Data & I/O (Remove Entirely)**
- [ ] `data/hiv/` - HIV-specific datasets
- [ ] `dataio/hiv/` - HIV data loading and processing
- [ ] `dataio/multi_organism/` - Multi-species data handling
- [ ] `data/autoimmunity.py` - Autoimmune data structures

#### 🧬 **Biological Analysis Modules (Remove Entirely)**
- [ ] `analysis/hiv/` - HIV-specific analysis (7 files)
- [ ] `analysis/immunology/` - Immunological system modeling (4 files)
- [ ] `analysis/crispr/` - CRISPR gene editing analysis (6 files)
- [ ] `analysis/ancestry/` - Population genetics (2 files)

#### 🧬 **Biological Analysis Files (Remove Individual Files)**
- [ ] `analysis/codon_optimization.py` - Codon usage optimization
- [ ] `analysis/crispr_offtarget.py` - CRISPR off-target analysis
- [ ] `analysis/evolution.py` - Evolutionary analysis
- [ ] `analysis/extraterrestrial_aminoacids.py` - Exotic amino acid analysis
- [ ] `analysis/extremophile_codons.py` - Extremophile genetic analysis
- [ ] `analysis/immune_validation.py` - Immune system validation
- [ ] `analysis/mrna_stability.py` - mRNA stability analysis
- [ ] `analysis/primer_stability_scanner.py` - PCR primer design
- [ ] `analysis/protein_landscape.py` - Protein fitness landscapes
- [ ] `analysis/resistance_analyzer.py` - Drug resistance analysis
- [ ] `analysis/rotamer_stability.py` - Protein rotamer analysis

#### 🧬 **Biological Encoders (Remove Individual Files)**
- [ ] `encoders/alphafold_encoder.py` - AlphaFold protein structure integration
- [ ] `encoders/circadian_encoder.py` - Circadian rhythm encoding
- [ ] `encoders/codon_encoder.py` - Genetic code encoding
- [ ] `encoders/geometric_vector_perceptron.py` - 3D protein structure encoding
- [ ] `encoders/hyperbolic_codon_encoder.py` - Hyperbolic codon relationships
- [ ] `encoders/motor_encoder.py` - Motor protein dynamics
- [ ] `encoders/multiscale_nucleotide_encoder.py` - Multi-scale genetic analysis
- [ ] `encoders/padic_amino_acid_encoder.py` - P-adic amino acid properties
- [ ] `encoders/peptide_encoder.py` - Peptide sequence encoding
- [ ] `encoders/ptm_encoder.py` - Post-translational modifications
- [ ] `encoders/segment_codon_encoder.py` - Segmented codon analysis
- [ ] `encoders/surface_encoder.py` - Protein surface properties
- [ ] `encoders/tam_aware_encoder.py` - T-cell activation markers
- [ ] `encoders/trainable_codon_encoder.py` - Trainable genetic encoders
- [ ] `encoders/hybrid_encoder.py` - Mixed biological-mathematical encoder

#### 🧬 **Biological Models (Remove Individual Files)**
- [ ] `models/cross_resistance_nnrti.py` - NNRTI drug resistance
- [ ] `models/cross_resistance_pi.py` - Protease inhibitor resistance
- [ ] `models/cross_resistance_vae.py` - General drug resistance VAE
- [ ] `models/epistasis_module.py` - Genetic epistatic interactions
- [ ] `models/gene_specific_vae.py` - Gene-specific modeling
- [ ] `models/pathogen_extension.py` - Pathogen-specific extensions
- [ ] `models/protein_lm_integration.py` - Protein language model integration
- [ ] `models/resistance_transformer.py` - Drug resistance transformers
- [ ] `models/structure_aware_vae.py` - Protein structure-aware VAE
- [ ] `models/subtype_specific.py` - Biological subtype modeling

#### 🧬 **Biological Model Subdirectories (Remove Entirely)**
- [ ] `models/plm/` - Protein language models (4 files)
- [ ] `models/predictors/` - Biological predictors (5 files)

#### 🧬 **Biological Loss Functions (Remove Individual Files)**
- [ ] `losses/autoimmunity.py` - Autoimmune response modeling
- [ ] `losses/codon_usage.py` - Codon optimization constraints
- [ ] `losses/coevolution_loss.py` - Evolutionary coevolution
- [ ] `losses/drug_interaction.py` - Drug interaction modeling
- [ ] `losses/epistasis_loss.py` - Genetic epistatic interactions
- [ ] `losses/glycan_loss.py` - Glycan structure constraints
- [ ] `losses/peptide_losses.py` - Peptide-specific objectives
- [ ] `losses/consequence_predictor.py` - Biological consequence prediction

#### 🧬 **Biological APIs & Interfaces (Remove Individual Files)**
- [ ] `api/drug_resistance_api.py` - Drug resistance API endpoints

#### 🧬 **Biological Utilities (Remove Individual Files)**
- [ ] `utils/padic_shift.py` - P-adic operations for biological sequences

#### 🧬 **Biological Visualization (Remove Individual Files)**
- [ ] `visualization/generate_hiv_papers.py` - HIV research visualization
- [ ] `visualization/generate_paper_charts.py` - Biological paper charts
- [ ] `visualization/generate_paper_diagrams.py` - Biological diagrams
- [ ] `visualization/generate_paper_flowcharts.py` - Biological flowcharts
- [ ] `visualization/hiv/` - HIV-specific visualizations (4 files)

#### 🧬 **Experimental Biological Components (Remove Individual Files)**
- [ ] `_experimental/contrastive/codon_sampler.py` - Codon contrastive sampling
- [ ] `_experimental/diffusion/codon_diffusion.py` - Codon diffusion models
- [ ] `_experimental/diffusion/structure_gen.py` - Biological structure generation
- [ ] `_experimental/equivariant/codon_symmetry.py` - Genetic code symmetries
- [ ] `_experimental/linguistics/peptide_grammar.py` - Peptide linguistic analysis
- [ ] `_experimental/quantum/biology.py` - Quantum effects in biology

#### 🧬 **Training Experiments (Remove Biological Files)**
- [ ] `training/experiments/disease_experiment.py` - Disease-specific experiments

#### 🧬 **Evaluation Modules (Remove Biological Files)**
- [ ] `evaluation/protein_metrics.py` - Protein-specific evaluation metrics

---

### ✅ **KEEP - Pure Mathematical Foundations**

#### 🧮 **Core Mathematics (Keep All)**
- [x] `core/ternary.py` - Ternary algebra and 3-adic operations
- [x] `core/padic_math.py` - P-adic number theory and ultrametric spaces
- [x] `core/geometry_utils.py` - Hyperbolic geometry operations (deprecated but mathematical)
- [x] `core/tensor_utils.py` - Mathematical tensor operations
- [x] `core/types.py` - Mathematical type definitions
- [x] `core/interfaces.py` - Mathematical operation interfaces
- [x] `core/metrics.py` - Mathematical metric computations

#### 🧮 **Geometric Foundations (Keep All)**
- [x] `geometry/poincare.py` - Poincaré ball hyperbolic geometry
- [x] `geometry/holographic_poincare.py` - Holographic principle in hyperbolic space

#### 🧮 **Mathematical Loss Functions (Keep All)**
- [x] `losses/fisher_rao.py` - Fisher-Rao information geometry
- [x] `losses/geometric_loss.py` - General geometric loss functions
- [x] `losses/hyperbolic_prior.py` - Hyperbolic space priors
- [x] `losses/hyperbolic_recon.py` - Hyperbolic reconstruction losses
- [x] `losses/hyperbolic_triplet_loss.py` - Hyperbolic metric learning
- [x] `losses/set_theory_loss.py` - Set-theoretic loss functions
- [x] `losses/padic_geodesic.py` - P-adic geodesic losses
- [x] `losses/radial_stratification.py` - Radial hierarchy enforcement
- [x] `losses/rich_hierarchy.py` - Hierarchical structure losses
- [x] `losses/adaptive_rich_hierarchy.py` - Adaptive hierarchy optimization
- [x] `losses/manifold_organization.py` - Manifold structure losses
- [x] `losses/zero_structure.py` - Zero-set structure enforcement

#### 🧮 **P-adic Mathematical Framework (Keep All)**
- [x] `losses/padic/norm_loss.py` - P-adic norm-based losses
- [x] `losses/padic/ranking_loss.py` - P-adic ranking with ultrametric
- [x] `losses/padic/ranking_v2.py` - Enhanced p-adic ranking
- [x] `losses/padic/ranking_hyperbolic.py` - Hyperbolic p-adic ranking
- [x] `losses/padic/metric_loss.py` - P-adic metric learning
- [x] `losses/padic/triplet_mining.py` - P-adic triplet mining

#### 🧮 **Mathematical Models (Keep All)**
- [x] `models/attention_encoder.py` - Mathematical attention mechanisms
- [x] `models/base_vae.py` - Core VAE mathematical framework
- [x] `models/curriculum.py` - Differentiable curriculum learning
- [x] `models/differentiable_controller.py` - Neural control systems
- [x] `models/homeostasis.py` - Mathematical homeostatic control
- [x] `models/improved_components.py` - Enhanced VAE components
- [x] `models/frozen_components.py` - Mathematical model freezing
- [x] `models/ternary_vae.py` - Ternary VAE implementation
- [x] `models/ternary_vae_optionc.py` - Enhanced ternary VAE
- [x] `models/simple_vae.py` - Basic VAE implementation
- [x] `models/optimal_vae.py` - Optimized VAE architecture

#### 🧮 **Mathematical Encoders (Keep All)**
- [x] `encoders/diffusion_encoder.py` - Mathematical diffusion map encoding
- [x] `encoders/generalized_padic_encoder.py` - Prime-agnostic p-adic encoding
- [x] `encoders/holographic_encoder.py` - Graph spectral + hyperbolic encoding

#### 🧮 **Experimental Mathematics (Keep All)**
- [x] `_experimental/categorical/category_theory.py` - Category theory for neural networks
- [x] `_experimental/category/functors.py` - Mathematical functors
- [x] `_experimental/category/sheaves.py` - Sheaf theory applications
- [x] `_experimental/diffusion/noise_schedule.py` - Mathematical noise scheduling
- [x] `_experimental/equivariant/se3_layer.py` - SE(3) equivariant layers
- [x] `_experimental/equivariant/so3_layer.py` - SO(3) equivariant operations
- [x] `_experimental/equivariant/spherical_harmonics.py` - Spherical harmonics
- [x] `_experimental/graphs/hyperbolic_gnn.py` - Hyperbolic graph neural networks
- [x] `_experimental/information/fisher_geometry.py` - Fisher information geometry
- [x] `_experimental/meta/meta_learning.py` - Mathematical meta-learning
- [x] `_experimental/physics/statistical_physics.py` - Statistical mechanics
- [x] `_experimental/topology/persistent_homology.py` - Topological data analysis
- [x] `_experimental/tropical/tropical_geometry.py` - Tropical geometry operations
- [x] `_experimental/quantum/descriptors.py` - Quantum mathematical descriptors

#### 🧮 **Mathematical Analysis (Keep All)**
- [x] `analysis/geometry.py` - Geometric analysis functions
- [x] `analysis/set_theory/formal_concepts.py` - Formal concept analysis
- [x] `analysis/set_theory/lattice.py` - Lattice theory operations
- [x] `analysis/set_theory/rough_sets.py` - Rough set theory
- [x] `analysis/base.py` - Base analysis framework
- [x] `analysis/classifiers.py` - General classification framework
- [x] `analysis/explainability.py` - Model explainability framework
- [x] `analysis/interpretability.py` - Model interpretability framework

#### 🧮 **Mathematical Training (Keep All)**
- [x] `training/hyperbolic_trainer.py` - Hyperbolic manifold training
- [x] `training/curriculum_trainer.py` - Mathematical curriculum learning
- [x] `training/adaptive_lr_scheduler.py` - Mathematical learning rate scheduling
- [x] `training/optimization/natural_gradient/fisher_optimizer.py` - Natural gradients
- [x] `training/grokking_detector.py` - Mathematical grokking detection

#### 🧮 **Mathematical Optimization (Keep All)**
- [x] `optimization/natural_gradient/` - Natural gradient methods

#### 🧮 **Mathematical Utilities (Keep All)**
- [x] `utils/ternary_lut.py` - Ternary operation lookup tables
- [x] `utils/metrics.py` - Mathematical metric computations
- [x] `utils/reproducibility.py` - Mathematical experiment reproducibility

#### 🧮 **Mathematical Visualization (Keep All)**
- [x] `visualization/plots/manifold.py` - Manifold structure visualization
- [x] `visualization/projections/poincare.py` - Poincaré ball projections

---

### 🔧 **KEEP - Infrastructure & Framework Components**

#### ⚙️ **Core Infrastructure (Keep All)**
- [x] `core/config_base.py` - Configuration base classes
- [x] `config/` - All configuration management (5 files)
- [x] `api/cli/` - Command-line interfaces (3 files)
- [x] `cli.py` - Main CLI entry point

#### ⚙️ **Training Infrastructure (Keep All)**
- [x] `training/base.py` - Base training framework
- [x] `training/callbacks/` - Training callback system (4 files)
- [x] `training/checkpoint_manager.py` - Checkpoint management
- [x] `training/config_schema.py` - Training configuration schemas
- [x] `training/curriculum.py` - Curriculum learning infrastructure
- [x] `training/data.py` - Data handling infrastructure
- [x] `training/environment.py` - Training environment setup
- [x] `training/experiments/base_experiment.py` - Experiment base framework
- [x] `training/gradient_checkpointing.py` - Memory optimization
- [x] `training/monitor.py` - Training monitoring
- [x] `training/monitoring/` - Monitoring infrastructure (4 files)
- [x] `training/optimizations.py` - Training optimizations
- [x] `training/optimizers/` - Custom optimizers (1 file)
- [x] `training/schedulers.py` - Learning rate schedulers
- [x] `training/trainer.py` - Main trainer implementation
- [x] `training/transfer_pipeline.py` - Transfer learning infrastructure

#### ⚙️ **Data Infrastructure (Keep All)**
- [x] `data/dataset.py` - Dataset base classes
- [x] `data/generation.py` - Data generation utilities
- [x] `data/gpu_resident.py` - GPU-resident datasets
- [x] `data/loaders.py` - Data loading infrastructure
- [x] `data/set_augmentation.py` - Data augmentation
- [x] `data/stratified.py` - Stratified sampling
- [x] `dataio/` - Data I/O infrastructure (equivalent files)

#### ⚙️ **Utilities Infrastructure (Keep All)**
- [x] `utils/checkpoint.py` - Checkpoint utilities
- [x] `utils/checkpoint_hub.py` - Checkpoint hub management
- [x] `utils/checkpoint_validator.py` - Checkpoint validation
- [x] `utils/nn_factory.py` - Neural network factory patterns
- [x] `utils/observability/` - Observability framework (5 files)

#### ⚙️ **Factory & Registry (Keep All)**
- [x] `factories/loss_factory.py` - Loss function factory
- [x] `factories/model_factory.py` - Model factory
- [x] `losses/registry.py` - Loss function registry
- [x] `losses/base.py` - Loss function base classes
- [x] `losses/components.py` - Loss function components
- [x] `losses/dual_vae_loss.py` - Dual VAE loss infrastructure

#### ⚙️ **Visualization Framework (Keep All)**
- [x] `visualization/config.py` - Visualization configuration
- [x] `visualization/core/` - Core visualization framework (4 files)
- [x] `visualization/plots/training.py` - Training plot utilities
- [x] `visualization/styles/` - Visualization styling (2 files)
- [x] `visualization/generate_missed.py` - Visualization utilities

#### ⚙️ **Evaluation Framework (Keep All)**
- [x] `evaluation/external_validator.py` - External validation framework
- [x] `evaluation/manifold_organization.py` - Manifold evaluation
- [x] `evaluation/temporal_split.py` - Temporal evaluation splits

#### ⚙️ **Model Infrastructure (Keep Framework Components)**
- [x] `models/ensemble.py` - Ensemble modeling framework
- [x] `models/multi_task_vae.py` - Multi-task learning framework
- [x] `models/uncertainty/` - Uncertainty quantification framework (7 files)
- [x] `models/contrastive/` - Contrastive learning framework (3 files)
- [x] `models/diffusion/` - Diffusion model framework (3 files)
- [x] `models/equivariant/` - Equivariant layer framework (2 files)
- [x] `models/fusion/` - Model fusion framework (2 files)
- [x] `models/holographic/` - Holographic model framework (2 files)
- [x] `models/tropical/` - Tropical geometry models (2 files)

#### ⚙️ **Placeholder Modules (Keep All)**
- [x] `categorical/` - Category theory placeholders
- [x] `contrastive/` - Contrastive learning placeholders
- [x] `diffusion/` - Diffusion model placeholders
- [x] `equivariant/` - Equivariance placeholders
- [x] `experimental/` - Experimental framework placeholders
- [x] `experiments/` - Experiments framework placeholders
- [x] `graphs/` - Graph neural network placeholders
- [x] `information/` - Information theory placeholders
- [x] `meta/` - Meta-learning placeholders
- [x] `objectives/` - Objectives framework placeholders
- [x] `physics/` - Physics-inspired placeholders
- [x] `quantum/` - Quantum computing placeholders (keep `descriptors.py`)
- [x] `topology/` - Topological method placeholders
- [x] `tropical/` - Tropical geometry placeholders

---

## Implementation Checklist

### Phase 1: Remove Bioinformatics Components
- [ ] Remove biological directories entirely (4 directories)
- [ ] Remove biological files from mixed directories (89+ files)
- [ ] Remove biological research scripts (120+ files)
- [ ] Update imports and dependencies

### Phase 2: Verify Mathematical Purity
- [ ] Ensure no biological terminology in remaining code
- [ ] Verify mathematical abstractions are domain-agnostic
- [ ] Test that all mathematical operations work independently
- [ ] Validate geometric operations on synthetic data

### Phase 3: Clean Infrastructure
- [ ] Ensure training infrastructure works with mathematical-only components
- [ ] Verify data loading works with generic tensors
- [ ] Test visualization with mathematical manifolds only
- [ ] Confirm API endpoints are domain-agnostic

### Phase 4: Documentation & Testing
- [ ] Update all docstrings to remove biological references
- [ ] Create mathematical-only examples and tutorials
- [ ] Add unit tests for mathematical operations
- [ ] Document pure ML pipeline usage

---

## Expected Results

**Before:** 630 Python files (mixed mathematical + bioinformatics + infrastructure)
**After:** ~290 Python files (pure mathematical + infrastructure only)
**Reduction:** ~54% fewer files

**Resulting Components:**
- Pure mathematical foundations (p-adic theory, hyperbolic geometry, category theory)
- Domain-agnostic ML infrastructure (training, optimization, monitoring)
- General-purpose utilities and visualization
- Zero biological coupling or terminology

**Use Cases for Pure Pipeline:**
- Financial modeling with geometric deep learning
- Physics simulations with hyperbolic embeddings
- Any domain requiring p-adic or tropical geometry
- Research in mathematical machine learning
- Category-theoretic neural network architectures

---

## References

- **Original Audit:** `audit-src.md` - Complete file-by-file analysis
- **Training-Opt Branch:** Contains already-cleaned mathematical pipeline
- **Complete File Index:** `complete-file-index.md` - All 639 files categorized

**Note:** This index serves as both a comprehensive catalog and an actionable checklist for creating a pure mathematical ML pipeline completely free of biological domain coupling.