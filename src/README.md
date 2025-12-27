# Ternary VAE Source Architecture

> **See also**: [Full Documentation Hub](../DOCUMENTATION/) | [Architecture Specs](../DOCUMENTATION/03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/05_SPECS_AND_GUIDES/ARCHITECTURE.md) | [Quick Start](../DOCUMENTATION/QUICK_START.md)

**Doc-Type:** Architecture Reference · Version 1.1 · Updated 2025-12-27

---

## Module Tiers

### Tier 1: CORE (Production Ready)
Fully implemented, tested, and used in production:

| Module | Files | Description |
|--------|-------|-------------|
| `core/` | 4 | Ternary arithmetic, p-adic math, interfaces |
| `losses/` | 23 | Loss functions and registry system |
| `training/` | 25 | Training loops, callbacks, monitoring |
| `models/` | 18 | VAE architectures (TernaryVAEV5_11 canonical) |
| `data/` | 13 | Datasets, loaders, generation |
| `config/` | 6 | Configuration and paths management |
| `metrics/` | 2 | Hyperbolic metrics computation |
| `utils/` | 6 | Checkpointing, reproducibility, helpers |

### Tier 2: EXTENDED (Working)
Implemented modules for extended functionality:

| Module | Files | Description |
|--------|-------|-------------|
| `visualization/` | 23 | Plotting, projections, paper figures |
| `analysis/` | 19 | CRISPR analysis, geometry analysis |
| `encoders/` | 9 | PTM encoder, diffusion encoder |
| `observability/` | 6 | Logging, metrics buffering, coverage |
| `objectives/` | 5 | Binding, solubility, manufacturability |
| `biology/` | 3 | Biological sequence utilities |
| `diseases/` | 5 | Disease-specific analysis (e.g., RA) |

### Tier 3: EXPERIMENTAL (Partial)
Modules with partial implementation:

| Module | Files | Description |
|--------|-------|-------------|
| `geometry/` | 3 | Poincare ball (geoopt backend) |
| `quantum/` | 3 | Quantum descriptors |
| `stability/` | 2 | Stability analysis |
| `evolution/` | 2 | Evolutionary analysis |
| `classifiers/` | 2 | Classification models |
| `cli/` | 4 | Command-line interface |

### Tier 4: FUTURE (Aspirational - Moved to `_future/`)
These modules have been moved to `src/_future/` to clarify they are **not implemented**. They contain only placeholder `__init__.py` with documentation describing *intended* future functionality.

See [`_future/README.md`](./_future/README.md) for implementation priorities and contributing guidelines.

| Module | Intended Purpose | Priority |
|--------|------------------|----------|
| `_future/topology/` | Persistent homology, TDA | Medium |
| `_future/categorical/` | Category theory abstractions | Low |
| `_future/tropical/` | Tropical semiring operations | Low |
| `_future/equivariant/` | SE(3) equivariant networks | **High** |
| `_future/information/` | Information-theoretic measures | Medium |
| `_future/graphs/` | Protein graph neural networks | **High** |
| `_future/meta/` | Meta-learning algorithms | Low |
| `_future/contrastive/` | Contrastive learning losses | Medium |
| `_future/diffusion/` | Diffusion models | Medium |
| `_future/physics/` | Statistical physics (spin glass) | Low |

---

## Wiring Overview

This document describes the wiring architecture of the Ternary VAE codebase, identifying **joints** (connection points that wire components together) and **non-joints** (leaf components with single responsibilities).

---

## Joints (Connection Points)

Components that wire other components together.

| Module | File | Type | Connects To |
|--------|------|------|-------------|
| `src/__init__.py` | Line 1-52 | Top-Level Joint | All submodules: data, models, losses, training, metrics, artifacts, utils |
| `training/__init__.py` | Line 1-30 | Module Joint | trainer, hyperbolic_trainer, schedulers, monitor, config_schema, environment |
| `training/trainer.py` | `TernaryVAETrainer` | Orchestrator Joint | schedulers, monitor, CheckpointManager, DualVAELoss, model |
| `training/hyperbolic_trainer.py` | `HyperbolicVAETrainer` | Orchestrator Joint | base_trainer, hyperbolic losses, metrics, monitor |
| `losses/__init__.py` | Line 1-26 | Module Joint | dual_vae_loss, hyperbolic_prior, hyperbolic_recon, padic_losses, appetitive_losses, consequence_predictor |
| `losses/dual_vae_loss.py` | `DualVAELoss` | Aggregator Joint | Aggregates: CE, KL, entropy, repulsion, p-adic losses |
| `models/__init__.py` | Line 1-25 | Module Joint | ternary_vae_v5_10, ternary_vae_v5_6, ternary_vae_v5_7, appetitive_vae |
| `data/__init__.py` | Line 1-28 | Module Joint | generation, dataset, loaders |
| `data/loaders.py` | `create_data_loaders()` | Factory Joint | TernaryOperationDataset, torch DataLoader |

---

## Non-Joints (Leaf Components)

Components with single responsibilities.

### Data Layer

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `data/generation.py` | `TernaryOperationGenerator` | Generate ternary operation lookup tables |
| `data/dataset.py` | `TernaryOperationDataset` | PyTorch dataset wrapper |

### Model Layer

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `models/ternary_vae_v5_10.py` | `DualNeuralVAEV5_10` | v5.10 Pure Hyperbolic VAE architecture |
| `models/ternary_vae_v5_6.py` | `DualNeuralVAEV5_6` | v5.6 VAE architecture |
| `models/ternary_vae_v5_7.py` | `DualNeuralVAEV5_7` | v5.7 VAE architecture |
| `models/appetitive_vae.py` | `AppetitiveDualVAE` | Wrapper adding appetitive drive |

### Loss Layer

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `losses/hyperbolic_prior.py` | `HomeostaticHyperbolicPrior` | Wrapped normal prior on Poincare ball |
| `losses/hyperbolic_recon.py` | `HomeostaticReconLoss` | Radius-weighted reconstruction loss |
| `losses/hyperbolic_recon.py` | `HyperbolicCentroidLoss` | Frechet mean centroid clustering |
| `losses/padic_losses.py` | `PAdicRankingLossHyperbolic` | Hyperbolic 3-adic triplet ranking loss |
| `losses/appetitive_losses.py` | `AppetitiveLoss` | Coverage-based appetitive drive loss |
| `losses/consequence_predictor.py` | `ConsequencePredictor` | Predict addition accuracy from metrics |
| `losses/consequence_predictor.py` | `PurposefulRankingLoss` | Ranking with consequence awareness |

### Training Support

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `training/schedulers.py` | `TemperatureScheduler` | Temperature scheduling (linear/cyclic) |
| `training/schedulers.py` | `BetaScheduler` | KL weight scheduling with warmup |
| `training/schedulers.py` | `LearningRateScheduler` | Learning rate scheduling |
| `training/monitor.py` | `TrainingMonitor` | Logging and TensorBoard observability |
| `training/config_schema.py` | `TrainingConfig` | Configuration validation |
| `training/environment.py` | `validate_environment()` | Pre-training environment validation |

### Metrics

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `metrics/hyperbolic.py` | `compute_ranking_correlation_hyperbolic()` | 3-adic ranking correlation eval |
| `metrics/hyperbolic.py` | `project_to_poincare()` | Project to Poincare ball |
| `metrics/hyperbolic.py` | `poincare_distance()` | Hyperbolic geodesic distance |
| `metrics/hyperbolic.py` | `compute_3adic_valuation()` | 3-adic valuation computation |

### Artifacts

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `artifacts/checkpoint_manager.py` | `CheckpointManager` | Save/load checkpoints |

### Utils

| File | Component | Single Responsibility |
|------|-----------|----------------------|
| `utils/reproducibility.py` | `set_seed()` | Random seed management |
| `utils/reproducibility.py` | `get_generator()` | Create seeded generator |

---

## Wiring Diagram

```
                          ENTRY POINTS (Scripts)
   scripts/train/train_ternary_v5_10.py -> Creates all components
                                |
                                v
+-------------------------------------------------------------------------+
|                    ORCHESTRATION LAYER (JOINTS)                          |
|                                                                          |
|  +---------------------+           +----------------------------+        |
|  |  TernaryVAETrainer  |<--------->|   HyperbolicVAETrainer     |        |
|  |  (trainer.py:26)    |           |   (hyperbolic_trainer:28)  |        |
|  +----------+----------+           +--------------+--------------+       |
|             |                                     |                      |
|  Wires: model, schedulers,          Wires: base_trainer, hyperbolic      |
|  monitor, checkpoint_manager,       losses, metrics, continuous          |
|  DualVAELoss                        feedback                             |
+-------------------------------------------------------------------------+
                                |
                +---------------+---------------+------------------+
                v               v               v                  v
    +-------------------+ +--------------+ +--------------+ +----------------+
    |    SCHEDULERS     | |   MONITOR    | |CHECKPOINT MGR| |  LOSS (Joint)  |
    |   (Non-Joints)    | |  (Non-Joint) | | (Non-Joint)  | | DualVAELoss    |
    +-------------------+ +--------------+ +--------------+ +-------+--------+
    | TempScheduler     |                                           |
    | BetaScheduler     |                                           |
    | LRScheduler       |                                           |
    +-------------------+                                           |
                                                                    v
+-------------------------------------------------------------------------+
|                      LOSS COMPONENTS (NON-JOINTS)                        |
|                                                                          |
|  +----------------------------------------------------------------------+|
|  | PAdicRankingLossHyperbolic | HomeostaticHyperbolicPrior              ||
|  | (padic_losses.py)          | (hyperbolic_prior.py)                   ||
|  +----------------------------+-----------------------------------------+|
|  | HomeostaticReconLoss       | HyperbolicCentroidLoss                  ||
|  | (hyperbolic_recon.py)      | (hyperbolic_recon.py)                   ||
|  +----------------------------------------------------------------------+|
+-------------------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------------------+
|                        MODEL LAYER (NON-JOINTS)                          |
|                                                                          |
|   +-------------------------------------------------------------------+  |
|   |  DualNeuralVAEV5_10  (ternary_vae_v5_10.py)                       |  |
|   |  - Encoder/Decoder networks                                       |  |
|   |  - StateNet v2 for adaptive corrections                           |  |
|   |  - Hyperbolic projection built-in                                 |  |
|   +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------------------+
|                         DATA LAYER (NON-JOINTS)                          |
|                                                                          |
|   +-------------------+    +--------------------+    +------------------+ |
|   | TernaryOperation  |--->| TernaryOperation   |<---| create_data      | |
|   | Generator         |    | Dataset            |    | _loaders()       | |
|   | (generation.py)   |    | (dataset.py)       |    | (loaders.py)     | |
|   +-------------------+    +--------------------+    +------------------+ |
+-------------------------------------------------------------------------+
```

---

## Key Observations

1. **Two Main Orchestrators** - `TernaryVAETrainer` and `HyperbolicVAETrainer` are the primary joints wiring all components together.

2. **Clean SRP** - Every non-joint has a documented single responsibility in its docstring.

3. **Composition Pattern** - `HyperbolicVAETrainer` wraps `TernaryVAETrainer` (base_trainer), extending rather than replacing.

4. **`DualVAELoss` is a Joint** - Aggregates multiple loss computations (CE, KL, entropy, repulsion, p-adic).

5. **Module `__init__.py` files are Joints** - They define public APIs and re-export components.

---

## Module Dependencies

```
src/
├── __init__.py          # Top-level joint
├── core/                # FOUNDATION (no dependencies)
│   ├── __init__.py      # Exports TERNARY singleton
│   └── ternary.py       # TernarySpace with precomputed LUTs
├── observability/       # Decoupled metrics (depends on core)
│   ├── __init__.py
│   ├── metrics_buffer.py    # Zero-I/O metrics collection
│   ├── async_writer.py      # Background TensorBoard writing
│   └── coverage.py          # Vectorized coverage evaluation
├── data/                # Data layer (depends on core)
│   ├── generation.py
│   ├── dataset.py
│   ├── loaders.py       # Depends on dataset
│   └── gpu_resident.py  # GPU-resident dataset (P2 optimization)
├── models/              # No internal dependencies
│   ├── ternary_vae_v5_10.py
│   ├── ternary_vae_v5_6.py
│   ├── ternary_vae_v5_7.py
│   └── appetitive_vae.py
├── losses/              # Internal dependencies between loss modules
│   ├── dual_vae_loss.py # Joint: aggregates others
│   ├── hyperbolic_prior.py
│   ├── hyperbolic_recon.py
│   ├── padic_losses.py
│   ├── appetitive_losses.py
│   └── consequence_predictor.py
├── training/            # Depends on losses, metrics, artifacts
│   ├── trainer.py       # Joint: wires schedulers, monitor, losses
│   ├── hyperbolic_trainer.py  # Joint: wraps trainer
│   ├── schedulers.py
│   ├── monitor.py
│   ├── config_schema.py
│   └── environment.py
├── metrics/             # No internal dependencies
│   └── hyperbolic.py
├── artifacts/           # No internal dependencies
│   └── checkpoint_manager.py
└── utils/               # No internal dependencies
    └── reproducibility.py
```
