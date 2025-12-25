# Research Implementation Roadmap

> **Date:** 2024-12-25
> **Objective:** Integrate the 12 Research Proposals into the existing codebase, addressing the "Documentation-Implementation Gap".

## 1. Strategy & Phasing

The implementation will be phased to build upon the core `Ternary VAE` stability. We will prioritize proposals that enhance the _current_ model's geometry and loss landscape (Phase 1) before adding complex auxiliary encoders or data modalities (Phase 2 & 3).

## 2. Phase 1: Geometric & Structural Foundations (Immediate)

**Goal:** Solidity the loss landscape and biological constraints.

### 1.1 Geometric Vaccine Design (Proposal 1)

- **Status:** Missing.
- **Action:** Create `src/losses/geometric_loss.py`.
- **Implementation:** Implement `GeometricAlignmentLoss` using `src/geometry/poincare.py` to calculate RMSD between latent representations and target nanoparticle symmetries (ferritin/mi3).
- **Integration:** Add to `CombinedGeodesicLoss` in `src/losses/padic_geodesic.py` or as a new loss term in `train.py`.

### 1.2 Codon-Space Exploration (Proposal 3)

- **Status:** Missing.
- **Action:** Create `src/encoders/codon_encoder.py`.
- **Implementation:** Implement a trainable embedding layer that maps codon bias (p-adic distance) to the VAE's latent space.
- **Dependency:** Requires `src/encoders/` directory creation.

## 3. Phase 2: Multi-Task & Interaction Modeling (Short-Term)

**Goal:** incorporating external biological/chemical constraints.

### 2.1 Drug-Interaction Modeling (Proposal 2)

- **Status:** Missing.
- **Action:** Create `src/losses/drug_interaction.py`.
- **Implementation:** Implement `DrugInteractionPenalty` as a contrastive loss term that pushes interacting drugs/targets apart (or together) in the hyperbolic space.

### 2.2 Autoimmunity & Codon Adaptation (Proposal 6)

- **Status:** Data-Dependent.
- **Action:** Create `src/data/autoimmunity.py`.
- **Implementation:** A data loader/processor that weights the training samples based on autoimmunity profiles (regularization term).

### 2.3 Multi-Objective Evolutionary Optimization (Proposal 7)

- **Status:** Missing.
- **Action:** Create `src/optimizers/multi_objective.py`.
- **Implementation:** Implement NSGA-II or similar Pareto-optimization wrapper around the trained VAE to select optimal candidates.

## 4. Phase 3: Advanced & Exploratory (Medium-Term)

**Goal:** New modalities and experimental architectures.

- **Spectral Bio-ML (Proposal 5):** Implement `src/models/spectral_encoder.py` for Graph Laplacian embeddings.
- **Swarm VAE (Proposal 11):** Implement `src/training/swarm_trainer.py` for distributed consensus training.
- **Extraterrestrial Analysis (Proposal 4):** Create isolated analysis scripts in `scripts/analysis/asteroid_amino_acids.py`.
- **Long-COVID & Huntington's (Proposals 9 & 10):** New specific model subclasses or latent dimension extensions.

## 5. Execution Checklist

- [ ] **Infrastructure:** Create `src/encoders/` and `src/optimizers/`.
- [ ] **Phase 1 Implementation:**
  - [ ] `src/losses/geometric_loss.py`
  - [ ] `src/encoders/codon_encoder.py`
- [ ] **Verification:** Add corresponding unit tests in `tests/unit/` (filling the testing gap).
