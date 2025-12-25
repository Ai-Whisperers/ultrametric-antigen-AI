# Phase 3 Walkthrough: Advanced & Exploratory Methods

## Goal

To implement experimental modalities that leverage Graph Signal Processing and Distributed Learning.

## Implementation Details

### 1. Spectral Graph Encoder

**File:** [spectral_encoder.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/models/spectral_encoder.py)

- **Class:** `SpectralGraphEncoder`
- **Method:** Graph Laplacian Eigenmaps.
- **Process:**
  1. Input: Adjacency Matrix $A$.
  2. Compute Normalized Laplacian $L = I - D^{-1/2} A D^{-1/2}$.
  3. Eigendecomposition $L v = \lambda v$.
  4. Select top-$k$ non-zero eigenvectors as spectral features.
  5. Project to Hyperbolic space.
- **Use Case:** Encoding protein contact maps or interaction networks where topology matters more than sequence.

### 2. Swarm Trainer

**File:** [swarm_trainer.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/training/swarm_trainer.py)

- **Class:** `SwarmTrainer`
- **Method:** Federated Averaging (FedAvg).
- **Process:**
  1. Maintain a population of $N$ independent VAE agents.
  2. `perform_consensus()`: Computes the average weight for every parameter across all agents.
  3. Redistributes the "consensus model" back to all agents.
- **Use Case:** Training on disjoint datasets (e.g., different hospitals/labs) without sharing raw data, or exploring different loss basins.

## Verification

### Unit Tests

Ran `python -m pytest`:

- `tests/unit/test_spectral_encoder.py`: Passed. Verified Laplacian shape and output dimensions.
- `tests/unit/test_swarm_trainer.py`: Passed. Verified that weights from 3 agents (1s, 3s, 5s) averaged correctly to 3s.

## Roadmap Completion

This concludes the planned implementation of the Research Roadmap.

- **Phase 1:** Geometric Loss & Codon Encoder (Complete)
- **Phase 2:** Drug Interaction & Autoimmunity (Complete)
- **Phase 3:** Spectral & Swarm (Complete)
