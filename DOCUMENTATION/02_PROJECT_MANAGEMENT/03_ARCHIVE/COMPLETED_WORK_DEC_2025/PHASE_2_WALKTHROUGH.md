# Phase 2 Walkthrough: Multi-Task & Interaction Modeling

## Goal

To incorporate biological and chemical constraints into the VAE via interaction-graph embeddings, autoimmunity penalties, and multi-objective optimization.

## Implementation Details

### 1. Drug Interaction Loss

**File:** [drug_interaction.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/losses/drug_interaction.py)

- **Class:** `DrugInteractionPenalty`
- **Mechanism:** Contrastive loss in hyperbolic space.
- **Formula:**
  - Interaction ($y=1$): $\mathcal{L} = d(z_1, z_2)^2$
  - No Interaction ($y=0$): $\mathcal{L} = \max(\text{margin} - d(z_1, z_2), 0)^2$
- **Integration:** Added to `train.py` with CLI arg `--drug_sim_weight`.

### 2. Autoimmunity Profiling

**File:** [autoimmunity.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/data/autoimmunity.py)

- **Class:** `AutoimmunityLoader`
- **Functionality:**
  - Estimates immunogenicity risk based on sequence complexity and motifs.
  - Returns a risk score $s \in [0, 1]$.
  - **Testing:** Verified against known high-risk (low complexity) and low-risk synthetic sequences.

### 3. Multi-Objective Optimization

**File:** [multi_objective.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/optimizers/multi_objective.py)

- **Class:** `ParetoFrontOptimizer`
- **Algorithm:** Non-dominated sorting to identify the Pareto Front from a batch of candidate solutions.
- **Use Case:** Selecting latent vectors that minimize VAE Loss AND Autoimmunity Risk simultaneously.

## Verification

### Unit Tests

Ran `python -m pytest` on:

- `tests/unit/test_drug_interaction.py`: Passed (verified contrastive logic).
- `tests/unit/test_autoimmunity.py`: Passed (verified risk scoring heuristics).
- `tests/unit/test_multi_objective.py`: Passed (verified Pareto dominance logic).

### CLI Verification

Ran `python scripts/train/train.py --help`:

- Confirmed `--drug_sim_weight` argument is present.
- Confirmed `drug_interaction_weight` integrated into training loop.
