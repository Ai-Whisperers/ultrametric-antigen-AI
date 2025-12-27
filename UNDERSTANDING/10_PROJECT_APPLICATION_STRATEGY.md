# Project Application Strategy: Universal Isomorphisms

**"What we need to do, improve, and run for each of these ideas."**

This document translates the theoretical findings from `03_UNIVERSAL_ISOMORPHISMS.md` into concrete engineering tasks for the Ternary VAEs project.

---

## 1. Physics: The Holographic VAE

**Concept**: The mapping between the hyperbolic latent space (Bulk) and peptide sequences (Boundary).

### Current State

- **Good**: `src/losses/padic/ranking_hyperbolic.py` already implements the radial hierarchy ($r \sim \text{valuation}$).
- **Missing**: We don't strictly enforce or measure the "Holographic Dictionary" (bulk fields $\leftrightarrow$ boundary operators).

### Action Items

1.  **Implement Holographic Reconstructor**:

    - Create `src/models/holographic_decoder.py`.
    - _Idea_: Instead of a standard MLP decoder, use a "Bulk-to-Boundary Propagator". The signal should decay from the origin (root) to the boundary (leaves) following the geodesic.
    - _Improvement_: This would make the decoder much more parameter-efficient and interpretible.

2.  **Run "Correlator Check"**:
    - _Experiment_: Verify if the mutual information between two sequences decays as a power law of their hyperbolic distance.
    - _Script_: `scripts/validation/verify_holographic_scaling.py`.

---

## 2. Spin Glasses: The Landscape of Immunity

**Concept**: Viral evolution happens on a rugged "energy landscape". The deep valleys are stable variants (strains).

### Current State

- **Missing**: We treat the latent space as continuous, but spin glass theory says it should be "clumpy" (ultrametric hierarchy of states).

### Action Items

1.  **Implement "Parisi Overlap" Metric**:

    - _New Module_: `src/analysis/physics/spin_glass_overlap.py`.
    - _Logic_: Sample $N$ latent vectors for a single viral sequence (using VAE reparameterization noise). Compute the distribution of distances between them.
    - _Goal_: If the distribution is non-trivial (not just Gaussian), it indicates "Replica Symmetry Breaking" – i.e., the virus has multiple distinct stable configurations (immune escape variants).

2.  **Run "Simulated Annealing" for Vaccine Design**:
    - _Run_: Use the "energy" (loss function) to run a standard simulated annealing search to find the "Ground State" (optimal vaccine target) that minimizes free energy across the landscape.

---

## 3. Networks: Popularity × Similarity

**Concept**: Scale-free networks emerge from hyperbolic geometry where connections trade off "popularity" (degree) and "similarity" (angular distance).

### Current State

- **Good**: We have `graphs/hyperbolic_gnn.py`.
- **Missing**: Connection to viral spreads (R0).

### Action Items

1.  **Model Viral Transmissibility as "Popularity"**:
    - _Improve_: In `src/analysis/evolution.py`, add a `transmissibility` score that maps to the radial coordinate ($r$).
    - _Logic_: High $R_0$ strains should be closer to the center of the disk (hubs), low $R_0$ strains at the periphery.
    - _Run_: Visualize the current variants. Are the "Pandemic" strains (Omicron, Delta) closer to the origin than rare variants?

---

## 4. Linguistics: The Peptide Grammar

**Concept**: Protein sequences aren't random; they have a "grammar" (syntax) like natural language.

### Current State

- **Missing**: Completely absent. We treat sequences as flat strings.

### Action Items

1.  **Implement "Peptide Merge" Operation**:

    - _New Module_: `src/linguistics/protein_grammar.py`.
    - _Idea_: Define a recursive operation that combines two domains into a protein.
    - _Run_: Train a "Tree-LSTM" or recursive neural network on the PDB dataset, explicitly forcing it to learn the "parse tree" of the protein.

2.  **Syntactic Mutation Analysis**:
    - _Improve_: Classify mutations not just by chemistry, but by "syntax error". Does a mutation break the "phrase structure" of the binding site?

---

## 5. Biology: Coalescent Reconstruction

**Concept**: We only see the "Boundary" (living sequences). We need to infer the "Bulk" (Ancestral Tree).

### Current State

- **Good**: `evolution.py` predicts likelihoods.
- **Missing**: Explicit Ancestral Reconstruction using the Hyperbolic Geodesics.

### Action Items

1.  **Implement "Geodesic Interpolation"**:
    - _Improve_: In `src/models/ternary_vae.py`, add method `interpolate_ancestor(seq_a, seq_b)`.
    - _Logic_: The ancestor of A and B is the "midpoint" of the geodesic connecting them in hyperbolic space.
    - _Run_: Take two divergent strains (e.g., SARS-CoV-1 and SARS-CoV-2). Compute the midpoint. Decode it. Does it look like the Bat-CoV RaTG13?

---

## Summary Checklist

- [ ] **Physics**: Create `verify_holographic_scaling.py`
- [ ] **Spin Glass**: Implement `spin_glass_overlap.py`
- [ ] **Networks**: Update `evolution.py` to map $R_0 \to \text{Radius}$
- [ ] **Biology**: Test "Geodesic Ancestor" decoding
