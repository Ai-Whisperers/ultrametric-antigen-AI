# Research Opportunities & Academic Context

> **Strategic Alignment**: Intersection of Geometric Deep Learning, p-adic Number Theory, and Viral Evolution.

This document identifies unproven theories that can be verified using the Ternary VAE framework, key open problems in the field, and the specific researchers and laboratories that would be most interested in this work.

---

## 1. Unproven Theories & Open Problems

_Theories that `ternary-vaes-bioinformatics` is uniquely positioned to prove or verify._

### A. The "Ultrametric Genome" Hypothesis

**The Theory**: The genetic code is not arbitrary but structured by an underlying ultrametric (tree-like) topology, specifically a p-adic topology where "closeness" is defined by functional similarity rather than sequence edit distance.
**Current State**: Primarily theoretical and descriptive (Khrennikov, Dragovich). Lacks a generative, predictive model.
**The Usage of This Repo**:

- **Verification**: Use the `Ternary VAE` to generate _de novo_ sequences. If the generated sequences are biologically viable (high folding stability), it proves that the latent space correctly captures the "hidden" ultrametric structure of biology.
- **Metric**: The project's **3-adic Banking Loss** directly optimizes for this.

### B. Hyperbolic Embeddings for Viral Escape

**The Theory**: Viral evolution (e.g., HIV, SARS-CoV-2) occurs on a hyperbolic manifold because the number of possible viable mutations expands exponentially, like the volume of a hyperbolic sphere.
**Current State**: Hyperbolic embeddings are used for static phylogenetic trees (Sarkar, De Sa), but not for _predicting future functional states_ of a virus in a generative manner.
**The Usage of This Repo**:

- **Prediction**: Use the **Poincaré Ball** embedding to map the "Escape Manifold". Existing tools (PhyloVAE) focus on tree topology reconstruction. This repo focuses on _operation space_—predicting which specific residue changes allow immune escape.

### C. The "Binary vs. Ternary" Code Evolution

**The Theory**: The genetic code evolved from a simpler binary system to the current ternary (A, C, G/T) combinatorial system to maximize error robustness.
**Current State**: Mathematical models exist (Rumer's symmetry), but no deep learning model has simulated this transition.
**The Usage of This Repo**:

- **Simulation**: Train the VAE with "masked" dimensionality (Binary Mode) vs. full Ternary Mode. Comparing reconstruction loss and error robustness between these modes provides empirical evidence for the evolutionary advantage of the ternary system.

---

## 2. Key Researchers & Laboratories

_Professionals who would be high-value targets for collaboration or review._

### 📍 Geometric Deep Learning & Bio

**Target**: Generative models for protein/molecule structure.

1.  **Marinka Zitnik (Harvard Medical School)**
    - **Focus**: Graph Neural Networks, Hyperbolic embeddings for drug repurposing.
    - **Relevance**: Her lab pioneered hyperbolic embeddings for hierarchical biological data. The Ternary VAE adds the _generative_ sequence component.
2.  **Matteo Dal Peraro (EPFL)**
    - **Focus**: Geometric Deep Learning for protein design (PeSTo, CARBonAra).
    - **Relevance**: His group creates geometric transformers. They would be interested in the "Frozen Encoder" approach for stable manifold projection.
3.  **Rex Ying (Yale)**
    - **Focus**: Graph Neural Networks, Hyperbolic ML.
    - **Relevance**: His group (Graph & Geometric Learning Lab) works on foundational hyperbolic architectures.

### 📍 p-adic Information Theory

**Target**: Mathematical foundations of the genetic code.

4.  **Andrei Khrennikov (Linnaeus University)**
    - **Focus**: Founder of p-adic information theory in biology.
    - **Relevance**: He has written extensively on the _theory_. This repo provides the _code_. He is the ideal theoretical partner.
5.  **Branko Dragovich (Institute of Physics Belgrade)**
    - **Focus**: p-adic modeling of the genetic code and protein distances.
    - **Relevance**: His work on "p-adic distances in the genetic code" is the direct theoretical ancestor of this project's embedding strategy.

### 📍 Viral Evolution & Deep Learning

**Target**: Practical application (HIV/COVID).

6.  **Debora Marks (Harvard Medical School)**
    - **Focus**: EVE (Evolutionary Model of Variant Effect). Uses VAEs for fitness prediction.
    - **Relevance**: EVE uses standard VAEs. Showing that a **Hyperbolic Ternary VAE** outperforms EVE on "long-tail" escape variants would be valid scientific breakthrough.
7.  **Sarah Teichmann (Sanger Institute / Cambridge)**
    - **Focus**: Single-cell genomics, mapping cellular dominance.
    - **Relevance**: Mapping cell lineages is inherently hyperbolic.

---

## 3. Recommended Thesis & Paper Topics

_Titles for academic output based on this repository._

1.  **"Generative Modeling of Viral Escape on a 3-adic Hyperbolic Manifold"**
    - _Venue_: ICLR / NeurIPS (AI for Science track).
    - _Hypothesis_: Hyperbolic latent spaces capture "escape trajectories" better than Euclidean ones.
2.  **"Is the Genetic Code Optimally Robust? A Deep Learning verification of Ultrametric Error Correction"**
    - _Venue_: Bioinformatics / PLoS Computational Biology.
    - _Hypothesis_: The standard genetic code minimizes "3-adic distance" errors during translation better than randomized codes.
3.  **"Frozen Encoders and Homeostatic Priors: Stabilizing Hyperbolic VAEs for Discrete Data"**
    - _Venue_: ICML.
    - _Hypothesis_: Fixing the encoder prevents "hyperbolic collapse" (a known issue where points crowd the origin) in discrete VAEs.

---

## 4. Relevant Papers to Cite

_Foundational reading for the bibliography._

- **p-adic Biology**:
  - Dragovich, B., & Dragovich, A. (2010). _p-Adic modelling of the genome and the genetic code_.
  - Khrennikov, A. (2010). _Gene expression from polynomial dynamics in the 2-adic information space_.
- **Hyperbolic Deep Learning**:
  - Nickel, M., & Kiela, D. (2017). _Poincaré Embeddings for Learning Hierarchical Representations_.
  - Mathieu, E., et al. (2019). _Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders_.
- **Geometric Protein Learning**:
  - Ingraham, J., et al. (2019). _Generative Models for Graph-Based Protein Design_.
  - Krapp, L. F., et al. (2023). _PeSTo: Parameter-free geometric deep learning for accurate protein structure analysis_.
