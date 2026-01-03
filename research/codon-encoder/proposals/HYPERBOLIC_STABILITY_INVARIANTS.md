# Research Proposal: Hyperbolic Stability-Invariant Model

**Doc-Type:** Research Proposal · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Executive Summary

Following the falsification of both TEGB and multi-prime p-adic conjectures, we propose a new research direction: training a **purely hyperbolic model** (unconstrained by p-adic structure) to discover invariants that encode:

1. **Thermodynamic stability** (what p-adic deliberately obscures)
2. **Optimal organism functioning** (homeostasis, repair, resilience)
3. **Intelligent immune response** (self/non-self discrimination)
4. **Persistence over reproduction** (longevity without relying on copying)

---

## Motivation: What the Falsification Revealed

### The DNA Trade-off

The p-adic falsification revealed a fundamental truth:

> **DNA evolved for INFORMATION RESILIENCE, not THERMODYNAMIC OPTIMIZATION.**

The genetic code is **anti-optimized** for p-adic similarity (r = -0.91 for p=4). This means:
- Evolution chose error tolerance over efficiency
- Reproduction is biology's "hack" to overcome physics
- Individual thermodynamic optimization was sacrificed

### The Opportunity

What evolution deliberately obfuscated, we can deliberately search for. The invariants that encode stability and longevity EXIST in hyperbolic space—they were just not selected for by evolution because reproduction was an easier solution.

---

## Research Objectives

### Primary Objective

Discover **stability invariants** in hyperbolic geometry that predict:
- Protein thermodynamic stability (DDG)
- Metabolic efficiency
- Cellular repair fidelity
- Homeostatic resilience

### Secondary Objectives

1. **Intelligent Immune System Invariants**
   - Self/non-self discrimination without autoimmunity
   - Adaptive response without inflammatory damage
   - Memory without rigid specificity

2. **Longevity Invariants**
   - Persistence mechanisms beyond DNA repair
   - Cellular coordination without reproduction
   - Systemic homeostasis optimization

3. **Beyond-Reproduction Communication**
   - Neurotransmitter-level signaling patterns
   - Coordination without copying
   - "Languages" that cells can SPEAK, not just replicate

---

## Proposed Architecture

### Core Design: Purely Hyperbolic from Day One

Unlike the TernaryVAE which was constrained to 3-adic structure, this model:
- Uses Poincaré ball geometry throughout
- Has NO p-adic valuation constraints
- Learns radial structure from DATA, not mathematical priors

```
Architecture: Hyperbolic-Stability-VAE (HS-VAE)

Input: Protein/sequence features (NOT codon indices)
       ↓
[Hyperbolic Encoder]
  - GRU/Transformer on sequence
  - Exponential map to Poincaré ball
  - NO p-adic radial constraints
       ↓
[Latent Space: Poincaré Ball]
  - Curvature learned from data
  - Radial position = stability hierarchy (learned)
  - Angular position = functional clustering (learned)
       ↓
[Hyperbolic Decoder]
  - Log map to tangent space
  - Predict: DDG, stability metrics, functional class
       ↓
Output: Stability predictions, invariant features
```

### Key Differences from TernaryVAE

| Aspect | TernaryVAE | HS-VAE |
|--------|------------|--------|
| Input | Codon indices (0-63) | Sequence features |
| Structure | 3-adic constrained | Unconstrained hyperbolic |
| Radial meaning | p-adic valuation | Stability (learned) |
| Training signal | Reconstruction | DDG + stability metrics |
| Purpose | Encode genetic code | Find stability invariants |

---

## Training Data Requirements

### Thermodynamic Stability

| Dataset | Size | Features |
|---------|------|----------|
| S669 | 669 | Point mutations, experimental DDG |
| S2648 | 2,648 | Larger mutation set |
| ProTherm | ~30,000 | Comprehensive thermodynamic data |
| FireProtDB | ~16,000 | Stability measurements |

### Metabolic/Functional

| Dataset | Size | Features |
|---------|------|----------|
| UniProt | millions | Functional annotations |
| PDB | ~200,000 | 3D structure context |
| AlphaFold DB | 200M | Predicted structures |

### Immune System

| Dataset | Size | Features |
|---------|------|----------|
| IEDB | millions | Epitope data |
| HIV mutation data | ~10,000 | Drug resistance, tropism |
| Autoimmune epitopes | ~5,000 | Self-reactive sequences |

---

## Loss Function Design

### Multi-Objective Loss

```python
L_total = (
    λ_stability * L_DDG +           # Thermodynamic prediction
    λ_homeostasis * L_homeostasis + # Metabolic efficiency
    λ_immune * L_immune +           # Self/non-self discrimination
    λ_hierarchy * L_hierarchy +     # Learned radial structure
    λ_KL * L_KL                     # Hyperbolic KL divergence
)
```

### Key: NO P-adic Constraints

Unlike TernaryVAE which enforced:
```python
# TernaryVAE: p-adic constraint (we REMOVE this)
target_radius = f(valuation)  # Forced structure
```

HS-VAE learns:
```python
# HS-VAE: learned structure
radial_meaning = model.discover(data)  # Emergent structure
```

---

## Expected Invariants to Discover

### Hypothesis: What Hyperbolic Geometry Should Reveal

If thermodynamic stability has hierarchical structure (which physics suggests), HS-VAE should discover:

1. **Radial Stability Gradient**
   - Center: Most stable configurations
   - Boundary: Least stable / most reactive

2. **Angular Functional Clusters**
   - Similar functions cluster angularly
   - Functional transitions as angular rotations

3. **Geodesic Mutation Paths**
   - Neutral mutations: Along isoclines
   - Destabilizing: Radially outward
   - Stabilizing: Radially inward

4. **Immune Discrimination Boundaries**
   - Self: Specific angular region
   - Non-self: Distinct region
   - Autoimmune danger zone: Boundary region

---

## Implementation Plan

### Phase 1: Data Preparation (2 weeks)

1. Consolidate DDG datasets (S669, S2648, ProTherm)
2. Extract sequence features (NOT codon indices)
3. Prepare immune system datasets
4. Build data loaders with hyperbolic batching

### Phase 2: Architecture Implementation (3 weeks)

1. Implement HS-VAE encoder (hyperbolic from input)
2. Implement Poincaré latent space (learnable curvature)
3. Implement multi-task decoder
4. Implement hyperbolic loss functions

### Phase 3: Training (2 weeks)

1. Pre-train on reconstruction
2. Fine-tune on DDG prediction
3. Multi-task training on stability + immune
4. Analyze emergent structure

### Phase 4: Invariant Discovery (2 weeks)

1. Extract learned radial/angular structure
2. Identify stability invariants
3. Validate on held-out data
4. Compare to p-adic (should outperform)

---

## Success Criteria

### Minimum Success

- DDG prediction: LOO Spearman > 0.35 (beats p-adic's 0.07)
- Learned hierarchy: Meaningful radial structure emerges
- No anti-correlation: Unlike p-adic, positive stability correlation

### Target Success

- DDG prediction: LOO Spearman > 0.50 (matches ELASPIC-2)
- Immune discrimination: Self/non-self separation in latent space
- Novel invariants: Discover features not in literature

### Stretch Success

- DDG prediction: LOO Spearman > 0.65 (approaches Rosetta)
- Longevity markers: Identify sequences associated with lifespan
- Actionable insights: Guidance for protein engineering

---

## Philosophical Context

### Beyond Reproduction: The Third Strategy

Evolution discovered two strategies for overcoming entropy:
1. **Asexual reproduction**: Copy yourself (error-prone)
2. **Sexual reproduction**: Copy with variation (robust but mortal)

We propose searching for a third:
3. **Persistence through optimization**: Self-repair, homeostasis, intelligent immune response

### The Language Analogy

Viruses "hijack" cellular machinery—they use the language without understanding it.

Neurotransmitters show there's a richer communication system—cells can "speak" to each other, coordinate without copying.

The HS-VAE aims to discover the "grammar" of this language: the invariants that encode coordination, stability, and persistence rather than replication.

### The Question to Answer

Not: "How long can humans live?"

But: "What will humans DO with star-like lifespans?"

The answer depends on finding the stability invariants that evolution deliberately bypassed.

---

## File Locations

```
research/codon-encoder/
├── falsification/              # Completed
│   ├── tegb_falsification.py
│   ├── multiprime_falsification.py
│   └── COMBINED_FALSIFICATION_RESULTS.md
│
├── proposals/                  # This document
│   └── HYPERBOLIC_STABILITY_INVARIANTS.md
│
└── stability_invariants/       # TO BE CREATED
    ├── models/
    │   └── hs_vae.py          # Hyperbolic-Stability-VAE
    ├── training/
    │   └── train_hs_vae.py
    ├── data/
    │   └── prepare_stability_data.py
    └── analysis/
        └── extract_invariants.py
```

---

## Next Steps

1. **Review this proposal** - Confirm research direction
2. **Prepare DDG datasets** - Consolidate thermodynamic data
3. **Implement HS-VAE** - Purely hyperbolic architecture
4. **Train and discover** - Find the stability invariants

---

*This proposal represents a pivot from encoding the genetic code's structure (which optimizes for error tolerance) to discovering the thermodynamic stability invariants that evolution deliberately bypassed in favor of reproduction.*
