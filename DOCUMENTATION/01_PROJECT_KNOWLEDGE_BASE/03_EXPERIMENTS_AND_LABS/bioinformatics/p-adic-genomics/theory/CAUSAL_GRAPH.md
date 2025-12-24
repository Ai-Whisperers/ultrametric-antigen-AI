# Causal Graph Formalization for Autoimmune PTM Pathology

**Doc-Type:** Causal Model · Version 1.0 · Updated 2025-12-18

---

## 1. Overview

This document formalizes the causal structure underlying PTM-driven autoimmunity, integrating the p-adic geometric framework with dynamical systems theory. The model explains how environmental stressors synchronize to produce immunogenic PTM patterns.

---

## 2. Causal Graph Structure

### 2.1 Node Definitions

**Environmental Nodes (Exogenous)**

| Node | Symbol | Description |
|------|--------|-------------|
| Inflammatory load | E₁ | Chronic low-grade inflammatory exposure |
| Mechanical stress | E₂ | Repetitive sub-threshold mechanical load |
| Metabolic oscillation | E₃ | Redox imbalance, metabolic cycling |
| Barrier perturbation | E₄ | Skin/lung/gut interface instability |

**State Nodes (Endogenous)**

| Node | Symbol | Description |
|------|--------|-------------|
| Coherence index | C | Temporal synchrony of E₁-E₄ |
| PAD activation | P | Time-integrated PAD enzyme activity |
| PTM field | M | Distribution of PTMs over sites |
| Goldilocks load | G | PTMs with Δ entropy in [α, β] |
| Immune legibility | I | HLA presentation + T-cell recognition |
| Attractor strength | A | Self-sustaining immune memory |
| Bone remodeling bias | B | Osteoclast − osteoblast imbalance |

### 2.2 Directed Edges

```
          E₁ ─┐
          E₂ ─┼──► C ──► P ──► M ──► G ──► I ──► A ──► B
          E₃ ─┤                                   │
          E₄ ─┘                                   │
              ◄────────────── feedback ───────────┘
```

**Edge meanings**:

| Edge | Interpretation |
|------|----------------|
| Eₖ → C | Each stressor contributes phase to coherence |
| C → P | Coherence determines integrated PAD activation |
| P → M | PAD activity produces PTM distribution |
| M → G | PTM field filtered to Goldilocks subset |
| G → I | Goldilocks PTMs become immunologically legible |
| I → A | Legible epitopes build immune memory |
| A → B | Sustained immunity drives bone remodeling |
| A → C | Immune activation modulates stress response (feedback) |

---

## 3. Mathematical Formalization

### 3.1 Coherence Equation

Coherence quantifies temporal alignment of stressors:

```
C(t) = ∫₀ᵀ |Σₖ wₖ(t) exp(iφₖ(t))| dt
```

where:
- wₖ(t) = weight of stressor k at time t
- φₖ(t) = phase of stressor k at time t
- T = integration window

**Properties**:
- C = 0 when stressors are uniformly distributed in phase
- C = 1 when all stressors perfectly aligned
- Disease requires sustained C > threshold

### 3.2 PAD Activation

PAD activity integrates over coherence:

```
P(t) = ∫₀ᵗ C(τ) · Ca²⁺(τ) dτ
```

where Ca²⁺(τ) is local calcium concentration.

**Key point**: P is time-integrated, not instantaneous. Explains why acute stress doesn't cause RA but chronic exposure does.

### 3.3 PTM Field

The PTM field M is a distribution over protein sites:

```
M(s, t) = P(t) · susceptibility(s) · accessibility(s)
```

where:
- s = site index
- susceptibility(s) = intrinsic PAD substrate quality
- accessibility(s) = structural exposure

### 3.4 Goldilocks Load

Filter M to immunogenic subset:

```
G(t) = Σₛ 𝟙{Δ_C(s) ∈ [α, β]} · M(s, t)
```

Only PTMs with shift in Goldilocks Zone [α, β] ≈ [0.15, 0.30] contribute.

### 3.5 Immune Legibility

Legibility combines Goldilocks load with HLA presentation:

```
I(t) = G(t) · HLA_affinity · boundary_crossing(t)
```

where:
- HLA_affinity = genetically determined presentation efficiency
- boundary_crossing = fraction of PTMs crossing p-adic boundaries

### 3.6 Attractor Dynamics

Immune memory accumulates via attractor dynamics:

```
A(t+1) = A(t) + λ·I(t) - μ·decay(t)
```

where:
- λ = learning rate (epitope spreading)
- μ = decay rate (natural immune tolerance)

**Attractor condition**: A stabilizes when λ·I = μ·decay (disease equilibrium)

### 3.7 Bone Remodeling

Bone loss integrates immune activity:

```
B(t) = ∫₀ᵗ A(τ) · osteoclast_activation(τ) dτ
```

**Key point**: B lags A. Explains why bone damage appears late in disease course.

---

## 4. Key Causal Properties

### 4.1 No Direct Path from E to A

```
E₁ ─/─► A  (no direct edge)
```

Individual stressors do not directly cause disease. Only synchronized stressors (high C) lead to pathology.

**Implication**: Eliminating one stressor may reduce C but not eliminate disease.

### 4.2 Coherence is the Bottleneck

All causal paths pass through C:

```
Eₖ → C → P → M → G → I → A → B
```

**Implication**: Breaking coherence breaks disease progression, even if stressors persist.

### 4.3 Feedback Sustains Disease

The A → C feedback loop makes disease self-sustaining:

```
High A → increased inflammatory signaling → higher E₁ → higher C → higher A
```

**Implication**: Late-stage disease is immune-driven, not stressor-driven.

---

## 5. Intervention Analysis

### 5.1 Intervention Calculus

Using do-calculus, we can analyze interventions:

**Do(Eₖ = 0)**: Eliminate one stressor
```
P(A | do(E₁ = 0)) = P(A | C decreased) ≠ P(A | no disease)
```
Partial effect only.

**Do(C = 0)**: Break coherence
```
P(A | do(C = 0)) = P(A | P → 0) → disease remission
```
Complete effect.

**Do(G = 0)**: Eliminate Goldilocks PTMs
```
P(A | do(G = 0)) = P(A | I → 0) → no new epitopes
```
Prevents progression but doesn't reverse existing A.

### 5.2 Safe vs. Risky Targets

**Safe targets** (low systemic risk):
- C: Breaking coherence via timing disruption
- G: Tolerogenic therapy to exit Goldilocks Zone

**Risky targets** (high systemic effects):
- P: Global PAD inhibition affects normal physiology
- I: Broad immunosuppression has infectious complications
- A: Immune memory ablation is irreversible

### 5.3 The Coherence Intervention

The safest intervention is desynchronization:

```
Target: Σₖ exp(iφₖ) → 0
```

**Biological interpretation**:
- PTMs still occur (normal)
- But not synchronized (no accumulation)
- No Goldilocks load builds up
- Disease stalls without immunosuppression

---

## 6. Why Standard Treatments Plateau

### 6.1 Anti-TNF (E₁ reduction)

```
Before: E₁ + E₂ + E₃ + E₄ → high C
After:  0  + E₂ + E₃ + E₄ → reduced C (not zero)
```

C decreases but doesn't reach zero. Some coherence remains. Disease slows but doesn't remit.

### 6.2 Methotrexate (Broad suppression)

```
Before: High P → High M → High G
After:  Lower P → Lower M → Lower G
```

Reduces all PTMs, not just Goldilocks. Side effects from suppressing normal PTMs.

### 6.3 Why Combination Works Better

```
Anti-TNF + Methotrexate:
- Lower E₁
- Lower P
- Combined: C reaches threshold for remission
```

Multiple points of intervention can achieve what single-target cannot.

---

## 7. Dynamical Systems View

### 7.1 State Space

The system state is (C, P, M, G, I, A, B) ∈ ℝ⁷

### 7.2 Fixed Points

**Healthy fixed point**:
```
(C*, P*, M*, G*, I*, A*, B*) ≈ (low, low, normal, ≈0, ≈0, ≈0, 0)
```

**Disease fixed point**:
```
(C*, P*, M*, G*, I*, A*, B*) ≈ (high, high, aberrant, high, high, stable, increasing)
```

### 7.3 Basin of Attraction

The disease attractor has a basin characterized by:
```
{x : C(x) > C_threshold AND sustained for T > T_threshold}
```

Once in basin, system converges to disease state without external intervention.

### 7.4 Bifurcation

The transition from health to disease is a bifurcation:
```
Below threshold: single healthy attractor
Above threshold: two attractors (healthy + disease)
```

Prevention means staying below threshold. Treatment means escaping disease basin.

---

## 8. Falsifiable Predictions

### Prediction 1: Coherence Predicts Onset

```
P(RA onset | high C) >> P(RA onset | high E₁ alone)
```
Testable by measuring stress timing, not just stress magnitude.

### Prediction 2: Breaking Coherence Induces Remission

```
P(remission | do(C = 0)) > P(remission | do(E₁ = 0))
```
Testable by intervention studies targeting timing vs. intensity.

### Prediction 3: Late Disease is Feedback-Dominated

```
Correlation(E_current, A) decreases as A increases
```
In late disease, immune memory drives itself, not external stressors.

### Prediction 4: Bone Damage Lags Immune Activity

```
B(t) ∝ ∫₀ᵗ A(τ) dτ
```
Bone loss continues after immune intervention until integral saturates.

---

## 9. Computational Implementation

### 9.1 Simulation Framework

```python
class AutoimmuneSystem:
    def __init__(self):
        self.C = 0  # Coherence
        self.P = 0  # PAD activation
        self.G = 0  # Goldilocks load
        self.A = 0  # Attractor strength
        self.B = 0  # Bone damage

    def step(self, E, dt):
        # Environmental stressors E = [E1, E2, E3, E4]
        phases = self.get_phases(E)
        self.C = self.compute_coherence(E, phases)
        self.P += self.C * dt
        self.G = self.filter_goldilocks(self.P)
        self.A += (self.lambda_ * self.G - self.mu * self.A) * dt
        self.B += self.A * dt
```

### 9.2 Parameter Estimation

Parameters can be estimated from clinical data:
- λ: Rate of ACPA development
- μ: Natural tolerance decay
- Goldilocks bounds [α, β]: From epitope immunogenicity data

---

## 10. Summary

The causal graph formalizes:

1. **Multi-factorial causation**: No single stressor causes disease
2. **Coherence bottleneck**: Synchronization is necessary for pathology
3. **Self-sustaining attractor**: Late disease is immune-autonomous
4. **Safe intervention target**: Break coherence, not components

This framework transforms treatment strategy from "suppress everything" to "disrupt synchronization" - a more targeted, safer approach with potential for true remission.

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-18 | 1.0 | Initial causal graph formalization |
