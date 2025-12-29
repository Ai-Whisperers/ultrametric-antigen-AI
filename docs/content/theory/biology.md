# Biological Context

> **Drug resistance, immune evasion, and evolutionary mechanisms.**

---

## Overview

Ternary VAE addresses key biological challenges:

1. **Drug Resistance Prediction** - Which mutations confer resistance?
2. **Immune Evasion** - How do pathogens escape antibodies?
3. **Vaccine Target Selection** - Which epitopes are evolutionarily stable?

---

## Drug Resistance

### Mechanisms

| Mechanism | Example | Drugs Affected |
|-----------|---------|----------------|
| Binding site mutation | K103N | NNRTIs |
| Steric exclusion | M184V | 3TC, FTC |
| Active site change | Q151M | All NRTIs |
| Excision enhancement | TAMs (T215Y) | AZT, d4T |

### HIV Drug Classes

| Class | Target | Example Drugs | Key Mutations |
|-------|--------|---------------|---------------|
| NRTI | Reverse transcriptase | AZT, 3TC, TDF | M184V, K65R, TAMs |
| NNRTI | RT binding pocket | EFV, NVP, RPV | K103N, Y181C, E138K |
| PI | Protease | LPV, DRV, ATV | I50L, I84V, L90M |
| INI | Integrase | DTG, RAL, EVG | R263K, Q148H, N155H |

### Cross-Resistance

Some mutations affect multiple drugs:

```
M184V → Resistant to: 3TC, FTC
      → Sensitive to: AZT, TDF (resensitization!)

TAMs  → Resistant to: AZT, d4T
      → Partially to: ABC, TDF
```

---

## Immune Evasion

### Glycan Shield

Viruses coat themselves with host sugars:

| Position | Glycan Type | Effect |
|----------|-------------|--------|
| N88 | High-mannose | Shields V1/V2 loop |
| N156 | Complex | Shields V2 apex |
| N332 | High-mannose | bnAb epitope, under selection |

**Sentinel glycans**: Positions where glycan removal exposes vulnerabilities.

### CTL Escape

CD8+ T cells recognize viral peptides on MHC:

```
HLA-A*02 presents: SLYNTVATL (Gag p17)
Escape mutations: Y→F, T→A

HLA-B*57 presents: KAFSPEVIPMF (Gag p24)
Escape mutations: A→T (with fitness cost)
```

### Antibody Escape

Broadly neutralizing antibodies (bnAbs) target:

| Target | bnAb Class | Escape Mutations |
|--------|------------|------------------|
| CD4 binding site | VRC01-class | D279N, N280D |
| V2 apex | PG9-class | K169Q, R166K |
| V3 glycan | PGT121-class | S334N, N332S |
| MPER | 10E8-class | N671D, W672L |

---

## Vaccine Target Selection

### Ideal Vaccine Target Properties

1. **Conserved**: Low variability across strains
2. **Accessible**: Surface-exposed, not shielded
3. **Functional**: Mutations reduce fitness
4. **Immunogenic**: Induces strong immune response

### Scoring

```python
def vaccine_priority(epitope):
    conservation = 1 - entropy(epitope) / max_entropy
    accessibility = surface_accessibility(epitope)
    fitness_cost = fitness_impact(epitope_mutations)
    immunogenicity = binding_affinity(epitope, HLA_alleles)

    return (
        0.3 * conservation +
        0.2 * accessibility +
        0.3 * fitness_cost +
        0.2 * immunogenicity
    )
```

### Top HIV Vaccine Candidates

| Epitope | Protein | Priority Score | Reason |
|---------|---------|----------------|--------|
| TPQDLNTML | Gag | 0.970 | Highly conserved, strong CTL |
| SLYNTVATL | Gag | 0.892 | HLA-A2 restricted, protective |
| GPGHKARVL | Pol | 0.856 | Conserved in RT |

---

## Multi-Disease Platform

### 11 Disease Domains

| Disease | Type | Key Targets | Resistance Concern |
|---------|------|-------------|-------------------|
| HIV | Virus | RT, PR, IN | MDR, transmitted resistance |
| SARS-CoV-2 | Virus | RdRp, Spike | mAb escape, Paxlovid |
| TB | Bacterium | rpoB, katG, gyrA | MDR-TB, XDR-TB |
| Influenza | Virus | NA, PB2 | NAI resistance, vaccine mismatch |
| HCV | Virus | NS3, NS5A, NS5B | DAA resistance |
| HBV | Virus | Pol | NA resistance |
| Malaria | Parasite | K13, Pfcrt | Artemisinin resistance |
| MRSA | Bacterium | mecA, pbp2a | Beta-lactam resistance |
| Candida auris | Fungus | ERG11, FKS1 | Pan-resistance |
| RSV | Virus | F protein | mAb escape |
| Cancer | Oncology | EGFR, BRAF, ALK | TKI resistance |

---

## Evolutionary Dynamics

### Fitness Landscape

```
Fitness
   ^
   |     /\
   |    /  \    /\
   |   /    \  /  \
   |  /      \/    \
   |_/________________> Sequence
   WT  M184V  TAMs
```

- **Peaks**: Stable genotypes
- **Valleys**: Unstable intermediates
- **Ridges**: Accessible evolutionary paths

### Epistasis

Mutations interact:
- **Positive**: Combined benefit > sum of individual
- **Negative**: Combined benefit < sum of individual
- **Sign**: Direction of effect changes

---

## References

1. Shafer, R.W. (2006). Rationale and uses of a public HIV drug-resistance database
2. Wei, X., et al. (2003). Antibody neutralization and escape by HIV-1
3. Bloom, J.D. (2017). Protein stability promotes evolvability

---

_Last updated: 2025-12-28_
