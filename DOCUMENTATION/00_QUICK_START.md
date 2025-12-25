# Quick Start Guide

> **Understand the Ternary VAE Bioinformatics project in 5 minutes.**

---

## What Is This?

A novel machine learning architecture that uses **hyperbolic geometry** and **3-adic number theory** to encode biological sequences in ways that naturally capture evolutionary relationships.

**Key insight**: Protein evolution has a tree-like structure. Hyperbolic space is uniquely suited to represent trees because distance grows exponentially (unlike flat Euclidean space).

---

## What Does It Do?

The system learns geometric representations of proteins that reveal:

1. **Drug resistance patterns** - Predicts which mutations will confer resistance
2. **Immune evasion strategies** - Maps how viruses hide from antibodies
3. **Vulnerability hotspots** - Identifies regions that cannot mutate without losing function
4. **Therapeutic targets** - Suggests intervention points

---

## Key Achievements

| Achievement | Result | Validation |
|:------------|:-------|:-----------|
| HIV glycan shield mapping | 7 sentinel glycan positions identified | Correlated with AlphaFold3 (r=-0.89) |
| Drug resistance prediction | 92.4% pairwise accuracy | Tested on known resistance profiles |
| Integrase vulnerability | "Achilles heel" region mapped | Consistent with clinical data |
| Viral escape prediction | 7 validated conjectures | Multi-source validation |

---

## Architecture Summary

```
Input: Protein Sequence (amino acids)
         ↓
    Frozen Encoders (VAE-A + VAE-B)
         ↓
    Hyperbolic Projection Layer (Poincare Ball)
         ↓
    Differentiable Controller
         ↓
Output: Geometric Embedding with hierarchical structure
```

**Key components:**
- **Dual VAE**: Two encoders capture different aspects (coverage + structure)
- **Hyperbolic projection**: Maps to Poincare ball where hierarchy emerges naturally
- **Frozen encoders**: Pre-trained, guaranteeing 100% coverage
- **Trainable projection**: Learns to organize the hyperbolic space

---

## Research Domains

| Domain | Status | Key Finding |
|:-------|:-------|:------------|
| **HIV** | Active | Glycan shield sentinel residues, integrase vulnerability |
| **SARS-CoV-2** | Active | Spike protein glycan analysis |
| **Rheumatoid Arthritis** | Active | HLA-autoantigen geometric relationships |
| **Neurodegeneration** | Exploratory | Tau phosphorylation patterns |

---

## Where to Go Next

### By Interest

| I want to... | Go to |
|:-------------|:------|
| Understand the math | [Theory & Foundations](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/) |
| See the research findings | [Experiments & Labs](01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/) |
| Evaluate for investment | [Investor Package](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/) |
| Run the code myself | [Setup Guide](03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/05_SPECS_AND_GUIDES/SETUP.md) |
| Validate claims | [Validation Suite](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/validation_suite/) |

### By Role

| Role | Start with |
|:-----|:-----------|
| Scientist | [Technical Whitepaper](03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/04_THEORY_DEEP_DIVES/THEORY_Technical_Whitepaper.md) |
| Investor | [Tier 2 Investor Package](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/README.md) |
| Developer | [Source README](../src/README.md) |
| Clinician | [Predictions](03_PRESENTATION_TIERS/01_TIER_1_PUBLIC/01_CLINICIANS_AND_VIROLOGISTS/PREDICTIONS.md) |

---

## Key Terms

| Term | Meaning |
|:-----|:--------|
| **Hyperbolic geometry** | Non-Euclidean geometry where parallel lines diverge; natural for tree structures |
| **3-adic / p-adic** | Alternative number system where "closeness" is defined by shared prefixes |
| **Poincare ball** | A model of hyperbolic space represented as a unit disk |
| **Glycan shield** | Sugar coating on viral proteins that helps evade immune detection |
| **Sentinel residue** | Position that signals structural or functional vulnerability |

---

## Documentation Map

```
DOCUMENTATION/
├── 00_MASTER_INDEX.md       ← Complete index
├── NAVIGATION_GUIDE.md      ← Role-based navigation
├── QUICK_START.md           ← You are here
│
├── 01_PROJECT_KNOWLEDGE_BASE/  ← Static knowledge
│   ├── 00_STRATEGY_AND_VISION/
│   ├── 02_THEORY_AND_FOUNDATIONS/
│   ├── 03_EXPERIMENTS_AND_LABS/
│   ├── 04_SCIENTIFIC_HISTORY/
│   └── 05_LEGAL_AND_IP/
│
├── 02_PROJECT_MANAGEMENT/   ← Active execution
│   ├── 00_TASKS/
│   ├── 01_ROADMAPS_AND_PLANS/
│   └── 02_CODE_HEALTH_METRICS/
│
├── 03_PRESENTATION_TIERS/   ← Stakeholder packages
│   ├── 01_TIER_1_PUBLIC/
│   ├── 02_TIER_2_INVESTOR/
│   └── 03_TIER_3_TECHNICAL/
│
├── 05_VALIDATION/           ← Testing docs
└── 06_DIAGRAMS/             ← Visual documentation
```

---

## Contact

For partnership inquiries, see [Contact Information](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/01_VENTURE_CAPITAL/CONTACT.md).

---

*Last updated: 2025-12-24*
