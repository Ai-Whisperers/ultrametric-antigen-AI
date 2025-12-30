# Carlos Brizuela - 10 Research Ideas

> **Future research directions for Antimicrobial Peptide Optimization**

**Document Version:** 1.0
**Last Updated:** December 29, 2025

---

## Overview

These 10 research ideas build on Carlos Brizuela's work on multi-objective antimicrobial peptide (AMP) optimization using NSGA-II in the Ternary VAE latent space, StarPepDB integration, and computational peptide design.

---

## Idea 1: Pathogen-Specific AMP Designer

### Concept
Train separate toxicity/activity predictors for specific pathogens and use targeted NSGA-II optimization to design AMPs against priority WHO pathogens.

### Target Pathogens (WHO Priority)
| Priority | Pathogen | Current AMPs | Resistance Level |
|----------|----------|--------------|------------------|
| Critical | *Acinetobacter baumannii* | Few | Carbapenem-resistant |
| Critical | *Pseudomonas aeruginosa* | Limited | MDR |
| Critical | *Enterobacteriaceae* | Polymyxins | Carbapenem-resistant |
| High | *Staphylococcus aureus* | Several | MRSA |
| High | *Helicobacter pylori* | Few | Clarithromycin-resistant |

### Technical Approach
```python
class PathogenSpecificOptimizer:
    def __init__(self, pathogen):
        self.pathogen = pathogen
        self.activity_model = load_model(f"activity_{pathogen}.pt")
        self.toxicity_model = load_model("universal_toxicity.pt")

    def optimize(self, population_size=200, generations=100):
        objectives = [
            self.reconstruction_loss,
            self.toxicity_model.predict,
            lambda z: -self.activity_model.predict(z)  # Maximize activity
        ]
        return NSGA2(objectives).run()
```

### Expected Output
- Pareto fronts per pathogen
- Top 10 candidates per pathogen for synthesis
- Cross-pathogen activity analysis

---

## Idea 2: Biofilm-Penetrating AMPs

### Concept
Optimize AMPs specifically for biofilm penetration by adding a biofilm disruption objective to the multi-objective optimization.

### The Biofilm Challenge
- Bacteria in biofilms are 10-1000x more resistant
- Most AMPs fail to penetrate the EPS matrix
- Biofilms cause 80% of chronic infections

### New Objectives
| Objective | Description | Measurement |
|-----------|-------------|-------------|
| **Biofilm penetration** | Ability to traverse EPS | MBEC assay |
| **Anti-attachment** | Prevent biofilm formation | Crystal violet assay |
| **Dispersal** | Break existing biofilms | Biomass reduction |
| **Low toxicity** | Safe for host | Hemolysis < 10% |

### Latent Space Strategy
```
Standard AMPs ────────────────────────────► High activity
                                           Low penetration
                     │
                     │ NSGA-II with
                     │ biofilm objectives
                     ▼
Biofilm AMPs ────────────────────────────► Moderate activity
                                           HIGH PENETRATION
```

### Validation Pipeline
1. Computational prediction (latent optimization)
2. *In vitro* biofilm assays (*P. aeruginosa* PAO1)
3. Confocal microscopy (penetration depth)
4. Animal model (catheter biofilm)

---

## Idea 3: Synergistic AMP Pairs

### Concept
Use multi-objective optimization to find pairs of AMPs that work synergistically, achieving better efficacy at lower individual doses.

### Synergy Types
| Type | Mechanism | Benefit |
|------|-----------|---------|
| **Sequential** | Pore former + intracellular | Enhanced killing |
| **Complementary** | Different targets | Reduced resistance |
| **Potentiating** | Membrane disruptor + antibiotic | Lower doses |

### Optimization Approach
```python
def synergy_score(z1, z2):
    """Compute synergy between two latent vectors."""
    seq1 = vae.decode(z1)
    seq2 = vae.decode(z2)

    # Individual activities
    mic1 = activity_model.predict(seq1)
    mic2 = activity_model.predict(seq2)

    # Combination activity (checkerboard prediction)
    mic_combo = combination_model.predict(seq1, seq2)

    # FIC index (< 0.5 = synergy)
    fic = mic_combo / mic1 + mic_combo / mic2
    return fic

# Optimize for low FIC
objectives = [
    synergy_score,
    lambda z1, z2: toxicity(z1) + toxicity(z2),
    lambda z1, z2: reconstruction_loss(z1) + reconstruction_loss(z2)
]
```

### Clinical Application
- Synergistic pairs for combination therapy
- Lower individual doses = reduced toxicity
- Delay resistance emergence

---

## Idea 4: Orally Bioavailable AMPs

### Concept
Add stability and permeability objectives to design AMPs that can be administered orally instead of requiring injection.

### Oral Delivery Challenges
| Challenge | Typical AMP | Goal |
|-----------|-------------|------|
| Protease stability | t½ < 30 min | t½ > 4 hours |
| Membrane permeability | Low | High (Papp > 1×10⁻⁶ cm/s) |
| Efflux susceptibility | High | Low |
| First-pass metabolism | High | < 50% |

### New Latent Space Objectives
```python
objectives = [
    antimicrobial_activity,      # MIC
    host_toxicity,               # Hemolysis
    protease_stability,          # Half-life prediction
    permeability_score,          # Caco-2 Papp prediction
    reconstruction_loss          # Sequence validity
]
```

### Peptide Engineering Strategies
- D-amino acid incorporation (predicted from latent)
- Cyclization propensity
- N-methylation sites
- Lipophilicity optimization

---

## Idea 5: Immunomodulatory AMP Design

### Concept
Design AMPs with dual function: direct antimicrobial activity PLUS beneficial immunomodulatory effects.

### Immunomodulatory Targets
| Effect | Mechanism | Benefit |
|--------|-----------|---------|
| Chemotaxis | Recruit immune cells | Enhanced clearance |
| Cytokine modulation | Balance inflammation | Reduced tissue damage |
| Neutralize LPS | Prevent septic shock | Lower mortality |
| Wound healing | Stimulate repair | Faster recovery |

### Multi-Objective Framework
```
5-Objective Optimization:
┌─────────────────────────────────────────────────────────────┐
│ 1. Antimicrobial activity (MIC)                             │
│ 2. Host toxicity (hemolysis)                                │
│ 3. LPS neutralization (LAL assay prediction)                │
│ 4. Chemotactic index (monocyte migration prediction)        │
│ 5. Wound healing score (keratinocyte proliferation)         │
└─────────────────────────────────────────────────────────────┘
```

### Validation Assays
- TNF-α/IL-6 release in THP-1 cells
- Transwell chemotaxis assay
- LAL endotoxin neutralization
- Scratch wound healing assay

---

## Idea 6: Resistance-Proof AMP Engineering

### Concept
Use evolutionary simulation in latent space to identify AMPs that bacteria cannot easily evolve resistance against.

### Resistance Mechanisms to Evade
| Mechanism | Bacterial Adaptation | Counter-Strategy |
|-----------|----------------------|------------------|
| Membrane modification | LPS/phospholipid changes | Multi-target design |
| Proteolytic degradation | Peptidase upregulation | D-aa incorporation |
| Efflux pumps | Transporter expression | Low pump affinity |
| Biofilm formation | EPS overproduction | Penetrating design |

### Evolutionary Stability Score
```python
def resistance_stability(z, n_generations=100):
    """Simulate bacterial evolution against AMP."""
    current_amp = vae.decode(z)

    for gen in range(n_generations):
        # Simulate bacterial mutations
        resistant_pop = evolve_bacteria(current_amp)

        # Check if resistance emerged
        if resistance_level(resistant_pop) > threshold:
            return gen  # Lower = easier to resist

    return n_generations  # Higher = harder to resist

# Optimize for HIGH resistance stability
objectives = [
    lambda z: -resistance_stability(z),  # Maximize stability
    activity_score,
    toxicity_score
]
```

### Validation
- Serial passage experiments (30+ generations)
- Genomic analysis of escape mutants
- Cross-resistance testing

---

## Idea 7: AMP-Antibiotic Hybrid Designer

### Concept
Design peptide-antibiotic conjugates by optimizing linker sequences and attachment points in latent space.

### Hybrid Architecture
```
[AMP Core] ─── [Linker] ─── [Antibiotic]

Linker options:
├── Cleavable (enzyme-sensitive)
├── Stable (permanent conjugate)
└── pH-sensitive (tumor microenvironment)
```

### Antibiotic Partners
| Antibiotic | Mechanism | Synergy Potential |
|------------|-----------|-------------------|
| Ciprofloxacin | DNA gyrase | Membrane + intracellular |
| Vancomycin | Cell wall | Dual wall attack |
| Rifampicin | RNA polymerase | Penetration + intracellular |
| Metronidazole | DNA damage | Anaerobes |

### Optimization Variables
```python
class HybridOptimizer:
    def __init__(self):
        self.amp_latent = z_amp  # 16D
        self.linker_latent = z_linker  # 8D
        self.antibiotic_choice = categorical  # 1-of-N

    def objective(self, z_combined):
        z_amp, z_linker, ab_idx = split(z_combined)
        hybrid = construct_hybrid(z_amp, z_linker, ab_idx)
        return [
            activity(hybrid),
            toxicity(hybrid),
            stability(hybrid),
            synthesis_cost(hybrid)
        ]
```

---

## Idea 8: Skin Microbiome-Safe AMPs

### Concept
Design AMPs that kill pathogens but spare beneficial skin microbiome members (*S. epidermidis*, *Cutibacterium acnes*).

### Selectivity Requirements
| Target | Desired MIC | Rationale |
|--------|-------------|-----------|
| *S. aureus* | < 4 μg/mL | Kill pathogen |
| MRSA | < 8 μg/mL | Kill resistant pathogen |
| *S. epidermidis* | > 64 μg/mL | Spare commensal |
| *C. acnes* | > 64 μg/mL | Spare commensal |
| *Malassezia* | > 64 μg/mL | Spare commensal |

### Selectivity Index
```python
def selectivity_index(z):
    seq = vae.decode(z)
    mic_pathogen = predict_mic(seq, "S_aureus")
    mic_commensal = predict_mic(seq, "S_epidermidis")
    return mic_commensal / mic_pathogen  # Higher = more selective

objectives = [
    lambda z: predict_mic(z, "S_aureus"),      # Minimize pathogen MIC
    lambda z: -predict_mic(z, "S_epidermidis"), # Maximize commensal MIC
    toxicity,
    reconstruction_loss
]
```

### Application
- Acne treatment (spare *C. acnes*)
- Wound care (spare protective microbiome)
- Atopic dermatitis management

---

## Idea 9: Temperature-Activated AMPs

### Concept
Design AMPs that are inactive at normal body temperature but activate at infection-associated elevated temperatures (fever).

### Temperature Activation Mechanism
```
37°C (Normal) ──────────────────► Inactive (folded, buried active site)
       │
       │  Fever / Infection
       ▼
39-40°C (Elevated) ─────────────► Active (unfolded, exposed)
```

### Design Strategy
- Incorporate temperature-sensitive motifs
- Optimize melting temperature (Tm) in latent space
- Balance stability vs. activity

### Latent Space Features
```python
def temperature_objective(z, T_activation=39.0):
    seq = vae.decode(z)

    # Activity at different temperatures
    activity_37 = predict_activity_at_T(seq, 37.0)
    activity_39 = predict_activity_at_T(seq, 39.0)

    # Activation ratio (higher = better switch)
    ratio = activity_39 / (activity_37 + 0.1)

    return ratio

objectives = [
    temperature_objective,      # Maximize activation
    lambda z: predict_activity_at_T(z, 39.0),  # High activity when needed
    toxicity,
    reconstruction_loss
]
```

### Advantages
- Reduced off-target effects in healthy tissue
- Self-targeting to infection sites
- Reduced systemic toxicity

---

## Idea 10: AI-Guided Peptide Synthesis Optimization

### Concept
Use the VAE latent space to guide solid-phase peptide synthesis (SPPS) by predicting synthesis difficulty and optimizing sequences for manufacturability.

### Synthesis Challenges
| Challenge | Cause | Prediction Target |
|-----------|-------|-------------------|
| Aggregation | Hydrophobic stretches | Aggregation propensity |
| Deletion peptides | Steric hindrance | Coupling efficiency |
| Racemization | Base-sensitive residues | Epimerization risk |
| Aspartimide formation | Asp-Xxx motifs | Side reaction probability |

### Synthesis Score
```python
def synthesis_difficulty(z):
    seq = vae.decode(z)

    scores = {
        'aggregation': predict_aggregation(seq),
        'coupling': predict_coupling_efficiency(seq),
        'racemization': predict_racemization(seq),
        'aspartimide': predict_aspartimide(seq),
        'cost': estimate_cost(seq)
    }

    return weighted_sum(scores)

# 4-objective optimization
objectives = [
    antimicrobial_activity,
    toxicity,
    synthesis_difficulty,  # NEW: manufacturability
    reconstruction_loss
]
```

### Industrial Application
- Reduce synthesis failures
- Lower production costs
- Faster time-to-clinic
- Scale-up prediction

---

## Summary Table

| # | Idea | Novelty | Feasibility | Impact | Priority |
|---|------|---------|-------------|--------|----------|
| 1 | Pathogen-Specific Design | Medium | High | High | 1 |
| 2 | Biofilm-Penetrating AMPs | High | Medium | Very High | 2 |
| 3 | Synergistic AMP Pairs | High | Medium | High | 3 |
| 4 | Orally Bioavailable AMPs | High | Low | Very High | 4 |
| 5 | Immunomodulatory Design | Medium | Medium | High | 5 |
| 6 | Resistance-Proof AMPs | Very High | Medium | Very High | 6 |
| 7 | AMP-Antibiotic Hybrids | High | Medium | High | 7 |
| 8 | Microbiome-Safe AMPs | Medium | High | High | 8 |
| 9 | Temperature-Activated AMPs | Very High | Low | Medium | 9 |
| 10 | Synthesis Optimization | Medium | High | Medium | 10 |

---

## Implementation Roadmap

### Phase 1 (Months 1-3): Foundation
- Implement Idea #1 (Pathogen-Specific) with 3 pathogens
- Train pathogen-specific activity predictors
- Validate 10 candidates per pathogen

### Phase 2 (Months 4-6): Advanced Objectives
- Add biofilm penetration objective (Idea #2)
- Implement synergy scoring (Idea #3)
- Begin experimental validation

### Phase 3 (Months 7-12): Translation
- Synthesis optimization integration (Idea #10)
- Partner with peptide manufacturer
- First clinical candidates

---

*Ideas developed based on the Ternary VAE Bioinformatics Partnership*
*For antimicrobial peptide research and development*
