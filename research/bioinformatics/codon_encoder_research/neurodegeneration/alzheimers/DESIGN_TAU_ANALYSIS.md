# Alzheimer's Tau Phosphorylation Analysis: Design Document

**Doc-Type:** Technical Design · Version 1.0 · Updated 2025-12-19 · Author AI Whisperers

---

## Conceptual Framework

### The Tau Problem vs The Viral Problem

| Aspect | Viral (SARS-CoV-2) | Tau (Alzheimer's) |
|:-------|:-------------------|:------------------|
| System | Two proteins (viral + host) | One protein (self) |
| Goal | Disrupt viral, preserve host | Understand dysfunction threshold |
| Modification | External perturbation | Internal state change |
| Outcome | Binding disruption | Aggregation propensity |
| Therapeutic | Block interaction | Restore function or prevent aggregation |

### Reframing Goldilocks for Neurodegeneration

In autoimmunity/viral contexts, Goldilocks Zone = **immune recognition threshold**.

In neurodegeneration, we propose a **Dysfunction Zone**:

```
Functional Tau                    Dysfunctional Tau
     │                                    │
     ▼                                    ▼
┌─────────┐    ┌─────────────┐    ┌──────────────┐
│  <15%   │    │   15-35%    │    │    >35%      │
│ Normal  │ →  │ Transition  │ →  │ Aggregation  │
│ binding │    │   zone      │    │    prone     │
└─────────┘    └─────────────┘    └──────────────┘
     │                │                   │
     ▼                ▼                   ▼
  Healthy        Early AD            Late AD
              (intervention          (rescue
               window)               difficult)
```

### Key Hypothesis

**Single phosphorylation** may cause small geometric shifts (<15%).
**Cumulative phosphorylation** pushes tau across dysfunction threshold.
**Specific combinations** may be synergistic "tipping points".

---

## Tau Biology

### Isoforms

| Isoform | Exons | Length | Brain Region |
|:--------|:------|:-------|:-------------|
| 0N3R | -2,-3,-10 | 352 aa | Fetal |
| 1N3R | +2,-3,-10 | 381 aa | Adult |
| 2N3R | +2,+3,-10 | 410 aa | Adult |
| 0N4R | -2,-3,+10 | 383 aa | Adult |
| 1N4R | +2,-3,+10 | 412 aa | Adult |
| **2N4R** | +2,+3,+10 | **441 aa** | Adult (longest) |

We will use **2N4R** (441 aa) as the reference - contains all domains.

### Functional Domains

```
N-terminus          Proline-rich       MTBR              C-terminus
  (1-150)            (151-243)       (244-369)           (370-441)
     │                   │               │                   │
     ▼                   ▼               ▼                   ▼
┌─────────┐    ┌─────────────────┐   ┌──────────┐    ┌─────────────┐
│ N1  N2  │    │  P1  P2  PRR   │   │ R1-R2-R3 │    │ Post-MTBR   │
│ inserts │    │ (SH3 binding)  │   │   -R4    │    │  region     │
└─────────┘    └─────────────────┘   └──────────┘    └─────────────┘
     │                   │               │                   │
Projection           Kinase          Microtubule        Aggregation
 domain             targets           binding             seed?
```

**MTBR** = Microtubule Binding Repeats (R1-R4)
- This is where tau binds tubulin
- Also the core of PHF (paired helical filament) aggregates

### Key Phosphorylation Sites (85 total S/T/Y)

**Early pathological markers**:
| Site | Epitope | Detection | Stage |
|:-----|:--------|:----------|:------|
| T181 | AT270 | CSF biomarker | Early |
| S202 | CP13 | Early tangles | Early |
| T205 | - | Part of AT8 | Early |
| S202+T205 | **AT8** | Gold standard | Early-Mid |
| T231 | TG3 | Conformational | Early |

**Late pathological markers**:
| Site | Epitope | Detection | Stage |
|:-----|:--------|:----------|:------|
| S262 | 12E8 | MTBR disruption | Mid |
| S396 | PHF-13 | PHF formation | Mid-Late |
| S404 | - | Part of PHF-1 | Late |
| S396+S404 | **PHF-1** | PHF marker | Late |
| S422 | - | Severe pathology | Late |

---

## Analysis Architecture

### Module 1: Single-Site Phosphorylation Sweep

For each of 85 S/T/Y sites:
1. Extract 15-mer context window
2. Encode wild-type with 3-adic encoder
3. Apply S→D, T→D, or Y→D (phosphomimic)
4. Compute centroid shift in hyperbolic space
5. Classify: <15% (tolerated), 15-35% (transition), >35% (disruptive)

**Output**: Ranked list of sites by geometric perturbation magnitude

### Module 2: Domain-Specific Analysis

Analyze phosphorylation impact by functional domain:
- **N-terminal projection**: How does phospho affect membrane interaction?
- **Proline-rich region**: Kinase docking sites, signaling
- **MTBR**: Microtubule binding - most critical for function
- **C-terminal**: Aggregation seeding region

### Module 3: Tau-Microtubule Interface Handshake

Encode both tau MTBR and tubulin binding surface:
1. Identify tau residues that contact tubulin (from cryo-EM structures)
2. Compute handshake geometry (like RBD-ACE2)
3. Test how phosphorylation at MTBR sites disrupts interface

**Key sites in MTBR**:
- S262 (R1) - Major detachment site
- S293 (R2)
- S324 (R3)
- S356 (R4)

### Module 4: Combinatorial Phosphorylation

Test pathologically relevant combinations:
| Combination | Name | Analysis |
|:------------|:-----|:---------|
| S202 + T205 | AT8 | Encode double-phospho |
| S396 + S404 | PHF-1 | PHF formation signature |
| T231 + S235 | TG3/AT180 | Conformational epitope |
| S202 + T205 + S208 | AT8 extended | Triple phospho |
| Full MTBR | All R1-R4 sites | Maximum disruption |

**Question**: Is geometric shift additive or synergistic?

### Module 5: Trajectory Analysis with VAE

Use ternary VAE (V5.11.3) to:
1. Encode tau sequences with varying phosphorylation states
2. Project to latent space
3. Visualize trajectory: Normal → Hyperphosphorylated
4. Identify "point of no return" in latent space

---

## Implementation Strategy

### Reuse from Existing Scripts

| Component | Source | Adaptation |
|:----------|:-------|:-----------|
| Encoder loading | `hyperbolic_utils.py` | Direct reuse |
| Context extraction | `01_spike_sentinel_analysis.py` | Modify window |
| Centroid shift | All scripts | Direct reuse |
| Asymmetric analysis | `03_deep_handshake_sweep.py` | Adapt for tau-tubulin |
| JSON output | All scripts | Direct reuse |

### New Components Needed

1. **Tau sequence database**: Full 2N4R sequence + all phospho-sites
2. **Tubulin interface**: Binding surface residues from structures
3. **Combinatorial generator**: Multi-site phosphorylation combinations
4. **Trajectory visualizer**: VAE latent space projection

### File Structure

```
research/bioinformatics/neurodegeneration/alzheimers/
├── DESIGN_TAU_ANALYSIS.md          # This document
├── 01_tau_phospho_sweep.py         # Single-site analysis
├── 02_tau_mtbr_interface.py        # Microtubule binding analysis
├── 03_tau_combinatorial.py         # Multi-site combinations
├── 04_tau_vae_trajectory.py        # VAE latent space analysis
├── data/
│   ├── tau_2n4r_sequence.fasta     # Reference sequence
│   ├── tau_phospho_sites.json      # All 85 sites + metadata
│   └── tubulin_interface.json      # Binding surface residues
├── results/
│   ├── single_site_results.json
│   ├── interface_analysis.json
│   ├── combinatorial_results.json
│   └── trajectory_analysis.json
└── FINDINGS.md                      # Results documentation
```

---

## Expected Outputs

### 1. Phospho-Site Risk Map

| Site | Domain | Shift | Zone | Known Pathology | Risk Score |
|:-----|:-------|:------|:-----|:----------------|:-----------|
| S262 | MTBR-R1 | 28% | Transition | 12E8 epitope | HIGH |
| T231 | PRR | 22% | Transition | TG3 epitope | HIGH |
| ... | ... | ... | ... | ... | ... |

### 2. Tipping Point Combinations

Identify which combinations cause non-linear (synergistic) geometric shifts:
- If S202 alone = 12% and T205 alone = 11%
- But S202+T205 = 35% (not 23%)
- Then AT8 is a **synergistic tipping point**

### 3. Intervention Targets

Sites where **dephosphorylation** (D→S reverse) would maximally restore functional geometry:
- Priority targets for phosphatase activation
- Or kinase inhibition targets

### 4. Biomarker Geometry

Correlate CSF biomarker sites (T181, T217) with geometric signatures:
- Do early biomarker sites have different geometric profiles than late sites?

---

## Validation Strategy

### Computational

1. AlphaFold3: Predict tau structure with/without phosphorylation
2. Literature correlation: Match predictions to known pathological sites

### Experimental (Future)

1. Phospho-mimetic tau constructs (S→D mutations)
2. Microtubule binding assays
3. Aggregation propensity assays
4. Cell-based tau spreading assays

---

## Key Questions to Answer

1. **Which single phospho-site causes the largest geometric shift?**
2. **Is there a threshold number of phosphorylations that triggers dysfunction?**
3. **Are AT8 and PHF-1 epitopes geometric "tipping points"?**
4. **Which MTBR sites most disrupt microtubule binding geometry?**
5. **Can we identify a "point of no return" in the phosphorylation trajectory?**
6. **Which sites are best targets for therapeutic intervention?**

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-19 | 1.0 | Initial design document |
