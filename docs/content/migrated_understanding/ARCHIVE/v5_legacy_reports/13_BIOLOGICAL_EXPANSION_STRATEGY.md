# Biological Expansion Strategy: Multi-Organism Integration

**Date:** 2025-12-27
**Scope:** Consolidation of Biological Expansion Analysis and Next Steps

---

## 1. Executive Summary

This document establishes the roadmap for expanding the Ternary VAE platform from a single-organism (HIV) tool to a **universal biological distance platform**.

Our analysis confirms that the p-adic/hyperbolic/tropical framework is mathematically sound and validated for HIV (202K+ records). The immediate opportunity is to leverage this framework to analyze **HBV, HCV, Influenza, SARS-CoV-2, TB, and Malaria**, supported by 8 existing but unintegrated advanced technical modules.

### Key Targets

- **Organisms**: HIV → HBV, HCV, Flu, COVID, TB, Malaria
- **Proteins**: Antibodies, TCRs, Kinases, GPCRs
- **Validation**: AlphaFold3 structure prediction, Cross-organism transfer
- **Architecture**: Unified `TropicalHyperbolicVAE`

---

## 2. Technical Capabilities & Gaps

### Current Validated State (HIV)

| Domain              | Capabilities    | Validation                |
| :------------------ | :-------------- | :------------------------ |
| **Drug Resistance** | 7,154 records   | r = 0.41 (Stanford HIVDB) |
| **Tropism**         | 2,932 records   | 85% Accuracy (V3 loop)    |
| **Neutralization**  | 189,879 records | r = 0.83 (CATNAP)         |

### Ready-to-Integrate Modules (Existing Code)

We have 8 advanced modules coded but not in the main pipeline. **Priority 1** is their integration.

1.  **Tropical Geometry** (`src/tropical/`): Phylogenetic tree inference without multiple alignment.
2.  **Persistent Homology** (`src/topology/`): Protein shape fingerprints (H0, H1, H2 features).
3.  **Information Geometry** (`src/information/`): Natural gradient optimization.
4.  **Statistical Physics** (`src/physics/`): Fitness landscape modeling (energy = fitness).
5.  **P-adic Contrastive** (`src/contrastive/`): Self-supervised pretraining.
6.  **Meta-Learning** (`src/meta/`): MAML for rapid pandemic adaptation.

---

## 3. Multi-Organism Expansion Plan

### Phase 1: Viral Expansion (Weeks 1-4)

#### A. Hepatitis B (HBV)

- **Source**: HBVdb (~10k sequences).
- **Why**: Overlapping reading frames (S/P genes) create strong p-adic constraints.
- **Validation**: Polymerase gene mutations (Lamivudine/Tenofovir).

#### B. Influenza A/B

- **Source**: GISAID/FluDB.
- **Why**: Antigenic drift is a "gradual p-adic movement"; Shift is a "jump".
- **Validation**: Antigenic cartography vs. P-adic distance.

#### C. SARS-CoV-2

- **Source**: GISAID (15M+ sequences).
- **Why**: Variant waves follow ultrametric tree structures.
- **Validation**: Pango lineage classification, Spike mutation clustering.

### Phase 2: Bacterial & Parasitic (Weeks 5-8)

#### A. Tuberculosis (TB)

- **Source**: BVBRC.
- **Why**: Slow evolution creates very distinct p-adic hierarchies.
- **Validation**: rpoB/katG mutations vs. WHO resistance catalog.

#### B. Malaria (P. falciparum)

- **Source**: MalariaGEN.
- **Why**: Strong geographic clustering fits hierarchical embedding.
- **Validation**: Artemisinin resistance markers (kelch13).

---

## 4. Technical Implementation Steps

### Step 1: Multi-Organism Data Loader (Immediate)

Create a unified loader that handles organism-specific "primes" (p=4 for nucleotides, p=20 for AA, p=64 for codons).

```python
from src.data.multi_organism import OrganismLoader, OrganismType

# HBV Loader (Prime=3 for codon overlap?)
hbv_loader = OrganismLoader(OrganismType.HBV)
sequences = hbv_loader.load_sequences()
# HIV Loader
hiv_loader = OrganismLoader(OrganismType.HIV)
```

### Step 2: Advanced Module Integration

Wire the `TropicalHyperbolicVAE` to use the topology and contrastive modules.

```python
from src.topology.persistent_homology import ProteinTopologyEncoder
from src.contrastive.padic_contrastive import PAdicContrastiveLoss

model = TropicalHyperbolicVAE(...)
# Add topology features to latent space
topology_encoder = ProteinTopologyEncoder(filtration_type="padic")
# Add contrastive loss for better structure
contrastive_loss = PAdicContrastiveLoss(temperature=0.1)
```

### Step 3: Meta-Learning for Rapid Response

Use MAML to adapt the HIV-trained model to new viruses (e.g., COVID) with few shots.

```python
from src.meta.meta_learning import MAML, PAdicTaskSampler

task_sampler = PAdicTaskSampler(organisms=[HIV, HBV, HCV, FLU])
maml = MAML(model, inner_lr=0.01)
# Train on knowns, adapt to novel pathogen
novel_model = maml.adapt(covid_data, n_steps=10)
```

---

## 5. Structural Validation (AlphaFold3)

**Goal**: Move beyond sequence to 3D structure validation.

- **Pipeline**: Predict structure with AlphaFold3 → Compute pLDDT → Correlate p-adic distance with structural RMSD.
- **Hypothesis**: Small p-adic changes can cause large structural changes (and vice versa); the VAE should capture "structural distance" not just "sequence edit distance".

---

## 6. Success Metrics (Target vs Current)

| Metric             | Current       | Target (3 Months)               |
| :----------------- | :------------ | :------------------------------ |
| **Organisms**      | 1 (HIV)       | **8+**                          |
| **Generalization** | N/A           | **70% Cross-organism Transfer** |
| **Structure**      | 0 Validations | **100+ Key Structures**         |
| **Pipeline**       | Manual        | **Unified API**                 |

---

## 7. Next Immediate Actions

1.  **Test Multi-Organism Framework**: Validate `src/data/multi_organism` with HBV data.
2.  **Run Hybrid VAE Tests**: Verify `TropicalHyperbolicVAE` performance on non-HIV data.
3.  **Deploy Curriculum**: Switch main training to the new `CurriculumTrainer` (Recon → KL → Structure).
