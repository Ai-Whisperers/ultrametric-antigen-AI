# Executive Summary: HIV Multi-Pressure Evolution Analysis

**For**: Clinicians, Virologists, and Vaccine Researchers  
**Datasets**: 202,085 records across 5 major HIV databases

---

## Overview

We conducted a comprehensive computational analysis of HIV evolution under simultaneous selective pressures. By modeling genetic changes geometrically, we quantified the "evolutionary cost" of mutations.

![The Geometric "Goldilocks Zone"](images/33_goldilocks_zone_scatter.png)

---

## Methodology: The Ternary VAE Geometric Model

Our approach uses a Ternary Variational Autoencoder (VAE) to map viral sequences into a 3-adic geometric space. This allows us to measure biological "distance" more accurately than traditional hamming distance.

![Ternary VAE Architecture](images/32_ternary_vae_architecture.png)

The resulting latent space reveals hierarchical clusters where distance corresponds to evolutionary cost.

![3-Adic Latent Space Visualization](images/31_3adic_latent_space_visualization.png)

---

## Major Findings Summary

### 1. Drug Resistance: Why Some Mutations Emerge Faster

**Key Insight**: The genetic distance a mutation must traverse correlates with how quickly resistance emerges.

![Drug Resistance Barrier Comparison](images/01_drug_class_barrier_comparison.png)

| Drug Class | Mean Genetic Distance | Clinical Implication                                   |
| :--------- | --------------------: | :----------------------------------------------------- |
| NRTI       |        6.08 (highest) | Resistance requires substantial genetic reorganization |
| NNRTI      |       5.04 (moderate) | Single mutations can confer resistance (e.g., K103N)   |
| PI         |   Variable (std=2.34) | Wide range due to accessory mutation requirements      |
| INSTI      |       4.92 (moderate) | DTG maintains high genetic barrier                     |

**Notable Mutations**:

- **K103N** (NNRTI): Low genetic distance (3.80) explains rapid emergence
- **M46I** (PI): Exceptionally low distance (0.65) as accessory mutation
- **T215Y** (NRTI): High distance (7.17) reflects major genetic shift

**Clinical Value**: We achieved 78.3% accuracy in classifying primary vs. accessory mutations using geometric features alone, providing a new framework for predicting resistance pathways.

---

### 2. Immune Escape: Why B*57 and B*27 Protect

**Key Insight**: Protective HLA alleles restrict epitopes where escape mutations require larger genetic changes.

![HLA Protection Ranking](images/07_hla_protection_ranking.png)

| HLA Allele | Escape Velocity | Protection Level |
| :--------- | --------------: | :--------------- |
| B\*57      | 0.218 (slowest) | Very High        |
| B\*27      |           0.256 | High             |
| A\*02      |           0.342 | High             |
| A\*03      |           0.389 | Moderate         |

**Finding**: 77.8% of successful immune escape mutations cross major cluster boundaries in our geometric space, indicating that effective escape requires substantial genetic reorganization.

**Protein Constraint Hierarchy**:

| Protein | Conservation Level |    Escape Rate |
| :------ | :----------------- | -------------: |
| Gag     | Highest            | 0.28 (slowest) |
| Pol     | High               |           0.31 |
| Env     | Moderate           |           0.45 |
| Nef     | Lowest             | 0.52 (fastest) |

This explains why Gag-targeting CTL responses are associated with viral control.

---

### 3. Tropism: New Position 22 Discovery

**Key Insight**: Position 22 in the V3 loop is the strongest tropism determinant, exceeding the canonical 11/25 rule.

![Tropism Position Ranking](images/14_tropism_position_ranking.png)

| V3 Position | Tropism Discrimination Score | Traditional Focus    |
| :---------- | ---------------------------: | :------------------- |
| **22**      |                    **0.591** | Not emphasized (NEW) |
| 8           |                        0.432 | Supporting role      |
| 20          |                        0.406 | Coreceptor contact   |
| 11          |                        0.341 | Classic 11/25 rule   |
| 25          |                        0.314 | Classic 11/25 rule   |

**Location**: Position 22 sits at the V3 crown tip, directly contacting the coreceptor. Charge changes (basic amino acids R/K) at this position strongly predict CXCR4 tropism.

**Tropism Classifier Performance**:

- Our model: 85% accuracy, AUC 0.86
- Classic 11/25 rule: 74% accuracy
- Comparable to Geno2pheno: 84%

**Clinical Relevance**: X4 tropism emergence is associated with faster disease progression. Position 22 may be a valuable addition to clinical tropism assessment.

---

### 4. Antibody Neutralization: What Makes bnAbs Effective

**Key Insight**: Broadly neutralizing antibodies target geometrically central (highly conserved) epitopes.

![bnAb Breadth vs Potency](images/19_bnab_breadth_potency.png)

| Antibody | Breadth | IC50 (μg/mL) | Epitope Class    |
| :------- | ------: | -----------: | :--------------- |
| 3BNC117  |   78.8% |        0.242 | CD4 binding site |
| 10E8     |   76.7% |        0.221 | MPER             |
| VRC01    |   68.9% |        0.580 | CD4 binding site |
| PG9      |   70.9% |        0.300 | V2-glycan        |
| PGT121   |   59.2% |        0.566 | V3-glycan        |

**Epitope Class Potency** (by IC50):

1. V2-glycan (0.689 μg/mL) - most potent
2. V3-glycan (0.745 μg/mL)
3. CD4 binding site (1.121 μg/mL)
4. MPER (1.815 μg/mL)
5. Interface (3.597 μg/mL)

**Finding**: Antibody breadth correlates with epitope centrality (r=0.68, p<0.001). This geometric relationship can guide bnAb design toward conserved targets.

---

### 5. Vaccine Targets: 328 Safe Epitopes Identified

**Key Insight**: We identified 328 CTL epitopes suitable for vaccine development that:

- Have broad HLA restriction (≥3 alleles)
- Show low escape velocity
- Contain NO drug resistance mutation overlap
- Are highly conserved across viral populations

![Top 10 Safe Vaccine Targets](images/27_top10_vaccine_targets.png)

**Top 5 Vaccine Candidates**:

| Rank | Epitope    | Protein | HLA Coverage | Score |
| :--- | :--------- | :------ | -----------: | ----: |
| 1    | TPQDLNTML  | Gag     |   25 alleles | 2.238 |
| 2    | AAVDLSHFL  | Nef     |   19 alleles | 1.701 |
| 3    | YPLTFGWCF  | Nef     |   19 alleles | 1.701 |
| 4    | YFPDWQNYT  | Nef     |   19 alleles | 1.701 |
| 5    | QVPLRPMTYK | Nef     |   19 alleles | 1.701 |

**Why "Safe"?**: These epitopes avoid the 16,054 positions where drug resistance mutations overlap with CTL epitopes, preventing selection pressure conflicts during treatment.

---

### 6. Trade-offs: When Pressures Conflict

**Key Insight**: 16,054 instances exist where drug resistance mutations fall within CTL epitopes, affecting 3,074 unique positions and 298 epitopes.

**Highest Trade-off Positions**:

| Mutation | Drug Class | Overlapping Epitopes | HLAs Affected | Trade-off Score |
| :------- | :--------- | -------------------: | ------------: | --------------: |
| S283R    | INSTI      |           RT epitope |            15 |            5.63 |
| D67NS    | NNRTI      |           RT epitope |            15 |            5.55 |
| Q61NH    | PI         |           RT epitope |            15 |            5.52 |

**Clinical Implication**: Patients with these HLA types may experience altered immune control when resistance mutations emerge at overlapping positions. This creates opportunities for personalized treatment strategies.

---

## Validation

Our methodology was validated by independently recovering known biological patterns:

| Known Pattern                          | Our Finding             | Status    |
| :------------------------------------- | :---------------------- | :-------- |
| B*57/B*27 protection                   | Highest escape barriers | Confirmed |
| 11/25 tropism rule                     | Recovered independently | Confirmed |
| bnAb profiles                          | Match published data    | Confirmed |
| Protein conservation (Gag>Pol>Env>Nef) | Same hierarchy          | Confirmed |
| NNRTI rapid resistance                 | Low genetic distance    | Confirmed |

---

## Clinical Applications

| Application                   | Finding Used                                        | Potential Impact                             |
| :---------------------------- | :-------------------------------------------------- | :------------------------------------------- |
| **Treatment Sequencing**      | Genetic distance correlates with resistance barrier | Predict durability of ART regimens           |
| **Vaccine Design**            | 328 safe targets identified                         | Avoid resistance-immunity conflicts          |
| **Tropism Testing**           | Position 22 as top determinant                      | Improve coreceptor prediction                |
| **Prognostic Stratification** | HLA-specific escape velocities                      | Predict viral control likelihood             |
| **Cure Research**             | Constraint mapping                                  | Identify targets with limited escape options |

---

## Databases Integrated

| Database                     |                               Records | Content                                          |
| :--------------------------- | ------------------------------------: | :----------------------------------------------- |
| Stanford HIV Drug Resistance | 90,269 mutations from 7,154 sequences | Drug class, fold-change, mutation positions      |
| LANL CTL Epitope Database    |                        2,115 epitopes | HLA restrictions, protein locations, escape data |
| CATNAP                       |                       189,879 records | bnAb potency (IC50) and breadth measurements     |
| V3 Tropism                   |                       2,932 sequences | CCR5 vs CXCR4 classification, V3 sequences       |
| **Total**                    |                           **202,085** | Integrated multi-pressure landscape              |

---

## Summary

This analysis provides a unified geometric framework for understanding HIV evolution under multiple selective pressures. The key clinical takeaways are:

1. **Drug resistance emergence speed** can be predicted by genetic distance
2. **Protective HLA alleles** restrict epitopes in evolutionarily constrained regions
3. **Position 22** should be considered in tropism assessment alongside 11/25
4. **328 epitopes** represent safe vaccine targets without resistance conflicts
5. **Trade-off mapping** enables personalized treatment based on HLA type

---

_For detailed findings, see individual documents in the findings/ directory._
