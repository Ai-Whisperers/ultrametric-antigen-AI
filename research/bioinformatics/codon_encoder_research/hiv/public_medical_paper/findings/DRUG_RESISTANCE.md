# Drug Resistance Findings

**Audience**: Clinicians, Pharmacologists, HIV Treatment Specialists  
**Data Source**: Stanford HIV Drug Resistance Database (7,154 sequences, 90,269 mutations)

---

## Summary

Our computational analysis reveals that the genetic "distance" a mutation must traverse correlates with clinical resistance levels and emergence speed. This provides a quantitative framework for understanding why some resistance mutations appear rapidly while others require prolonged selective pressure.

---

## Core Finding

**Correlation**: Genetic distance correlates with fold-change resistance (r = 0.34-0.41, p < 10^-50)

Mutations conferring higher-level resistance require larger genetic reorganization at the codon level. This "evolutionary cost" varies by drug class and helps explain clinical observations about resistance emergence patterns.

![Resistance Emergence Timeline](../images/04_resistance_emergence_timeline.png)

---

## Drug Class Analysis

![Genetic Barrier to Resistance by Drug Class](../images/01_drug_class_barrier_comparison.png)

### NRTI (Nucleoside Reverse Transcriptase Inhibitors)

| Metric         | Value | Clinical Interpretation                                  |
| :------------- | ----: | :------------------------------------------------------- |
| Mean Distance  |  6.08 | Highest barrier - requires substantial genetic change    |
| Notable: M184V |  5.67 | High resistance (3TC/FTC) despite moderate distance      |
| Notable: T215Y |  7.17 | Major genetic shift explains need for multiple mutations |
| Notable: K65R  |  5.52 | TDF resistance with moderate fitness cost                |

**Why highest distance?** NRTIs directly compete with natural nucleotides at the polymerase active site. Resistance mutations must alter binding while preserving enzymatic function, requiring coordinated codon-level changes.

**Clinical implication**: NRTI-based regimens tend to have durable efficacy because resistance emergence requires multiple sequential mutations, each with associated fitness costs.

---

### NNRTI (Non-Nucleoside Reverse Transcriptase Inhibitors)

| Metric         | Value | Clinical Interpretation                                |
| :------------- | ----: | :----------------------------------------------------- |
| Mean Distance  |  5.04 | Moderate barrier                                       |
| Notable: K103N |  3.80 | Low distance explains rapid single-mutation resistance |
| Notable: Y181C |  4.12 | Moderate distance with cross-resistance                |
| Notable: G190A |  3.89 | Low barrier mutation                                   |

**Why rapid resistance?** NNRTIs bind an allosteric pocket that doesn't require the same precision as the active site. Single mutations like K103N require minimal genetic reorganization while abolishing drug binding.

**Clinical implication**: The low genetic barrier of K103N (distance 3.80) explains why first-generation NNRTI resistance emerges rapidly under suboptimal adherence. Newer agents (doravirine, rilpivirine) require mutations with higher genetic distances.

---

### Protease Inhibitors (PI)

| Metric        |    Value | Clinical Interpretation                                  |
| :------------ | -------: | :------------------------------------------------------- |
| Mean Distance | Variable | Wide range reflects primary/accessory distinction        |
| Std Deviation |     2.34 | Highest variance of any drug class                       |
| Notable: M46I |     0.65 | Exceptionally low - explains ease of accessory emergence |
| Notable: D30N |     3.53 | Primary mutation with moderate barrier                   |
| Notable: I84V |     3.89 | Cross-resistance primary mutation                        |

**Primary vs. Accessory Mutations**:

| Type      | Mean Distance | Characteristic                               |
| :-------- | ------------: | :------------------------------------------- |
| Primary   |     4.2 ± 1.1 | Directly reduce drug binding                 |
| Accessory |     2.8 ± 1.5 | Compensate for primary mutation fitness cost |

**Clinical implication**: The high variance in PI genetic distances reflects the complex interplay between primary and accessory mutations. Boosted PI regimens maintain efficacy because accumulating the full resistance genotype requires traversing substantial total genetic distance.

---

### INSTI (Integrase Strand Transfer Inhibitors)

| Metric         | Value | Clinical Interpretation                        |
| :------------- | ----: | :--------------------------------------------- |
| Mean Distance  |  4.92 | Moderate-high barrier                          |
| Notable: Y143R |  5.08 | High barrier explains raltegravir durability   |
| Notable: R263K |  4.40 | DTG-associated, high barrier with fitness cost |
| Notable: N155H |  4.19 | First-generation INSTI pathway                 |

**Why DTG has high barrier?** Dolutegravir resistance requires mutations (like R263K) with high genetic distances AND substantial fitness costs. This dual barrier explains the exceptional durability observed clinically.

**Clinical implication**: DTG-based regimens represent optimal genetic barrier, requiring both large genetic distances and fitness-costly mutations for escape.

---

## Primary vs. Accessory Classification

![Primary vs Accessory Mutation Diagram](../images/03_primary_vs_accessory_diagram.png)

Using geometric features alone, we achieved **78.3% accuracy** in classifying mutations as primary or accessory:

| Feature            | Primary Mutations | Accessory Mutations |
| :----------------- | ----------------: | ------------------: |
| Geometric Position | Peripheral (edge) |  Central (interior) |
| Mean Radius        |       0.82 ± 0.11 |         0.64 ± 0.15 |
| Cluster Boundary   |       Often cross |        Usually same |

**Interpretation**: Primary mutations occupy peripheral positions in the evolutionary landscape, representing large jumps away from the conserved viral core. Accessory mutations cluster internally, making smaller adjustments to restore fitness.

---

## Cross-Resistance Patterns

![Cross-Resistance Heatmap](../images/05_cross_resistance_heatmap.png)

Geometric analysis reveals shared evolutionary pathways between drug classes:

| Drug Class Pair | Geometric Overlap | Clinical Cross-Resistance                 |
| :-------------- | ----------------: | :---------------------------------------- |
| NRTI-NRTI       |       High (0.78) | Well-documented (TAMs affect class)       |
| NNRTI-NNRTI     |   Moderate (0.61) | K103N, Y181C affect multiple agents       |
| PI-PI           |          Variable | Boosted PIs share resistance pathways     |
| INSTI-INSTI     |          Moderate | First-gen INSTIs share Q148/N155 pathways |
| NNRTI-NRTI      |        Low (0.23) | Limited cross-class overlap               |

---

## Clinical Applications

### Treatment Sequencing

![Geometric Barrier Clinical Decision Guide](../images/06_barrier_clinical_decision.png)

| Scenario                  | Genetic Distance Implication                     |
| :------------------------ | :----------------------------------------------- |
| High adherence expected   | Any regimen suitable                             |
| Adherence concerns        | Prefer high-distance mutations (DTG, boosted PI) |
| Prior NNRTI exposure      | Check for low-barrier mutations (K103N)          |
| Simplification candidates | Consider accumulated genetic distance            |

### Resistance Prediction

| Genetic Distance | Resistance Risk                       | Recommended Action          |
| :--------------- | :------------------------------------ | :-------------------------- |
| < 3.0            | High risk, rapid emergence            | Avoid if alternatives exist |
| 3.0 - 5.0        | Moderate risk                         | Standard monitoring         |
| > 5.0            | Low risk, requires multiple mutations | Durable option              |

---

## Key Mutation Reference Table

![Key Resistance Mutations Reference Card](../images/02_resistance_mutations_reference.png)

| Mutation | Drug Class | Distance | Fold-Change | Type      | Notes                            |
| :------- | :--------- | -------: | ----------: | :-------- | :------------------------------- |
| M184V    | NRTI       |     5.67 |        >100 | Primary   | 3TC/FTC resistance, fitness cost |
| K65R     | NRTI       |     5.52 |        5-10 | Primary   | TDF resistance                   |
| T215Y    | NRTI       |     7.17 |       10-50 | Primary   | TAM, requires precursor          |
| K103N    | NNRTI      |     3.80 |         >50 | Primary   | Low barrier, rapid emergence     |
| Y181C    | NNRTI      |     4.12 |       10-50 | Primary   | ETV cross-resistance             |
| M46I     | PI         |     0.65 |         2-5 | Accessory | Very low barrier                 |
| I84V     | PI         |     3.89 |        5-20 | Primary   | Multi-PI resistance              |
| R263K    | INSTI      |     4.40 |         2-5 | Primary   | DTG, severe fitness cost         |
| N155H    | INSTI      |     4.19 |       10-50 | Primary   | First-gen INSTI pathway          |

---

## Statistical Validation

| Metric                           |          Value | Significance            |
| :------------------------------- | -------------: | :---------------------- |
| Distance-Resistance Correlation  |  r = 0.34-0.41 | p < 10^-50              |
| Primary/Accessory Classification | 78.3% accuracy | Geometric features only |
| Cross-Resistance Prediction      |       AUC 0.82 | Independent validation  |

---

## Summary

1. **Genetic distance predicts resistance barrier**: Higher distance = more durable regimen
2. **Drug classes differ systematically**: NRTIs highest, NNRTIs lowest
3. **Primary/accessory distinction is geometric**: 78% classification accuracy
4. **Clinical utility**: Framework for treatment sequencing and resistance prediction

---

_Data derived from Stanford HIV Drug Resistance Database. Individual mutation data available upon request._
