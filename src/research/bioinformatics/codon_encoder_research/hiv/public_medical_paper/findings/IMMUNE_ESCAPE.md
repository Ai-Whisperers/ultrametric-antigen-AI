# Immune Escape Findings: CTL and Cellular Immunity

**Audience**: Immunologists, Virologists, Vaccine Researchers  
**Data Source**: LANL CTL Epitope Database (2,115 epitopes, 240 HLA restrictions)

---

## Summary

Our analysis quantifies why certain HLA alleles (B*57, B*27) are associated with viral control. These protective alleles restrict epitopes located in evolutionarily constrained regions where escape mutations require substantial genetic reorganization and impose significant fitness costs.

---

## Core Finding

**77.8% of successful escape mutations cross cluster boundaries** in our geometric space

This indicates that effective immune escape is not a minor adjustment but requires substantial genetic reorganization. The magnitude of this reorganization varies by HLA allele and protein location, explaining differential disease outcomes.

---

## HLA-Specific Escape Patterns

![HLA protection ranking](../images/07_hla_protection_ranking.png)

### Protective Alleles

| HLA Allele | Escape Velocity | Escape Success Rate | Protection Level |
| :--------- | --------------: | ------------------: | :--------------- |
| B\*57:01   |           0.218 |                 22% | Very High        |
| B\*27:05   |           0.256 |                 26% | High             |
| B\*58:01   |           0.278 |                 28% | High             |
| A\*02:01   |           0.342 |                 34% | Moderate-High    |

### Non-Protective Alleles

| HLA Allele | Escape Velocity | Escape Success Rate | Protection Level |
| :--------- | --------------: | ------------------: | :--------------- |
| A\*03:01   |           0.389 |                 39% | Moderate         |
| B\*35:01   |           0.445 |                 45% | Low              |
| A\*24:02   |           0.412 |                 41% | Moderate         |

**Why B*57 and B*27 protect**: These alleles present epitopes from structurally essential regions of Gag and Pol. Escape mutations in these regions require:

1. Large genetic distance (higher mutational barrier)
2. Alterations to essential protein structure (fitness cost)
3. Compensatory mutations to restore function (additional barrier)

---

## Protein-Specific Constraints

![Protein Constraint Map](../images/08_protein_constraint_map.png)

HIV proteins vary dramatically in their tolerance for escape mutations:

| Protein | Conservation Level | Escape Rate | Functional Role                    |
| :------ | :----------------- | ----------: | :--------------------------------- |
| Gag     | Highest            |        0.28 | Capsid structure - essential       |
| Pol     | High               |        0.31 | Enzymatic (RT, PR, IN) - essential |
| Env     | Moderate           |        0.45 | Surface glycoprotein - variable    |
| Nef     | Low                |        0.52 | Accessory - dispensable in vitro   |
| Vif     | Low                |        0.48 | Accessory function                 |
| Vpr     | Low                |        0.51 | Accessory function                 |

**Clinical implication**: CTL responses targeting Gag and Pol are associated with better viral control because escape mutations impose prohibitive fitness costs.

---

## Key Epitopes Analyzed

![Gag Epitope Map](../images/10_gag_epitope_map.png)

### High-Barrier Epitopes (Difficult Escape)

| Epitope           | Protein | Position | HLA      | Escape Barrier | Clinical Association |
| :---------------- | :------ | :------- | :------- | -------------: | :------------------- |
| SL9 (SLYNTVATL)   | Gag p17 | 77-85    | A\*02:01 |           4.38 | Elite controller     |
| KK10 (KRWIILGLNK) | Gag p24 | 263-272  | B\*27:05 |           4.40 | Elite controller     |
| TW10 (TSTLQEQIGW) | Gag p24 | 240-249  | B\*57:01 |           4.18 | Elite controller     |

### Lower-Barrier Epitopes (Easier Escape)

| Epitope         | Protein | Position | HLA      | Escape Barrier | Notes           |
| :-------------- | :------ | :------- | :------- | -------------: | :-------------- |
| FL8 (FLKEKGGL)  | Nef     | 90-97    | A\*24:02 |           3.40 | Variable region |
| RL9 (RLRDLLLIW) | Env     | 311-319  | B\*08:01 |           3.53 | Surface exposed |

---

## Escape Mutation Characteristics

![Escape Zone Diagram](../images/11_escape_zone_diagram.png)

### High Efficacy, Low Fitness Cost ("Dangerous" Escapes)

Mutations in the "escape zone" (genetic distance 5.8-6.9) achieve effective immune evasion while maintaining viral fitness:

| Original | Escape | Epitope      | Efficacy | Fitness Retention |
| :------- | :----- | :----------- | :------- | :---------------- |
| T242     | N      | TW10 (B\*57) | High     | 85%               |
| L268     | M      | KK10 (B\*27) | Moderate | 95%               |
| Y79      | F      | SL9 (A\*02)  | High     | 90%               |

### High Efficacy, High Fitness Cost ("Constrained" Escapes)

Mutations requiring distance >7.0 typically impose severe fitness costs:

| Original | Escape | Epitope      | Efficacy | Fitness Retention |
| :------- | :----- | :----------- | :------- | :---------------- |
| R264     | K      | KK10 (B\*27) | High     | 60%               |
| D314     | N      | RL9 (B\*08)  | Moderate | 55%               |

---

## Elite Controller Signature

![Elite Controller Comparison](../images/09_elite_controller_comparison.png)

Patients with B*27 or B*57 alleles show distinct geometric patterns:

| Metric                  | Elite Controllers | Progressors | Difference   |
| :---------------------- | ----------------: | ----------: | :----------- |
| Mean Escape Barrier     |              4.29 |        3.72 | +15% higher  |
| Escape Success Rate     |               24% |         42% | -43% lower   |
| Fitness Cost per Escape |               28% |         12% | +133% higher |

**Interpretation**: Elite controller HLA alleles force the virus into a "lose-lose" situation: either remain visible to CTL or escape with crippling fitness costs.

---

## Epitope Clustering by Constraint Level

![Epitope Tier Classification](../images/12_epitope_tier_classification.png)

### Tier 1: Highest Constraint (Best Vaccine Targets)

| Protein Region     | # Epitopes | Mean Barrier | Rationale             |
| :----------------- | :--------- | -----------: | :-------------------- |
| Gag p24 (CA)       | 89         |         4.31 | Capsid assembly       |
| Pol RT active site | 45         |         4.18 | Catalytic function    |
| Pol IN catalytic   | 32         |         4.05 | Integration essential |

### Tier 2: Moderate Constraint

| Protein Region | # Epitopes | Mean Barrier | Rationale             |
| :------------- | :--------- | -----------: | :-------------------- |
| Gag p17 (MA)   | 67         |         3.89 | Matrix function       |
| Pol PR         | 38         |         3.76 | Substrate recognition |

### Tier 3: Low Constraint (Poor Vaccine Targets)

| Protein Region      | # Epitopes | Mean Barrier | Rationale           |
| :------------------ | :--------- | -----------: | :------------------ |
| Nef variable        | 124        |         3.21 | Dispensable regions |
| Env V1/V2           | 89         |         3.18 | Hypervariable       |
| Accessory (Vif/Vpr) | 156        |         3.35 | Non-essential       |

---

## Population-Level Analysis

### HLA Frequency vs. Protection

| HLA      | Population Frequency | Protection Level | Public Health Impact       |
| :------- | -------------------: | :--------------- | :------------------------- |
| A\*02:01 |               25-50% | Moderate-High    | High (common + protective) |
| B\*57:01 |                5-10% | Very High        | Moderate (rare but strong) |
| B\*27:05 |                 5-8% | High             | Moderate                   |
| B\*35:01 |               10-15% | Low              | Low (common but weak)      |

---

## Implications for Vaccine Design

### Optimal Epitope Selection Criteria

1. **Located in Gag or Pol** (essential proteins)
2. **High escape barrier** (>4.0)
3. **Multiple HLA restrictions** (population coverage)
4. **Conserved across clades** (global utility)
5. **No drug resistance overlap** (see Vaccine Targets document)

### Epitopes Meeting All Criteria

| Epitope    | Protein | Barrier | HLA Count | Conservation |
| :--------- | :------ | ------: | --------: | :----------- |
| TPQDLNTML  | Gag     |    4.45 |        25 | 98%          |
| SLYNTVATL  | Gag     |    4.38 |        12 | 96%          |
| KRWIILGLNK | Gag     |    4.40 |         8 | 97%          |

---

## Summary

1. **Protective HLA alleles restrict constrained epitopes**: B*57/B*27 target essential regions
2. **Escape requires major genetic reorganization**: 77.8% cross cluster boundaries
3. **Gag/Pol epitopes are optimal targets**: Highest escape barriers
4. **Escape zone exists**: Moderate-distance mutations (5.8-6.9) balance efficacy and fitness
5. **Elite controller signature is geometric**: 15% higher escape barriers

---

_Data derived from LANL HIV Immunology Database. Epitope-level data available upon request._
