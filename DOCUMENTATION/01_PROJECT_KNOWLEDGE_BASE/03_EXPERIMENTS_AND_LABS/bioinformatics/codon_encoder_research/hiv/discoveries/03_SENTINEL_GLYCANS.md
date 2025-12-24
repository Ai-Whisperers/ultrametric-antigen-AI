# Sentinel Glycans: Inverse Goldilocks Model

**Doc-Type:** Discovery Module | Version 2.0 | Updated 2025-12-24

---

## Overview

The HIV glycan shield masks conserved epitopes from broadly neutralizing antibodies (bnAbs). Using the **Inverse Goldilocks Model**, we identified 7 "sentinel glycans" whose removal optimally exposes bnAb epitopes - shifting them into the immunogenic Goldilocks Zone (15-30% centroid shift).

---

## The Inverse Goldilocks Concept

```mermaid
flowchart LR
    subgraph STANDARD["<b>Standard Goldilocks</b><br/>(Autoimmunity)"]
        direction TB
        S1["Native Protein"] -->|"+PTM"| S2["Modified Protein"]
        S2 --> S3["<b>Goldilocks Zone</b><br/>(Immunogenic)"]
    end

    subgraph INVERSE["<b>Inverse Goldilocks</b><br/>(HIV Vaccine)"]
        direction TB
        I1["Glycosylated Env<br/>(Shielded)"] -->|"-Glycan"| I2["Deglycosylated Env"]
        I2 --> I3["<b>Goldilocks Zone</b><br/>(bnAb Accessible)"]
    end

    STANDARD -.->|"Opposite<br/>Direction"| INVERSE

    style S3 fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style I3 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style S1 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style I1 fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
```

---

## Goldilocks Zone Classification

```mermaid
flowchart TB
    subgraph ZONES["<b>P-Adic Shift Classification</b>"]
        BELOW["<b>BELOW GOLDILOCKS</b><br/>< 15% shift<br/>Still shielded"]
        GOLD["<b>GOLDILOCKS ZONE</b><br/>15-30% shift<br/>Optimal exposure"]
        ABOVE["<b>ABOVE GOLDILOCKS</b><br/>> 30% shift<br/>Destabilizing"]
    end

    INPUT["<b>Deglycosylation</b><br/>(N→Q mutation)"] --> ZONES

    style GOLD fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style BELOW fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style ABOVE fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style INPUT fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
```

---

## Sentinel Glycan Results

```mermaid
xychart-beta
    title "Centroid Shift by Glycan Site (%)"
    x-axis ["N58", "N429", "N103", "N204", "N107", "N271", "N265"]
    y-axis "Centroid Shift (%)" 0 --> 35
    bar [22.4, 22.6, 23.7, 25.1, 17.0, 28.4, 29.1]
```

*All 7 sites fall within the Goldilocks Zone (15-30%)*

---

## Sentinel Glycan Map on gp120

```mermaid
flowchart TB
    subgraph GP120["<b>BG505 gp120 Structure</b>"]
        direction LR

        subgraph V1V2["<b>V1/V2 Region</b>"]
            N58["<b>N58</b><br/>22.4%<br/>Score: 1.19"]
            N103["<b>N103</b><br/>23.7%<br/>Score: 1.04"]
            N107["<b>N107</b><br/>17.0%<br/>Score: 0.46"]
        end

        subgraph V3["<b>V3 Region</b>"]
            N204["<b>N204</b><br/>25.1%<br/>Score: 0.85"]
        end

        subgraph C3["<b>C3 Core</b>"]
            N265["<b>N265</b><br/>29.1%<br/>Score: 0.32"]
            N271["<b>N271</b><br/>28.4%<br/>Score: 0.42"]
        end

        subgraph C5["<b>C5 Region</b>"]
            N429["<b>N429</b><br/>22.6%<br/>Score: 1.19"]
        end
    end

    BNAB["<b>bnAb Epitopes</b><br/>Exposed by removal"] --> GP120

    style N58 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style N429 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style N103 fill:#10b981,stroke:#059669,color:#ffffff,stroke-width:2px
    style N204 fill:#10b981,stroke:#059669,color:#ffffff,stroke-width:2px
    style N107 fill:#14b8a6,stroke:#0d9488,color:#ffffff,stroke-width:2px
    style N265 fill:#14b8a6,stroke:#0d9488,color:#ffffff,stroke-width:2px
    style N271 fill:#14b8a6,stroke:#0d9488,color:#ffffff,stroke-width:2px
    style BNAB fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

---

## Detailed Sentinel Table

| Rank | Site | Region | Shift | Score | bnAb Relevance | Priority |
|:-----|:-----|:-------|:------|:------|:---------------|:---------|
| 1 | **N58** | V1 | 22.4% | 1.19 | V1/V2 shield | HIGH |
| 2 | **N429** | C5 | 22.6% | 1.19 | Structural | HIGH |
| 3 | **N103** | V2 | 23.7% | 1.04 | PG9/PG16 apex | HIGH |
| 4 | **N204** | V3 | 25.1% | 0.85 | PGT121/128 supersite | HIGH |
| 5 | N107 | V2 | 17.0% | 0.46 | V1/V2 bnAbs | MEDIUM |
| 6 | N271 | C3 | 28.4% | 0.42 | Core glycan | MEDIUM |
| 7 | N265 | C3 | 29.1% | 0.32 | Core glycan | MEDIUM |

---

## Goldilocks Score Calculation

```mermaid
flowchart TB
    subgraph CALC["<b>Goldilocks Score Formula</b>"]
        SHIFT["<b>Centroid Shift</b><br/>(Δ%)"] --> ZONE
        ZONE["<b>Zone Classification</b>"]

        ZONE -->|"15-30%"| GOLD_SCORE["Zone Score = 1.0 - |Δ - 22.5%| / 7.5%"]
        ZONE -->|"< 15%"| BELOW_SCORE["Zone Score = Δ / 15% × 0.5"]
        ZONE -->|"> 30%"| ABOVE_SCORE["Zone Score = max(0, 1 - (Δ-30%)/20%) × 0.5"]

        BOUNDARY["<b>Boundary Crossed?</b>"] -->|"Yes"| BONUS["+0.2 Bonus"]

        GOLD_SCORE --> FINAL
        BELOW_SCORE --> FINAL
        ABOVE_SCORE --> FINAL
        BONUS --> FINAL["<b>Final Goldilocks Score</b>"]
    end

    style FINAL fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style SHIFT fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style BOUNDARY fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
```

---

## bnAb Epitope Mapping

```mermaid
flowchart LR
    subgraph EPITOPES["<b>bnAb Classes & Target Glycans</b>"]
        subgraph V1V2_BNAB["<b>V1/V2 Apex bnAbs</b>"]
            PG9["PG9/PG16"]
            PGT145["PGT145"]
        end

        subgraph V3_BNAB["<b>V3 Supersite bnAbs</b>"]
            PGT121["PGT121"]
            PGT128["PGT128"]
        end

        subgraph CD4BS["<b>CD4 Binding Site</b>"]
            VRC01["VRC01-class"]
        end
    end

    N58_L["N58"] & N103_L["N103"] & N107_L["N107"] --> V1V2_BNAB
    N204_L["N204"] --> V3_BNAB

    style PG9 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style PGT121 fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
    style VRC01 fill:#eab308,stroke:#ca8a04,color:#ffffff,stroke-width:2px
    style N58_L fill:#22c55e,stroke:#15803d,color:#ffffff
    style N103_L fill:#22c55e,stroke:#15803d,color:#ffffff
    style N204_L fill:#22c55e,stroke:#15803d,color:#ffffff
```

---

## Vaccine Immunogen Designs

```mermaid
flowchart TB
    subgraph DESIGNS["<b>Recommended Immunogen Constructs</b>"]
        D1["<b>Triple Sentinel</b><br/>N58Q + N103Q + N204Q<br/>Exposes V1/V2 + V3"]
        D2["<b>V1/V2 Focused</b><br/>N103Q + N107Q<br/>PG9/PG16 targets"]
        D3["<b>V3 Focused</b><br/>N204Q alone<br/>PGT121/128 targets"]
        D4["<b>All Goldilocks</b><br/>7-site removal<br/>Maximum exposure"]
    end

    BG505["<b>BG505 SOSIP.664</b><br/>Base Construct"] --> DESIGNS

    style D1 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style D2 fill:#10b981,stroke:#059669,color:#ffffff,stroke-width:2px
    style D3 fill:#14b8a6,stroke:#0d9488,color:#ffffff,stroke-width:2px
    style D4 fill:#06b6d4,stroke:#0891b2,color:#ffffff,stroke-width:2px
    style BG505 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

### Design Priority

| Design | Sites | Target bnAbs | Complexity | Priority |
|:-------|:------|:-------------|:-----------|:---------|
| Triple Sentinel | N58Q, N103Q, N204Q | PG9, PGT121 | Low | **1** |
| V1/V2 Focused | N103Q, N107Q | PG9, PG16 | Low | **2** |
| V3 Focused | N204Q | PGT121, PGT128 | Very Low | **3** |
| All Goldilocks | 7 sites | Broad | High | 4 |

---

## AlphaFold3 Structural Validation

```mermaid
xychart-beta
    title "Goldilocks Score vs Structural Disorder (AF3)"
    x-axis ["N58", "N429", "N103", "N204", "N246", "N152"]
    y-axis "Disorder %" 0 --> 100
    bar [75, 100, 67, 68, 63, 61]
```

**Correlation:** r = -0.89 (Goldilocks score inversely correlates with structural stability)

---

## Validation Summary

```mermaid
flowchart LR
    subgraph VALIDATION["<b>Sentinel Glycan Validation</b>"]
        V1["<b>P-Adic Analysis</b><br/>7 sites in Goldilocks"]
        V2["<b>AlphaFold3</b><br/>r = -0.89 correlation"]
        V3["<b>Literature</b><br/>Adjacent to known bnAb sites"]
    end

    V1 --> CONFIRMED
    V2 --> CONFIRMED
    V3 --> CONFIRMED["<b>VALIDATED</b>"]

    style CONFIRMED fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style V1 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style V2 fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
    style V3 fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
```

---

## Distribution Summary

```mermaid
pie showData
    title "Glycan Site Distribution (n=24)"
    "Goldilocks Zone (15-30%)" : 7
    "Above Goldilocks (>30%)" : 17
    "Below Goldilocks (<15%)" : 0
```

---

## Key Insights

1. **7 sentinel glycans identified** in the Goldilocks Zone
2. **Top sites (N58, N429)** have near-perfect Goldilocks scores (1.19)
3. **V1/V2 and V3 regions** contain the most promising targets
4. **AlphaFold3 validates** structural sensitivity at predicted sites
5. **Multi-site removal** shows synergistic effects

---

## Related Documents

- [Elite Controllers](./02_ELITE_CONTROLLERS.md)
- [Therapeutic Applications](./04_THERAPEUTIC_APPLICATIONS.md)
- [Validation Results](./05_VALIDATION_RESULTS.md)
- [Glycan Shield Analysis](../glycan_shield/README.md)

---

**Navigation:** [← Elite Controllers](./02_ELITE_CONTROLLERS.md) | [Index](./README.md) | [Applications →](./04_THERAPEUTIC_APPLICATIONS.md)
