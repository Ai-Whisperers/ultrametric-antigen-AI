# Drug Resistance Geometric Profiles

**Doc-Type:** Discovery Module | Version 2.0 | Updated 2025-12-24

---

## Overview

Each antiretroviral drug class has a characteristic p-adic distance profile reflecting evolutionary constraint on its target site. Drugs targeting conserved active sites force the virus to make larger geometric "jumps" to escape - jumps that carry significant fitness costs.

---

## Drug Class Distance Hierarchy

```mermaid
flowchart TB
    subgraph HIERARCHY["P-Adic Distance by Drug Class"]
        direction TB
        NRTI["<b>NRTI</b><br/>d = 6.05 ± 1.28<br/>RT Active Site"]
        NNRTI["<b>NNRTI</b><br/>d = 5.34 ± 1.40<br/>Allosteric Pocket"]
        INSTI["<b>INSTI</b><br/>d = 5.16 ± 1.45<br/>Integrase Active Site"]
        PI["<b>PI</b><br/>d = 3.60 ± 2.01<br/>Protease"]
    end

    subgraph CONSTRAINT["Constraint Level"]
        HIGH["<b>HIGH CONSTRAINT</b><br/>Active Sites"]
        LOW["<b>LOW CONSTRAINT</b><br/>Flexible Regions"]
    end

    HIGH --> NRTI
    HIGH --> INSTI
    LOW --> NNRTI
    LOW --> PI

    style NRTI fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:3px
    style INSTI fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
    style NNRTI fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style PI fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style HIGH fill:#eab308,stroke:#ca8a04,color:#ffffff,stroke-width:2px
    style LOW fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
```

---

## Distance Bar Chart

```mermaid
xychart-beta
    title "Mean Escape Distance by Drug Class"
    x-axis ["NRTI", "NNRTI", "INSTI", "PI"]
    y-axis "Poincare Distance (d)" 0 --> 8
    bar [6.05, 5.34, 5.16, 3.60]
```

---

## Target Site Constraint Map

```mermaid
quadrantChart
    title Drug Escape Distance vs Target Conservation
    x-axis Low Conservation --> High Conservation
    y-axis Low Escape Distance --> High Escape Distance
    quadrant-1 "High Barrier (Optimal Targets)"
    quadrant-2 "Unexpected Low Distance"
    quadrant-3 "Easy Escape (Avoid)"
    quadrant-4 "Moderate Barrier"

    NRTI: [0.85, 0.80]
    INSTI: [0.80, 0.68]
    NNRTI: [0.45, 0.70]
    PI: [0.35, 0.48]
```

---

## Detailed Mutation Analysis

### NRTI Mutations (Nucleoside RT Inhibitors)

```mermaid
flowchart LR
    subgraph NRTI["NRTI Resistance Mutations"]
        M184V["<b>M184V</b><br/>d = 4.00<br/>3TC/FTC"]
        K65R["<b>K65R</b><br/>d = 7.41<br/>TDF/ABC"]
        K70R["<b>K70R</b><br/>d = 7.41<br/>AZT/D4T"]
        T215Y["<b>T215Y</b><br/>d = 6.06<br/>TAM"]
        L74V["<b>L74V</b><br/>d = 4.63<br/>ABC/DDI"]
    end

    RT["<b>RT Active Site</b><br/>Highly Conserved"] --> NRTI

    style K65R fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style K70R fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style T215Y fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
    style RT fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| K65R | 7.41 | TDF, ABC | Moderate |
| K70R | 7.41 | AZT, D4T | Moderate |
| T215Y | 6.06 | AZT, D4T | Minimal |
| L74V | 4.63 | ABC, DDI | Moderate |
| M184V | 4.00 | 3TC, FTC | Moderate |

**Mean: 6.05 ± 1.28**

---

### INSTI Mutations (Integrase Inhibitors)

```mermaid
flowchart LR
    subgraph INSTI["INSTI Resistance Mutations"]
        R263K["<b>R263K</b><br/>d = 7.41<br/>DTG"]
        Y143R["<b>Y143R</b><br/>d = 5.72<br/>RAL"]
        N155H["<b>N155H</b><br/>d = 4.19<br/>RAL/EVG"]
        Q148H["<b>Q148H</b><br/>d = 4.27<br/>RAL/EVG/DTG"]
        E92Q["<b>E92Q</b><br/>d = 4.19<br/>RAL/EVG"]
    end

    IN["<b>Integrase Active Site</b><br/>DDE Catalytic Triad"] --> INSTI

    style R263K fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style Y143R fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
    style IN fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:2px
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| R263K | 7.41 | DTG | High |
| Y143R | 5.72 | RAL | Moderate |
| Q148H | 4.27 | RAL, EVG, DTG | Moderate |
| N155H | 4.19 | RAL, EVG | Moderate |
| E92Q | 4.19 | RAL, EVG | Minimal |

**Mean: 5.16 ± 1.45**

---

### NNRTI Mutations (Non-Nucleoside RT Inhibitors)

```mermaid
flowchart LR
    subgraph NNRTI["NNRTI Resistance Mutations"]
        K103N["<b>K103N</b><br/>d = 6.89<br/>EFV/NVP"]
        Y181C["<b>Y181C</b><br/>d = 5.27<br/>NVP/EFV"]
        G190A["<b>G190A</b><br/>d = 4.63<br/>NVP/EFV"]
        K101E["<b>K101E</b><br/>d = 4.58<br/>NVP/EFV"]
    end

    POCKET["<b>Allosteric Pocket</b><br/>(Not Active Site)"] --> NNRTI

    style K103N fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
    style POCKET fill:#06b6d4,stroke:#0891b2,color:#ffffff,stroke-width:2px
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| K103N | 6.89 | EFV, NVP | Minimal |
| Y181C | 5.27 | NVP, EFV | Minimal |
| G190A | 4.63 | NVP, EFV | Minimal |
| K101E | 4.58 | NVP, EFV | Minimal |

**Mean: 5.34 ± 1.40**

---

### PI Mutations (Protease Inhibitors)

```mermaid
flowchart LR
    subgraph PI["PI Resistance Mutations"]
        I84V["<b>I84V</b><br/>d = 6.43<br/>DRV/ATV"]
        M46I["<b>M46I</b><br/>d = 3.39<br/>IDV/NFV"]
        V82A["<b>V82A</b><br/>d = 2.41<br/>IDV/RTV"]
        L90M["<b>L90M</b><br/>d = 2.18<br/>SQV/NFV"]
    end

    PROT["<b>Protease</b><br/>(More Flexible)"] --> PI

    style I84V fill:#f97316,stroke:#ea580c,color:#ffffff,stroke-width:2px
    style L90M fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style PROT fill:#ec4899,stroke:#db2777,color:#ffffff,stroke-width:2px
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| I84V | 6.43 | DRV, ATV | Moderate |
| M46I | 3.39 | IDV, NFV | Minimal |
| V82A | 2.41 | IDV, RTV | Minimal |
| L90M | 2.18 | SQV, NFV | Minimal |

**Mean: 3.60 ± 2.01**

---

## Combination Therapy Implications

```mermaid
flowchart TB
    subgraph HIGH["<b>HIGH BARRIER COMBINATIONS</b>"]
        direction LR
        H1["INSTI + NRTI"] --> H1R["<b>d ≈ 11.2</b>"]
        H2["INSTI + NNRTI"] --> H2R["<b>d ≈ 10.5</b>"]
    end

    subgraph LOW["<b>LOWER BARRIER COMBINATIONS</b>"]
        direction LR
        L1["NNRTI + PI"] --> L1R["d ≈ 8.9"]
        L2["PI + PI"] --> L2R["d ≈ 7.2"]
    end

    REC["<b>RECOMMENDATION</b><br/>Use high-barrier combinations<br/>for durable suppression"] --> HIGH

    style HIGH fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style LOW fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:2px
    style REC fill:#eab308,stroke:#ca8a04,color:#ffffff,stroke-width:2px
    style H1R fill:#10b981,stroke:#059669,color:#ffffff
    style H2R fill:#10b981,stroke:#059669,color:#ffffff
```

---

## Summary Metrics

```mermaid
flowchart LR
    subgraph METRICS["<b>Drug Class Summary</b>"]
        M1["<b>NRTI</b><br/>d = 6.05<br/>HIGHEST"]
        M2["<b>NNRTI</b><br/>d = 5.34"]
        M3["<b>INSTI</b><br/>d = 5.16"]
        M4["<b>PI</b><br/>d = 3.60<br/>LOWEST"]
    end

    M1 --> M2 --> M3 --> M4

    style M1 fill:#ef4444,stroke:#dc2626,color:#ffffff,stroke-width:3px
    style M4 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
```

---

## Key Insights

1. **NRTIs have highest constraint** - RT active site is catalytically essential
2. **INSTIs target conserved DDE triad** - Metal coordination must be preserved
3. **NNRTIs escape more easily** - Allosteric pocket tolerates substitutions
4. **PIs show high variability** - Protease is structurally more flexible

---

## Related Documents

- [Main Discovery Report](./DISCOVERY_HIV_PADIC_RESISTANCE.md)
- [Elite Controller Mechanism](./02_ELITE_CONTROLLERS.md)
- [Therapeutic Applications](./04_THERAPEUTIC_APPLICATIONS.md)

---

**Navigation:** [← Back to Index](./README.md) | [Next: Elite Controllers →](./02_ELITE_CONTROLLERS.md)
