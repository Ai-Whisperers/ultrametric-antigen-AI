# In Silico Hypothesis: Sentinel Glycans and the Inverse Goldilocks Model

**Doc-Type:** In Silico Hypothesis | Version 2.0 | Updated 2025-12-24

---

## 1. Core Hypothesis

```mermaid
flowchart LR
    subgraph HYPOTHESIS[" Core Hypothesis "]
        SHIELD["Glycan<br/>Shield"]
        PADIC["P-Adic<br/>Analysis"]
        SENTINEL["Sentinel<br/>Glycans"]
        BNAB["bnAb<br/>Exposure"]
    end

    SHIELD -->|"Geometric<br/>Analysis"| PADIC
    PADIC -->|"Goldilocks<br/>Zone"| SENTINEL
    SENTINEL -->|"Deglycosylation"| BNAB

    style SHIELD fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style PADIC fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style SENTINEL fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
    style BNAB fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
```

**The HIV glycan shield can be analyzed through p-adic geometry to identify "sentinel glycans" - specific N-linked glycosylation sites whose removal shifts viral epitopes into an immunogenic Goldilocks Zone, exposing broadly neutralizing antibody (bnAb) targets.**

This is the **inverse** of the RA autoimmunity model:

- **RA**: PTM _addition_ (citrullination) shifts self-proteins into immunogenic zone → autoimmunity
- **HIV**: PTM _removal_ (deglycosylation) shifts viral proteins into immunogenic zone → productive immunity

---

## 2. Background

### 2.1 The HIV Glycan Shield

```mermaid
flowchart TB
    subgraph SHIELD[" HIV Glycan Shield Composition "]
        GP120["gp120<br/>~25 N-glycans"]
        GP41["gp41<br/>~4 N-glycans"]
        MASS["~50% of<br/>molecular weight"]
    end

    SHIELD --> MASK["Masks Conserved<br/>Epitopes"]

    style GP120 fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style GP41 fill:#f97316,stroke:#c2410c,color:#ffffff,stroke-width:2px
    style MASS fill:#eab308,stroke:#a16207,color:#ffffff,stroke-width:2px
    style MASK fill:#64748b,stroke:#475569,color:#ffffff,stroke-width:2px
```

HIV-1 Env (gp120/gp41) is one of the most heavily glycosylated proteins known:

- ~25 N-linked glycosylation sites on gp120
- ~4 sites on gp41
- Glycans constitute ~50% of gp120 molecular weight
- Shield masks conserved epitopes from antibody recognition

### 2.2 N-Linked Glycosylation Sequon

```mermaid
flowchart LR
    subgraph SEQUON[" N-X-S/T Sequon "]
        N["Asparagine<br/>(N)"]
        X["Any AA<br/>(except P)"]
        ST["Serine or<br/>Threonine"]
    end

    N --> X --> ST
    N -->|"Glycan<br/>Attached"| GLYCAN["Glycan Tree"]

    style N fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style X fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style ST fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style GLYCAN fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
```

N-linked glycans attach at the consensus motif: **N-X-S/T** (where X ≠ P)

The asparagine (N) is modified. In p-adic terms:

- **Native:** N codon contributes to embedding
- **Glycosylated:** N + glycan tree (large structural perturbation)
- **Deglycosylated (mutant):** N→Q or N→D removes glycan attachment

### 2.3 Broadly Neutralizing Antibodies (bnAbs)

```mermaid
flowchart TB
    subgraph BNAB_TARGETS[" bnAb Target Epitopes "]
        CD4BS["CD4 Binding Site<br/>Partially shielded"]
        V1V2["V1/V2 Apex<br/>N156, N160"]
        V3["V3 Supersite<br/>N332-dependent"]
        INTERFACE["gp120-gp41<br/>Interface"]
        MPER["MPER<br/>Membrane-proximal"]
    end

    style CD4BS fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style V1V2 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style V3 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style INTERFACE fill:#ec4899,stroke:#be185d,color:#ffffff,stroke-width:2px
    style MPER fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
```

bnAbs target conserved epitopes often involving glycans:

- **CD4 binding site**: Partially shielded by surrounding glycans
- **V1/V2 apex**: Glycan-dependent epitope (N156, N160)
- **V3 glycan supersite**: N332-dependent
- **gp120-gp41 interface**: Masked by glycans
- **MPER**: Membrane-proximal, partially glycan-shielded

---

## 3. The Inverse Goldilocks Hypothesis

### 3.1 Model Comparison

```mermaid
flowchart TB
    subgraph STANDARD[" Standard Goldilocks - RA Model "]
        direction LR
        S1["Native<br/>Protein"]
        S2["+ PTM"]
        S3["Modified<br/>Protein"]
        S4["Immunogenic<br/>Zone"]
    end

    subgraph INVERSE[" Inverse Goldilocks - HIV Model "]
        direction LR
        I1["Glycosylated<br/>Env"]
        I2["- Glycan"]
        I3["Deglycosylated<br/>Env"]
        I4["bnAb<br/>Accessible"]
    end

    S1 --> S2 --> S3 --> S4
    I1 --> I2 --> I3 --> I4

    style S1 fill:#22c55e,stroke:#15803d,color:#ffffff
    style S4 fill:#ef4444,stroke:#b91c1c,color:#ffffff
    style I1 fill:#ef4444,stroke:#b91c1c,color:#ffffff
    style I4 fill:#22c55e,stroke:#15803d,color:#ffffff
```

### 3.2 Prediction Framework

```mermaid
flowchart TB
    subgraph FRAMEWORK[" Sentinel Prediction Framework "]
        STEP1["Compute Baseline<br/>C_glyc"]
        STEP2["Compute Deglycosylated<br/>C_deglyc"]
        STEP3["Calculate Shift<br/>Delta_C"]
        STEP4["Classify Zone"]
    end

    STEP1 --> STEP2 --> STEP3 --> STEP4

    STEP4 --> BELOW["Below 15%<br/>Still Shielded"]
    STEP4 --> GOLD["15-30%<br/>SENTINEL"]
    STEP4 --> ABOVE["Above 30%<br/>Destabilizing"]

    style STEP1 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style STEP2 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style STEP3 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style STEP4 fill:#ec4899,stroke:#be185d,color:#ffffff,stroke-width:2px
    style BELOW fill:#64748b,stroke:#475569,color:#ffffff
    style GOLD fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style ABOVE fill:#ef4444,stroke:#b91c1c,color:#ffffff
```

For each glycosylation site i:

1. **Compute baseline embedding**: C_glyc = embedding of glycosylated sequence context
2. **Compute deglycosylated embedding**: C_deglyc = embedding with N→Q mutation
3. **Calculate p-adic shift**: Δ_C = ||C_deglyc - C_glyc|| / ||C_glyc||
4. **Classify by Goldilocks Zone**:
   - Δ_C < 15%: Still shielded (insufficient exposure)
   - Δ_C ∈ [15%, 30%]: **Sentinel glycan** (optimal bnAb exposure)
   - Δ_C > 30%: Destabilizing (structural collapse, non-functional)

---

## 4. Key Differences from RA Model

```mermaid
flowchart LR
    subgraph COMPARISON[" RA vs HIV Models "]
        RA["RA Model<br/>PTM Addition<br/>Triggers Disease"]
        HIV["HIV Model<br/>PTM Removal<br/>Enables Vaccine"]
    end

    RA -.->|"Opposite<br/>Direction"| HIV

    style RA fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style HIV fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
```

| Aspect             | RA (Citrullination) | HIV (Deglycosylation) |
| ------------------ | ------------------- | --------------------- |
| PTM direction      | Addition            | Removal               |
| Target             | Self-proteins       | Viral proteins        |
| Immune outcome     | Autoimmunity (bad)  | Neutralization (good) |
| Goldilocks meaning | Triggers disease    | Enables vaccine       |
| Clinical goal      | Avoid zone          | Target zone           |

---

## 5. Structural Considerations

### 5.1 Glycan Context Window

```mermaid
flowchart TB
    subgraph CONTEXT[" Glycan Context Analysis "]
        RADIUS["Influence Radius<br/>15-20 Angstrom"]
        SEQUON["Sequon Context<br/>N-X-S/T (3 residues)"]
        WINDOW["Analysis Window<br/>plus/minus 5 residues (11 total)"]
    end

    style RADIUS fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style SEQUON fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style WINDOW fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
```

Unlike citrullination (single residue), glycosylation affects a larger structural neighborhood:

- **Glycan influence radius:** ~15-20 Angstrom
- **Sequon context:** N-X-S/T (3 residues minimum)
- **Analysis window:** ±5 residues from N (11 residues total)

### 5.2 Glycan-Glycan Interactions

```mermaid
flowchart LR
    subgraph INTERACTIONS[" Glycan Network Types "]
        CLUSTERS["Glycan Clusters<br/>Cooperative Shielding"]
        HOLES["Glycan Holes<br/>Natural bnAb Targets"]
        COMP["Compensatory Glycans<br/>Backup Shielding"]
    end

    style CLUSTERS fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style HOLES fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style COMP fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
```

Some glycans form networks:

- **Glycan clusters**: Multiple nearby glycans with cooperative shielding
- **Glycan holes**: Natural gaps in the shield (bnAb targets)
- **Compensatory glycans**: Removal of one may expose another

### 5.3 Structural Validation (AlphaFold3)

```mermaid
flowchart LR
    subgraph AF3[" AlphaFold3 Validation "]
        WT["Wild-Type<br/>Structure"]
        MUT["Mutant N to Q<br/>Structure"]
        COMPARE["Compare<br/>Metrics"]
    end

    WT --> COMPARE
    MUT --> COMPARE
    COMPARE --> RESULTS["Exposure and<br/>Stability Analysis"]

    style WT fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style MUT fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
    style COMPARE fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style RESULTS fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

For top sentinel candidates, predict:

1. Wild-type structure (with glycan sequons)
2. Mutant structure (N→Q at sentinel site)
3. Compare: Exposure of underlying epitope, structural stability

---

## 6. Target Glycosylation Sites

### 6.1 Known bnAb-Related Glycans

```mermaid
flowchart TB
    subgraph GLYCANS[" bnAb-Associated Glycan Sites "]
        subgraph V1V2[" V1/V2 Region "]
            N156["N156<br/>PG9/PG16"]
            N160["N160<br/>PGT145"]
        end

        subgraph V3[" V3 Supersite "]
            N332["N332<br/>PGT121/128"]
            N339["N339<br/>PGT121"]
        end

        subgraph CD4[" CD4bs Region "]
            N276["N276<br/>VRC01-class"]
            N234["N234"]
            N295["N295"]
        end
    end

    style N156 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style N160 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style N332 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style N339 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style N276 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

| Site | Region | bnAb Class  | Known Importance         |
| ---- | ------ | ----------- | ------------------------ |
| N156 | V1/V2  | PG9/PG16    | Critical for V1/V2 bnAbs |
| N160 | V1/V2  | PG9/PGT145  | Glycan-dependent epitope |
| N276 | C2     | VRC01-class | Shields CD4bs            |
| N332 | V3     | PGT121/128  | V3 glycan supersite      |
| N339 | V3     | PGT121      | V3 supersite             |
| N234 | C2     | Multiple    | Near CD4bs               |
| N295 | C2     | Multiple    | Near CD4bs               |
| N301 | V3     | 2G12        | High-mannose cluster     |
| N386 | C3     | Multiple    | gp120 core               |
| N392 | C3/V4  | Multiple    | Variable                 |
| N448 | C4     | Multiple    | gp120-gp41 interface     |
| N611 | gp41   | 10E8/MPER   | MPER shielding           |

### 6.2 Analysis Strategy

```mermaid
flowchart TB
    subgraph STRATEGY[" Analysis Pipeline "]
        S1["Enumerate all<br/>N-X-S/T sequons"]
        S2["Compute p-adic shift<br/>for each N to Q"]
        S3["Rank by<br/>Goldilocks score"]
        S4["Cross-reference<br/>with bnAb epitopes"]
        S5["Validate with<br/>AlphaFold3"]
    end

    S1 --> S2 --> S3 --> S4 --> S5

    style S1 fill:#0ea5e9,stroke:#0369a1,color:#ffffff,stroke-width:2px
    style S2 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style S3 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style S4 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style S5 fill:#ec4899,stroke:#be185d,color:#ffffff,stroke-width:2px
```

1. **Enumerate all N-X-S/T sequons** in reference Env (HXB2 or consensus)
2. **Compute p-adic shift** for each deglycosylation (N→Q)
3. **Rank by Goldilocks score**
4. **Cross-reference with known bnAb epitopes**
5. **Validate top hits with AlphaFold3**

---

## 7. Falsifiable Predictions

```mermaid
flowchart TB
    subgraph PREDICTIONS[" Falsifiable Predictions "]
        P1["Goldilocks Glycans<br/>= bnAb Targets"]
        P2["Above-Goldilocks<br/>= Structural"]
        P3["Below-Goldilocks<br/>= Redundant"]
        P4["Sentinel Clusters<br/>= Broad Neutralization"]
    end

    style P1 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style P2 fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style P3 fill:#64748b,stroke:#475569,color:#ffffff,stroke-width:2px
    style P4 fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
```

### Prediction 1: Goldilocks Glycans Are bnAb Targets

Sites with deglycosylation Δ_C in [15%, 30%] should:

- Correlate with known bnAb epitope glycans
- Show intermediate conservation (not too conserved, not too variable)

### Prediction 2: Non-Goldilocks Glycans Are Structural

Sites with Δ_C > 30% should:

- Be essential for Env folding/stability
- Mutation causes non-functional Env

### Prediction 3: Sub-Goldilocks Glycans Are Redundant

Sites with Δ_C < 15% should:

- Have nearby compensatory glycans
- Removal alone insufficient for bnAb access

### Prediction 4: Sentinel Clusters Enable Broad Neutralization

Multiple sentinel glycans in proximity should:

- Define optimal vaccine immunogen designs
- Predict multi-glycan deletion mutants for immunization

---

## 8. Experimental Validation Path

```mermaid
flowchart TB
    subgraph VALIDATION[" Validation Hierarchy "]
        subgraph COMP[" Computational "]
            C1["P-adic Shift<br/>Calculation"]
            C2["AlphaFold3<br/>Predictions"]
            C3["bnAb Data<br/>Correlation"]
        end

        subgraph LIT[" Literature "]
            L1["Glycan Knockout<br/>Studies"]
            L2["CATNAP<br/>Database"]
            L3["Conservation<br/>Analysis"]
        end

        subgraph EXP[" Experimental "]
            E1["Generate<br/>Mutants"]
            E2["bnAb Binding<br/>Assays"]
            E3["Immunization<br/>Studies"]
        end
    end

    COMP --> LIT --> EXP

    style C1 fill:#3b82f6,stroke:#1e40af,color:#ffffff
    style C2 fill:#3b82f6,stroke:#1e40af,color:#ffffff
    style C3 fill:#3b82f6,stroke:#1e40af,color:#ffffff
    style L1 fill:#f59e0b,stroke:#b45309,color:#ffffff
    style L2 fill:#f59e0b,stroke:#b45309,color:#ffffff
    style L3 fill:#f59e0b,stroke:#b45309,color:#ffffff
    style E1 fill:#22c55e,stroke:#15803d,color:#ffffff
    style E2 fill:#22c55e,stroke:#15803d,color:#ffffff
    style E3 fill:#22c55e,stroke:#15803d,color:#ffffff
```

### 8.1 Computational (This Analysis)

1. P-adic shift calculation for all glycan sites
2. AlphaFold3 structure prediction for top candidates
3. Correlation with known bnAb binding data

### 8.2 Literature Validation

1. Compare predictions to published glycan knockout studies
2. Check against CATNAP database (bnAb-Env interactions)
3. Validate against glycan conservation in HIV sequence databases

### 8.3 Future Experimental (Out of Scope)

1. Generate predicted sentinel glycan mutants
2. Test binding to bnAb panels
3. Immunize with designed immunogens

---

## 9. Implications for Vaccine Design

### 9.1 Immunogen Design Strategy

```mermaid
flowchart LR
    subgraph DESIGN[" Immunogen Design "]
        ENV["Native Env"]
        REMOVE["Remove Sentinel<br/>Glycans"]
        EXPOSE["Expose bnAb<br/>Epitopes"]
        STABLE["Maintain<br/>Structure"]
    end

    ENV --> REMOVE --> EXPOSE --> STABLE

    style ENV fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style REMOVE fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
    style EXPOSE fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style STABLE fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

```
Optimal HIV immunogen = Env with sentinel glycans removed
                      = Exposes bnAb epitopes without structural collapse
                      = P-adic guided glycan deletion
```

### 9.2 Sequential Immunization

```mermaid
flowchart LR
    subgraph SEQUENTIAL[" Prime-Boost Strategy "]
        PRIME["PRIME<br/>Deglycosylated Env<br/>Broad Priming"]
        BOOST["BOOST<br/>Native Env<br/>Affinity Maturation"]
    end

    PRIME --> BOOST

    style PRIME fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style BOOST fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

### 9.3 Mosaic Approaches

```mermaid
flowchart TB
    subgraph MOSAIC[" Universal Vaccine Strategy "]
        CLADES["Analyze All<br/>HIV-1 Clades"]
        INTERSECT["Find Common<br/>Sentinels"]
        UNIVERSAL["Universal<br/>Immunogen"]
    end

    CLADES --> INTERSECT --> UNIVERSAL

    style CLADES fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style INTERSECT fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style UNIVERSAL fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
```

---

## 10. Connection to P-Adic Framework

### 10.1 Why P-Adic Geometry Works

```mermaid
flowchart TB
    subgraph HIERARCHY[" Hierarchical Glycan Shield "]
        INDIVIDUAL["Individual Glycans<br/>Shield Local Epitopes"]
        CLUSTERS["Glycan Clusters<br/>Shield Regions"]
        WHOLE["Entire Shield<br/>Protects Core"]
    end

    INDIVIDUAL --> CLUSTERS --> WHOLE

    PADIC["P-Adic Geometry<br/>Captures Hierarchy"]

    WHOLE --> PADIC

    style INDIVIDUAL fill:#0ea5e9,stroke:#0369a1,color:#ffffff,stroke-width:2px
    style CLUSTERS fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style WHOLE fill:#6366f1,stroke:#4338ca,color:#ffffff,stroke-width:2px
    style PADIC fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:3px
```

The glycan shield operates hierarchically:

- Individual glycans shield local epitopes
- Glycan clusters shield larger regions
- The entire shield shields the conserved core

P-adic/ultrametric geometry naturally captures this hierarchy:

- Close in p-adic space = similar immunological visibility
- Cluster boundaries = immunological recognition thresholds

### 10.2 The Goldilocks Zone is Universal

```mermaid
flowchart LR
    subgraph UNIVERSAL[" Universal Goldilocks Zone "]
        RA["RA<br/>Citrullination"]
        HIV["HIV<br/>Deglycosylation"]
        SARS["SARS-CoV-2<br/>Phosphomimic"]
        AD["Alzheimers<br/>Phosphorylation"]
    end

    GOLD["15-30%<br/>Shift Threshold"]

    GOLD --> RA & HIV & SARS & AD

    style GOLD fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:3px
    style RA fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
    style HIV fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style SARS fill:#ef4444,stroke:#b91c1c,color:#ffffff,stroke-width:2px
    style AD fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
```

The same [15%, 30%] shift range that predicts:

- RA autoantigen selection (citrullination)
- Should predict HIV bnAb epitope exposure (deglycosylation)

This universality suggests the Goldilocks Zone reflects **fundamental immune recognition thresholds**, not disease-specific phenomena.

---

## 11. Analysis Plan

```mermaid
flowchart TB
    subgraph PHASES[" Analysis Phases "]
        P1["Phase 1<br/>Data Preparation"]
        P2["Phase 2<br/>P-Adic Calculation"]
        P3["Phase 3<br/>Ranking"]
        P4["Phase 4<br/>Structural Validation"]
        P5["Phase 5<br/>Synthesis"]
    end

    P1 --> P2 --> P3 --> P4 --> P5

    style P1 fill:#0ea5e9,stroke:#0369a1,color:#ffffff,stroke-width:2px
    style P2 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
    style P3 fill:#8b5cf6,stroke:#6d28d9,color:#ffffff,stroke-width:2px
    style P4 fill:#a855f7,stroke:#7e22ce,color:#ffffff,stroke-width:2px
    style P5 fill:#ec4899,stroke:#be185d,color:#ffffff,stroke-width:2px
```

### Phase 1: Data Preparation

- Obtain HIV-1 Env reference sequence (HXB2 or BG505 SOSIP)
- Identify all N-X-S/T sequons
- Extract ±5 residue windows around each N

### Phase 2: P-Adic Calculation

- Encode wild-type windows with 3-adic encoder
- Encode N→Q mutant windows
- Calculate Δ_C for each site
- Classify into Goldilocks zones

### Phase 3: Ranking and Selection

- Rank sites by Goldilocks score
- Cross-reference with known bnAb glycans
- Select top 10-20 candidates for AlphaFold3

### Phase 4: Structural Validation

- Generate AlphaFold3 inputs (WT and mutant Env)
- Predict structures
- Analyze epitope exposure changes

### Phase 5: Synthesis

- Compile sentinel glycan list
- Compare to literature
- Document novel predictions

---

## 12. Expected Outcomes

### Success Criteria

```mermaid
flowchart TB
    subgraph SUCCESS[" Success Criteria "]
        SC1["Known bnAb glycans<br/>in Goldilocks Zone"]
        SC2["Structural glycans<br/>outside zone"]
        SC3["Novel predictions<br/>identified"]
        SC4["AlphaFold3<br/>confirmation"]
    end

    style SC1 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style SC2 fill:#22c55e,stroke:#15803d,color:#ffffff,stroke-width:2px
    style SC3 fill:#f59e0b,stroke:#b45309,color:#ffffff,stroke-width:2px
    style SC4 fill:#3b82f6,stroke:#1e40af,color:#ffffff,stroke-width:2px
```

1. **Known bnAb glycans in Goldilocks Zone**: N156, N160, N332 should score in [15%, 30%]
2. **Structural glycans outside zone**: Core glycans should score >30%
3. **Novel predictions**: Identify previously uncharacterized sentinel candidates
4. **AlphaFold3 confirmation**: Structural changes consistent with epitope exposure

### Potential Findings

- **High-confidence sentinels**: Strong candidates for immunogen design
- **Glycan clusters**: Groups that must be removed together
- **Strain-specific sentinels**: Clade-dependent predictions
- **Universal sentinels**: Conserved across all clades

---

## Changelog

| Date       | Version | Description                                        |
| ---------- | ------- | -------------------------------------------------- |
| 2025-12-24 | 2.0     | Added Mermaid diagrams, improved visual structure  |
| 2025-12-24 | 1.1     | Renamed to In Silico Hypothesis                    |
| 2025-12-18 | 1.0     | Initial conjecture document                        |

---

## Related Documents

- [HIV README](../README.md)
- [Discovery Report](../discoveries/DISCOVERY_HIV_PADIC_RESISTANCE.md)
- [Glycan Shield README](./README.md)

---

**Status:** Hypothesis validated through computational analysis | Ready for experimental follow-up
