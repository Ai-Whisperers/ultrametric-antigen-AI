# Image Specifications for HIV Medical Paper

**Total Images**: 30  
**Target Audience**: Clinicians, Virologists, Medical Researchers  
**Design Principle**: Clear, explanatory, clinically actionable

---

## Design Guidelines

### Color Palette

| Purpose | Color | Hex Code | Usage |
|:--------|:------|:---------|:------|
| Primary Blue | Medical blue | #2E86AB | Main elements, headers |
| Secondary Teal | Trust/health | #1B998B | Supporting elements |
| Accent Orange | Attention | #E94F37 | Warnings, resistance |
| Accent Green | Safe/positive | #4CAF50 | Safe targets, protection |
| Neutral Gray | Background | #F5F5F5 | Backgrounds |
| Dark Gray | Text | #333333 | Labels, text |
| Light Gray | Borders | #E0E0E0 | Dividers, grids |

### Drug Class Colors (Consistent Across All Images)

| Drug Class | Color | Hex Code |
|:-----------|:------|:---------|
| NRTI | Blue | #2196F3 |
| NNRTI | Orange | #FF9800 |
| PI | Purple | #9C27B0 |
| INSTI | Green | #4CAF50 |

### Protein Colors (Consistent Across All Images)

| Protein | Color | Hex Code |
|:--------|:------|:---------|
| Gag | Dark Blue | #1565C0 |
| Pol | Teal | #00897B |
| Env | Orange | #EF6C00 |
| Nef | Red | #C62828 |
| Accessory | Gray | #757575 |

### Typography

| Element | Font | Size | Weight |
|:--------|:-----|:-----|:-------|
| Title | Arial/Helvetica | 24pt | Bold |
| Subtitle | Arial/Helvetica | 18pt | Regular |
| Axis Labels | Arial/Helvetica | 14pt | Regular |
| Data Labels | Arial/Helvetica | 12pt | Regular |
| Legend | Arial/Helvetica | 11pt | Regular |
| Footnotes | Arial/Helvetica | 10pt | Italic |

### Standard Dimensions

| Format | Width | Height | Use Case |
|:-------|:------|:-------|:---------|
| Full Width | 1200px | 800px | Complex diagrams |
| Half Width | 600px | 500px | Simple charts |
| Square | 800px | 800px | Circular diagrams |
| Banner | 1200px | 400px | Timelines, flows |

### File Format

- **Primary**: PNG (300 DPI for print)
- **Web**: PNG or SVG
- **Editable**: SVG source files retained

---

## Section 1: Drug Resistance (Images 01-06)

### Image 01: Genetic Distance by Drug Class

**Filename**: `01_drug_class_barrier_comparison.png`  
**Dimensions**: 800 x 600 px  
**Type**: Horizontal bar chart

**Data**:
| Drug Class | Mean Distance | Error (±) |
|:-----------|:-------------:|:---------:|
| NRTI | 6.08 | 1.42 |
| NNRTI | 5.04 | 1.28 |
| INSTI | 4.92 | 1.15 |
| PI | 4.35 | 2.34 |

**Layout**:
```
+------------------------------------------+
|  Genetic Barrier to Resistance by        |
|  Drug Class                              |
+------------------------------------------+
|                                          |
|  NRTI   ████████████████████████  6.08   |
|                                          |
|  NNRTI  ██████████████████       5.04    |
|                                          |
|  INSTI  █████████████████        4.92    |
|                                          |
|  PI     ██████████████  ←→       4.35    |
|                    (high variance)       |
|                                          |
|         0    2    4    6    8            |
|              Genetic Distance            |
+------------------------------------------+
|  Higher distance = more durable regimen  |
+------------------------------------------+
```

**Notes**:
- Use drug class colors defined above
- Add error bars for PI to show high variance
- Include interpretive footer

---

### Image 02: Key Resistance Mutations Reference Card

**Filename**: `02_resistance_mutations_reference.png`  
**Dimensions**: 1000 x 700 px  
**Type**: Color-coded table/infographic

**Data**:
| Mutation | Class | Distance | Barrier Level | Clinical Note |
|:---------|:------|:--------:|:--------------|:--------------|
| K103N | NNRTI | 3.80 | LOW | Rapid emergence |
| M46I | PI | 0.65 | VERY LOW | Accessory |
| Y181C | NNRTI | 4.12 | MODERATE | Cross-resistance |
| M184V | NRTI | 5.67 | HIGH | Common, fitness cost |
| K65R | NRTI | 5.52 | HIGH | TDF resistance |
| T215Y | NRTI | 7.17 | VERY HIGH | Major shift |
| R263K | INSTI | 4.40 | HIGH | DTG, fitness cost |

**Color Coding**:
- Distance < 3: Red background (LOW barrier)
- Distance 3-5: Yellow background (MODERATE)
- Distance > 5: Green background (HIGH barrier)

**Layout**:
```
+--------------------------------------------------+
|  Key Drug Resistance Mutations                   |
|  Color indicates genetic barrier to emergence    |
+--------------------------------------------------+
|                                                  |
|  [RED]    K103N  NNRTI  3.80  Rapid emergence   |
|  [RED]    M46I   PI     0.65  Accessory mut.    |
|  [YELLOW] Y181C  NNRTI  4.12  Cross-resistance  |
|  [GREEN]  M184V  NRTI   5.67  Common mutation   |
|  [GREEN]  K65R   NRTI   5.52  TDF resistance    |
|  [GREEN]  T215Y  NRTI   7.17  Major genetic     |
|  [GREEN]  R263K  INSTI  4.40  DTG pathway       |
|                                                  |
+--------------------------------------------------+
|  Legend: [RED]=Low barrier  [YELLOW]=Moderate   |
|          [GREEN]=High barrier (durable)         |
+--------------------------------------------------+
```

---

### Image 03: Primary vs Accessory Mutation Diagram

**Filename**: `03_primary_vs_accessory_diagram.png`  
**Dimensions**: 900 x 700 px  
**Type**: Conceptual diagram with two zones

**Concept**:
- Show a circular "target" diagram
- Center = conserved viral core (high fitness)
- Edge = variable periphery (escape zone)
- Primary mutations shown at periphery
- Accessory mutations shown internally

**Layout**:
```
+--------------------------------------------------+
|  Primary vs Accessory Mutations                  |
+--------------------------------------------------+
|                                                  |
|              ┌─────────────────┐                 |
|         ╱    │   ACCESSORY     │    ╲            |
|       ╱      │   MUTATIONS     │      ╲          |
|      │       │  (compensatory) │       │         |
|      │       │   M46I (0.65)   │       │         |
|      │       │   L10I (1.23)   │       │         |
|      │       └────────┬────────┘       │         |
|      │                │                │         |
|      │    ←──── Fitness ────→          │         |
|      │                                 │         |
|  PRIMARY                           PRIMARY       |
|  MUTATIONS                         MUTATIONS     |
|  K103N (3.80)                     T215Y (7.17)  |
|  I84V (3.89)                      M184V (5.67)  |
|                                                  |
|  PERIPHERY ←─────────────────────→ PERIPHERY    |
|  (direct drug resistance)                        |
+--------------------------------------------------+
|  Primary: Directly reduce drug binding           |
|  Accessory: Compensate for fitness cost          |
+--------------------------------------------------+
```

**Visual Elements**:
- Concentric circles (3 rings)
- Inner ring: labeled "Core (Accessory)"
- Outer ring: labeled "Periphery (Primary)"
- Mutations placed with distance from center proportional to genetic distance

---

### Image 04: Resistance Emergence Timeline

**Filename**: `04_resistance_emergence_timeline.png`  
**Dimensions**: 1200 x 500 px  
**Type**: Horizontal timeline

**Concept**: Show typical order of resistance emergence under selection pressure

**Data**:
```
Timeline: Weeks on suboptimal therapy
Week 0    Week 4    Week 8    Week 12   Week 16   Week 20+
   |         |         |         |         |         |
   ▼         ▼         ▼         ▼         ▼         ▼
         [NNRTI]              [NRTI]           [PI/INSTI]
         K103N                M184V            Multiple
         (rapid)              (moderate)       (slow)
```

**Layout**:
```
+------------------------------------------------------------------+
|  Typical Resistance Emergence Timeline                           |
+------------------------------------------------------------------+
|                                                                  |
|  Weeks:  0      4      8      12     16     20     24+          |
|          |      |      |      |      |      |      |            |
|          ├──────┼──────┼──────┼──────┼──────┼──────┤            |
|                                                                  |
|  NNRTI:  ████████                                                |
|          K103N emerges (distance 3.80)                          |
|                                                                  |
|  NRTI:          ░░░░░░████████████                              |
|                 M184V emerges (distance 5.67)                   |
|                                                                  |
|  PI:                          ░░░░░░░░████████████              |
|                               Requires multiple mutations       |
|                                                                  |
|  INSTI:                             ░░░░░░░░░░████████          |
|                                     Rare with DTG               |
|                                                                  |
+------------------------------------------------------------------+
|  Lower genetic distance = faster emergence                       |
+------------------------------------------------------------------+
```

---

### Image 05: Cross-Resistance Heatmap

**Filename**: `05_cross_resistance_heatmap.png`  
**Dimensions**: 800 x 800 px  
**Type**: Matrix heatmap

**Data** (correlation of resistance patterns):
|  | NRTI | NNRTI | PI | INSTI |
|:--|:----:|:-----:|:--:|:-----:|
| NRTI | 1.00 | 0.23 | 0.18 | 0.12 |
| NNRTI | 0.23 | 1.00 | 0.15 | 0.08 |
| PI | 0.18 | 0.15 | 1.00 | 0.11 |
| INSTI | 0.12 | 0.08 | 0.11 | 1.00 |

**Color Scale**: White (0) → Dark Blue (1.0)

**Layout**:
```
+------------------------------------------+
|  Cross-Resistance Between Drug Classes  |
+------------------------------------------+
|                                          |
|         NRTI  NNRTI   PI   INSTI        |
|                                          |
|  NRTI   ████   ░░░    ░     ░           |
|                                          |
|  NNRTI  ░░░    ████   ░     ░           |
|                                          |
|  PI      ░      ░    ████   ░           |
|                                          |
|  INSTI   ░      ░     ░    ████         |
|                                          |
+------------------------------------------+
|  Low cross-class resistance allows       |
|  effective drug class switching          |
+------------------------------------------+
```

---

### Image 06: Genetic Barrier Clinical Decision Guide

**Filename**: `06_barrier_clinical_decision.png`  
**Dimensions**: 1000 x 600 px  
**Type**: Decision flowchart

**Layout**:
```
+----------------------------------------------------------+
|  Selecting ART Based on Genetic Barrier                  |
+----------------------------------------------------------+
|                                                          |
|                   ┌─────────────────┐                    |
|                   │ Assess Patient  │                    |
|                   │ Adherence Risk  │                    |
|                   └────────┬────────┘                    |
|                            │                             |
|              ┌─────────────┼─────────────┐               |
|              ▼             ▼             ▼               |
|       ┌──────────┐  ┌──────────┐  ┌──────────┐          |
|       │   LOW    │  │ MODERATE │  │   HIGH   │          |
|       │ Adherence│  │ Adherence│  │ Adherence│          |
|       │   Risk   │  │   Risk   │  │   Risk   │          |
|       └────┬─────┘  └────┬─────┘  └────┬─────┘          |
|            │             │             │                 |
|            ▼             ▼             ▼                 |
|       ┌──────────┐  ┌──────────┐  ┌──────────┐          |
|       │   Any    │  │ Prefer   │  │ Require  │          |
|       │ Regimen  │  │ INSTI or │  │ DTG or   │          |
|       │ Suitable │  │ boosted  │  │ boosted  │          |
|       │          │  │ PI based │  │ PI only  │          |
|       └──────────┘  └──────────┘  └──────────┘          |
|                                                          |
|  Genetic Barrier:  NNRTI < PI < INSTI (DTG highest)     |
+----------------------------------------------------------+
```

---

## Section 2: Immune Escape (Images 07-12)

### Image 07: HLA Protection Hierarchy

**Filename**: `07_hla_protection_ranking.png`  
**Dimensions**: 800 x 600 px  
**Type**: Horizontal bar chart with protection levels

**Data**:
| HLA Allele | Escape Velocity | Protection Level |
|:-----------|:---------------:|:-----------------|
| B*57:01 | 0.218 | Very High |
| B*27:05 | 0.256 | High |
| B*58:01 | 0.278 | High |
| A*02:01 | 0.342 | Moderate-High |
| A*03:01 | 0.389 | Moderate |
| B*35:01 | 0.445 | Low |

**Layout**:
```
+------------------------------------------+
|  HLA Alleles Ranked by HIV Protection    |
+------------------------------------------+
|                                          |
|  B*57:01  ██████████████████████  0.218  |
|           [Very High Protection]         |
|                                          |
|  B*27:05  ████████████████████    0.256  |
|           [High Protection]              |
|                                          |
|  B*58:01  ███████████████████     0.278  |
|           [High Protection]              |
|                                          |
|  A*02:01  ████████████████        0.342  |
|           [Moderate-High]                |
|                                          |
|  A*03:01  ██████████████          0.389  |
|           [Moderate]                     |
|                                          |
|  B*35:01  ███████████             0.445  |
|           [Low Protection]               |
|                                          |
|           0.2   0.3   0.4   0.5          |
|           ← Slower escape = Protection → |
+------------------------------------------+
```

**Color Coding**:
- Very High: Dark Green
- High: Light Green
- Moderate: Yellow
- Low: Orange/Red

---

### Image 08: Protein Constraint Map

**Filename**: `08_protein_constraint_map.png`  
**Dimensions**: 1200 x 500 px  
**Type**: HIV genome schematic with color overlay

**Concept**: Linear representation of HIV genome with proteins colored by constraint level

**Layout**:
```
+------------------------------------------------------------------+
|  HIV Protein Constraint Hierarchy                                |
|  (Darker = More constrained = Better vaccine target)             |
+------------------------------------------------------------------+
|                                                                  |
|  5'LTR  ┌──────────────────────────────────────────────┐  3'LTR |
|         │                                              │         |
|         │ GAG        POL              ENV    NEF ACC   │         |
|         │ ████████   ████████████     ░░░░░  ░░  ░░░   │         |
|         │ p17|p24|   PR|RT|IN        gp120|  Nef Vif   │         |
|         │    |NC |                   gp41 |      Vpr   │         |
|         │                                              │         |
|         └──────────────────────────────────────────────┘         |
|                                                                  |
|  Escape Rate:  0.28    0.31           0.45   0.52               |
|                                                                  |
|  ████ Highly Constrained (escape = fitness cost)                |
|  ░░░░ Variable (escape tolerated)                               |
|                                                                  |
+------------------------------------------------------------------+
|  Target Gag and Pol for durable immune responses                 |
+------------------------------------------------------------------+
```

---

### Image 09: Elite Controller vs Progressor Comparison

**Filename**: `09_elite_controller_comparison.png`  
**Dimensions**: 1000 x 600 px  
**Type**: Side-by-side comparison infographic

**Layout**:
```
+----------------------------------------------------------+
|  Elite Controller vs Typical Progressor                  |
+----------------------------------------------------------+
|                                                          |
|    ELITE CONTROLLER           TYPICAL PROGRESSOR         |
|    (B*57/B*27)                (Other HLA)                |
|                                                          |
|  ┌──────────────────┐     ┌──────────────────┐          |
|  │ Escape Barrier   │     │ Escape Barrier   │          |
|  │     4.29         │     │     3.72         │          |
|  │  ████████████    │     │  ████████        │          |
|  │  [+15% higher]   │     │  [baseline]      │          |
|  └──────────────────┘     └──────────────────┘          |
|                                                          |
|  ┌──────────────────┐     ┌──────────────────┐          |
|  │ Escape Success   │     │ Escape Success   │          |
|  │     24%          │     │     42%          │          |
|  │  ████            │     │  ████████        │          |
|  │  [43% lower]     │     │  [baseline]      │          |
|  └──────────────────┘     └──────────────────┘          |
|                                                          |
|  ┌──────────────────┐     ┌──────────────────┐          |
|  │ Fitness Cost     │     │ Fitness Cost     │          |
|  │ per Escape: 28%  │     │ per Escape: 12%  │          |
|  │  ████████████    │     │  ████            │          |
|  │  [Higher cost]   │     │  [Lower cost]    │          |
|  └──────────────────┘     └──────────────────┘          |
|                                                          |
+----------------------------------------------------------+
|  Protective HLA forces costly escape mutations           |
+----------------------------------------------------------+
```

---

### Image 10: Epitope Location on Gag Protein

**Filename**: `10_gag_epitope_map.png`  
**Dimensions**: 1200 x 600 px  
**Type**: Linear protein diagram with epitope annotations

**Data** (key epitopes on Gag):
| Epitope | Position | HLA | Barrier |
|:--------|:---------|:----|:-------:|
| SLYNTVATL | 77-85 | A*02 | 4.38 |
| KIRLRPGGK | 18-26 | B*27 | 4.12 |
| KRWIILGLNK | 263-272 | B*27 | 4.40 |
| TSTLQEQIGW | 240-249 | B*57 | 4.18 |

**Layout**:
```
+------------------------------------------------------------------+
|  Key CTL Epitopes on HIV Gag Protein                             |
+------------------------------------------------------------------+
|                                                                  |
|  Gag Protein (500 amino acids)                                   |
|                                                                  |
|  ├───────────┼───────────────────┼─────────────────────┼────────┤|
|  0    MA    132      CA         363      NC     432  p6  500    |
|       p17             p24                                        |
|                                                                  |
|       ▼               ▼                   ▼        ▼             |
|  ┌─────────┐    ┌───────────┐      ┌───────────┐                |
|  │KIRLRPGGK│    │ SLYNTVATL │      │KRWIILGLNK │                |
|  │ 18-26   │    │  77-85    │      │ 263-272   │                |
|  │ B*27    │    │  A*02     │      │  B*27     │                |
|  │[4.12]   │    │ [4.38]    │      │ [4.40]    │                |
|  └─────────┘    └───────────┘      └───────────┘                |
|                                                                  |
|                        ▼                                         |
|                  ┌───────────┐                                   |
|                  │TSTLQEQIGW │                                   |
|                  │ 240-249   │                                   |
|                  │  B*57     │                                   |
|                  │ [4.18]    │                                   |
|                  └───────────┘                                   |
|                                                                  |
|  [#.##] = Escape barrier score (higher = more durable)          |
+------------------------------------------------------------------+
```

---

### Image 11: Escape Zone Diagram

**Filename**: `11_escape_zone_diagram.png`  
**Dimensions**: 900 x 700 px  
**Type**: Conceptual zone diagram

**Concept**: Show the "Goldilocks zone" where escape mutations balance efficacy and fitness

**Layout**:
```
+----------------------------------------------------------+
|  The Escape Zone: Balancing Efficacy and Fitness         |
+----------------------------------------------------------+
|                                                          |
|  Fitness    ▲                                            |
|  Retention  │                                            |
|             │                                            |
|    100% ────┤  ●●●  Too close                           |
|             │       (ineffective escape)                 |
|             │                                            |
|     85% ────┤       ╔══════════════════╗                |
|             │       ║   ESCAPE ZONE    ║                |
|             │       ║  Distance 5.8-6.9║                |
|     70% ────┤       ║  ● T242N (B*57)  ║                |
|             │       ║  ● Y79F (A*02)   ║                |
|             │       ╚══════════════════╝                |
|     55% ────┤                                            |
|             │            ●●●  Too far                    |
|             │                 (fitness cost prohibitive) |
|     40% ────┤                                            |
|             │                                            |
|             └────┬────┬────┬────┬────┬────┬────→        |
|                  3    4    5    6    7    8             |
|                      Genetic Distance                    |
|                                                          |
+----------------------------------------------------------+
|  Effective escape requires moderate genetic distance     |
+----------------------------------------------------------+
```

---

### Image 12: CTL Epitope Tier Classification

**Filename**: `12_epitope_tier_classification.png`  
**Dimensions**: 1000 x 700 px  
**Type**: Tiered pyramid/hierarchy diagram

**Layout**:
```
+----------------------------------------------------------+
|  CTL Epitope Classification for Vaccine Design           |
+----------------------------------------------------------+
|                                                          |
|                    ╱╲                                    |
|                   ╱  ╲                                   |
|                  ╱    ╲                                  |
|                 ╱ TIER ╲                                 |
|                ╱   1    ╲                                |
|               ╱ Gag p24  ╲  Barrier: 4.31               |
|              ╱  Pol RT    ╲  89 epitopes                |
|             ╱   Pol IN     ╲ BEST TARGETS               |
|            ╱────────────────╲                            |
|           ╱                  ╲                           |
|          ╱      TIER 2        ╲                          |
|         ╱    Gag p17 (MA)      ╲  Barrier: 3.89         |
|        ╱      Pol PR            ╲  105 epitopes         |
|       ╱      GOOD TARGETS        ╲                       |
|      ╱────────────────────────────╲                      |
|     ╱                              ╲                     |
|    ╱          TIER 3                ╲                    |
|   ╱    Nef, Env V1/V2, Accessory     ╲  Barrier: 3.25   |
|  ╱         AVOID FOR VACCINES         ╲  469 epitopes   |
| ╱──────────────────────────────────────╲                 |
|                                                          |
+----------------------------------------------------------+
```

---

## Section 3: Tropism Analysis (Images 13-18)

### Image 13: V3 Loop Structure with Position 22

**Filename**: `13_v3_loop_position22.png`  
**Dimensions**: 800 x 800 px  
**Type**: Structural diagram of V3 hairpin

**Concept**: Show the V3 loop as a hairpin structure with key positions highlighted

**Layout**:
```
+------------------------------------------+
|  V3 Loop Structure                       |
|  Position 22: Top Tropism Determinant    |
+------------------------------------------+
|                                          |
|                  ┌───┐                   |
|                  │22 │ ← POSITION 22     |
|                  │   │   (Crown tip)     |
|              ┌───┴───┴───┐               |
|              │  20   24  │               |
|          ┌───┤           ├───┐           |
|          │18 │           │ 25│ ← Classic |
|      ┌───┤   │           │   ├───┐  rule |
|      │16 │   │           │   │   │       |
|  ┌───┤   │   │           │   │   ├───┐   |
|  │   │   │   │           │   │   │   │   |
|  │11 │   │   │   V3 LOOP │   │   │   │   |
|  │ ↑ │   │   │           │   │   │   │   |
|  │   │   │   │           │   │   │   │   |
|  └───┴───┴───┴───────────┴───┴───┴───┘   |
|     │                               │     |
|     └──────── Disulfide Bond ───────┘     |
|              (Cys 1 - Cys 35)             |
|                                          |
|  ████ Position 22 (score: 0.591)         |
|  ░░░░ Positions 11/25 (classic rule)     |
+------------------------------------------+
```

---

### Image 14: Position Importance Ranking

**Filename**: `14_tropism_position_ranking.png`  
**Dimensions**: 800 x 600 px  
**Type**: Horizontal bar chart

**Data**:
| Position | Score | Traditional Focus |
|:--------:|:-----:|:------------------|
| 22 | 0.591 | NEW - Not emphasized |
| 8 | 0.432 | Supporting |
| 20 | 0.406 | Coreceptor contact |
| 11 | 0.341 | Classic 11/25 rule |
| 16 | 0.314 | Glycan proximity |
| 25 | 0.298 | Classic 11/25 rule |

**Layout**:
```
+------------------------------------------+
|  V3 Position Importance for Tropism      |
+------------------------------------------+
|                                          |
|  Pos 22 █████████████████████████ 0.591  |
|         [NEW DISCOVERY]        ↑         |
|                                │         |
|  Pos 8  ████████████████       │  0.432  |
|                                │         |
|  Pos 20 ██████████████         │  0.406  |
|                                │         |
|  Pos 11 ████████████           │  0.341  |
|         [Classic 11/25 rule]   │         |
|                                │         |
|  Pos 16 ██████████             │  0.314  |
|                            73% higher    |
|  Pos 25 █████████              │  0.298  |
|         [Classic 11/25 rule] ──┘         |
|                                          |
|         0    0.2   0.4   0.6             |
|         Tropism Discrimination Score     |
+------------------------------------------+
```

---

### Image 15: Amino Acid Distribution at Position 22

**Filename**: `15_position22_amino_acids.png`  
**Dimensions**: 900 x 600 px  
**Type**: Grouped bar chart (R5 vs X4)

**Data**:
| Amino Acid | R5 % | X4 % | Charge |
|:-----------|:----:|:----:|:-------|
| T (Thr) | 48% | 12% | Neutral |
| A (Ala) | 22% | 8% | Neutral |
| I (Ile) | 15% | 18% | Neutral |
| R (Arg) | 3% | 31% | Positive |
| K (Lys) | 2% | 19% | Positive |
| H (His) | 1% | 8% | Positive |

**Layout**:
```
+--------------------------------------------------+
|  Amino Acid Distribution at Position 22          |
+--------------------------------------------------+
|                                                  |
|  60% ─┤                                          |
|       │  ██                                      |
|  50% ─┤  ██                                      |
|       │  ██                                      |
|  40% ─┤  ██                                      |
|       │  ██                          ░░          |
|  30% ─┤  ██  ██                      ░░          |
|       │  ██  ██                      ░░  ░░      |
|  20% ─┤  ██  ██  ██  ░░              ░░  ░░      |
|       │  ██  ██  ██  ░░  ░░          ░░  ░░  ░░  |
|  10% ─┤  ██  ██  ██  ░░  ░░  ██  ██  ░░  ░░  ░░  |
|       │  ██  ░░  ██  ██  ░░  ██  ░░  ██  ██  ██  |
|   0% ─┼──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──|
|          T     A     I     R     K     H         |
|       NEUTRAL         │    POSITIVE (Basic)      |
|                       │                          |
|       ██ R5 (CCR5)    ░░ X4 (CXCR4)             |
|                                                  |
+--------------------------------------------------+
|  Basic amino acids (R, K, H) predict X4 tropism  |
+--------------------------------------------------+
```

---

### Image 16: Tropism Prediction Accuracy Comparison

**Filename**: `16_tropism_prediction_comparison.png`  
**Dimensions**: 800 x 500 px  
**Type**: Bar chart comparing methods

**Data**:
| Method | Accuracy | AUC |
|:-------|:--------:|:---:|
| Classic 11/25 Rule | 74% | 0.72 |
| PSSM (LANL) | 82% | 0.81 |
| Geno2pheno | 84% | 0.84 |
| Our Model | 85% | 0.86 |

**Layout**:
```
+------------------------------------------+
|  Tropism Prediction Method Comparison    |
+------------------------------------------+
|                                          |
|  100% ─┤                                 |
|        │                                 |
|   90% ─┤                                 |
|        │                          ██     |
|   85% ─┤               ██  ██     ██     |
|        │          ██   ██  ██     ██     |
|   80% ─┤          ██   ██  ██     ██     |
|        │     ██   ██   ██  ██     ██     |
|   75% ─┤     ██   ██   ██  ██     ██     |
|        │     ██   ██   ██  ██     ██     |
|   70% ─┤     ██   ██   ██  ██     ██     |
|        │     ██   ██   ██  ██     ██     |
|        └─────┴────┴────┴────┴─────┴─────|
|            11/25 PSSM G2P  Ours         |
|             74%  82%  84%  85%          |
|                                          |
|  Position 22 adds 11% over classic rule  |
+------------------------------------------+
```

---

### Image 17: Tropism Clinical Decision Flowchart

**Filename**: `17_tropism_clinical_flowchart.png`  
**Dimensions**: 1000 x 700 px  
**Type**: Decision flowchart

**Layout**:
```
+----------------------------------------------------------+
|  Tropism Assessment: Enhanced 11/22/25 Rule              |
+----------------------------------------------------------+
|                                                          |
|                ┌─────────────────┐                       |
|                │ Obtain V3 Loop  │                       |
|                │    Sequence     │                       |
|                └────────┬────────┘                       |
|                         │                                |
|                         ▼                                |
|                ┌─────────────────┐                       |
|                │ Check Position  │                       |
|                │      22         │                       |
|                └────────┬────────┘                       |
|                         │                                |
|           ┌─────────────┼─────────────┐                  |
|           ▼             ▼             ▼                  |
|    ┌───────────┐ ┌───────────┐ ┌───────────┐            |
|    │ T, A, S   │ │ I, V, L   │ │ R, K, H   │            |
|    │ (Neutral  │ │ (Neutral  │ │ (Basic/   │            |
|    │  small)   │ │  bulky)   │ │ Positive) │            |
|    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘            |
|          │             │             │                   |
|          ▼             ▼             ▼                   |
|    ┌───────────┐ ┌───────────┐ ┌───────────┐            |
|    │   R5      │ │  Check    │ │   X4      │            |
|    │  LIKELY   │ │ Pos 11/25 │ │  LIKELY   │            |
|    │           │ │           │ │           │            |
|    │ Maraviroc │ │ Continue  │ │  Avoid    │            |
|    │ suitable  │ │ algorithm │ │ Maraviroc │            |
|    └───────────┘ └───────────┘ └───────────┘            |
|                                                          |
+----------------------------------------------------------+
```

---

### Image 18: R5 to X4 Transition Pathway

**Filename**: `18_tropism_transition_pathway.png`  
**Dimensions**: 1100 x 500 px  
**Type**: Timeline/pathway diagram

**Layout**:
```
+------------------------------------------------------------------+
|  Typical R5 to X4 Tropism Transition                             |
+------------------------------------------------------------------+
|                                                                  |
|  Disease Stage:  EARLY          INTERMEDIATE         LATE        |
|                    │                  │                │          |
|                    ▼                  ▼                ▼          |
|                                                                  |
|  V3 Sequence:   CTRPNNNTR...     CTRPNNNTR...     CTRPNNNTR...  |
|  Position 11:      N (neutral)      R (basic) ←      R (basic)  |
|  Position 22:      T (neutral)      T (neutral)      R (basic) ←|
|  Position 25:      N (neutral)      N (neutral)      K (basic) ←|
|                                                                  |
|  Tropism:          R5               R5/X4            X4          |
|                                  (transitional)                  |
|                                                                  |
|  CD4 Count:       >500             200-500           <200        |
|                                                                  |
|  ────────────────────────────────────────────────────────────── |
|                                                                  |
|  Position 22 change (T→R) often marks the "tipping point"       |
|  that completes X4 transition                                    |
+------------------------------------------------------------------+
```

---

## Section 4: Antibody Neutralization (Images 19-24)

### Image 19: bnAb Breadth vs Potency Scatter Plot

**Filename**: `19_bnab_breadth_potency.png`  
**Dimensions**: 900 x 700 px  
**Type**: Scatter plot with quadrants

**Data**:
| Antibody | Breadth (%) | IC50 (μg/mL) | Class |
|:---------|:-----------:|:------------:|:------|
| 3BNC117 | 78.8 | 0.242 | CD4bs |
| 10E8 | 76.7 | 0.221 | MPER |
| VRC01 | 68.9 | 0.580 | CD4bs |
| PG9 | 70.9 | 0.300 | V2-glycan |
| PGT121 | 59.2 | 0.566 | V3-glycan |
| N6 | 75.2 | 0.198 | CD4bs |

**Layout**:
```
+--------------------------------------------------+
|  bnAb Breadth vs Potency                         |
+--------------------------------------------------+
|                                                  |
|  Potency    ▲  (IC50, lower = more potent)      |
|  (IC50)     │                                    |
|             │                                    |
|     0.1 ────┤                     ● N6          |
|             │                 ● 10E8             |
|     0.2 ────┤             ● 3BNC117              |
|             │         ● PG9                      |
|     0.3 ────┤                                    |
|             │                                    |
|     0.4 ────┤                                    |
|             │                                    |
|     0.5 ────┤     ● PGT121                       |
|             │                 ● VRC01            |
|     0.6 ────┤                                    |
|             │                                    |
|             └────┬────┬────┬────┬────┬────→     |
|                 50   60   70   80   90          |
|                      Breadth (%)                 |
|                                                  |
|  ● CD4bs  ● MPER  ● V2-glycan  ● V3-glycan     |
|                                                  |
|  Upper right = Ideal (broad + potent)           |
+--------------------------------------------------+
```

---

### Image 20: Env Trimer with Epitope Regions

**Filename**: `20_env_trimer_epitopes.png`  
**Dimensions**: 800 x 900 px  
**Type**: Simplified structural diagram (side view of trimer)

**Concept**: Show the HIV Env trimer from the side with epitope regions highlighted

**Layout**:
```
+------------------------------------------+
|  HIV Envelope Trimer: bnAb Epitope Sites |
+------------------------------------------+
|                                          |
|              V2-apex (PG9)               |
|                 ▼ ▼ ▼                    |
|              ╭─────────╮                 |
|             ╱           ╲                |
|            │  ●     ●    │  ← V3-glycan  |
|            │    gp120    │    (PGT121)   |
|  CD4bs →   │      ●      │               |
|  (VRC01)   │             │               |
|            │             │               |
|             ╲           ╱                |
|              ╰────┬────╯                 |
|                   │                      |
|            ╭──────┴──────╮               |
|            │    gp41    │                |
|            │            │                |
|            │     ●      │ ← MPER (10E8)  |
|            ╰────────────╯                |
|                   │                      |
|            ═══════════════               |
|              Viral membrane              |
|                                          |
|  ● = bnAb binding site                   |
+------------------------------------------+
```

---

### Image 21: Epitope Class Comparison Table

**Filename**: `21_epitope_class_comparison.png`  
**Dimensions**: 1000 x 600 px  
**Type**: Visual comparison table

**Layout**:
```
+------------------------------------------------------------------+
|  bnAb Epitope Class Comparison                                   |
+------------------------------------------------------------------+
|                                                                  |
|              │ Breadth │ Potency │ Access │ Conservation │       |
|  ────────────┼─────────┼─────────┼────────┼──────────────┼       |
|              │         │         │        │              │       |
|  CD4bs       │ █████   │ ███     │ ██     │ █████        │ BEST  |
|  (VRC01)     │ 72%     │ 1.1     │ Low    │ Very High    │ for   |
|              │         │         │        │              │ breadth|
|  ────────────┼─────────┼─────────┼────────┼──────────────┼       |
|              │         │         │        │              │       |
|  MPER        │ ████    │ ███     │ █      │ █████        │       |
|  (10E8)      │ 68%     │ 1.8     │ V.Low  │ Very High    │       |
|              │         │         │        │              │       |
|  ────────────┼─────────┼─────────┼────────┼──────────────┼       |
|              │         │         │        │              │       |
|  V2-glycan   │ ███     │ █████   │ ███    │ ███          │ BEST  |
|  (PG9)       │ 58%     │ 0.69    │ Mod    │ Moderate     │ for   |
|              │         │         │        │              │ potency|
|  ────────────┼─────────┼─────────┼────────┼──────────────┼       |
|              │         │         │        │              │       |
|  V3-glycan   │ ██      │ ████    │ ███    │ ███          │       |
|  (PGT121)    │ 51%     │ 0.75    │ Mod    │ Moderate     │       |
|              │         │         │        │              │       |
+------------------------------------------------------------------+
```

---

### Image 22: bnAb Combination Coverage Venn Diagram

**Filename**: `22_bnab_combination_coverage.png`  
**Dimensions**: 900 x 700 px  
**Type**: Venn diagram showing overlapping coverage

**Data**:
- 3BNC117 alone: 78.8%
- 10-1074 alone: 54.3%
- 10E8 alone: 76.7%
- 3BNC117 + 10-1074: 91%
- 3BNC117 + 10-1074 + 10E8: 96%

**Layout**:
```
+--------------------------------------------------+
|  bnAb Combination Coverage                       |
+--------------------------------------------------+
|                                                  |
|            ╭───────────────────╮                 |
|           ╱    3BNC117 (CD4bs) ╲                 |
|          ╱        78.8%         ╲                |
|         │                       │                |
|         │    ╭───────────╮     │                |
|         │   ╱   10E8    ╲     │                 |
|         │  ╱    (MPER)    ╲    │                |
|          ╲│     76.7%      │╱                   |
|           ╲       ╭───────╱                     |
|            ╲     ╱  ▼                           |
|             ╲   ╱  96%                          |
|              ╲ ╱  combined                      |
|          ╭────╳────╮                            |
|         ╱    ╱ ╲    ╲                           |
|        ╱    ╱   ╲    ╲                          |
|       │ 10-1074  │                              |
|       │(V3-glycan)│                             |
|       │  54.3%   │                              |
|        ╲        ╱                               |
|         ╲      ╱                                |
|          ╲    ╱                                 |
|           ╲  ╱                                  |
|            ╲╱                                   |
|                                                  |
|  Triple combination achieves 96% coverage        |
+--------------------------------------------------+
```

---

### Image 23: Virus Susceptibility by Clade

**Filename**: `23_clade_susceptibility.png`  
**Dimensions**: 900 x 600 px  
**Type**: World map or bar chart by clade

**Data**:
| Clade | Susceptibility | Resistant % | Region |
|:------|:--------------:|:-----------:|:-------|
| B | 0.72 | 12% | Americas, Europe |
| C | 0.68 | 18% | Sub-Saharan Africa |
| A | 0.65 | 22% | East Africa |
| CRF01_AE | 0.58 | 31% | Southeast Asia |
| D | 0.61 | 26% | East Africa |

**Layout (Bar Chart Option)**:
```
+--------------------------------------------------+
|  bnAb Susceptibility by HIV Clade                |
+--------------------------------------------------+
|                                                  |
|  Clade B    ████████████████████████  72%       |
|  (Americas) [12% resistant]                      |
|                                                  |
|  Clade C    ██████████████████████    68%       |
|  (Africa)   [18% resistant]                      |
|                                                  |
|  Clade A    ████████████████████      65%       |
|  (E.Africa) [22% resistant]                      |
|                                                  |
|  Clade D    ██████████████████        61%       |
|  (E.Africa) [26% resistant]                      |
|                                                  |
|  CRF01_AE   ████████████████          58%       |
|  (SE Asia)  [31% resistant]                      |
|                                                  |
|             0%   25%   50%   75%  100%          |
|             Mean Susceptibility to bnAbs        |
+--------------------------------------------------+
```

---

### Image 24: bnAb Selection Guide for Clinical Use

**Filename**: `24_bnab_clinical_selection.png`  
**Dimensions**: 1000 x 700 px  
**Type**: Decision matrix/table

**Layout**:
```
+------------------------------------------------------------------+
|  bnAb Selection Guide for Clinical Applications                  |
+------------------------------------------------------------------+
|                                                                  |
|  CLINICAL GOAL          RECOMMENDED bnAb(s)         RATIONALE    |
|  ─────────────────────────────────────────────────────────────── |
|                                                                  |
|  Prevention             3BNC117 + 10-1074           High breadth,|
|  (PrEP-like)            [CD4bs + V3-glycan]         91% coverage |
|                                                                  |
|  ─────────────────────────────────────────────────────────────── |
|                                                                  |
|  Reservoir              N6 + 10E8                   Broadest     |
|  Reduction              [CD4bs + MPER]              93% coverage |
|                                                                  |
|  ─────────────────────────────────────────────────────────────── |
|                                                                  |
|  Cure                   Triple combination          Minimize     |
|  Strategies             3BNC117 + 10-1074 + 10E8    escape, 96%  |
|                                                                  |
|  ─────────────────────────────────────────────────────────────── |
|                                                                  |
|  Pediatric              VRC01-LS                    Safety data, |
|                         [Long half-life]            extended t½  |
|                                                                  |
|  ─────────────────────────────────────────────────────────────── |
|                                                                  |
|  AVOID: Same-class combinations (VRC01+3BNC117, PGT121+10-1074) |
|  REASON: Overlapping escape pathways reduce durability          |
+------------------------------------------------------------------+
```

---

## Section 5: Vaccine Targets (Images 25-30)

### Image 25: Safe vs Unsafe Epitope Genome Map

**Filename**: `25_safe_unsafe_genome_map.png`  
**Dimensions**: 1200 x 600 px  
**Type**: HIV genome linear diagram with overlays

**Layout**:
```
+------------------------------------------------------------------+
|  Safe vs Unsafe Vaccine Targets: HIV Genome Map                  |
+------------------------------------------------------------------+
|                                                                  |
|  5'LTR  ┌──────────────────────────────────────────────┐  3'LTR |
|         │                                              │         |
|  SAFE   │ ████  ████████████████                       │         |
|  TARGETS│ Gag   Pol (RT, IN)                           │         |
|  (328)  │ (no resistance                               │         |
|         │  overlap)                                    │         |
|         │                                              │         |
|  UNSAFE │       ░░░░░░░░░░░░    ░░░░░                  │         |
|  TARGETS│       Pol (PR, RT)   Env                     │         |
|  (298)  │       (K103N, M184V  (variable               │         |
|         │        overlap)       regions)               │         |
|         │                                              │         |
|         └──────────────────────────────────────────────┘         |
|                                                                  |
|  ████ SAFE: No drug resistance overlap (use for vaccines)       |
|  ░░░░ UNSAFE: Contains resistance positions (avoid)             |
|                                                                  |
+------------------------------------------------------------------+
|  16,054 conflict positions identified where resistance and      |
|  immunity overlap - these should be excluded from vaccine design |
+------------------------------------------------------------------+
```

---

### Image 26: Trade-off Conflict Illustration

**Filename**: `26_tradeoff_conflict_diagram.png`  
**Dimensions**: 900 x 700 px  
**Type**: Conceptual diagram showing opposing pressures

**Layout**:
```
+----------------------------------------------------------+
|  The Drug Resistance - Immune Escape Conflict            |
+----------------------------------------------------------+
|                                                          |
|                     OVERLAPPING                          |
|                      POSITION                            |
|                         │                                |
|                         ▼                                |
|                    ┌─────────┐                           |
|                    │  e.g.   │                           |
|                    │  RT 103 │                           |
|                    │ (K103N) │                           |
|                    └────┬────┘                           |
|                         │                                |
|           ┌─────────────┼─────────────┐                  |
|           │             │             │                  |
|           ▼             │             ▼                  |
|   ┌───────────────┐     │     ┌───────────────┐         |
|   │     DRUG      │ ←───┴───→ │    IMMUNE     │         |
|   │   PRESSURE    │           │   PRESSURE    │         |
|   │               │           │               │         |
|   │ Selects FOR   │           │ Selects       │         |
|   │ K103N         │           │ AGAINST       │         |
|   │ (resistance)  │           │ K103N         │         |
|   │               │           │ (escape)      │         |
|   └───────────────┘           └───────────────┘         |
|                                                          |
|   RESULT: Drug treatment may inadvertently               |
|   select for immune escape at these positions            |
|                                                          |
+----------------------------------------------------------+
|  Solution: Use vaccine targets that avoid these overlaps |
+----------------------------------------------------------+
```

---

### Image 27: Top 10 Vaccine Targets Infographic

**Filename**: `27_top10_vaccine_targets.png`  
**Dimensions**: 1000 x 800 px  
**Type**: Ranked list with metrics bars

**Layout**:
```
+------------------------------------------------------------------+
|  Top 10 Safe Vaccine Targets                                     |
+------------------------------------------------------------------+
|                                                                  |
|  RANK  EPITOPE       PROTEIN  HLA COVERAGE          SCORE       |
|  ──────────────────────────────────────────────────────────────  |
|                                                                  |
|   1    TPQDLNTML     Gag      █████████████████████████  2.24   |
|                               25 HLA alleles                    |
|                                                                  |
|   2    AAVDLSHFL     Nef      ███████████████████        1.70   |
|                               19 HLA alleles                    |
|                                                                  |
|   3    YPLTFGWCF     Nef      ███████████████████        1.70   |
|                               19 HLA alleles                    |
|                                                                  |
|   4    YFPDWQNYT     Nef      ███████████████████        1.70   |
|                               19 HLA alleles                    |
|                                                                  |
|   5    QVPLRPMTYK    Nef      ███████████████████        1.70   |
|                               19 HLA alleles                    |
|                                                                  |
|   6    SLYNTVATL     Gag      ███████████████            1.61   |
|                               15 HLA alleles                    |
|                                                                  |
|   7    KIRLRPGGK     Gag      ██████████████             1.55   |
|                               14 HLA alleles                    |
|                                                                  |
|   8    FLGKIWPSH     Gag      ████████████               1.49   |
|                               12 HLA alleles                    |
|                                                                  |
|   9    TSTLQEQIGW    Gag      ███████████                1.46   |
|                               11 HLA alleles                    |
|                                                                  |
|  10    KRWIILGLNK    Gag      ██████████                 1.41   |
|                               10 HLA alleles                    |
|                                                                  |
+------------------------------------------------------------------+
|  All targets: No resistance overlap, >90% conservation          |
+------------------------------------------------------------------+
```

---

### Image 28: Population Coverage World Map

**Filename**: `28_population_coverage_map.png`  
**Dimensions**: 1200 x 700 px  
**Type**: World map with region coloring

**Data**:
| Region | Coverage | Color |
|:-------|:--------:|:------|
| North America | 89% | Dark Green |
| Europe | 87% | Dark Green |
| Sub-Saharan Africa | 82% | Light Green |
| South Asia | 81% | Light Green |
| East Asia | 78% | Yellow |
| Southeast Asia | 75% | Yellow |

**Layout**:
```
+------------------------------------------------------------------+
|  Estimated Vaccine Coverage with Top Safe Epitopes               |
+------------------------------------------------------------------+
|                                                                  |
|           ┌─────────────────────────────────────────────┐        |
|           │                                             │        |
|           │      [WORLD MAP OUTLINE]                    │        |
|           │                                             │        |
|           │   N.America    Europe                       │        |
|           │    (89%)       (87%)                        │        |
|           │     ████        ████                        │        |
|           │                           East Asia         │        |
|           │                            (78%)            │        |
|           │                             ░░░░            │        |
|           │                       S.Asia  SE Asia       │        |
|           │    Sub-Saharan        (81%)   (75%)         │        |
|           │     Africa             ███     ░░░          │        |
|           │      (82%)                                  │        |
|           │       ███                                   │        |
|           │                                             │        |
|           └─────────────────────────────────────────────┘        |
|                                                                  |
|  ████ >85%    ███ 80-85%    ░░░ 75-80%                          |
+------------------------------------------------------------------+
```

---

### Image 29: Safe Target Protein Distribution

**Filename**: `29_safe_target_protein_distribution.png`  
**Dimensions**: 800 x 600 px  
**Type**: Pie chart or horizontal stacked bar

**Data**:
| Protein | Safe Targets | Percentage |
|:--------|:------------:|:----------:|
| Gag | 112 | 34% |
| Nef | 89 | 27% |
| Pol | 67 | 20% |
| Env | 38 | 12% |
| Accessory | 22 | 7% |

**Layout**:
```
+------------------------------------------+
|  Safe Vaccine Targets by Protein         |
|  (328 total epitopes)                    |
+------------------------------------------+
|                                          |
|         ╭────────────────╮               |
|        ╱        Gag       ╲              |
|       ╱         34%        ╲             |
|      ╱      (112 epitopes)  ╲            |
|     │                        │           |
|     │   ╭────────────╮      │           |
|     │  ╱    Nef      ╲     │            |
|     │ │     27%       │    │            |
|     │ │  (89 epitopes)│    │            |
|     │  ╲             ╱     │            |
|     │   ╰─────┬─────╯      │            |
|     │         │             │            |
|      ╲   Pol  │   Env      ╱             |
|       ╲  20%  │   12%     ╱              |
|        ╲ (67) │  (38)    ╱               |
|         ╲     │         ╱                |
|          ╲ Acc│essory  ╱                 |
|           ╲7% │ (22)  ╱                  |
|            ╲  │      ╱                   |
|             ╲ │     ╱                    |
|              ╲│    ╱                     |
|               ╰───╯                      |
|                                          |
|  Gag dominates: highest conservation,    |
|  no resistance overlap                   |
+------------------------------------------+
```

---

### Image 30: Vaccine Design Decision Framework

**Filename**: `30_vaccine_design_framework.png`  
**Dimensions**: 1100 x 800 px  
**Type**: Decision flowchart/framework

**Layout**:
```
+------------------------------------------------------------------+
|  Vaccine Target Selection Framework                              |
+------------------------------------------------------------------+
|                                                                  |
|                    ┌─────────────────┐                           |
|                    │ Candidate       │                           |
|                    │ Epitope         │                           |
|                    └────────┬────────┘                           |
|                             │                                    |
|                             ▼                                    |
|  STEP 1:           ┌─────────────────┐                           |
|  Conservation      │ Conservation    │                           |
|  Check             │ >90%?           │                           |
|                    └────────┬────────┘                           |
|                     YES │         │ NO                           |
|                         ▼         ▼                              |
|                    Continue    REJECT                            |
|                         │                                        |
|                         ▼                                        |
|  STEP 2:           ┌─────────────────┐                           |
|  Resistance        │ Drug resistance │                           |
|  Check             │ overlap?        │                           |
|                    └────────┬────────┘                           |
|                      NO │         │ YES                          |
|                         ▼         ▼                              |
|                    Continue    REJECT                            |
|                         │                                        |
|                         ▼                                        |
|  STEP 3:           ┌─────────────────┐                           |
|  HLA Coverage      │ Restricted by   │                           |
|  Check             │ ≥3 HLA alleles? │                           |
|                    └────────┬────────┘                           |
|                     YES │         │ NO                           |
|                         ▼         ▼                              |
|                    Continue    REJECT                            |
|                         │                                        |
|                         ▼                                        |
|  STEP 4:           ┌─────────────────┐                           |
|  Escape Barrier    │ Escape velocity │                           |
|  Check             │ <0.3?           │                           |
|                    └────────┬────────┘                           |
|                     YES │         │ NO                           |
|                         ▼         ▼                              |
|               ┌─────────────┐   REJECT                           |
|               │   ACCEPT    │                                    |
|               │ Safe Target │                                    |
|               │  (n=328)    │                                    |
|               └─────────────┘                                    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Image Checklist

| # | Filename | Section | Status |
|:-:|:---------|:--------|:------:|
| 01 | drug_class_barrier_comparison.png | Drug Resistance | ⬜ |
| 02 | resistance_mutations_reference.png | Drug Resistance | ⬜ |
| 03 | primary_vs_accessory_diagram.png | Drug Resistance | ⬜ |
| 04 | resistance_emergence_timeline.png | Drug Resistance | ⬜ |
| 05 | cross_resistance_heatmap.png | Drug Resistance | ⬜ |
| 06 | barrier_clinical_decision.png | Drug Resistance | ⬜ |
| 07 | hla_protection_ranking.png | Immune Escape | ⬜ |
| 08 | protein_constraint_map.png | Immune Escape | ⬜ |
| 09 | elite_controller_comparison.png | Immune Escape | ⬜ |
| 10 | gag_epitope_map.png | Immune Escape | ⬜ |
| 11 | escape_zone_diagram.png | Immune Escape | ⬜ |
| 12 | epitope_tier_classification.png | Immune Escape | ⬜ |
| 13 | v3_loop_position22.png | Tropism | ⬜ |
| 14 | tropism_position_ranking.png | Tropism | ⬜ |
| 15 | position22_amino_acids.png | Tropism | ⬜ |
| 16 | tropism_prediction_comparison.png | Tropism | ⬜ |
| 17 | tropism_clinical_flowchart.png | Tropism | ⬜ |
| 18 | tropism_transition_pathway.png | Tropism | ⬜ |
| 19 | bnab_breadth_potency.png | Antibodies | ⬜ |
| 20 | env_trimer_epitopes.png | Antibodies | ⬜ |
| 21 | epitope_class_comparison.png | Antibodies | ⬜ |
| 22 | bnab_combination_coverage.png | Antibodies | ⬜ |
| 23 | clade_susceptibility.png | Antibodies | ⬜ |
| 24 | bnab_clinical_selection.png | Antibodies | ⬜ |
| 25 | safe_unsafe_genome_map.png | Vaccine Targets | ⬜ |
| 26 | tradeoff_conflict_diagram.png | Vaccine Targets | ⬜ |
| 27 | top10_vaccine_targets.png | Vaccine Targets | ⬜ |
| 28 | population_coverage_map.png | Vaccine Targets | ⬜ |
| 29 | safe_target_protein_distribution.png | Vaccine Targets | ⬜ |
| 30 | vaccine_design_framework.png | Vaccine Targets | ⬜ |

---

## Production Notes

### Priority Order

**High Priority** (create first):
1. Image 14: Position importance ranking (novel finding)
2. Image 13: V3 loop with Position 22
3. Image 01: Drug class barrier comparison
4. Image 07: HLA protection ranking
5. Image 27: Top 10 vaccine targets

**Medium Priority**:
6-15: Remaining section-specific images

**Lower Priority** (enhance later):
16-30: Supporting and supplementary images

### Tools Recommended

- **Diagrams**: draw.io, Lucidchart, or Figma
- **Charts**: Python (matplotlib/seaborn) or R (ggplot2)
- **Structural**: PyMOL or ChimeraX for protein structures
- **Maps**: QGIS or web-based map tools

### File Delivery

- All images in `/public_medical_paper/images/` folder
- PNG format, 300 DPI
- SVG source files retained for editing
- Naming convention: `XX_descriptive_name.png`

---

*Specification version 1.0 - Ready for image production*
