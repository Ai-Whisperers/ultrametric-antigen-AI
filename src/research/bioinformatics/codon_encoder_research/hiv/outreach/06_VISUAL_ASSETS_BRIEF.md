# Visual Assets Brief: "The Eyes"

To effectively communicate our findings, we need to create the following visual assets. This document serves as the "Spec Sheet" for our designers (or for us to build).

## 1. The Visual Abstract (Single Slide)

**Goal:** Explain the entire discovery in < 5 seconds.
**Format:** Landscape 16:9, High Res PNG.

**Layout:**

- **Left Panel (The "Old Way"):** A flat sequence string `...ATG-CCT-GGA...`. Text: "Standard Bio - Flat, Linear, Blind to Structure".
- **Center Panel (The "Innovation"):** A 3D Hyperbolic Disk (Poincaré Ball). Points plotted inside.
  - Show a "Cloud" of red dots at distance $r \approx 6.0$.
  - Label: "**The Goldilocks Escape Zone** (High Efficacy, Low Cost)".
- **Right Panel (The "Result"):** A molecule (Protease) with a "Shield" icon. Text: "Predicting Resistance Before It Happens".

## 2. The Pitch Deck (5 Slides)

**Audience:** Commercial/Strategic Partners.

- **Slide 1: The Hook.**
  - Title: "Geometric Virology."
  - Subtitle: "Predicting the Next Pandemic Variant using 3-adic Number Theory."
  - Image: Beautiful render of the HIV capsid inside a hyperbolic manifold.
- **Slide 2: The Problem.**
  - "Biological evolution is faster than Clinical Trials."
  - "We chase variants. We need to predict them."
- **Slide 3: The Solution (The Hiding Matrix).**
  - Show the matrix heatmap.
  - Highlight **M46I** (Protease) -> Distance 0.65.
  - Callout: "The virus thinks these are the same. We know they are hidden doors."
- **Slide 4: Validation.**
  - Three charts:
    1.  **HIV:** 78% accuracy on Escape Zones.
    2.  **DTG:** Accurate prediction of High Genetic Barrier.
    3.  **Expansion:** "Ready for COVID-19, TB, and Malaria."
- **Slide 5: The Ask (Open Medicine).**
  - "License the Platform."
  - "Commit to Access."
  - "Lead the Field."

## 3. The Data Product (CSV Release)

**Filename:** `hiv_hiding_distance_matrix.csv`
**Columns:**

- `Codon_A`: Original Codon (e.g., AUG)
- `Codon_B`: Target Codon (e.g., AUC)
- `Amino_A`: Methionine
- `Amino_B`: Isoleucine
- `Hyperbolic_Distance`: 0.654
- `3_Adic_Valuation_Diff`: 0.0
- `Risk_Score`: **EXTREME** (Low distance = High Risk)

_> This file serves as the "Proof of Work" that researchers can download immediately._
