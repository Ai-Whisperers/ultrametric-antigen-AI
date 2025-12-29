# HIV Approach Clustering Summary

## Key Findings

### Cross-Category Connections (Hyperbolic Distance)

| Approach 1 | Approach 2 | Distance | Implication |
|------------|------------|----------|-------------|
| DTG/BIC 2-Drug | EC Immune Escape | 0.472 | Treatment targets overlap with immune escape pathways |
| Third-Gen INSTI | Lymphoid Reservoir | 0.580 | Novel drugs target reservoir-associated mutations |
| CTL Targets | CNS Sanctuary | 0.525 | Immune surveillance shares codon space with tissue sanctuary |

### Category Cohesion

| Category | Within-Category Distance | Interpretation |
|----------|--------------------------|----------------|
| Treatment | 0.720 | Tight clustering - drugs target similar codon changes |
| Immune | 0.949 | Moderate spread - diverse immune targets |
| Reservoir | 1.030 | Widest spread - tissue-specific adaptation |

### Notable Mutations by Distance

- **R263K (INSTI)**: d=7.413 - DTG-selected, high fitness cost
- **K65R (NRTI)**: d=7.413 - TAF resistance
- **R264K (Gag)**: d=7.413 - KK10 epitope escape
- **Q148H (INSTI)**: d=2.978 - Primary resistance, moderate distance
- **Y181C (NNRTI)**: d=3.079 - Persistent reservoir variant

## Interpretation

The hyperbolic embedding reveals that:

1. **Treatment-Immune Connection**: DTG/BIC regimens target the same codon
   space as immune escape mutations, suggesting these drugs may
   exploit evolutionary constraints on viral immune evasion.

2. **INSTI-Reservoir Link**: Third-gen INSTIs target mutations also
   found in lymphoid tissue reservoirs, making them potentially
   effective for reservoir reduction.

3. **R→K Transitions**: Multiple high-distance mutations involve
   arginine-to-lysine changes (R263K, R264K, K65R), suggesting
   this transition occupies a distant region of codon space.
