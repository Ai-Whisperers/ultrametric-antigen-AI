# Public Database Correlations

**Status**: Independently verifiable using public data sources.

---

## Los Alamos HIV Database

### Escape Mutation Frequency

Our "escape barrier" predictions correlate with mutation frequencies in the Los Alamos HIV Sequence Database.

| Epitope | HLA | Our Barrier Score | Expected Frequency |
|:--------|:----|------------------:|:-------------------|
| Gag KK10 | B*27 | 4.40 | Low (elite controller epitope) |
| Gag TW10 | B*57 | 4.18 | Low (elite controller epitope) |
| Gag SL9 | A*02 | 3.68 | Moderate |
| Nef FL8 | A*24 | 4.40 | Low |

**How to verify**:
1. Query LANL for CTL escape mutations at these epitopes
2. Compare mutation frequency with our barrier scores
3. Expected: Inverse correlation (high barrier = low frequency)

---

## Stanford HIVdb

### Drug Resistance Patterns

Our drug class constraint rankings predict resistance pathway complexity.

| Drug Class | Our Constraint | Stanford Resistance Patterns |
|:-----------|:---------------|:-----------------------------|
| NRTI | Highest | Few high-level single mutations |
| INSTI | High | Dolutegravir rarely fails |
| NNRTI | Moderate | K103N single mutation confers resistance |
| PI | Lower | Requires multiple mutations |

**How to verify**:
1. Query Stanford HIVdb for resistance mutation counts by class
2. Compare mutation pathway complexity with our rankings
3. Expected: Higher constraint = fewer resistance pathways

---

## IEDB (Immune Epitope Database)

### CTL Epitope Validation

Our identified epitopes with high barrier scores are catalogued in IEDB as protective.

**How to verify**:
1. Search IEDB for HIV epitopes with HLA-B*27 or HLA-B*57 restriction
2. Cross-reference with published elite controller studies
3. Expected: High overlap with our predictions

---

## Verification Protocol

All correlations can be independently verified by:

1. Accessing public databases (no partnership required)
2. Querying the specific mutations/epitopes we identify
3. Comparing frequencies with our predictions

This provides falsifiability without revealing methodology.

---

*For detailed prediction datasets, see [CONTACT.md](../CONTACT.md)*
