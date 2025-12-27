# Technical Brief: Arbovirus Hyperbolic Surveillance

## Executive Summary

This project provides a novel approach to arbovirus surveillance using **hyperbolic trajectory analysis**. By embedding viral genomes into hyperbolic space and tracking their evolution over time, we can:

1. **Predict serotype dominance** before outbreaks occur
2. **Identify stable genomic regions** for robust RT-PCR primers
3. **Quantify evolutionary velocity** to assess pandemic risk

---

## Problem Statement

### Current Surveillance Limitations

| Challenge | Current Approach | Limitation |
|-----------|------------------|------------|
| Serotype shifts | Reactive monitoring | No prediction |
| Primer failure | Redesign after failure | Costly delays |
| Risk assessment | Case counting | Lags behind virus |

### Paraguay Context

- **4 Dengue serotypes** (DENV-1, 2, 3, 4) circulate with cyclical dominance
- **Serotype shifts** linked to severe dengue (secondary infections)
- **RT-PCR primers** must match circulating strains for accurate diagnosis

---

## Technical Approach

### 1. Hyperbolic Genome Embedding

Viral genomes are embedded into 6D hyperbolic space using:

```
Genome → Sliding Windows → Codon Indices → P-adic Valuations → Hyperbolic Coordinates
```

**Window Features (6 dimensions):**
1. Mean p-adic valuation of codons
2. Standard deviation of valuations
3. Maximum valuation
4. Fraction with valuation > 0
5. Normalized mean codon index
6. Standard deviation of codon indices

### 2. Trajectory Computation

For each serotype:
```
2015: [c₁] → 2016: [c₂] → 2017: [c₃] → ... → 2024: [cₙ]
         ↓          ↓          ↓               ↓
      centroid   centroid   centroid        centroid
```

**Trajectory** = time-ordered sequence of centroids

### 3. Velocity and Forecasting

```
Velocity = (centroid_current - centroid_previous) / Δt

Predicted_position = centroid_current + velocity × forecast_horizon
```

**Risk Score**:
```
risk = |predicted_position| / |current_position|

> 1.0 = diverging from origin (evolving rapidly)
< 1.0 = converging to origin (stabilizing)
```

### 4. Primer Stability Analysis

For each genomic position:
```
Stability = 1 / (1 + variance_of_embeddings_over_time)
```

**High stability** = region resists evolutionary change = good primer target

---

## Key Innovations

### Why Hyperbolic?

Evolutionary trees are naturally hyperbolic:
- Branches diverge exponentially
- Distance between clades grows rapidly
- Euclidean metrics underestimate divergence

### Why P-adic?

Codon structure has hierarchical organization:
- Synonymous codons cluster
- P-adic valuations capture this grouping
- Reveals patterns invisible to standard metrics

### Predictive Power

Traditional: "DENV-2 was dominant last year"
Our approach: "DENV-3 is moving fastest → likely dominant next season"

---

## Validation Strategy

### Retrospective Analysis

1. Use 2011-2022 data to predict 2023
2. Compare prediction with actual 2023 surveillance
3. Measure accuracy of serotype dominance prediction

### Primer Validation

1. Identify top 10 stable regions from 2015-2024 data
2. Compare with published pan-Dengue primer sites
3. Test stability of predictions on 2024 sequences

### IICS-UNA Integration

1. Connect to surveillance database
2. Automated monthly trajectory updates
3. Alert system for rapid serotype shifts

---

## Expected Outcomes

| Deliverable | Description | Timeline |
|-------------|-------------|----------|
| Forecast JSON | Trajectory predictions for all serotypes | Immediate |
| Primer CSV | Top 50 stable primer candidates | Immediate |
| Dashboard | Interactive Jupyter visualization | Immediate |
| Validation | Retrospective accuracy assessment | With real data |

---

## Technical Requirements

### Software
- Python 3.8+
- NumPy, Pandas
- Biopython
- NCBI Datasets CLI (optional, for real data)

### Data Sources
- NCBI Virus database
- GenBank arbovirus sequences
- IICS-UNA surveillance records (for validation)

---

## References

1. NCBI Virus: https://www.ncbi.nlm.nih.gov/labs/virus/
2. Dengue serotype distribution in Paraguay (IICS-UNA reports)
3. Pan-Dengue primer design literature

---

*Prepared for IICS-UNA Arbovirus Surveillance Program*
