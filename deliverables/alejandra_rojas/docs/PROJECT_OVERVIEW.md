# Project Overview: Ternary VAE for Arbovirus Surveillance

## The Big Picture

This project applies **novel mathematical frameworks** (p-adic numbers and hyperbolic geometry) to computational biology problems. Your deliverable focuses on arbovirus surveillance and trajectory forecasting using these techniques.

---

## Why P-adic and Hyperbolic Geometry?

### Traditional Approaches
- Phylogenetic trees (static snapshots)
- Sequence alignment (pairwise comparisons)
- Consensus sequences (ignore temporal dynamics)

### Our Innovation
- **P-adic embeddings**: Capture codon structure hierarchically
- **Hyperbolic trajectories**: Track evolution in curved space
- **Velocity vectors**: Predict future positions

---

## Your Project: Hyperbolic Trajectory Forecasting

### The Problem
Dengue surveillance in Paraguay faces:
- Unpredictable serotype dominance shifts
- Primer failures when viruses mutate
- Reactive (not predictive) public health response

### Our Solution
Track viral evolution in **hyperbolic space** to:
1. **Predict** which serotype will dominate next season
2. **Identify** stable regions for robust RT-PCR primers
3. **Quantify** evolutionary velocity as early warning

### Why This Works
- Hyperbolic geometry matches evolutionary tree structure
- P-adic encoding captures codon hierarchy
- Temporal analysis reveals trends invisible to snapshots

---

## Key Results in Your Package

### `results/dengue_forecast.json`

**Serotype Trajectory Analysis:**

For each of the 4 Dengue serotypes (DENV-1, 2, 3, 4), the file contains:

```json
{
  "DENV-1": {
    "trajectory": [
      {"time": "2015", "centroid": [...], "n_sequences": 5},
      {"time": "2020", "centroid": [...], "n_sequences": 8}
    ],
    "velocity": {
      "direction": [0.068, 0.203, 0.972, ...],
      "magnitude": 0.0215
    },
    "forecast": {
      "confidence": 0.456,
      "risk_score": 1.007
    }
  }
}
```

**How to interpret:**

| Field | Meaning | Action |
|-------|---------|--------|
| `trajectory` | Path through hyperbolic space over time | Visualize evolution |
| `velocity.magnitude` | Speed of change | Higher = faster evolution |
| `velocity.direction` | Which way it's moving | Towards/away from other serotypes |
| `risk_score` | Divergence from origin | >1.0 = concerning divergence |
| `confidence` | Reliability of forecast | Higher = more consistent trajectory |

**Demo Results Summary:**
```
Fastest moving serotype: DENV-3
Highest risk serotype: DENV-3
```

### `results/primer_candidates.csv`

**Top Primer Candidates:**

| Rank | Position | Sequence | Stability | GC% | Tm |
|------|----------|----------|-----------|-----|-----|
| 1 | 7268 | GAAATGAGCAGCGGTGTCGC | 0.991 | 60% | 55.9C |
| 2 | 3600 | GAGTTGCGCGGTATTGGTGC | 0.989 | 60% | 55.9C |
| 3 | 1723 | CCCGGGAGGCAGACAATAGT | 0.987 | 60% | 55.9C |

**How to interpret:**

| Column | Meaning | Ideal Value |
|--------|---------|-------------|
| `stability_score` | Resistance to evolutionary change | Higher = better |
| `conservation_score` | Fraction of sequences with exact match | Higher = better |
| `gc_content` | GC percentage | 40-60% optimal |
| `tm_estimate` | Melting temperature | 55-65C optimal |
| `variance_over_time` | Change across years | Lower = better |

---

## Understanding the Hyperbolic Embedding

### Visualization

```
           DENV-2 (fast)
              /
     Origin *------> DENV-3 (fastest)
              \
           DENV-1 (slow)

DENV-4 (stationary) *
```

- **Origin**: Average of all serotypes (reference point)
- **Distance from origin**: How divergent the serotype is
- **Velocity vector**: Direction and speed of evolution
- **Clustering**: Serotypes close together may be similar

### The 6D Embedding

Each genome is represented by 6 features:

| Dimension | Feature | Captures |
|-----------|---------|----------|
| 1 | Mean p-adic valuation | Codon structure depth |
| 2 | Std of valuations | Variability in structure |
| 3 | Max valuation | Presence of special patterns |
| 4 | Fraction with v>0 | Proportion of "structured" codons |
| 5 | Normalized codon mean | Average codon usage |
| 6 | Codon std | Diversity of codon usage |

---

## Primer Stability Methodology

### The Problem with Traditional Primers

```
2015: Primer matches all sequences (OK)
2020: Primer matches 95% (Warning)
2023: Primer fails on 30% (false negatives!)
```

### Our Solution

Track **embedding variance** over time:

```
Position X:
  2015 embedding: [0.5, 0.3, ...]
  2018 embedding: [0.5, 0.3, ...]  <- Low variance = stable
  2023 embedding: [0.5, 0.3, ...]

Position Y:
  2015 embedding: [0.5, 0.3, ...]
  2018 embedding: [0.8, 0.1, ...]  <- High variance = unstable
  2023 embedding: [0.2, 0.9, ...]
```

**Stability Score** = 1 / (1 + variance)

- High stability = region resists change = good primer site
- Low stability = region evolves = avoid for primers

---

## Integration with IICS-UNA Surveillance

### Proposed Workflow

```
+-------------------------------------+
|         Monthly Update              |
+-------------------------------------+
| 1. Download new NCBI sequences      |
| 2. Run trajectory analysis          |
| 3. Update primer stability scores   |
| 4. Generate risk assessment report  |
+-------------------------------------+
                 |
                 v
+-------------------------------------+
|         Alert Triggers              |
+-------------------------------------+
| * Serotype velocity > threshold     |
| * Risk score > 1.5                  |
| * Primer stability drops            |
| * New serotype trajectory detected  |
+-------------------------------------+
                 |
                 v
+-------------------------------------+
|         Public Health Action        |
+-------------------------------------+
| * Update surveillance protocols     |
| * Validate/replace RT-PCR primers   |
| * Prepare for serotype shift        |
| * Communicate to health authorities |
+-------------------------------------+
```

### Data Sources

1. **NCBI Virus Database**: Historical and current sequences
2. **IICS-UNA Records**: Local Paraguay surveillance
3. **PAHO**: Regional arbovirus data

---

## Validation Plan

### Retrospective Testing

1. Use 2011-2022 data only
2. Predict 2023 dominant serotype
3. Compare with actual 2023 surveillance
4. Measure prediction accuracy

### Primer Validation

1. Identify top 10 stable regions
2. Compare with published pan-Dengue primers
3. Check if our stable regions include known primers
4. Test on 2024 sequences

### Forward Deployment

1. Set up automated monthly analysis
2. Track prediction accuracy over 12 months
3. Refine model based on results
4. Scale to Zika and Chikungunya

---

## Next Steps

1. **Run with demo data** to verify installation
2. **Download real Paraguay data** from NCBI
3. **Validate 2023 predictions** retrospectively
4. **Test primer candidates** in lab
5. **Integrate with IICS-UNA** dashboard

---

## References

1. NCBI Virus Database: https://www.ncbi.nlm.nih.gov/labs/virus/
2. Dengue serotype surveillance (PAHO)
3. RT-PCR primer design guidelines (WHO)

---

*Prepared for Alejandra Rojas (IICS-UNA) - Ternary VAE Bioinformatics Partnership*
*For integration with IICS-UNA arbovirus surveillance program*
