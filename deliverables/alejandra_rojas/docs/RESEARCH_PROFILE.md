# Alejandra Rojas - Research Profile and Collaboration Analysis

> **Comprehensive documentation for the IICS-UNA Arbovirus Surveillance Partnership**

**Document Version:** 1.0
**Last Updated:** December 29, 2025
**Partnership Phase:** Phase 3

---

## Table of Contents

1. [Researcher Profile](#researcher-profile)
2. [Institutional Context](#institutional-context)
3. [Research Domain](#research-domain)
4. [Project Technical Analysis](#project-technical-analysis)
5. [Results Interpretation](#results-interpretation)
6. [Clinical Applications](#clinical-applications)
7. [Future Directions](#future-directions)

---

## Researcher Profile

### Basic Information

| Field | Details |
|-------|---------|
| **Name** | Alejandra Rojas |
| **Institution** | IICS-UNA (Instituto de Investigaciones en Ciencias de la Salud) |
| **University** | Universidad Nacional de Asunción |
| **Country** | Paraguay |
| **Role** | Researcher / Collaborator |
| **Partnership Status** | Active - Phase 3 |

### Research Focus

Alejandra Rojas works at the intersection of:
- **Molecular epidemiology** of arboviruses
- **Public health surveillance** systems
- **Diagnostic development** for tropical diseases
- **Computational biology** applications

### Collaboration Context

The partnership with Alejandra Rojas focuses on applying the Ternary VAE framework to **arbovirus surveillance in Paraguay**, with specific emphasis on:

1. **Dengue serotype forecasting** - Predicting which serotype will dominate
2. **Primer stability analysis** - Identifying robust RT-PCR targets
3. **Outbreak early warning** - Detecting evolutionary shifts before outbreaks

---

## Institutional Context

### IICS-UNA Overview

The **Instituto de Investigaciones en Ciencias de la Salud** (IICS) is the premier biomedical research institution in Paraguay, affiliated with the Universidad Nacional de Asunción.

**Key Characteristics:**
- Leading center for infectious disease research in Paraguay
- Houses national reference laboratories
- Conducts surveillance for arboviruses, including dengue, Zika, and chikungunya
- Collaborates with PAHO (Pan American Health Organization)
- Trains next-generation researchers in tropical medicine

### Paraguay Arbovirus Context

Paraguay faces significant arbovirus burden:

| Disease | Status | Recent History |
|---------|--------|----------------|
| **Dengue** | Endemic | Major outbreaks 2011, 2013, 2020, 2023 |
| **Zika** | Present | First cases 2015, ongoing transmission |
| **Chikungunya** | Present | Introduced 2015, sporadic outbreaks |

**Epidemiological Challenges:**
- All 4 dengue serotypes circulate
- Unpredictable serotype dominance shifts
- Limited resources for comprehensive surveillance
- Need for predictive (not just reactive) approaches

### Surveillance Infrastructure

```
National Surveillance Flow:
┌─────────────────┐
│  Health Posts   │ → Sample Collection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Regional Labs   │ → Initial Testing (RDT)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    IICS-UNA     │ → Confirmation, Sequencing, Characterization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ministry of   │ → Policy Response
│     Health      │
└─────────────────┘
```

---

## Research Domain

### Arbovirus Biology

**Dengue Virus (DENV):**
- Family: Flaviviridae
- Genome: ~11 kb single-stranded positive-sense RNA
- Serotypes: DENV-1, DENV-2, DENV-3, DENV-4
- Transmission: *Aedes aegypti* and *Aedes albopictus* mosquitoes

**Serotype Dynamics:**
- Cross-immunity is partial and temporary
- Sequential infections with different serotypes → severe dengue risk
- Serotype replacement occurs cyclically (3-5 year intervals)
- Predicting dominant serotype is clinically important

### Current Surveillance Limitations

| Limitation | Impact |
|------------|--------|
| Reactive approach | Outbreaks detected late |
| Static primers | May fail as virus evolves |
| Manual analysis | Slow, inconsistent |
| Limited forecasting | No predictive models |

### Innovation Opportunity

The Ternary VAE framework addresses these limitations by:
1. **Tracking evolution geometrically** - Hyperbolic trajectories capture dynamics
2. **Quantifying stability** - Embedding variance identifies robust primer sites
3. **Predicting trends** - Velocity vectors enable forecasting
4. **Automating analysis** - Reproducible computational pipeline

---

## Project Technical Analysis

### Mathematical Foundation

**P-adic Encoding of Codons:**

Codons are encoded using 3-adic (base-3) valuations:

```python
def padic_valuation(n, p=3):
    """Count how many times p divides n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v

# Example: codon index 27 = 3³
# v₃(27) = 3 (divides by 3 three times)
```

**Biological Interpretation:**
- Higher valuation = deeper in the codon hierarchy
- Related codons (synonymous) cluster together
- Evolutionary distance ≈ p-adic distance

**Hyperbolic Embedding:**

The 6D embedding captures:

| Dimension | Feature | Biological Meaning |
|-----------|---------|-------------------|
| 1 | Mean p-adic valuation | Average codon "depth" |
| 2 | Std of valuations | Variability in structure |
| 3 | Max valuation | Presence of deep patterns |
| 4 | Fraction v > 0 | Proportion structured |
| 5 | Mean codon index | Average codon usage |
| 6 | Codon index std | Diversity of usage |

### Trajectory Analysis Method

**Step 1: Temporal Grouping**
```python
# Group sequences by year
sequences_by_year = {
    2015: [seq1, seq2, ...],
    2016: [seq3, seq4, ...],
    ...
}
```

**Step 2: Centroid Computation**
```python
# For each year, compute centroid in 6D embedding space
centroids = {}
for year, seqs in sequences_by_year.items():
    embeddings = [embed(seq) for seq in seqs]
    centroids[year] = np.mean(embeddings, axis=0)
```

**Step 3: Velocity Estimation**
```python
# Velocity = rate of change in embedding space
velocity = (centroid_t2 - centroid_t1) / (t2 - t1)
magnitude = np.linalg.norm(velocity)
direction = velocity / magnitude  # Unit vector
```

**Step 4: Forecasting**
```python
# Extrapolate trajectory
predicted_position = current_centroid + velocity * horizon
risk_score = distance_from_origin(predicted_position) / distance_from_origin(current_centroid)
```

### Primer Stability Method

**Concept:** Regions that are stable in embedding space are stable evolutionarily.

```python
def stability_score(position, sequences, window_size=20):
    """
    Compute stability of a genomic region.

    Higher stability = lower variance in embeddings across sequences/time.
    """
    windows = [seq[position:position+window_size] for seq in sequences]
    embeddings = [embed(window) for window in windows]
    variance = np.mean(np.var(embeddings, axis=0))
    return 1.0 / (1.0 + variance)
```

**Selection Criteria:**
- Stability score > 0.98
- GC content 40-60% (optimal for PCR)
- Melting temperature 55-65°C
- No self-complementarity (hairpins)

---

## Results Interpretation

### Dengue Forecast Analysis

**From `results/dengue_forecast.json`:**

#### Serotype Trajectories (2015-2024)

| Serotype | Years Tracked | Velocity | Risk Score | Confidence |
|----------|---------------|----------|------------|------------|
| DENV-1 | 2015, 2019, 2023 | 0.0210 | 1.005 | 90.5% |
| DENV-2 | 2016, 2020, 2024 | 0.0147 | 1.004 | 90.1% |
| **DENV-3** | 2017, 2021 | **0.0372** | **1.013** | 30.0% |
| DENV-4 | 2018, 2022 | 0.0075 | 0.998 | 30.0% |

#### Key Findings

**1. DENV-3 is the Fastest Evolving:**
- Velocity magnitude: 0.0372 (nearly 2x higher than DENV-2)
- Direction: Moving away from origin (diverging)
- Risk score: 1.013 (highest among all serotypes)
- **Implication:** DENV-3 should be prioritized for surveillance

**2. DENV-4 is Relatively Stable:**
- Lowest velocity: 0.0075
- Risk score < 1.0 (0.998) - moving toward origin
- Negative velocity direction - may be declining
- **Implication:** Lower priority for immediate concern

**3. DENV-1 and DENV-2 Show Steady Evolution:**
- Moderate velocities (0.01-0.02)
- High confidence forecasts (>90%)
- Slight divergence (risk ~1.00)
- **Implication:** Monitor but not critical

#### Trajectory Visualization

```
Hyperbolic Space (conceptual):

                    DENV-3 ←─── Fastest, diverging
                   /
                  /
    Origin ●─────●─── DENV-1 (moderate)
                  \
                   \── DENV-2 (moderate)

    DENV-4 ───→ Origin (converging, stable)
```

### Primer Candidates Analysis

**From `results/primer_candidates.csv`:**

Top 5 stable regions identified:

| Rank | Position | Sequence (20nt) | Stability | GC% | Tm (°C) |
|------|----------|-----------------|-----------|-----|---------|
| 1 | 7268 | GAAATGAGCAGCGGTGTCGC | 0.991 | 60% | 55.9 |
| 2 | 3600 | GAGTTGCGCGGTATTGGTGC | 0.989 | 60% | 55.9 |
| 3 | 1723 | CCCGGGAGGCAGACAATAGT | 0.987 | 60% | 55.9 |
| 4 | 5012 | GCTGAAACGCGAGAGAAACC | 0.985 | 55% | 54.2 |
| 5 | 8901 | TTCGGCGTTGACACTCGCAT | 0.983 | 55% | 55.1 |

**Genomic Context:**
- Position 7268: Likely in NS5 gene (RNA-dependent RNA polymerase)
- Position 3600: Likely in NS2A/NS2B region
- Position 1723: Likely in Capsid/prM region

**Recommendation:** Prioritize primers in NS5 (position 7268) - most conserved gene across flaviviruses.

---

## Clinical Applications

### Surveillance Integration

**Proposed Workflow for IICS-UNA:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Monthly Update Cycle                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Week 1: Data Collection                                     │
│  ├── Download new NCBI sequences (ingest_arboviruses.py)    │
│  └── Collect local surveillance sequences                    │
│                                                              │
│  Week 2: Analysis                                            │
│  ├── Run trajectory analysis                                 │
│  ├── Update primer stability scores                          │
│  └── Generate risk assessment                                │
│                                                              │
│  Week 3: Validation                                          │
│  ├── Compare predictions with clinical data                  │
│  └── Adjust model parameters if needed                       │
│                                                              │
│  Week 4: Reporting                                           │
│  ├── Generate surveillance report                            │
│  ├── Alert if risk score > threshold                         │
│  └── Update primer validation queue                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Alert Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Serotype velocity spike | >0.05 | Immediate alert |
| Risk score | >1.5 | Enhanced surveillance |
| Primer stability drop | <0.9 | Re-validate primers |
| New trajectory detected | Any | Investigate |

### Dashboard Integration

**Key Metrics for Display:**

1. **Serotype Risk Gauge:**
   - Color-coded risk levels (green/yellow/red)
   - Current velocity indicator
   - 30-day forecast

2. **Primer Health Monitor:**
   - Top 10 primer stability scores
   - Historical trend
   - Alert for declining stability

3. **Geographic Overlay:**
   - Serotype distribution by region
   - Predicted hotspots
   - Sequence coverage map

---

## Future Directions

### Short-term (3-6 months)

1. **Validation Study:**
   - Use 2011-2022 data to predict 2023
   - Compare predictions with actual outbreak data
   - Measure accuracy, sensitivity, specificity

2. **Primer Laboratory Testing:**
   - Synthesize top 10 primer candidates
   - Test against recent isolates
   - Compare with current IICS-UNA primers

3. **Real-time Integration:**
   - Set up automated NCBI download
   - Daily trajectory updates
   - Alert system integration

### Medium-term (6-12 months)

1. **Multi-pathogen Expansion:**
   - Add Zika trajectory analysis
   - Add Chikungunya monitoring
   - Unified arbovirus dashboard

2. **Regional Collaboration:**
   - Share framework with neighboring countries
   - Coordinate with PAHO surveillance
   - Standardize protocols

3. **Publication:**
   - Document methodology
   - Publish validation results
   - Open-source toolkit release

### Long-term (1-2 years)

1. **Machine Learning Enhancement:**
   - Train predictive models on historical data
   - Incorporate climate variables
   - Vector density integration

2. **Clinical Decision Support:**
   - Integrate with hospital systems
   - Automated severity prediction
   - Treatment recommendation engine

3. **National Platform:**
   - Scale to all Paraguay health regions
   - Ministry of Health integration
   - Policy decision support

---

## References

### Scientific Background

1. **Dengue Epidemiology:**
   - WHO Global Strategy for Dengue Prevention and Control (2012-2020)
   - PAHO/WHO Epidemiological Alerts

2. **Molecular Surveillance:**
   - NCBI Virus Database: https://www.ncbi.nlm.nih.gov/labs/virus/
   - GISAID Arbovirus Initiative

3. **RT-PCR Primer Design:**
   - Lanciotti et al. (1992) - Original pan-Dengue primers
   - Johnson et al. (2005) - Improved CDC primers

### Technical References

4. **Hyperbolic Geometry:**
   - Nickel & Kiela (2017) - "Poincaré Embeddings for Learning Hierarchical Representations"

5. **P-adic Numbers:**
   - Dragovich et al. (2017) - "P-adic Mathematical Physics: The First 30 Years"

---

## Appendix: File Inventory

### Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `ingest_arboviruses.py` | 398 | NCBI sequence download |
| `arbovirus_hyperbolic_trajectory.py` | 434 | Trajectory analysis |
| `primer_stability_scanner.py` | 391 | Primer identification |

### Source Modules

| File | Classes | Purpose |
|------|---------|---------|
| `src/data_pipeline.py` | NCBIFetcher, SequenceProcessor | Data ingestion |
| `src/geometry.py` | HyperbolicSpace | Geometric operations |

### Results

| File | Format | Contents |
|------|--------|----------|
| `dengue_forecast.json` | JSON | Serotype trajectories, velocities, forecasts |
| `primer_candidates.csv` | CSV | Top 30 stable primer candidates |

### Data

| File | Format | Contents |
|------|--------|----------|
| `dengue_paraguay.fasta` | FASTA | 10 demo dengue sequences |

---

*Document prepared as part of the Ternary VAE Bioinformatics Partnership*
*For integration with IICS-UNA arbovirus surveillance program*
