# Deliverable Package: Alejandra Rojas
## Hyperbolic Trajectory Analysis for Arbovirus Surveillance

**Prepared for:** Alejandra Rojas (IICS-UNA)
**Project:** Ternary VAE Bioinformatics - Partnership Phase 3
**Date:** December 26, 2024

---

## Overview

This package contains all materials for validating the **Hyperbolic Trajectory Forecasting System** for dengue and arbovirus surveillance. The system tracks viral evolution in hyperbolic space to:

1. **Predict serotype dominance** for the upcoming season
2. **Identify stable genomic regions** for RT-PCR primer design

---

## What's Included

### 1. Core Scripts

| File | Description |
|------|-------------|
| `scripts/ingest_arboviruses.py` | NCBI virus genome download (398 lines) |
| `scripts/arbovirus_hyperbolic_trajectory.py` | Serotype trajectory analysis (434 lines) |
| `scripts/primer_stability_scanner.py` | Stable primer identification (391 lines) |

### 2. Interactive Notebook

| File | Description |
|------|-------------|
| `notebooks/rojas_serotype_forecast.ipynb` | Surveillance dashboard notebook |

### 3. Results

| File | Description |
|------|-------------|
| `results/dengue_forecast.json` | Trajectory analysis for 4 serotypes |
| `results/primer_candidates.csv` | 30 stable primer candidates |

### 4. Reference Data

| File | Description |
|------|-------------|
| `data/dengue_paraguay.fasta` | 10 demo dengue sequences |

### 5. Documentation

| File | Description |
|------|-------------|
| `docs/IMPLEMENTATION_GUIDE.md` | Implementation specifications |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy biopython pandas matplotlib seaborn

# Optional: For downloading real NCBI data
# Windows: winget install NCBI.Datasets
# macOS: brew install ncbi-datasets
# Linux: conda install -c conda-forge ncbi-datasets-cli
```

### Step 2: Generate Demo Data

```bash
cd scripts
python ingest_arboviruses.py --demo \
    --output ../data/dengue_demo.fasta
```

**Expected Output:**
```
Created demo data at data/dengue_demo.fasta
```

### Step 3: Run Trajectory Analysis

```bash
python arbovirus_hyperbolic_trajectory.py \
    --input ../data/dengue_demo.fasta \
    --output ../results/dengue_forecast.json
```

**Expected Output:**
```
Loading sequences from data/dengue_demo.fasta...
Found 4 serotypes
  DENV-1: 3 sequences
  DENV-2: 3 sequences
  DENV-3: 2 sequences
  DENV-4: 2 sequences

Computing trajectories...
Computing velocities and forecasts...

Results saved to results/dengue_forecast.json

=== Forecast Summary ===
Fastest moving serotype: DENV-3
Highest risk serotype: DENV-3
```

### Step 4: Find Stable Primers

```bash
python primer_stability_scanner.py \
    --input ../data/dengue_demo.fasta \
    --output ../results/primer_candidates.csv \
    --window_size 20 \
    --top_n 30
```

**Expected Output:**
```
Scanning sequences...
Processed 10 sequences, 9981 window positions
Computing stability scores...

=== Top 10 Primer Candidates ===
 1. Pos  7268: GAAATGAGCAGCGGTGTCGC
    Stability=0.991 Conservation=0.100 GC=0.60 Tm=55.9
...
Exported 30 primer candidates to results/primer_candidates.csv
```

### Step 5: Explore Dashboard

```bash
jupyter notebook notebooks/rojas_serotype_forecast.ipynb
```

---

## Technical Details

### Supported Viruses

| Virus | NCBI Taxon ID | Serotypes |
|-------|---------------|-----------|
| Dengue | 12637 | DENV-1, DENV-2, DENV-3, DENV-4 |
| Zika | 64320 | Single serotype |
| Chikungunya | 37124 | Single serotype |

### Hyperbolic Embedding Method

Viral genomes are embedded using sliding window p-adic encoding:

```
Genome → Codons → P-adic Valuations → Hyperbolic Coordinates
```

**Window Features (6D embedding):**
1. Mean p-adic valuation
2. Std of p-adic valuations
3. Max valuation
4. Fraction with valuation > 0
5. Normalized mean codon index
6. Std of codon indices

### Trajectory Computation

For each serotype and time point (year):

```python
# Compute centroid of all sequences from that year
centroid = mean(embeddings_for_year)

# Track trajectory over time
trajectory = [centroid_2015, centroid_2016, ..., centroid_2024]
```

### Velocity and Forecasting

```python
# Velocity = direction of recent movement
velocity = (centroid_current - centroid_previous) / time_delta

# Forecast = extrapolate trajectory
predicted_position = centroid_current + velocity * steps_ahead

# Risk score = divergence from origin
risk_score = distance(predicted_position, origin) / distance(current, origin)
```

### Primer Stability Score

For each genomic window position:

```python
# Collect all sequences at this position
sequences = [seq[pos:pos+window_size] for seq in all_sequences]

# Compute embedding variance
embeddings = [embed(seq) for seq in sequences]
variance = mean(var(embeddings, axis=0))

# Stability = inverse of variance
stability_score = 1.0 / (1.0 + variance)
```

**Higher stability** = less evolutionary change = better primer target

---

## Output Formats

### Trajectory JSON

```json
{
  "serotypes": {
    "DENV-1": {
      "trajectory": [
        {
          "time": "2015",
          "centroid": [0.458, 0.752, 2.949, 0.324, 0.492, 0.286],
          "n_sequences": 5,
          "variance": 0.023
        },
        {
          "time": "2020",
          "centroid": [0.465, 0.769, 2.980, 0.321, 0.495, 0.288],
          "n_sequences": 8,
          "variance": 0.031
        }
      ],
      "velocity": {
        "direction": [0.068, 0.203, 0.972, -0.005, 0.033, 0.024],
        "magnitude": 0.0215,
        "time_window": "2015 to 2020"
      },
      "forecast": {
        "current_position": [0.465, 0.769, 2.980, ...],
        "predicted_position": [0.466, 0.773, 3.000, ...],
        "confidence": 0.456,
        "risk_score": 1.007
      }
    }
  },
  "summary": {
    "total_serotypes": 4,
    "fastest_moving": "DENV-3",
    "highest_risk": "DENV-3"
  }
}
```

### Primer Candidates CSV

| Column | Description |
|--------|-------------|
| `rank` | Priority ranking (1 = best) |
| `position` | Start position in genome |
| `sequence` | 20nt primer sequence |
| `stability_score` | Evolutionary stability (0-1) |
| `conservation_score` | Sequence identity fraction |
| `combined_score` | stability × conservation |
| `gc_content` | GC percentage (0.4-0.6 optimal) |
| `tm_estimate` | Melting temperature (°C) |
| `variance_over_time` | Embedding variance across years |
| `n_sequences` | Number of sequences analyzed |

---

## Using with Real NCBI Data

### Step 1: Download Paraguay Dengue Data

```bash
python ingest_arboviruses.py \
    --virus dengue \
    --geo_location "Paraguay" \
    --output ../data/dengue_paraguay_real.fasta
```

This wraps the NCBI Datasets CLI command:
```bash
datasets download virus genome taxon 12637 \
    --geo-location "Paraguay" \
    --include genome
```

### Step 2: Analyze Trajectories

```bash
python arbovirus_hyperbolic_trajectory.py \
    --input ../data/dengue_paraguay_real.fasta \
    --output ../results/paraguay_forecast.json
```

### Step 3: Find Paraguay-Specific Primers

```bash
python primer_stability_scanner.py \
    --input ../data/dengue_paraguay_real.fasta \
    --output ../results/paraguay_primers.csv \
    --window_size 20 \
    --top_n 50 \
    --min_gc 0.4 \
    --max_gc 0.6
```

---

## Integration with IICS-UNA Surveillance

### Suggested Workflow

```
                    ┌─────────────────────────────────────┐
                    │         NCBI Virus Database          │
                    │    (Updated weekly/monthly)          │
                    └───────────────┬─────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      ingest_arboviruses.py          │
                    │   Download Paraguay sequences        │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │  Trajectory Analysis      │   │   Primer Stability Scan   │
    │  arbovirus_hyperbolic_    │   │   primer_stability_       │
    │  trajectory.py            │   │   scanner.py              │
    └───────────────┬───────────┘   └───────────────┬───────────┘
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │  dengue_forecast.json     │   │  primer_candidates.csv    │
    │  • Risk scores            │   │  • Top 50 primers         │
    │  • Velocity vectors       │   │  • GC/Tm properties       │
    │  • Predicted positions    │   │  • Stability scores       │
    └───────────────┬───────────┘   └───────────────┬───────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │     IICS-UNA Surveillance           │
                    │         Dashboard                    │
                    │  • Serotype risk alerts              │
                    │  • Primer validation queue           │
                    │  • Historical trend visualization    │
                    └─────────────────────────────────────┘
```

### API for Dashboard Integration

```python
import json

# Load forecast results
with open('dengue_forecast.json') as f:
    forecast = json.load(f)

# Get current risk assessment
highest_risk = forecast['summary']['highest_risk']
fastest_moving = forecast['summary']['fastest_moving']

# Get detailed serotype data
for serotype, data in forecast['serotypes'].items():
    if 'forecast' in data:
        risk = data['forecast']['risk_score']
        confidence = data['forecast']['confidence']
        print(f"{serotype}: Risk={risk:.3f}, Confidence={confidence:.3f}")
```

---

## Key Innovation

**Traditional Surveillance:**
- Counts sequences per serotype
- Reactive to outbreaks
- Primers designed once, may drift

**Our Hyperbolic Approach:**
- Tracks evolutionary trajectories
- Predictive risk assessment
- Continuously monitors primer stability
- Identifies regions resistant to mutation

---

## Validation Checklist

- [ ] Demo data generates correctly
- [ ] Trajectory analysis runs for all 4 serotypes
- [ ] Velocity vectors computed
- [ ] Risk scores generated
- [ ] Primer scanner finds candidates
- [ ] All primers meet GC/Tm criteria
- [ ] Notebook dashboard visualizes correctly
- [ ] Real NCBI download works (requires CLI installed)

---

## Expected Findings with Real Data

When run on actual Paraguay surveillance sequences (2011-2024):

1. **Serotype cycles** visible in trajectory patterns
2. **DENV-2 and DENV-3** likely highest velocity (historically dominant)
3. **Conserved regions** in NS3 and NS5 genes (replication machinery)
4. **Primer candidates** should overlap with published pan-Dengue primers

---

## Questions?

- See docstrings in each script for implementation details
- NCBI Virus database: https://www.ncbi.nlm.nih.gov/labs/virus/
- NCBI Datasets CLI: https://www.ncbi.nlm.nih.gov/datasets/

---

*Prepared as part of the Ternary VAE Bioinformatics Partnership*
*For integration with IICS-UNA arbovirus surveillance program*
