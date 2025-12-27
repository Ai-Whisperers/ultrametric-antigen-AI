# Implementation Plan: Arbovirus Hyperbolic Surveillance

> **Technical Roadmap for Dengue Trajectory Forecasting**

This document details the technical implementation steps for your arbovirus surveillance project using hyperbolic trajectory analysis and primer stability scanning.

---

## Project Focus

| Aspect | Details |
|--------|---------|
| **Domain** | Arboviruses (Dengue, Zika, Chikungunya) |
| **Key Deliverable** | `Dengue_Serotype_Forecast_Dashboard` |
| **Core Scripts** | `ingest_arboviruses.py`, `arbovirus_hyperbolic_trajectory.py`, `primer_stability_scanner.py` |

---

## Data Acquisition (NCBI Virus)

We need complete genomes of Dengue (DENV-1, 2, 3, 4), Zika, and Chikungunya, specifically chronological series from Paraguay/South America.

- **Source:** NCBI Virus / GenBank
- **Tool:** NCBI Datasets CLI (Recommended) or `Bio.Entrez`
- **Query Parameters:**
  - Organism: Dengue virus (taxid:12637)
  - Geo Location: Paraguay/South America
  - Collection Date: 2011-2025

### Download Command

```bash
datasets download virus genome taxon 12637 \
    --geo-location "Paraguay" \
    --include genome,biosample
```

### Data Format

```fasta
>NC_001477.1 Dengue virus 1, complete genome
ATGAATAACCAACGGAAAAAGGCGAAAAACACGCCTTTCAATATGCTG...

>NC_001474.2 Dengue virus 2, complete genome
AGTTGTTAGTCTACGTGGACCGACAAAGACAGATTCTTTGAGGGAGCT...
```

---

## Implementation Components

### 1. Arbovirus Data Ingestion

**Script:** `scripts/ingest/ingest_arboviruses.py`

**Logic:**
- Wrapper around NCBI Datasets CLI or Bio.Entrez
- Fetch FASTA sequences and metadata (date, location)
- Parse collection dates for temporal analysis
- Filter by geographic region

### 2. Hyperbolic Trajectory Analysis

**Script:** `scripts/analysis/arbovirus_hyperbolic_trajectory.py`

**Logic:**
1. Embed genomes into Hyperbolic Space using p-adic codon encoding
2. Group sequences by Date (Month/Year)
3. Calculate serotype centroids for each time point
4. Compute the "Velocity Vector" of the clade centroid
5. Predict next-month/season trajectory

### 3. Primer Stability Scanner

**Script:** `scripts/analysis/primer_stability_scanner.py`

**Logic:**
1. Scan genome with rolling window (default: 20bp)
2. Compute embedding for each window position
3. Calculate variance across all sequences
4. Identify regions with minimal hyperbolic movement over time
5. Rank primer candidates by stability score

---

## Hyperbolic Embedding Method

### Codon to P-adic Encoding

```python
CODON_MAP = {'GCT': 0, 'GCC': 1, 'GCA': 2, 'GCG': 3, ...}  # 64 codons

def padic_valuation(n, p=3):
    if n == 0:
        return 0
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v
```

### Window Features (6D Embedding)

For each window of codons:
1. Mean p-adic valuation
2. Std of p-adic valuations
3. Max valuation
4. Fraction with valuation > 0
5. Normalized mean codon index
6. Std of codon indices

---

## Trajectory Computation

### Centroid Tracking

```python
# For each serotype and time point
for serotype in ['DENV-1', 'DENV-2', 'DENV-3', 'DENV-4']:
    for year in range(2015, 2024):
        sequences = get_sequences(serotype, year)
        embeddings = [embed(seq) for seq in sequences]
        centroid = np.mean(embeddings, axis=0)
        trajectory[serotype].append(centroid)
```

### Velocity Estimation

```python
# Velocity = direction of recent movement
velocity = (centroid_current - centroid_previous) / time_delta

# Forecast = extrapolate trajectory
predicted_position = centroid_current + velocity * steps_ahead

# Risk score = divergence from origin
risk_score = distance(predicted_position, origin) / distance(current, origin)
```

---

## Primer Stability Score

### Calculation

```python
def primer_stability(position, window_size=20):
    # Collect all sequences at this position
    windows = [seq[position:position+window_size] for seq in all_sequences]

    # Compute embedding for each
    embeddings = [embed_window(w) for w in windows]

    # Calculate variance across sequences
    variance = np.mean(np.var(embeddings, axis=0))

    # Stability = inverse of variance
    return 1.0 / (1.0 + variance)
```

### Output Format

```csv
rank,position,sequence,stability_score,conservation_score,gc_content,tm_estimate
1,7268,GAAATGAGCAGCGGTGTCGC,0.991,0.100,0.60,55.9
2,3600,GAGTTGCGCGGTATTGGTGC,0.989,0.100,0.60,55.9
```

---

## Output Specifications

### Trajectory JSON

```json
{
  "serotypes": {
    "DENV-1": {
      "trajectory": [
        {"time": "2015", "centroid": [...], "n_sequences": 5, "variance": 0.023}
      ],
      "velocity": {
        "direction": [...],
        "magnitude": 0.0215,
        "time_window": "2015 to 2020"
      },
      "forecast": {
        "current_position": [...],
        "predicted_position": [...],
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

---

## IICS-UNA Integration

### Surveillance Workflow

```
                    +-------------------------------------+
                    |         NCBI Virus Database         |
                    |    (Updated weekly/monthly)         |
                    +-------------------+-----------------+
                                        |
                                        v
                    +-------------------------------------+
                    |      ingest_arboviruses.py          |
                    |   Download Paraguay sequences       |
                    +-------------------+-----------------+
                                        |
                    +-----------------+-+-----------------+
                    v                                     v
    +---------------------------+       +---------------------------+
    |  Trajectory Analysis      |       |   Primer Stability Scan   |
    |  arbovirus_hyperbolic_    |       |   primer_stability_       |
    |  trajectory.py            |       |   scanner.py              |
    +-------------+-------------+       +-------------+-------------+
                  |                                   |
                  v                                   v
    +---------------------------+       +---------------------------+
    |  dengue_forecast.json     |       |  primer_candidates.csv    |
    |  * Risk scores            |       |  * Top 50 primers         |
    |  * Velocity vectors       |       |  * GC/Tm properties       |
    +-------------+-------------+       +-------------+-------------+
                  |                                   |
                  +-------------------+---------------+
                                      v
                    +-------------------------------------+
                    |     IICS-UNA Surveillance           |
                    |         Dashboard                   |
                    |  * Serotype risk alerts             |
                    |  * Primer validation queue          |
                    +-------------------------------------+
```

---

## Validation Workflow

1. **Demo Run**: Generate synthetic data and verify pipeline
2. **Real NCBI Download**: Fetch actual Paraguay sequences
3. **Retrospective Validation**: Predict 2023 using 2011-2022 data
4. **Primer Comparison**: Check against published pan-Dengue primers
5. **Forward Deployment**: Monthly automated analysis

---

## Expected Results

With real Paraguay dengue surveillance data:

- **Serotype cycles** visible in trajectory patterns
- **DENV-2 and DENV-3** likely highest velocity (historically dominant)
- **Conserved regions** in NS3 and NS5 genes (replication machinery)
- **Primer candidates** should overlap with published pan-Dengue primers

---

## Dependencies

```bash
pip install numpy biopython pandas matplotlib seaborn

# For NCBI Datasets CLI
# Windows: winget install NCBI.Datasets
# macOS: brew install ncbi-datasets
# Linux: conda install -c conda-forge ncbi-datasets-cli
```

---

*Implementation Plan for Alejandra Rojas (IICS-UNA) - Arbovirus Hyperbolic Surveillance*
