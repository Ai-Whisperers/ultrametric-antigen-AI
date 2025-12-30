# Alejandra Rojas - 10 Research Ideas

> **Future research directions based on the IICS-UNA Arbovirus Surveillance Partnership**

**Document Version:** 1.0
**Last Updated:** December 29, 2025

---

## Overview

These 10 research ideas build on Alejandra Rojas' work at IICS-UNA on arbovirus surveillance, leveraging the Ternary VAE framework's p-adic trajectory analysis and hyperbolic embeddings for dengue forecasting and primer design.

---

## Idea 1: Real-Time Serotype Dominance Prediction System

### Concept
Deploy a real-time surveillance dashboard that continuously ingests new dengue sequences from Paraguay and predicts which serotype will dominate in the next epidemic season.

### Technical Approach
```
Data Flow:
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ NCBI/GISAID │ ──► │ Daily Ingest │ ──► │ Update         │
│ New Seqs    │     │ Pipeline     │     │ Trajectory     │
└─────────────┘     └──────────────┘     └────────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ Alert       │ ◄── │ Risk Score   │ ◄── │ Velocity       │
│ Dashboard   │     │ Computation  │     │ Extrapolation  │
└─────────────┘     └──────────────┘     └────────────────┘
```

### Key Deliverables
- Automated NCBI sequence ingestion script
- 14-day, 30-day, 90-day forecast models
- Alert thresholds for Ministry of Health integration
- Historical validation (2015-2024 data)

### Impact
- 2-4 weeks earlier outbreak detection
- Optimized hospital resource allocation
- Evidence-based public health response

### Resources Needed
- Compute: 1 GPU server for daily updates
- Data: NCBI API access, local sequence database
- Collaboration: IICS-UNA epidemiology team

---

## Idea 2: Pan-Arbovirus Primer Library

### Concept
Extend the primer stability scanner to design a comprehensive primer library covering Dengue (all 4 serotypes), Zika, Chikungunya, and Mayaro virus - all arboviruses circulating in Paraguay.

### Technical Approach
1. **Sequence Collection**: Download all available sequences per virus
2. **Embedding Generation**: Apply 6D p-adic embedding to each
3. **Stability Scoring**: Compute multi-year stability for each 20nt window
4. **Cross-Virus Validation**: Ensure no cross-reactivity between targets

### Primer Design Criteria
| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Stability Score | > 0.95 | Evolutionary conservation |
| GC Content | 40-60% | PCR efficiency |
| Tm | 55-65°C | Annealing compatibility |
| Self-complementarity | < 3bp | No hairpins |
| Cross-reactivity | < 70% homology | Specificity |

### Expected Output
```
Pan-Arbovirus Primer Library:
├── Dengue/
│   ├── DENV-1_primers.fasta (10 pairs)
│   ├── DENV-2_primers.fasta (10 pairs)
│   ├── DENV-3_primers.fasta (10 pairs)
│   └── DENV-4_primers.fasta (10 pairs)
├── Zika/
│   └── ZIKV_primers.fasta (10 pairs)
├── Chikungunya/
│   └── CHIKV_primers.fasta (10 pairs)
└── Mayaro/
    └── MAYV_primers.fasta (10 pairs)
```

### Impact
- Single multiplex panel for all regional arboviruses
- Reduced diagnostic costs
- Faster differential diagnosis

---

## Idea 3: Antigenic Evolution Tracking

### Concept
Map the antigenic evolution of dengue envelope protein using p-adic embeddings to identify when immune escape variants emerge.

### Technical Approach
- Focus on E protein domain III (neutralizing epitopes)
- Track trajectory in embedding space over time
- Identify "antigenic jumps" - sudden trajectory direction changes
- Correlate with secondary infection severity data

### Key Metrics
| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| Antigenic velocity | Δembedding / Δtime | Evolution rate |
| Antigenic acceleration | Δvelocity / Δtime | Selection pressure |
| Direction change | cos(v_t, v_{t-1}) | Immune escape signal |

### Clinical Application
- Predict secondary infection severity risk
- Guide tetravalent vaccine strain selection
- Identify high-risk clades before outbreaks

---

## Idea 4: Mosquito Vector Surveillance Integration

### Concept
Combine viral genomic trajectories with *Aedes aegypti* population dynamics and climate data for comprehensive outbreak prediction.

### Data Sources
| Data Type | Source | Update Frequency |
|-----------|--------|------------------|
| Viral sequences | NCBI, local labs | Weekly |
| Mosquito density | Ovitraps, larval surveys | Weekly |
| Climate | Weather stations | Daily |
| Historical outbreaks | Ministry of Health | Annual |

### Integrated Model
```python
def outbreak_risk(region, date):
    # Viral trajectory component
    viral_risk = serotype_velocity(region, date)

    # Vector component
    vector_density = mosquito_index(region, date)

    # Climate component
    climate_factor = temp_humidity_index(region, date)

    # Combined risk
    return viral_risk * vector_density * climate_factor
```

### Output
- Regional risk maps updated weekly
- 2-week outbreak probability forecasts
- Resource pre-positioning recommendations

---

## Idea 5: Vaccine Strain Selection Advisor

### Concept
Use trajectory forecasting to recommend optimal vaccine strain composition for Paraguay, similar to influenza vaccine strain selection.

### Methodology
1. **Trajectory Projection**: Extrapolate each serotype 12-18 months forward
2. **Representative Selection**: Identify sequences closest to projected centroids
3. **Coverage Analysis**: Compute expected population immunity coverage
4. **Recommendation Report**: Generate annual vaccine composition guidance

### Example Output
```
PARAGUAY DENGUE VACCINE RECOMMENDATION 2026

Serotype | Recommended Strain | Projection Confidence | Notes
---------|-------------------|----------------------|-------
DENV-1   | PY/2024/ABC123    | 85%                  | Stable trajectory
DENV-2   | PY/2023/DEF456    | 78%                  | Consider alternative
DENV-3   | BR/2024/GHI789    | 62%                  | Rapid evolution, monitor
DENV-4   | PY/2022/JKL012    | 91%                  | Very stable

OVERALL COVERAGE: 79% (95% CI: 72-86%)
```

### Impact
- Data-driven vaccine formulation
- Improved population protection
- Regional collaboration model

---

## Idea 6: Cross-Border Surveillance Network

### Concept
Establish a p-adic trajectory sharing network with neighboring countries (Brazil, Argentina, Bolivia) for early warning of imported serotype introductions.

### Network Architecture
```
┌─────────────────────────────────────────────────────────────┐
│              REGIONAL ARBOVIRUS SURVEILLANCE                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ Paraguay │   │ Brazil  │   │Argentina│   │ Bolivia │    │
│  │ IICS-UNA│   │ FIOCRUZ │   │ ANLIS   │   │ INLASA  │    │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘    │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │   Central   │                         │
│                    │   Hub       │                         │
│                    │  (PAHO)     │                         │
│                    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Sharing Protocol
- Weekly trajectory centroid exchange
- Velocity vector alerts (threshold > 0.05)
- New serotype detection notifications
- Standardized p-adic embedding format

### Impact
- 4-6 weeks earlier detection of imported variants
- Coordinated regional response
- Shared primer validation

---

## Idea 7: Machine Learning Severity Predictor

### Concept
Train a model to predict dengue severity (DF vs. DHF vs. DSS) based on viral sequence p-adic embeddings plus patient factors.

### Features
| Category | Features |
|----------|----------|
| **Viral** | P-adic embedding (6D), serotype, trajectory velocity |
| **Patient** | Age, sex, prior infection history, comorbidities |
| **Clinical** | Day of illness, platelet count, hematocrit |
| **Environmental** | Season, region, concurrent cases |

### Model Architecture
```python
class SeverityPredictor(nn.Module):
    def __init__(self):
        self.viral_encoder = PadicEncoder(input_dim=6, hidden_dim=32)
        self.patient_encoder = MLP(input_dim=10, hidden_dim=32)
        self.classifier = MLP(input_dim=64, output_dim=3)  # DF, DHF, DSS

    def forward(self, viral_embedding, patient_features):
        v = self.viral_encoder(viral_embedding)
        p = self.patient_encoder(patient_features)
        combined = torch.cat([v, p], dim=-1)
        return self.classifier(combined)
```

### Validation
- Retrospective validation on 2020-2023 Paraguay cases
- Prospective validation in 2025 outbreak season
- Compare with WHO warning signs algorithm

---

## Idea 8: Evolutionary Escape Pathway Mapping

### Concept
Map all possible evolutionary pathways from current circulating strains using p-adic distance constraints.

### Methodology
1. **Current State**: Embed all 2024 sequences
2. **Mutation Graph**: Generate all single-nucleotide neighbors
3. **Constraint Filtering**: Remove lethal/highly deleterious mutations
4. **Pathway Analysis**: Find shortest paths to immune escape

### Visualization
```
Current Strain (2024)
       │
       ├──► Pathway A (3 mutations) → Escape Variant A
       │         └── d_padic = 4.2
       │
       ├──► Pathway B (5 mutations) → Escape Variant B
       │         └── d_padic = 6.8 (unlikely, high cost)
       │
       └──► Pathway C (2 mutations) → Partial Escape
                 └── d_padic = 2.1 (monitor closely)
```

### Clinical Application
- Pre-emptive primer updates
- Vaccine escape early warning
- Drug resistance pathway prediction (for antivirals)

---

## Idea 9: Wastewater Surveillance Integration

### Concept
Apply p-adic analysis to wastewater sequencing data for community-level arbovirus monitoring.

### Advantages of Wastewater
| Advantage | Explanation |
|-----------|-------------|
| Population-level | Single sample represents thousands |
| Early detection | Viral shedding precedes symptoms |
| Unbiased | Captures asymptomatic infections |
| Cost-effective | Fewer samples than clinical surveillance |

### Technical Pipeline
```
Wastewater Sample → RNA Extraction → Amplicon Sequencing
                                            │
                                            ▼
                   ┌────────────────────────────────────┐
                   │  Variant Deconvolution Algorithm   │
                   │  (identify mixed serotype signals) │
                   └────────────────────────────────────┘
                                            │
                                            ▼
                   ┌────────────────────────────────────┐
                   │  P-adic Trajectory Integration     │
                   │  (update regional centroids)       │
                   └────────────────────────────────────┘
```

### Implementation
- Partner with Asunción sanitation authority
- Establish 10 sentinel wastewater sites
- Weekly sampling during high season

---

## Idea 10: Training Platform for Regional Labs

### Concept
Develop a training platform to teach p-adic surveillance methods to other Latin American laboratories.

### Platform Components

**1. Online Course Modules:**
- Module 1: Introduction to P-adic Numbers
- Module 2: Hyperbolic Geometry for Biologists
- Module 3: VAE Fundamentals
- Module 4: Trajectory Analysis in Practice
- Module 5: Primer Design Optimization

**2. Hands-On Workshop:**
```
Day 1: Theory
  ├── P-adic distance intuition
  ├── Hyperbolic embeddings
  └── Trajectory interpretation

Day 2: Practice
  ├── Running the ingest pipeline
  ├── Analyzing your own sequences
  └── Generating primer candidates

Day 3: Integration
  ├── Setting up automated surveillance
  ├── Dashboard deployment
  └── Regional data sharing
```

**3. Open-Source Toolkit:**
```bash
pip install ternary-surveillance

# Simple API
from ternary_surveillance import ArbovirusTracker

tracker = ArbovirusTracker(region="Paraguay")
tracker.ingest_sequences("local_sequences.fasta")
tracker.update_trajectories()
tracker.generate_report()
```

### Impact
- Capacity building across Latin America
- Standardized surveillance methodology
- Network of trained collaborators

---

## Summary Table

| # | Idea | Effort | Impact | Priority |
|---|------|--------|--------|----------|
| 1 | Real-Time Prediction System | High | Very High | 1 |
| 2 | Pan-Arbovirus Primer Library | Medium | High | 2 |
| 3 | Antigenic Evolution Tracking | Medium | High | 3 |
| 4 | Mosquito Vector Integration | High | Very High | 4 |
| 5 | Vaccine Strain Selection | Medium | High | 5 |
| 6 | Cross-Border Network | High | Very High | 6 |
| 7 | Severity Predictor | High | Very High | 7 |
| 8 | Escape Pathway Mapping | Medium | Medium | 8 |
| 9 | Wastewater Surveillance | High | High | 9 |
| 10 | Training Platform | Medium | High | 10 |

---

## Next Steps

1. **Prioritize** based on IICS-UNA resources and strategic goals
2. **Pilot** Idea #1 (Real-Time System) with 2024 data
3. **Validate** primer candidates from Idea #2 in laboratory
4. **Propose** regional collaboration for Ideas #4 and #6

---

*Ideas developed based on the Ternary VAE Bioinformatics Partnership*
*For discussion with IICS-UNA research team*
