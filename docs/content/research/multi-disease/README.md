# Multi-Disease Platform

> **Unified drug resistance and escape prediction across 11 disease domains.**

---

## Overview

A unified framework for predicting drug resistance and immune escape across viral, bacterial, fungal, and oncology domains.

---

## Supported Diseases

| Disease | Type | Analyzer | Key Features |
|---------|------|----------|--------------|
| **HIV** | Viral | `HIVAnalyzer` | 23 ARVs, r=0.89 |
| **SARS-CoV-2** | Viral | `SARSCoV2Analyzer` | Paxlovid, mAb escape |
| **Tuberculosis** | Bacterial | `TuberculosisAnalyzer` | 13 drugs, MDR/XDR |
| **Influenza** | Viral | `InfluenzaAnalyzer` | NAIs, vaccine selection |
| **HCV** | Viral | `HCVAnalyzer` | DAAs (NS3/NS5A/NS5B) |
| **HBV** | Viral | `HBVAnalyzer` | Nucleos(t)ide analogues |
| **Malaria** | Parasitic | `MalariaAnalyzer` | K13 artemisinin |
| **MRSA** | Bacterial | `MRSAAnalyzer` | mecA/mecC, MDR |
| **Candida auris** | Fungal | `CandidaAnalyzer` | Pan-resistance alerts |
| **RSV** | Viral | `RSVAnalyzer` | Nirsevimab, palivizumab |
| **Cancer** | Oncology | `CancerAnalyzer` | EGFR/BRAF/KRAS/ALK |

---

## Usage

### Basic Analysis

```python
from src.diseases import TuberculosisAnalyzer, TBGene, TBDrug

analyzer = TuberculosisAnalyzer()

results = analyzer.analyze(
    sequences={TBGene.RPOB: ["...rpoB sequence..."]},
)

print(results["mdr_classification"])  # DS-TB, MDR-TB, pre-XDR-TB, or XDR-TB
print(results["drug_resistance"])     # Per-drug resistance scores
```

### With Uncertainty

```python
from src.diseases.uncertainty_aware_analyzer import (
    UncertaintyAwareAnalyzer,
    UncertaintyConfig,
)

config = UncertaintyConfig(method="evidential", calibrate=True)
analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config, model=model)

results = analyzer.analyze_with_uncertainty(sequences, encodings=x)
# Returns predictions with confidence intervals
```

### Cross-Disease Transfer

```python
from src.training.transfer_pipeline import TransferLearningPipeline

pipeline = TransferLearningPipeline(config)

# Pre-train on all diseases
pipeline.pretrain({
    "hiv": hiv_data,
    "tb": tb_data,
    "influenza": flu_data,
})

# Fine-tune on low-data disease
pipeline.finetune("candida", candida_data)
```

---

## Architecture

### Unified Base Class

All analyzers inherit from `DiseaseAnalyzer`:

```python
class DiseaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, sequences, **kwargs) -> Dict[str, Any]:
        """Analyze sequences for drug resistance."""

    @abstractmethod
    def get_supported_drugs(self) -> List[str]:
        """Return list of supported drugs."""
```

### Disease Registry

```python
from src.diseases import DiseaseRegistry

# Get all registered diseases
diseases = DiseaseRegistry.list_diseases()

# Get analyzer by name
analyzer = DiseaseRegistry.get_analyzer("tuberculosis")
```

---

## Performance

| Disease | Samples | Drugs | Correlation |
|---------|---------|-------|-------------|
| HIV | 200K+ | 23 | +0.890 |
| TB | 20K | 13 | +0.82* |
| Influenza | 10K | 4 | +0.78* |
| SARS-CoV-2 | 50K+ | 3 | +0.85* |

*Preliminary results, validation ongoing

---

## Drug Databases

| Disease | Database | URL |
|---------|----------|-----|
| HIV | Stanford HIVDB | hivdb.stanford.edu |
| TB | MUBII-TB | mubii.u-bordeaux.fr |
| Influenza | IRD | fludb.org |
| SARS-CoV-2 | GISAID | gisaid.org |

---

_Last updated: 2025-12-28_
