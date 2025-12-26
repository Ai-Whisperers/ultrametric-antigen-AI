# Data Access Module

Unified API-based access to biological databases for bioinformatics research.

## Overview

This module provides Python clients for accessing multiple biological databases via their APIs, eliminating the need for bulk data downloads.

### Supported Databases

| Database | Data Type | Auth Required | Package |
|----------|-----------|---------------|---------|
| NCBI/Entrez | Sequences, PubMed | Email (API key optional) | biopython |
| Stanford HIVDB | HIV drug resistance | None | requests |
| cBioPortal | Cancer mutations | None | requests |
| MalariaGEN | Malaria genomics | None | malariagen_data |
| CARD | Antibiotic resistance | None | requests |
| BV-BRC | Bacterial/viral genomes | None | requests |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For MalariaGEN (optional - cloud-based data)
pip install malariagen_data
```

## Configuration

1. Copy the environment template:
```bash
cp config/.env.template config/.env
```

2. Edit `.env` with your credentials:
```ini
# Required for NCBI access
NCBI_EMAIL=your.email@example.com

# Optional: for higher rate limits (10/sec instead of 3/sec)
NCBI_API_KEY=your_api_key
```

## Quick Start

```python
from data_access import DataHub

# Initialize
hub = DataHub()

# Check configuration
warnings = hub.validate()
for w in warnings:
    print(f"Warning: {w}")

# Test connections
status = hub.test_connections()
print(status)

# Access databases
hiv_drugs = hub.hivdb.get_drug_classes()
cancer_types = hub.cbioportal.get_cancer_types()
malaria_samples = hub.malariagen.get_pf_sample_metadata()
eskape = hub.card.get_eskape_pathogens()
tb_genomes = hub.bvbrc.get_tb_genomes()
```

## Client Usage Examples

### NCBI/Entrez

```python
from data_access import NCBIClient

ncbi = NCBIClient()

# Search for HIV sequences
results = ncbi.search_hiv_sequences(subtype="B", max_results=100)
print(f"Found {results['count']} sequences")

# Fetch sequence metadata
summary = ncbi.get_sequence_summary(results['ids'][:10])

# Search for SARS-CoV-2
sars = ncbi.search_sars_cov2_sequences(lineage="XBB")

# Search for TB
tb = ncbi.search_tuberculosis_sequences(gene="rpoB")
```

### Stanford HIVDB (Drug Resistance)

```python
from data_access import HIVDBClient

hivdb = HIVDBClient()

# Get drug classes
drugs = hivdb.get_drug_classes()

# Analyze mutations directly
result = hivdb.get_mutations_analysis(["M184V", "K65R"], gene="RT")

# Analyze a sequence
resistance = hivdb.analyze_sequence(hiv_sequence)
summary = hivdb.get_resistance_summary(hiv_sequence)
```

### cBioPortal (Cancer Genomics)

```python
from data_access import CBioPortalClient

cbio = CBioPortalClient()

# Get cancer types
cancer_types = cbio.get_cancer_types()

# Search studies
lung_studies = cbio.get_studies(keyword="lung")

# Get mutations for a gene
tp53_mutations = cbio.get_mutations_by_gene("TP53")

# Get top mutated genes in a study
top_genes = cbio.get_top_mutated_genes("brca_tcga", top_n=20)
```

### MalariaGEN (Cloud-Based)

```python
from data_access import MalariaGENClient

malaria = MalariaGENClient()

# Get P. falciparum metadata (20,864 samples)
pf_samples = malaria.get_pf_sample_metadata()

# Filter by country
kenya_samples = malaria.get_pf_samples_by_country("Kenya")

# Get country summary
summary = malaria.get_pf_country_summary()

# Get drug resistance variants
dr_variants = malaria.get_pf_drug_resistance_variants(gene="kelch13")
```

### CARD (Antibiotic Resistance)

```python
from data_access import CARDClient

card = CARDClient()

# Get ESKAPE pathogens
eskape = card.get_eskape_pathogens()

# Search resistance ontology
aro = card.search_aro("carbapenem")

# Get resistance mechanisms
mechanisms = card.get_resistance_mechanisms()

# Get drug classes
drug_classes = card.get_drug_classes()
```

### BV-BRC (Bacteria & Viruses)

```python
from data_access import BVBRCClient

bvbrc = BVBRCClient()

# Get TB genomes
tb = bvbrc.get_tb_genomes(limit=100)

# Get syphilis genomes
syphilis = bvbrc.get_syphilis_genomes()

# Get AMR phenotypes
amr = bvbrc.get_amr_phenotypes(antibiotic="rifampicin")

# Get virulence factors
vf = bvbrc.get_virulence_factors(genome_id="83332.12")

# Get ESKAPE pathogen genomes
mrsa = bvbrc.get_eskape_genomes("Staphylococcus aureus")
```

## Module Structure

```
data_access/
├── __init__.py          # DataHub and exports
├── requirements.txt     # Dependencies
├── README.md           # This file
├── config/
│   ├── __init__.py
│   ├── .env.template   # Configuration template
│   └── settings.py     # Configuration management
├── clients/
│   ├── __init__.py
│   ├── ncbi_client.py      # NCBI/Entrez
│   ├── hivdb_client.py     # Stanford HIVDB
│   ├── cbioportal_client.py # cBioPortal
│   ├── malariagen_client.py # MalariaGEN
│   ├── card_client.py      # CARD
│   └── bvbrc_client.py     # BV-BRC
└── notebooks/
    └── data_access_examples.py  # Usage examples
```

## Rate Limits

| API | Rate Limit | Notes |
|-----|------------|-------|
| NCBI | 3/sec (10 with key) | Email required |
| HIVDB | No limit | Public GraphQL |
| cBioPortal | No limit | Public REST |
| MalariaGEN | No limit | Cloud data |
| CARD | No limit | Public REST |
| BV-BRC | No limit | Public REST |

## Error Handling

All clients include proper error handling:

```python
from data_access import DataHub

hub = DataHub()

try:
    result = hub.hivdb.analyze_sequence(sequence)
except requests.exceptions.HTTPError as e:
    print(f"API error: {e}")
except requests.exceptions.Timeout:
    print("Request timed out")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Caching

The module supports caching to reduce API calls:

```python
from data_access.config import settings

# Set cache directory
settings.cache_dir = Path("./my_cache")
settings.ensure_cache_dir()
```

## License

This module is part of the bioinformatics research project.
Individual databases have their own terms of use.
