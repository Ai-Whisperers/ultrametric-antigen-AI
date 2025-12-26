"""
Data Access Module - Example Usage

This script demonstrates how to use all the API clients
in the data_access module.

To run as a notebook:
    jupytext --to notebook data_access_examples.py
    jupyter notebook data_access_examples.ipynb

Or run directly:
    python data_access_examples.py
"""

# %% [markdown]
# # Data Access Module Examples
#
# This notebook demonstrates how to use the unified data access module
# to query biological databases via their APIs without downloading bulk data.

# %% Setup and imports
import sys
sys.path.insert(0, "../..")  # Add project root to path

from data_access import DataHub

# Initialize the data hub
hub = DataHub()

# Check configuration
print("Configuration Warnings:")
warnings = hub.validate()
for w in warnings:
    print(f"  - {w}")

# %% [markdown]
# ## Test API Connections

# %% Test connections
print("\nTesting API connections...")
status = hub.test_connections()
print(status.to_string(index=False))

# %% [markdown]
# ## 1. Stanford HIVDB - HIV Drug Resistance Analysis

# %% HIVDB - Get drug classes
print("\n" + "="*60)
print("Stanford HIVDB - HIV Drug Resistance")
print("="*60)

# Get available drug classes
drug_classes = hub.hivdb.get_drug_classes()
print("\nDrug Classes:")
print(drug_classes.head(10).to_string(index=False))

# %% HIVDB - Analyze mutations
# Analyze a set of mutations for drug resistance
mutations = ["M184V", "K65R", "K103N"]
print(f"\nAnalyzing mutations: {mutations}")

result = hub.hivdb.get_mutations_analysis(mutations, gene="RT")
print("\nResistance analysis completed!")

# %% HIVDB - Example sequence analysis (commented to avoid API call)
# Example: Analyze a real HIV sequence
# sequence = "ATGCCC..."  # Your HIV sequence here
# resistance = hub.hivdb.get_resistance_summary(sequence)
# print(resistance)

# %% [markdown]
# ## 2. cBioPortal - Cancer Genomics

# %% cBioPortal - Get cancer types
print("\n" + "="*60)
print("cBioPortal - Cancer Genomics")
print("="*60)

cancer_types = hub.cbioportal.get_cancer_types()
print(f"\nTotal cancer types: {len(cancer_types)}")
print(cancer_types.head(10).to_string(index=False))

# %% cBioPortal - Search studies
# Search for lung cancer studies
lung_studies = hub.cbioportal.get_studies(keyword="lung")
print(f"\nLung cancer studies: {len(lung_studies)}")
if not lung_studies.empty:
    print(lung_studies[["studyId", "name"]].head(5).to_string(index=False))

# %% cBioPortal - Search genes
# Search for TP53
tp53_info = hub.cbioportal.search_genes("TP53")
print("\nTP53 gene info:")
print(tp53_info.to_string(index=False))

# %% [markdown]
# ## 3. CARD - Antibiotic Resistance Database

# %% CARD - Drug classes
print("\n" + "="*60)
print("CARD - Antibiotic Resistance")
print("="*60)

drug_classes = hub.card.get_drug_classes()
print("\nAntibiotic drug classes:")
print(drug_classes.head(10).to_string(index=False))

# %% CARD - ESKAPE pathogens
eskape = hub.card.get_eskape_pathogens()
print("\nESKAPE Pathogens:")
print(eskape.to_string(index=False))

# %% CARD - Resistance mechanisms
mechanisms = hub.card.get_resistance_mechanisms()
print("\nResistance mechanisms:")
print(mechanisms.to_string(index=False))

# %% [markdown]
# ## 4. BV-BRC - Bacterial and Viral Genomes

# %% BV-BRC - Search for TB genomes
print("\n" + "="*60)
print("BV-BRC - Bacterial and Viral Bioinformatics")
print("="*60)

tb_genomes = hub.bvbrc.get_tb_genomes(limit=10)
print(f"\nM. tuberculosis genomes found: {len(tb_genomes)}")
if not tb_genomes.empty:
    print(tb_genomes[["genome_id", "genome_name", "genome_status"]].head().to_string(index=False))

# %% BV-BRC - AMR phenotypes
amr_data = hub.bvbrc.get_amr_phenotypes(limit=20)
print(f"\nAMR phenotype records: {len(amr_data)}")
if not amr_data.empty:
    print(amr_data.head().to_string(index=False))

# %% [markdown]
# ## 5. MalariaGEN - Malaria Genomics (Cloud-Based)
#
# Note: Requires `malariagen_data` package:
# ```bash
# pip install malariagen_data
# ```

# %% MalariaGEN - Sample metadata
print("\n" + "="*60)
print("MalariaGEN - Malaria Genomics")
print("="*60)

try:
    # Get dataset summary
    summary = hub.malariagen.get_dataset_summary()
    print("\nMalariaGEN Datasets:")
    print(summary.to_string(index=False))

    # Get country summary
    country_summary = hub.malariagen.get_pf_country_summary()
    print("\nTop 10 countries by sample count:")
    print(country_summary.head(10).to_string(index=False))

except ImportError:
    print("\nMalariaGEN package not installed.")
    print("Install with: pip install malariagen_data")

# %% [markdown]
# ## 6. NCBI/Entrez - Sequence Database
#
# Note: Requires NCBI_EMAIL to be configured in .env

# %% NCBI - Search sequences
print("\n" + "="*60)
print("NCBI/Entrez - Sequence Database")
print("="*60)

try:
    # Search for HIV-1 sequences
    hiv_results = hub.ncbi.search_hiv_sequences(subtype="B", max_results=10)
    print(f"\nHIV-1 subtype B sequences found: {hiv_results['count']}")

    # Get summary of first few
    if hiv_results['ids']:
        summary = hub.ncbi.get_sequence_summary(hiv_results['ids'][:5])
        print("\nFirst 5 sequences:")
        print(summary[["accession", "title", "length"]].to_string(index=False))

except ValueError as e:
    print(f"\nNCBI Error: {e}")
    print("Configure NCBI_EMAIL in .env file")

# %% [markdown]
# ## Cross-Database Queries
#
# The DataHub provides convenience methods for common cross-database queries.

# %% Cross-database - ESKAPE summary
print("\n" + "="*60)
print("Cross-Database Queries")
print("="*60)

# Get ESKAPE pathogen summary
eskape_summary = hub.get_eskape_summary()
print("\nESKAPE Pathogen Summary:")
print(eskape_summary.to_string(index=False))

# %% [markdown]
# ## Configuration Summary

# %% Configuration info
print("\n" + "="*60)
print("Configuration Summary")
print("="*60)

from data_access.config import settings

print(f"\nCache directory: {settings.cache_dir}")
print(f"Request timeout: {settings.timeout}s")
print(f"Debug mode: {settings.debug}")
print(f"\nNCBI email configured: {bool(settings.ncbi.email)}")
print(f"NCBI API key configured: {bool(settings.ncbi.api_key)}")
print(f"HIVDB endpoint: {settings.hivdb.endpoint}")

# %% [markdown]
# ## Next Steps
#
# 1. Configure your `.env` file with NCBI credentials
# 2. Install optional packages: `pip install malariagen_data`
# 3. Explore specific databases based on your research needs
# 4. Use the `DataHub` for unified access or individual clients for more control
