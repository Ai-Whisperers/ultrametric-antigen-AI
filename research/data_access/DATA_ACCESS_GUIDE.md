# Complete Data Access Guide

This document explains all data accessible through the unified data access module.

## Quick Start

```python
from data_access import DataHub

hub = DataHub()

# Check all connections
status = hub.test_connections()
print(status)
```

---

## 1. NCBI/Entrez - Sequence & Literature Database

**Total Records Available:**
- HIV-1 sequences: **1,316,345+**
- SARS-CoV-2 sequences: **9,183,629+**
- Influenza A sequences: **1,458,429+**
- M. tuberculosis sequences: **1,478,393+**
- PubMed articles: **36+ million**

### What You Can Get

#### Sequence Data
```python
# Search for HIV sequences
results = hub.ncbi.search_hiv_sequences(
    subtype="B",           # A, B, C, D, etc.
    gene="pol",            # pol, env, gag, etc.
    country="USA",         # Country filter
    max_results=100
)

# Search SARS-CoV-2 by variant
results = hub.ncbi.search_sars_cov2_sequences(
    lineage="XBB.1.5",     # BA.1, BA.2, XBB, etc.
    gene="S",              # Spike protein
    max_results=100
)

# Search Influenza
results = hub.ncbi.search_influenza_sequences(
    type_="A",             # A, B, C
    subtype="H3N2",        # H1N1, H3N2, H5N1
    segment=4,             # 1-8 (HA is 4, NA is 6)
    max_results=100
)

# Search Tuberculosis
results = hub.ncbi.search_tuberculosis_sequences(
    gene="rpoB",           # rpoB, katG, inhA, gyrA
    max_results=100
)

# Fetch actual sequences
for record in hub.ncbi.fetch_sequences(results['ids'][:10]):
    print(f"{record.id}: {len(record.seq)} bp")
    print(record.seq[:100])

# Get GenBank record with full annotations
gb_record = hub.ncbi.fetch_genbank("NC_001802")  # HIV-1 reference
print(gb_record.features)
```

#### Literature (PubMed)
```python
# Search PubMed
articles = hub.ncbi.fetch_pubmed_abstracts(
    "HIV drug resistance mutations",
    max_results=50
)
print(articles[['pmid', 'title', 'year']])
```

### Available Search Filters
| Organism | Filters Available |
|----------|------------------|
| HIV-1 | subtype, gene, country |
| SARS-CoV-2 | lineage, gene, country |
| Influenza | type, subtype, segment, host |
| M. tuberculosis | gene |

---

## 2. Stanford HIVDB - HIV Drug Resistance

**Current Algorithm:** HIVDB_9.8 (January 2025)

### What You Can Get

#### Drug Resistance Analysis
```python
# Analyze a sequence
sequence = "ATGCCC..."  # Your HIV pol sequence
result = hub.hivdb.analyze_sequence(sequence)

# Get simplified resistance summary
summary = hub.hivdb.get_resistance_summary(sequence)
print(summary[['drug', 'score', 'interpretation']])

# Extract detected mutations
mutations = hub.hivdb.get_mutations_list(sequence)
print(mutations[['gene', 'position', 'mutation']])
```

#### Analyze Mutations Directly
```python
# Analyze specific mutations without sequence
result = hub.hivdb.get_mutations_analysis(
    mutations=["M184V", "K65R", "K103N"],
    gene="RT"
)
# Returns resistance scores for all drugs
```

#### Drug Information
```python
# Get all drugs and classes
drugs = hub.hivdb.get_drug_classes()
print(drugs)  # 27 drugs in 5 classes

# Get gene information
genes = hub.hivdb.get_genes()
print(genes)  # 13 HIV genes monitored
```

### Drug Classes Monitored
| Class | Drugs | Target Gene |
|-------|-------|-------------|
| **NRTI** | ABC, AZT, D4T, DDI, FTC, 3TC, TDF | RT |
| **NNRTI** | DOR, EFV, ETR, NVP, RPV, DPV | RT |
| **PI** | ATV/r, DRV/r, FPV/r, IDV/r, LPV/r, NFV, SQV/r, TPV/r | PR |
| **INSTI** | BIC, CAB, DTG, EVG, RAL | IN |
| **CAI** | LEN (Lenacapavir) | CA |

### Resistance Interpretation Levels
- 1 = Susceptible
- 2 = Potential Low-Level Resistance
- 3 = Low-Level Resistance
- 4 = Intermediate Resistance
- 5 = High-Level Resistance

---

## 3. cBioPortal - Cancer Genomics

**Available Data:**
- **516 cancer studies**
- **898 cancer types**
- Millions of mutation records
- Clinical data, CNV, expression

### What You Can Get

#### Studies & Cancer Types
```python
# Get all studies
studies = hub.cbioportal.get_studies()
print(f"Total: {len(studies)} studies")

# Search specific cancers
breast = hub.cbioportal.get_studies(keyword="breast")
lung = hub.cbioportal.get_studies(keyword="lung")
melanoma = hub.cbioportal.get_studies(keyword="melanoma")

# Get study details
study = hub.cbioportal.get_study("brca_tcga")
```

#### Mutation Data
```python
# Get mutations for a gene across studies
tp53_mutations = hub.cbioportal.get_mutations_by_gene("TP53")
brca1_mutations = hub.cbioportal.get_mutations_by_gene("BRCA1")

# Get mutations from specific study
mutations = hub.cbioportal.get_mutations("brca_tcga")

# Get top mutated genes in a study
top_genes = hub.cbioportal.get_top_mutated_genes("brca_tcga", top_n=20)
```

#### Clinical & Molecular Data
```python
# Get patients and samples
patients = hub.cbioportal.get_patients("brca_tcga")
samples = hub.cbioportal.get_samples("brca_tcga")

# Get clinical data
clinical = hub.cbioportal.get_clinical_data("brca_tcga")

# Get copy number alterations
cna = hub.cbioportal.get_copy_number_alterations("brca_tcga")

# Get gene expression
expression = hub.cbioportal.get_gene_expression(
    "brca_tcga",
    gene_ids=[7157]  # TP53 entrez ID
)
```

### Major Cancer Types Available
- Breast Cancer (37 studies)
- Lung Cancer (41 studies)
- Colorectal Cancer
- Prostate Cancer
- Melanoma
- Leukemia/Lymphoma
- Brain Tumors
- Pancreatic Cancer
- And 800+ more types

---

## 4. CARD - Antibiotic Resistance Database

### What You Can Get

#### ESKAPE Pathogens (Priority Threats)
```python
eskape = hub.card.get_eskape_pathogens()
```

| Code | Pathogen | Type | Key Resistance |
|------|----------|------|----------------|
| E | Enterococcus faecium | Gram+ | Vancomycin (VRE) |
| S | Staphylococcus aureus | Gram+ | Methicillin (MRSA) |
| K | Klebsiella pneumoniae | Gram- | Carbapenems (KPC) |
| A | Acinetobacter baumannii | Gram- | Multi-drug |
| P | Pseudomonas aeruginosa | Gram- | Multi-drug |
| E | Enterobacter species | Gram- | Beta-lactams |

#### Resistance Mechanisms
```python
mechanisms = hub.card.get_resistance_mechanisms()
```
- Antibiotic efflux
- Antibiotic inactivation
- Antibiotic target alteration
- Antibiotic target protection
- Antibiotic target replacement
- Reduced permeability

#### Drug Classes & Genes
```python
# Get drug classes
drug_classes = hub.card.get_drug_classes()
# aminoglycoside, beta-lactam, fluoroquinolone, 
# glycopeptide, macrolide, tetracycline, etc.

# Search resistance ontology
aro_terms = hub.card.search_aro("carbapenem")

# Get resistance genes for pathogen
genes = hub.card.get_resistance_genes(pathogen="Escherichia coli")

# Get SNPs in resistance gene
snps = hub.card.get_snp_data("rpoB")
```

---

## 5. BV-BRC - Bacterial & Viral Genomes

**Contains:** Thousands of complete bacterial and viral genomes with annotations

### What You Can Get

#### Pathogen Genomes
```python
# Mycobacterium tuberculosis (TB)
tb = hub.bvbrc.get_tb_genomes(limit=100)

# Treponema pallidum (Syphilis)
syphilis = hub.bvbrc.get_syphilis_genomes(limit=100)

# ESKAPE pathogens
mrsa = hub.bvbrc.get_eskape_genomes("Staphylococcus aureus")
klebsiella = hub.bvbrc.get_eskape_genomes("Klebsiella pneumoniae")
pseudomonas = hub.bvbrc.get_eskape_genomes("Pseudomonas aeruginosa")

# Custom organism search
genomes = hub.bvbrc.search_genomes(
    organism="Neisseria gonorrhoeae",
    genome_status="Complete",
    limit=50
)
```

#### AMR Phenotype Data
```python
# Get AMR phenotypes (resistance testing results)
amr = hub.bvbrc.get_amr_phenotypes(
    antibiotic="rifampicin",
    phenotype="Resistant",
    limit=500
)
# Returns: genome_name, antibiotic, resistant_phenotype, measurement
```

#### Genome Features & Genes
```python
# Get genome details
genome = hub.bvbrc.get_genome("1773.228")

# Get genes/features
features = hub.bvbrc.get_genome_features(
    genome_id="1773.228",
    feature_type="CDS"
)

# Get specialty genes (virulence, resistance)
virulence = hub.bvbrc.get_virulence_factors(genome_id="1773.228")
resistance = hub.bvbrc.get_resistance_genes(genome_id="1773.228")

# Get pathways
pathways = hub.bvbrc.get_pathways("1773.228")

# Get subsystems
subsystems = hub.bvbrc.get_subsystems("1773.228")
```

### Organisms with Pre-configured Taxon IDs
| Organism | Taxon ID | Use Case |
|----------|----------|----------|
| M. tuberculosis | 1773 | TB drug resistance |
| T. pallidum | 160 | Syphilis |
| S. aureus | 1280 | MRSA |
| E. coli | 562 | General AMR |
| K. pneumoniae | 573 | Carbapenem resistance |
| P. aeruginosa | 287 | Multi-drug resistance |
| A. baumannii | 470 | Hospital infections |
| E. faecium | 1352 | Vancomycin resistance |
| N. gonorrhoeae | 485 | Gonorrhea AMR |

---

## 6. MalariaGEN - Malaria Genomics

**Note:** Requires Google Cloud authentication for data access.

### Setup for MalariaGEN
```bash
# Install gcloud CLI and authenticate
gcloud auth application-default login
```

### What You Can Get (Once Authenticated)

#### Plasmodium falciparum (Pf7)
- **20,864 samples** from malaria-endemic regions
- SNP calls and variant data
- Drug resistance markers
- Geographic distribution

```python
# Sample metadata
samples = hub.malariagen.get_pf_sample_metadata()

# Samples by country
kenya = hub.malariagen.get_pf_samples_by_country("Kenya")

# Drug resistance variants
dr_variants = hub.malariagen.get_pf_drug_resistance_variants(gene="kelch13")

# SNP calls for genomic region
snps = hub.malariagen.get_pf_snp_calls(region="Pf3D7_13_v3:1-100000")
```

#### Plasmodium vivax (Pv4)
```python
pv_samples = hub.malariagen.get_pv_sample_metadata()
```

#### Anopheles Mosquitoes (Ag3)
```python
mosquito_samples = hub.malariagen.get_ag_sample_metadata()
```

---

## Research Application Matrix

| Research Area | Primary APIs | Key Data |
|---------------|--------------|----------|
| **HIV Drug Resistance** | HIVDB, NCBI | Mutations, scores, sequences |
| **HIV Evolution** | NCBI, HIVDB | Subtype sequences, phylogeny |
| **SARS-CoV-2 Variants** | NCBI | Variant sequences, lineages |
| **Influenza Surveillance** | NCBI | Seasonal strains, segments |
| **Cancer Genomics** | cBioPortal | Mutations, clinical data |
| **TB Drug Resistance** | BV-BRC, NCBI | Genomes, AMR phenotypes |
| **Syphilis** | BV-BRC, NCBI | Genomes, strain typing |
| **ESKAPE Pathogens** | CARD, BV-BRC | Resistance genes, mechanisms |
| **Malaria** | MalariaGEN | Samples, drug resistance |
| **General AMR** | CARD, BV-BRC | Ontology, phenotypes |

---

## Usage Examples for Your Research

### HIV Codon Analysis Extension
```python
# Get HIV sequences for codon analysis
hiv_results = hub.ncbi.search_hiv_sequences(subtype="B", gene="pol", max_results=500)
for record in hub.ncbi.fetch_sequences(hiv_results['ids']):
    # Process codon structure
    sequence = str(record.seq)
    # Apply your geometric analysis...
```

### Drug Resistance Correlation Study
```python
# Get mutations and their resistance scores
mutations = ["M184V", "K65R", "K103N", "Y181C", "G190A"]
for combo in combinations(mutations, 2):
    result = hub.hivdb.get_mutations_analysis(list(combo), gene="RT")
    # Analyze synergy/antagonism patterns
```

### Cross-Pathogen Resistance Analysis
```python
# Compare resistance mechanisms across pathogens
eskape_data = hub.card.get_eskape_pathogens()
for _, pathogen in eskape_data.iterrows():
    amr = hub.bvbrc.get_amr_phenotypes(limit=100)
    # Compare resistance patterns
```

---

---

## 7. UniProt - Protein Sequence & Function Database

**Contains:** 250+ million protein sequences with annotations

### What You Can Get

#### Protein Search
```python
# Search proteins by keyword
kinases = hub.uniprot.search_proteins("kinase", organism="human", limit=100)

# Get human proteins by gene
tp53 = hub.uniprot.get_human_proteins(gene="TP53")

# Get viral proteins
hiv_proteins = hub.uniprot.get_hiv_proteins(gene="gag", strain="HIV-1")
hbv_proteins = hub.uniprot.get_viral_proteins("HBV")
covid_proteins = hub.uniprot.get_viral_proteins("SARS-CoV-2")
```

#### Disease & Drug Targets
```python
# Proteins associated with diseases
cancer_proteins = hub.uniprot.get_cancer_proteins(cancer_type="breast")
disease_proteins = hub.uniprot.get_disease_proteins("Alzheimer disease")

# Drug targets
drug_targets = hub.uniprot.get_drug_targets(organism="human")
```

#### Enzymes & Functions
```python
# Get enzymes by EC number
proteases = hub.uniprot.get_enzymes(ec_number="3.4", organism="human")

# Search by function
dna_repair = hub.uniprot.search_by_function("DNA repair", organism="human")
kinases = hub.uniprot.search_by_keyword("Kinase", organism="human")
```

#### Protein Features & Cross-References
```python
# Get protein details
protein = hub.uniprot.get_protein("P04637")  # p53

# Get sequence
fasta = hub.uniprot.get_protein_sequence("P04637")

# Get features (domains, sites)
features = hub.uniprot.get_protein_features("P04637")

# Get cross-references (PDB, InterPro, etc.)
xrefs = hub.uniprot.get_protein_xrefs("P04637")

# Get proteins with structures
proteins_with_pdb = hub.uniprot.get_proteins_with_structure(organism="human")
```

---

## 8. IEDB - Immune Epitope Database

**Contains:** 1.5+ million epitopes from 4,000+ species

### What You Can Get

#### Epitope Search
```python
# Search epitopes by organism
hiv_epitopes = hub.iedb.get_hiv_epitopes(protein="Gag")
covid_epitopes = hub.iedb.get_covid_epitopes(protein="Spike")
tb_epitopes = hub.iedb.get_tb_epitopes()
malaria_epitopes = hub.iedb.get_malaria_epitopes()

# T-cell epitopes
tcell = hub.iedb.search_tcell_epitopes(organism="HIV-1", mhc_class="I")
ctl = hub.iedb.get_hiv_ctl_epitopes(gene="nef")

# B-cell epitopes
bcell = hub.iedb.search_bcell_epitopes(organism="SARS-CoV-2")
neutralizing = hub.iedb.get_neutralizing_epitopes("HIV-1")
```

#### MHC Binding Predictions
```python
# Predict MHC class I binding
predictions = hub.iedb.predict_mhc_binding(
    sequence="SLYNTVATL",  # HIV Gag epitope
    alleles=["HLA-A*02:01", "HLA-A*03:01"]
)

# Predict MHC class II binding
mhc2 = hub.iedb.predict_mhc_class_ii_binding(
    sequence="AVDLSHFLKEKGGL",
    alleles=["HLA-DRB1*01:01"]
)

# Predict immunogenicity
immunogenicity = hub.iedb.predict_immunogenicity(
    peptides=["SLYNTVATL", "ILKEPVHGV"],
    mhc_allele="HLA-A*02:01"
)
```

#### Population Coverage
```python
# Calculate epitope population coverage
coverage = hub.iedb.calculate_population_coverage(
    epitopes=["SLYNTVATL", "ILKEPVHGV"],
    alleles=["HLA-A*02:01", "HLA-A*24:02"],
    population="World"
)
```

### Common HLA Alleles Available
| Class I | Class II |
|---------|----------|
| HLA-A*02:01 | HLA-DRB1*01:01 |
| HLA-A*01:01 | HLA-DRB1*03:01 |
| HLA-A*03:01 | HLA-DRB1*04:01 |
| HLA-A*24:02 | HLA-DRB1*07:01 |
| HLA-B*07:02 | HLA-DRB1*15:01 |
| HLA-B*08:01 | HLA-DQB1*02:01 |

---

## 9. LANL HIV Database - Curated HIV Data

**Contains:** Curated HIV-1/2/SIV sequences, drug resistance, immunology

### What You Can Get

#### Drug Resistance Mutations
```python
# Get all resistance mutations
all_mutations = hub.lanl.get_resistance_mutations()

# Filter by drug class
nrti_mutations = hub.lanl.get_resistance_mutations(drug_class="NRTI")
pi_mutations = hub.lanl.get_resistance_mutations(drug_class="PI")
insti_mutations = hub.lanl.get_resistance_mutations(drug_class="INSTI")

# Get drug information
drugs = hub.lanl.get_drug_info(drug_class="NNRTI")
```

#### Major Resistance Mutations
| Drug Class | Key Mutations |
|------------|---------------|
| NRTI | M184V/I, K65R, TAMs (41L, 67N, 210W, 215Y/F, 219Q) |
| NNRTI | K103N, Y181C, G190A, E138K |
| PI | I84V, L90M, M46I/L, V82A |
| INSTI | N155H, Q148H/R/K, R263K, G118R |

#### Antibody & Epitope Data
```python
# Get bnAb target sites
bnab_sites = hub.lanl.get_bnab_targets()
# CD4bs, V2 apex, V3 glycan, MPER, fusion peptide

# CTL epitope guidance
ctl_info = hub.lanl.get_ctl_epitopes(protein="Gag", hla="HLA-A*02")

# Antibody epitope guidance
ab_info = hub.lanl.get_antibody_epitopes(protein="Env")
```

#### Subtype Information
```python
# Get subtype descriptions
subtypes = hub.lanl.get_subtype_info()

# Get global distribution
distribution = hub.lanl.get_subtype_distribution()

# Get reference strains
references = hub.lanl.get_reference_strains()
# HXB2, NL4-3, SIVmac239, etc.
```

#### Sequence & Analysis Tools
```python
# Get HXB2 reference positions
ref = hub.lanl.get_reference_sequence(gene="pol")

# Convert amino acid to nucleotide position
position = hub.lanl.get_codon_position(gene="RT", aa_position=184)

# Get alignment tool parameters
alignment = hub.lanl.get_alignment_tool_params(gene="ENV")

# Get phylogenetic analysis info
phylo = hub.lanl.get_phylogenetic_analysis_info()
```

---

## Extended cBioPortal Capabilities

### Structural Variants (Gene Fusions)
```python
# Get structural variants
fusions = hub.cbioportal.get_structural_variants("brca_tcga")

# Get gene fusions (ALK, ROS1, etc.)
alk_fusions = hub.cbioportal.get_gene_fusions("lung_tcga", gene_symbols=["ALK"])

# Get fusion summary
fusion_summary = hub.cbioportal.get_fusion_genes_summary("lung_tcga")
```

### Methylation Data
```python
# Get methylation data for genes
methylation = hub.cbioportal.get_methylation_data(
    "gbm_tcga",
    gene_ids=[4297]  # MGMT
)

# Get methylation across studies
mgmt_meth = hub.cbioportal.get_methylation_by_gene("MGMT")
```

### Protein Levels (RPPA)
```python
# Get protein expression data
protein = hub.cbioportal.get_protein_data("brca_tcga")

# Get protein by gene
egfr_protein = hub.cbioportal.get_protein_by_gene("EGFR")
```

### Survival & Clinical Data
```python
# Get survival data
survival = hub.cbioportal.get_survival_data("brca_tcga")

# Get clinical attributes
attributes = hub.cbioportal.get_clinical_attributes("brca_tcga")

# Get sample clinical data
sample_clinical = hub.cbioportal.get_sample_clinical_data("brca_tcga")
```

### Find Studies by Data Type
```python
# Find studies with methylation data
meth_studies = hub.cbioportal.find_studies_with_data_type("methylation")

# Find studies with protein data
protein_studies = hub.cbioportal.find_studies_with_data_type("protein")

# Check available data types for a study
available = hub.cbioportal.get_available_data_types("brca_tcga")
```

---

## Extended BV-BRC Capabilities

### Epitope Data
```python
# Get epitopes by organism
epitopes = hub.bvbrc.get_epitopes(organism="Mycobacterium tuberculosis")

# Get epitope summary
summary = hub.bvbrc.get_epitope_summary(organism="Staphylococcus aureus")
```

### Protein Structures
```python
# Search structures
structures = hub.bvbrc.search_protein_structures(
    organism="Mycobacterium tuberculosis",
    gene="rpoB"
)

# Get structure details
details = hub.bvbrc.get_structure_details("1HNJ")
```

### Gene Ontology
```python
# Get GO annotations
go = hub.bvbrc.get_gene_ontology(genome_id="1773.228")

# Get GO enrichment
enrichment = hub.bvbrc.get_go_enrichment(
    genome_id="1773.228",
    ontology="biological_process"
)
```

### Pan-Genome & Comparative Genomics
```python
# Compare genomes
comparison = hub.bvbrc.compare_genomes(
    genome_ids=["1773.228", "1773.229", "1773.230"]
)

# Pan-genome analysis
pan = hub.bvbrc.get_pan_genome("Staphylococcus aureus", limit=20)
```

### Extended Specialty Genes
```python
# Drug targets
targets = hub.bvbrc.get_drug_targets(genome_id="1773.228")

# Essential genes
essential = hub.bvbrc.get_essential_genes(genome_id="1773.228")

# Transporters
transporters = hub.bvbrc.get_transporters()
```

---

## Extended NCBI Capabilities

### ClinVar (Clinical Variants)
```python
# Search clinical variants
variants = hub.ncbi.search_clinvar(gene="BRCA1", significance="pathogenic")

# Get variant details
details = hub.ncbi.get_clinvar_details(variant_ids)
```

### dbSNP
```python
# Search SNPs
snps = hub.ncbi.search_snp(gene="TP53", clinical=True)
```

### Additional Virus Searches
```python
# Hepatitis B
hbv = hub.ncbi.search_hbv_sequences(genotype="D", gene="S")

# Hepatitis C
hcv = hub.ncbi.search_hcv_sequences(genotype="1a", region="NS5B")

# Feline Immunodeficiency Virus
fiv = hub.ncbi.search_fiv_sequences(subtype="A", gene="pol")

# Syphilis (Treponema pallidum)
syphilis = hub.ncbi.search_treponema_sequences(subspecies="pallidum")
```

### Protein & Gene Databases
```python
# Search proteins
proteins = hub.ncbi.search_proteins(organism="HIV-1", gene="gag")
protein_seqs = hub.ncbi.fetch_protein_sequences(ids)

# Search genes
genes = hub.ncbi.search_genes(symbol="TP53", organism="human")
gene_details = hub.ncbi.get_gene_details(gene_ids)

# Search structures
structures = hub.ncbi.search_structures(protein="HIV protease")
```

---

## Research Application Matrix (Updated)

| Research Area | Primary APIs | Key Data |
|---------------|--------------|----------|
| **HIV Drug Resistance** | HIVDB, LANL, NCBI | Mutations, scores, curated data |
| **HIV Immunology** | IEDB, LANL | CTL epitopes, bnAb targets, MHC binding |
| **HIV Evolution** | NCBI, LANL | Subtype sequences, phylogeny |
| **Vaccine Design** | IEDB, LANL, UniProt | Epitopes, population coverage, proteins |
| **Cancer Genomics** | cBioPortal | Mutations, fusions, survival |
| **Cancer Proteomics** | cBioPortal, UniProt | Protein levels, PTMs |
| **Clinical Variants** | NCBI (ClinVar) | Pathogenic variants |
| **TB Drug Resistance** | BV-BRC, NCBI | Genomes, AMR phenotypes |
| **Pathogen Proteins** | UniProt, BV-BRC | Virulence factors, drug targets |
| **Malaria** | MalariaGEN | Samples, drug resistance |
| **General AMR** | CARD, BV-BRC | Ontology, phenotypes, genes |

---

## Rate Limits & Best Practices

| API | Rate Limit | Recommendation |
|-----|------------|----------------|
| NCBI | 3/sec (10 with key) | Get an API key |
| HIVDB | None | Batch sequences |
| cBioPortal | None | Cache results |
| MalariaGEN | None (Cloud) | Stream large data |
| CARD | None | Cache ontology |
| BV-BRC | None | Use pagination |
| UniProt | None | Batch queries |
| IEDB | None | Cache predictions |
| LANL | Web-based | Use curated data |

### Tips
1. Cache frequently-used data locally
2. Use batch methods when available
3. Implement retry logic for network issues
4. Store API keys in `.env` (never commit)
5. Use the unified DataHub interface for convenience
6. Check `hub.get_available_data_summary()` for all options
