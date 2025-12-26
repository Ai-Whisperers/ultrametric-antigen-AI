# Master Dataset List for Codon Encoder Research

**Purpose**: Comprehensive catalog of all available datasets for research expansion  
**Last Updated**: December 2024

---

## Quick Reference Summary

| Research Area | Primary Database | Records | Access |
|:--------------|:-----------------|--------:|:-------|
| **SARS-CoV-2** | GISAID EpiCoV | 16.9M sequences | Registration |
| **Influenza** | GISAID EpiFlu | 1M+ sequences | Registration |
| **HIV** | Stanford HIVDB | 90K+ mutations | Public |
| **Tuberculosis** | WHO Mutation Catalogue | 52K isolates | Public |
| **Cancer** | TCGA | 11K+ tumors | Public |
| **ESKAPE Bacteria** | CARD + BV-BRC | 6.4K genes, 172K genomes | Public |
| **Malaria** | MalariaGEN Pf7 | 20K samples | Public |
| **HBV** | HBVdb | 23K+ sequences | Public |
| **HCV** | HCV Sequence DB | 5.5K+ NS5A sequences | Public |
| **Syphilis** | NCBI GenBank | ~500 genomes | Public |
| **FIV** | NCBI GenBank | Limited | Public |

---

## 1. SARS-CoV-2 / Coronaviruses

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **GISAID EpiCoV** | [gisaid.org](https://gisaid.org/) | 16.9M+ sequences, variants, metadata | Registration required |
| **NCBI SARS-CoV-2** | [ncbi.nlm.nih.gov/sars-cov-2](https://www.ncbi.nlm.nih.gov/sars-cov-2/) | 8.9M sequences | Public |
| **Nextstrain** | [nextstrain.org/ncov](https://nextstrain.org/ncov/) | Phylogenetic analysis, daily updates | Public |
| **Stanford CoVDB** | [covdb.stanford.edu](https://covdb.stanford.edu/) | Drug resistance data | Public |

### Specialized Resources

| Resource | Content | Use Case |
|:---------|:--------|:---------|
| **CoV-GLUE** | Mutation analysis tool | Variant annotation |
| **GESS Database** | Global sequence evaluation | Quality assessment |
| **CoCoPUTs** | Codon usage statistics | Codon-level analysis |

### Key Data Fields
- Spike protein sequences
- Variant classifications (Pango lineages)
- Drug resistance mutations (nsp5 for Paxlovid)
- Neutralization data (bnAbs, convalescent sera)
- Geographic and temporal metadata

### Download Notes
```
GISAID requires:
1. Registration with institutional email
2. Agreement to data access terms
3. No redistribution without permission
```

---

## 2. Influenza

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **GISAID EpiFlu** | [gisaid.org](https://gisaid.org/) | 1M+ sequences, all subtypes | Registration |
| **NCBI Influenza Virus DB** | [ncbi.nlm.nih.gov/genomes/FLU](https://www.ncbi.nlm.nih.gov/genomes/FLU/Database/nph-select.cgi) | Complete genomes | Public |
| **FluSurver** | [flusurver.bii.a-star.edu.sg](https://flusurver.bii.a-star.edu.sg/) | Mutation analysis | Public |

### Specialized Resources

| Resource | Content | Use Case |
|:---------|:--------|:---------|
| **CDC FluView** | Surveillance data | Epidemic tracking |
| **WHO Vaccine Composition** | Annual strain recommendations | Vaccine selection |
| **Nextflu** | Phylodynamic forecasting | Strain prediction |

### Key Data Fields
- HA/NA sequences (antigenic sites)
- Drug resistance (NA inhibitors, cap-dependent endonuclease)
- Antigenic characterization data
- Vaccine strain selections (historical)

### Subtypes Available
| Subtype | Relevance | Data Volume |
|:--------|:----------|:------------|
| H1N1 | Seasonal, pandemic | High |
| H3N2 | Seasonal, rapid evolution | Very High |
| H5N1 | Avian, pandemic potential | Moderate |
| H7N9 | Avian, emerging | Moderate |
| Influenza B | Seasonal | High |

---

## 3. Tuberculosis (Mycobacterium tuberculosis)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **WHO Mutation Catalogue v2** | [who.int](https://www.who.int/publications/i/item/9789240082410) | 52K+ isolates, 13 drugs, 30K variants | Public |
| **ReSeqTB** | [platform.reseqtb.org](https://platform.reseqtb.org/) | 4.6K+ isolates with WGS + DST | Public |
| **BV-BRC (PATRIC)** | [bv-brc.org](https://www.bv-brc.org/) | Genomes + phenotypes | Public |
| **TB Portals** | [tbportals.niaid.nih.gov](https://tbportals.niaid.nih.gov/) | Clinical + genomic data | Public |

### Specialized Resources

| Resource | Content | Use Case |
|:---------|:--------|:---------|
| **TBProfiler** | Resistance prediction tool | Genotype-phenotype |
| **Mykrobe Predictor** | WGS analysis | Clinical use |
| **PhyResSE** | Phylogenetic + resistance | Transmission tracking |
| **Afro-TB Dataset** | African isolates | Geographic representation |

### Drug Resistance Data
| Drug | Gene(s) | WHO Catalogue Status |
|:-----|:--------|:---------------------|
| Rifampicin | rpoB | Well characterized |
| Isoniazid | katG, inhA | Well characterized |
| Ethambutol | embB | Moderate |
| Pyrazinamide | pncA | Moderate |
| Fluoroquinolones | gyrA, gyrB | Well characterized |
| Aminoglycosides | rrs, eis | Well characterized |
| Bedaquiline | atpE, Rv0678 | Emerging |
| Pretomanid | ddn, fgd1 | Limited |
| Linezolid | rrl, rplC | Emerging |

---

## 4. Cancer / Tumor Evolution

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **TCGA** | [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov/) | 11K+ tumors, 33 cancer types | Public |
| **ICGC** | [dcc.icgc.org](https://dcc.icgc.org/) | International cancer genomes | Public |
| **cBioPortal** | [cbioportal.org](https://www.cbioportal.org/) | Integrated cancer genomics | Public |
| **COSMIC** | [cancer.sanger.ac.uk/cosmic](https://cancer.sanger.ac.uk/cosmic) | Somatic mutations | Registration |

### Immunotherapy-Specific Resources

| Resource | Content | Use Case |
|:---------|:--------|:---------|
| **TCIA** | Imaging + genomics | Radiogenomics |
| **IEDB** | Epitope database | Neoantigen prediction |
| **dbGaP** | Clinical + genomic | Immunotherapy response |
| **GEO** | Gene expression | Immune signatures |

### Key Data for Immunotherapy Analysis
| Data Type | Source | Relevance |
|:----------|:-------|:----------|
| Tumor Mutation Burden (TMB) | TCGA, cBioPortal | Response prediction |
| Neoantigen predictions | pVACtools, NetMHC | T-cell response |
| HLA typing | OptiType, HLA-HD | Epitope restriction |
| Immune infiltration | CIBERSORT, xCell | TME characterization |
| PD-L1 expression | IHC data | Checkpoint response |

### Priority Cancer Types
| Cancer | Immunotherapy Data | TMB Range |
|:-------|:-------------------|:----------|
| Melanoma | Excellent | High |
| NSCLC | Excellent | Moderate-High |
| Bladder | Good | High |
| MSI-high tumors | Excellent | Very High |
| Renal cell | Good | Low-Moderate |

---

## 5. ESKAPE Bacteria / Antimicrobial Resistance

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **CARD** | [card.mcmaster.ca](https://card.mcmaster.ca/) | 6,442 resistance genes, 4,480 SNPs | Public |
| **BV-BRC** | [bv-brc.org](https://www.bv-brc.org/) | 172K+ genomes, phenotypes | Public |
| **NDARO** | [ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance](https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/) | Resistance gene finder | Public |
| **ESKtides** | [phageonehealth.cn](http://www.phageonehealth.cn:9000/ESKtides) | Phage-derived peptides | Public |

### ESKAPE Pathogen Data Availability

| Pathogen | Genomes in BV-BRC | Key Resistance | Priority |
|:---------|------------------:|:---------------|:---------|
| *Enterococcus faecium* | 5,000+ | Vancomycin | High |
| *Staphylococcus aureus* | 50,000+ | Methicillin, Vancomycin | Critical |
| *Klebsiella pneumoniae* | 20,000+ | Carbapenems | Critical |
| *Acinetobacter baumannii* | 10,000+ | Pan-drug | Critical |
| *Pseudomonas aeruginosa* | 15,000+ | Carbapenems | Critical |
| *Enterobacter* spp. | 5,000+ | Cephalosporins | High |

### CARD Statistics
```
Total entries: 8,582 ontology terms
Reference sequences: 6,442
SNP models: 4,480
Publications curated: 3,354
AMR detection models: 6,480
```

### Tools Available
| Tool | Function |
|:-----|:---------|
| RGI (Resistance Gene Identifier) | Predict resistome from sequences |
| BLAST against CARD | Homology search |
| AMRFinderPlus | NCBI resistance detection |
| Abricate | Multiple database search |

---

## 6. Malaria (Plasmodium falciparum)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **MalariaGEN Pf7** | [malariagen.net/data_package/pf7](https://www.malariagen.net/data_package/open-dataset-plasmodium-falciparum-v70/) | 20,864 samples, 33 countries | Public |
| **PlasmoDB** | [plasmodb.org](https://plasmodb.org/) | Reference genomes, annotations | Public |
| **WWARN** | [wwarn.org](https://www.wwarn.org/) | Drug resistance surveillance | Public |

### Pf7 Dataset Contents
| Data Type | Details |
|:----------|:--------|
| Samples | 20,864 from 33 countries |
| SNP calls | Genome-wide variants |
| Drug resistance markers | crt, dhfr, dhps, mdr1, kelch13, plasmepsin2-3 |
| Copy number variants | mdr1 amplification |
| Population structure | Principal components, admixture |

### Kelch13 Resistance Data
| Mutation | Region | Resistance Level |
|:---------|:-------|:-----------------|
| C580Y | SE Asia | High |
| R539T | SE Asia | High |
| Y493H | SE Asia | High |
| R561H | Rwanda (de novo) | Emerging |
| I543T | SE Asia | High |

### Key Genes for Analysis
| Gene | Drug | Function |
|:-----|:-----|:---------|
| kelch13 | Artemisinin | Resistance marker |
| pfcrt | Chloroquine | Transporter |
| pfmdr1 | Multiple | Multidrug resistance |
| dhfr | Pyrimethamine | Folate pathway |
| dhps | Sulfadoxine | Folate pathway |

---

## 7. Hepatitis B (HBV)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **HBVdb** | [hbvdb.lyon.inserm.fr](https://hbvdb.lyon.inserm.fr/) | Annotated sequences, resistance profiles | Public |
| **HBVrtDB** | NCBI-derived | 23,871 RT sequences | Public |
| **Stanford HBV** | [hivdb.stanford.edu/HBV](https://hivdb.stanford.edu/HBV/HBVseq/development/HBVseq.html) | Drug resistance tool | Public |

### Key Features
- **Overlapping reading frames**: Unique constraint analysis opportunity
- **8 genotypes** (A-H): ~10% nucleotide difference
- **Drug resistance profiling**: Lamivudine, Entecavir, Tenofovir, Adefovir

### Resistance Mutations
| Mutation | Drug(s) Affected | Position |
|:---------|:-----------------|:---------|
| rtL180M | Lamivudine, Entecavir, Telbivudine | RT domain |
| rtM204I/V | Lamivudine, Entecavir, Telbivudine | RT domain |
| rtA181T/V | Adefovir, Tenofovir | RT domain |
| rtN236T | Adefovir | RT domain |
| rtT184S | Entecavir | RT domain |

### Unique Analytical Opportunity
```
HBV Overlapping Frames:
Pol (RT) ←→ Surface (HBsAg)

Single nucleotide change affects BOTH proteins!
This creates ultra-constrained positions.
```

---

## 8. Hepatitis C (HCV)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **HCV Sequence Database** | Los Alamos / euHCVdb | 5,465+ NS5A sequences | Public |
| **HCV Guidance** | [hcvguidelines.org](https://www.hcvguidelines.org/evaluate/resistance) | Clinical resistance info | Public |
| **GenBank HCV** | NCBI | Complete genomes | Public |

### DAA Resistance Data
| Drug Class | Target | Key RASs |
|:-----------|:-------|:---------|
| NS3/4A inhibitors | Protease | Q80K (GT1a), D168V/A |
| NS5A inhibitors | NS5A | Y93H, L31M, Q30R |
| NS5B inhibitors | Polymerase | S282T (rare) |

### Genotype Distribution
| Genotype | Global Prevalence | DAA Response |
|:---------|:------------------|:-------------|
| GT1a | 46% (Americas) | Good, some RAS issues |
| GT1b | 22% (Europe, Asia) | Excellent |
| GT3 | 22% (South Asia) | More challenging |
| GT2 | 9% | Excellent |
| GT4-6 | <5% | Variable |

---

## 9. Syphilis (Treponema pallidum)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **NCBI GenBank** | ncbi.nlm.nih.gov | ~500 genomes, 196 near-complete | Public |
| **BioProject PRJNA723099** | NCBI | Multi-continent sequences | Public |

### Available Data
| Data Type | Count | Source |
|:----------|------:|:-------|
| Near-complete genomes | 196 | 8 countries, 6 continents |
| Nichols clade genomes | 90 | Madagascar dominant |
| SS14 clade genomes | ~100 | Global distribution |

### Research Potential
- **Antibiotic resistance**: Azithromycin resistance emerging
- **Vaccine targets**: No current vaccine exists
- **Immune evasion**: Master of persistence
- **Co-infection studies**: Strong HIV overlap

### Key Accession Numbers
```
Colombia isolate: MN630242
Brazil isolate: MF370550
SRA data: PRJNA645463
Consensus genomes: CP073381-CP073576
```

---

## 10. FIV (Feline Immunodeficiency Virus)

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **NCBI GenBank** | ncbi.nlm.nih.gov | Complete genomes, partial sequences | Public |
| **NCBI Virus** | ncbi.nlm.nih.gov/labs/virus | Lentivirus collection | Public |

### Available Sequences
| Region | Accession | Source |
|:-------|:----------|:-------|
| Colombia (Subtype A) | MN630242 | Complete genome |
| Brazil | MF370550 | Near-complete |
| Reference (Petaluma) | M25381 | Original isolate |

### FIV Genome Structure
```
Length: ~9,472 bp
Genes: gag, pol, env, vif, ORF-A, rev

Subtypes: A-F (A and B most prevalent)
```

### Research Value
| Aspect | HIV Comparison | Research Opportunity |
|:-------|:---------------|:---------------------|
| Genome | ~9.4 kb vs ~9.7 kb | Similar structure |
| Pathology | AIDS-like in cats | Natural model |
| Tenofovir response | May work as vaccine! | Unique finding |
| Escape routes | Potentially simpler | Easier analysis |

---

## 11. Drug-Drug Interactions

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **Liverpool HIV Interactions** | [hiv-druginteractions.org](https://www.hiv-druginteractions.org/checker) | ART + concomitant drugs | Public |
| **NIH HIV Guidelines** | [clinicalinfo.hiv.gov](https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv/drug-interactions-overview) | Official recommendations | Public |
| **DrugBank** | [drugbank.com](https://go.drugbank.com/) | Comprehensive drug data | Registration |

### Key Interaction Categories
| ART Class | CYP Interaction | Psychiatric Drug Concern |
|:----------|:----------------|:-------------------------|
| PIs (boosted) | CYP3A4 inhibition | Major - dose adjustments |
| NNRTIs | CYP3A4 induction | Moderate |
| INSTIs (unboosted) | Minimal | Preferred for psych combo |
| Lenacapavir | Under study | Sertraline interaction noted |

### Lenacapavir-Sertraline Interaction
```
Observation: Lenacapavir affects reuptake mechanisms
Impact: Diminished sertraline (serotonin) reuptake
Clinical relevance: Many HIV patients on SSRIs
Research need: Mechanism at molecular level
```

---

## 12. Immunology / CD4-CD8 Data

### Primary Databases

| Database | URL | Content | Access |
|:---------|:----|:--------|:-------|
| **ImmPort** | [immport.org](https://www.immport.org/) | Shared immunology data | Public |
| **IEDB** | [iedb.org](https://www.iedb.org/) | Epitope database | Public |
| **FlowRepository** | [flowrepository.org](https://flowrepository.org/) | Flow cytometry data | Public |
| **GEO** | [ncbi.nlm.nih.gov/geo](https://www.ncbi.nlm.nih.gov/geo/) | Gene expression | Public |

### HIV-Specific Immunology Data
| Source | Data Type | Access |
|:-------|:----------|:-------|
| MACS/WIHS cohorts | Longitudinal CD4/CD8 | dbGaP |
| ACTG trials | Treatment response | Collaboration |
| LANL HIV Immunology | CTL epitopes | Public |

### Key Metrics Available
| Metric | Clinical Relevance | Data Source |
|:-------|:-------------------|:------------|
| CD4 count | Disease progression | Clinical cohorts |
| CD4/CD8 ratio | Immune status | Flow cytometry |
| CD8 activation (CD38/HLA-DR) | Inflammation | Research studies |
| Viral load | Treatment response | Clinical data |

---

## Data Access Summary

### Fully Public (No Registration)
- NCBI (GenBank, SRA, Virus)
- WHO TB Mutation Catalogue
- CARD
- BV-BRC
- MalariaGEN
- PlasmoDB
- HBVdb
- Liverpool HIV Interactions
- IEDB

### Registration Required (Free)
- GISAID (sequences)
- COSMIC (cancer mutations)
- DrugBank
- ImmPort

### Collaboration/Application Required
- TCGA (controlled access for some data)
- dbGaP (clinical data)
- ACTG trials

---

## Recommended Download Priority

### Phase 1: Immediate
| Dataset | Size Estimate | Priority |
|:--------|:--------------|:---------|
| Stanford CoVDB (resistance) | Small | Immediate |
| GISAID SARS-CoV-2 (subset) | Variable | Immediate |
| NCBI Influenza DB | ~50 GB full | Immediate |
| WHO TB Catalogue | Small | Immediate |

### Phase 2: Near-term
| Dataset | Size Estimate | Priority |
|:--------|:--------------|:---------|
| MalariaGEN Pf7 | ~100 GB | High |
| CARD complete | ~1 GB | High |
| HBVdb sequences | Small | High |
| TCGA (selected cancers) | Variable | High |

### Phase 3: Extended
| Dataset | Size Estimate | Priority |
|:--------|:--------------|:---------|
| FIV (NCBI) | Small | Medium |
| Syphilis genomes | Small | Medium |
| BV-BRC ESKAPE | Large | Medium |

---

## Sources

### SARS-CoV-2
- [GISAID](https://gisaid.org/)
- [NCBI SARS-CoV-2 Resources](https://www.ncbi.nlm.nih.gov/sars-cov-2/)
- [Nextstrain](https://nextstrain.org/ncov/)
- [Stanford CoVDB](https://covdb.stanford.edu/)

### Influenza
- [NCBI Influenza Virus Database](https://www.ncbi.nlm.nih.gov/genomes/FLU/Database/nph-select.cgi)
- [GISAID Human Influenza Vaccine Composition](https://gisaid.org/resources/human-influenza-vaccine-composition/)

### Tuberculosis
- [WHO TB Mutation Catalogue](https://www.who.int/publications/i/item/9789240082410)
- [ReSeqTB Benchmarking Study](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2658-z)
- [BV-BRC AMR Documentation](https://www.bv-brc.org/docs/data_protocols/antimicrobial_resistance.html)

### Cancer
- [TCGA GDC Portal](https://portal.gdc.cancer.gov/)
- [cBioPortal](https://www.cbioportal.org/)
- [Neoantigens and TMB in Immunotherapy](https://jhoonline.biomedcentral.com/articles/10.1186/s13045-025-01732-z)

### AMR/ESKAPE
- [CARD Database](https://card.mcmaster.ca/)
- [BV-BRC Introduction](https://academic.oup.com/nar/article/51/D1/D678/6814465)

### Malaria
- [MalariaGEN Pf7 Dataset](https://www.malariagen.net/data_package/open-dataset-plasmodium-falciparum-v70/)
- [Kelch13 Secrets](https://www.malariagen.net/blog/secrets-kelch13)

### Hepatitis
- [HBVdb](https://hbvdb.lyon.inserm.fr/)
- [HBV RT Database Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4374605/)
- [HCV Guidance](https://www.hcvguidelines.org/evaluate/resistance)

### Syphilis
- [T. pallidum Genome Sequencing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8735616/)

### FIV
- [FIV Molecular Biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC3230847/)

### Drug Interactions
- [Liverpool HIV Drug Interactions](https://www.hiv-druginteractions.org/checker)
- [NIH HIV Drug Interaction Guidelines](https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv/drug-interactions-overview)

---

*This document catalogs publicly available datasets for expanding codon encoder research beyond HIV.*
