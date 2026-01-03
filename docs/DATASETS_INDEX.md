# Centralized Datasets Index

**Doc-Type:** Reference Index · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

This document provides a **single source of truth** for all external datasets, databases, and resources used or planned for the Ternary VAE project. It distinguishes between:

- **VALIDATED** - Datasets we have used and validated
- **PLANNED** - Datasets identified for future validation
- **REFERENCE** - External resources for context/literature

---

## 1. Protein Stability (ΔΔG)

### VALIDATED

| Dataset | URL | Description | Usage | Status |
|---------|-----|-------------|-------|--------|
| **ProThermDB** | https://web.iitm.ac.in/bioinfo2/prothermdb/ | 32,000+ thermodynamic entries (Tm, ΔG, ΔΔG) | `jose_colbes/` DDG training | ✅ Online |

### PLANNED

| Dataset | URL | Description | Priority | Status |
|---------|-----|-------------|----------|--------|
| **DDG EMB** | https://ddgemb.biocomp.unibo.it/datasets/ | S2450 (2450 mutations), S669 (669 mutations), ptMUL-NR (82 multi-point) | HIGH | ✅ Online |
| **ThermoMutDB** | https://biosig.lab.uq.edu.au/thermomutdb/ | Curated mutation effects | MEDIUM | Unverified |
| **FireProtDB** | https://loschmidt.chemi.muni.cz/fireprotdb/ | Protein stability engineering | LOW | Unverified |

---

## 2. Contact Prediction & Structure

### PLANNED

| Dataset | URL | Description | Priority | Status |
|---------|-----|-------------|----------|--------|
| **PSICOV Benchmark** | http://bioinfadmin.cs.ucl.ac.uk/downloads/PSICOV/ | 150 proteins with contact maps (.aln, .contacts) | HIGH | ⚠️ SSL issues |
| **PconsC4** | https://github.com/ElofssonLab/PconsC4 | Contact prediction tool (requires external benchmarks) | HIGH | ✅ Online |
| **AlphaFold Database** | https://alphafold.ebi.ac.uk/download | 214M protein structures, CC-BY-4.0 | MEDIUM | ✅ Online |

---

## 3. Codon/CDS Databases

### PLANNED

| Dataset | URL | Description | Priority | Status |
|---------|-----|-------------|----------|--------|
| **CoDNaS** | http://ufq.unq.edu.ar/codnas/ | Curated CDS-structure pairs (actual codons) | HIGH | ❌ Offline |
| **UniProt CDS** | https://www.uniprot.org/downloads | Complete proteomes with EMBL/GenBank CDS | MEDIUM | ✅ Online |
| **NCBI CDS** | https://www.ncbi.nlm.nih.gov/datasets/ | Coding sequences via datasets API | MEDIUM | ✅ Online |

> ⚠️ **CoDNaS Alternative**: If CoDNaS remains offline, use UniProt → EMBL cross-references to obtain CDS sequences.

---

## 4. Antimicrobial Peptides (AMP)

### VALIDATED

| Dataset | URL | Description | Usage | Status |
|---------|-----|-------------|-------|--------|
| **APD3** | https://aps.unmc.edu/ | >3,000 validated AMPs | `carlos_brizuela/` training | ⚠️ SSL issues |
| **DRAMP** | http://dramp.cpu-bioinfor.org/ | 30,260 AMPs (v4.0, Apr 2024), activity + clinical data | `carlos_brizuela/` training | ✅ Online |
| **DBAASP** | https://dbaasp.org/ | AMP activity, structure, hemolysis, synergy data | Hemolysis data | ✅ Online |

### REFERENCE

| Dataset | URL | Description | Status |
|---------|-----|-------------|--------|
| **HemoPI** | https://webs.iiitd.edu.in/raghava/hemopi/ | Hemolytic peptide database | Unverified |
| **CAMPR3** | http://www.camp.bicnirrh.res.in/ | Collection of AMPs | Unverified |

---

## 5. HIV/Drug Resistance

### VALIDATED

| Dataset | URL | Description | Usage | Status |
|---------|-----|-------------|-------|--------|
| **Stanford HIVdb** | https://hivdb.stanford.edu/ | Drug resistance scores, mutation comments, GraphQL API | `hiv_research_package/` | ✅ Online (JS required) |
| **Los Alamos HIV** | https://www.hiv.lanl.gov/ | Sequences, epitopes, alignments, neutralization | Vaccine targets | ⚠️ SSL issues |

### VALIDATED (Downloaded)

| Source | Repository/URL | Local Path | Status |
|--------|----------------|------------|--------|
| **HuggingFace** | `damlab/human_hiv_ppi` | `data/external/huggingface/human_hiv_ppi/` | ✅ Downloaded |
| **HuggingFace** | `damlab/HIV_V3_coreceptor` | `data/external/huggingface/HIV_V3_coreceptor/` | ✅ Downloaded |
| **GitHub** | `lucblassel/HIV-DRM-machine-learning` | Drug resistance ML data (African & UK) | ✅ Available |

### PLANNED

| Dataset | URL | Description | Priority | Status |
|---------|-----|-------------|----------|--------|
| **Kaggle HIV-1/2** | https://www.kaggle.com/datasets/protobioengineering/hiv-1-and-hiv-2-rna-sequences | RNA sequences (FASTA/GenBank) | LOW | Unverified |
| **Zenodo Tropism** | https://zenodo.org/record/6475667 | CCR5/CXCR4 sequences | MEDIUM | Unverified |

See [`data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md`](../data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md) for complete HIV dataset guide.

---

## 6. Arbovirus (DENV, ZIKV, CHIKV)

### REFERENCE

| Dataset | URL | Description | Usage |
|---------|-----|-------------|-------|
| **NCBI Virus** | https://www.ncbi.nlm.nih.gov/labs/virus/ | Viral sequences by taxon | `alejandra_rojas/` primers |
| **ViPR** | https://www.viprbrc.org/ | Virus Pathogen Resource | Sequence retrieval |

---

## 7. Protein Language Models

### REFERENCE

| Model | URL | Description |
|-------|-----|-------------|
| **ESM-2** | https://github.com/facebookresearch/esm | Meta AI protein embeddings |
| **ProtTrans** | https://github.com/agemagician/ProtTrans | Protein transformers |

---

## 8. General Bioinformatics

### REFERENCE

| Database | URL | Description |
|----------|-----|-------------|
| **UniProt** | https://www.uniprot.org/ | Protein sequences and annotations |
| **PDB** | https://www.rcsb.org/ | Protein structures |
| **NCBI** | https://www.ncbi.nlm.nih.gov/ | Sequences, literature, taxonomy |
| **Primer3** | https://primer3.org/ | Primer design tool |

---

## Download Locations

All external data should be downloaded to standardized paths:

```
data/
├── external/
│   ├── github/           # Git repositories
│   ├── kaggle/           # Kaggle datasets
│   ├── huggingface/      # HuggingFace datasets
│   ├── zenodo/           # Zenodo archives
│   └── manual/           # Manual downloads
│
research/
└── contact-prediction/
    └── data/
        └── validation/   # PSICOV, PconsC benchmarks
```

---

## Validation Status Legend

| Category | Meaning |
|----------|---------|
| **VALIDATED** | Downloaded, tested, integrated into pipeline |
| **PLANNED** | Identified for future use, not yet downloaded |
| **REFERENCE** | External resource for context, not direct input |

| URL Status | Meaning |
|------------|---------|
| ✅ Online | URL verified working (Jan 2026) |
| ⚠️ SSL issues | Works in browser, certificate warnings |
| ❌ Offline | Connection refused, site may be down |
| Unverified | Not yet tested |

---

## Adding New Datasets

When adding a new dataset:

1. **Categorize** - Determine which section it belongs to
2. **Document** - Add URL, description, and intended usage
3. **Download** - Place in appropriate `data/external/` subdirectory
4. **Validate** - Test loading and format compatibility
5. **Update Status** - Move from PLANNED to VALIDATED

---

## Related Documentation

- [`data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md`](../data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md) - HIV dataset details
- [`deliverables/docs/CURATED_DATABASES.md`](../deliverables/docs/CURATED_DATABASES.md) - Curated training data
- [`deliverables/docs/INDEX.md`](../deliverables/docs/INDEX.md) - Deliverables documentation index

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.1 | Added URL verification status, detailed dataset descriptions |
| 2026-01-03 | 1.0 | Initial centralized index with 8 categories |
