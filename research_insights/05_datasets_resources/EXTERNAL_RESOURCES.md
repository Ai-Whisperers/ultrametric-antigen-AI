# External Datasets and Resources

Comprehensive list of datasets, tools, and resources for expanding Ternary VAEs Bioinformatics research.

---

## 1. Genomic Sequence Databases

### Primary Databases

| Database | URL | Data Type | Relevance |
|----------|-----|-----------|-----------|
| **NCBI GenBank** | https://www.ncbi.nlm.nih.gov/genbank/ | DNA/RNA sequences | All genomic analysis |
| **UniProt** | https://www.uniprot.org/ | Protein sequences + PTMs | PTM analysis, autoimmune |
| **Ensembl** | https://www.ensembl.org/ | Annotated genomes | Comparative genomics |
| **RefSeq** | https://www.ncbi.nlm.nih.gov/refseq/ | Curated sequences | Reference genomes |

### Specialized Databases

| Database | URL | Focus |
|----------|-----|-------|
| **Codon Usage Database** | https://www.kazusa.or.jp/codon/ | Codon frequencies by organism |
| **tRNAdb** | http://trna.bioinf.uni-leipzig.de/ | tRNA sequences and structures |
| **SILVA** | https://www.arb-silva.de/ | Ribosomal RNA sequences |

---

## 2. Protein Structure Resources

### Structure Databases

| Resource | URL | Use Case |
|----------|-----|----------|
| **PDB** | https://www.rcsb.org/ | Experimental structures |
| **AlphaFold DB** | https://alphafold.ebi.ac.uk/ | AI-predicted structures |
| **SWISS-MODEL** | https://swissmodel.expasy.org/ | Homology models |
| **ESMFold** | https://esmatlas.com/explore | Fast structure prediction |

### HIV-Specific Structures

| Resource | URL | Content |
|----------|-----|---------|
| **HIV Structural Database** | https://www.hiv.lanl.gov/content/sequence/HIV/mainpage.html | HIV protein structures |
| **PDB HIV entries** | https://www.rcsb.org/search?q=HIV | All HIV structures |
| **BG505 SOSIP** | PDB: 5FYL, 5FYJ, 5FYK | gp140 trimer structures |

---

## 3. Virus-Specific Databases

### HIV Databases

| Database | URL | Content | API |
|----------|-----|---------|-----|
| **Los Alamos HIV DB** | https://www.hiv.lanl.gov/ | Sequences, immunology | Yes |
| **Stanford HIVDB** | https://hivdb.stanford.edu/ | Drug resistance | Yes |
| **CATNAP** | https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/ | Neutralization data | Yes |

### SARS-CoV-2 Databases

| Database | URL | Content |
|----------|-----|---------|
| **GISAID** | https://www.gisaid.org/ | Viral genomes |
| **NextStrain** | https://nextstrain.org/ncov | Evolution tracking |
| **CoVariants** | https://covariants.org/ | Variant tracking |
| **COVID-19 Data Portal** | https://www.covid19dataportal.org/ | Multi-omic data |

### General Virus Databases

| Database | URL | Content |
|----------|-----|---------|
| **ViPR** | https://www.viprbrc.org/ | Viral genomes |
| **RVDB** | https://rvdb.dbi.udel.edu/ | Reference viral database |

---

## 4. PTM and Modification Databases

| Database | URL | PTM Types |
|----------|-----|-----------|
| **PhosphoSitePlus** | https://www.phosphosite.org/ | Phosphorylation, ubiquitination |
| **dbPTM** | https://awi.cuhk.edu.cn/dbPTM/ | All PTM types |
| **UniProt PTM** | https://www.uniprot.org/ptm/ | Curated PTMs |
| **GlyConnect** | https://glyconnect.expasy.org/ | Glycosylation sites |
| **GlycoPATH** | https://glycopath.expasy.org/ | Glycan pathways |

---

## 5. Extremophile Resources

### Genome Databases

| Organism Type | Database | URL |
|---------------|----------|-----|
| **Thermophiles** | GTDB | https://gtdb.ecogenomic.org/ |
| **Archaea** | UCSC Archaeal Browser | https://archaea.ucsc.edu/ |
| **Extremophiles** | EzBioCloud | https://www.ezbiocloud.net/ |

### Specific Organism Data

| Organism | Relevance | Data Source |
|----------|-----------|-------------|
| **Pyrococcus furiosus** | Thermophile model | NCBI: NC_003413 |
| **Deinococcus radiodurans** | Radiation resistant | NCBI: NC_001263 |
| **Tardigrade** | Multi-extremophile | NCBI: GCA_001949185 |
| **Thermococcus kodakarensis** | Hyperthermophile | NCBI: NC_006624 |

---

## 6. Autoimmune Disease Resources

### Rheumatoid Arthritis

| Resource | URL | Content |
|----------|-----|---------|
| **ACPA antigens** | PubMed search | Citrullinated peptide data |
| **RA Gene DB** | https://bioinfo.uth.edu/RA | RA-associated genes |

### General Autoimmune

| Resource | URL | Content |
|----------|-----|---------|
| **AutoAB** | http://autoab.bmb.msu.edu/ | Autoantibody database |
| **IEDB** | https://www.iedb.org/ | Immune epitope database |
| **HLA Database** | https://www.ebi.ac.uk/ipd/imgt/hla/ | HLA allele data |

---

## 7. Origin of Life Resources

### Prebiotic Chemistry

| Resource | URL | Content |
|----------|-----|---------|
| **NASA Astrobiology** | https://astrobiology.nasa.gov/ | Origin of life research |
| **OSIRIS-REx Data** | https://sbn.psi.edu/pds/resource/orex/ | Bennu sample data |
| **Meteorite Database** | https://www.lpi.usra.edu/meteor/ | Meteorite composition |

### Early Life

| Resource | URL | Content |
|----------|-----|---------|
| **Ribosome Evolution** | Georgia Tech | Evolution papers |
| **LUCA Project** | Various | Last Universal Common Ancestor |

---

## 8. Machine Learning Resources

### Pre-trained Models

| Model | Source | Use Case |
|-------|--------|----------|
| **ESM-2** | https://github.com/facebookresearch/esm | Protein embeddings |
| **ProtTrans** | https://github.com/agemagician/ProtTrans | Protein language models |
| **TAPE** | https://github.com/songlab-cal/tape | Protein benchmarks |
| **geoopt** | https://github.com/geoopt/geoopt | Riemannian optimization |

### Datasets for Training

| Dataset | Content | URL |
|---------|---------|-----|
| **ProteinNet** | Protein structures | https://github.com/aqlaboratory/proteinnet |
| **UniRef** | Clustered proteins | https://www.uniprot.org/uniref/ |
| **Pfam** | Protein families | https://www.ebi.ac.uk/interpro/entry/pfam/ |

---

## 9. Video-Specific Data Sources

Based on Anton Petrov videos, specific data to collect:

### Nobel Prize 2025 (Immune System)
- **Source**: Nobel Prize website + laureate papers
- **URL**: https://www.nobelprize.org/prizes/medicine/2025/
- **Data Needed**: Molecular distance thresholds

### Bennu Asteroid
- **Source**: NASA OSIRIS-REx
- **URL**: https://www.nasa.gov/osiris-rex
- **Data Needed**: Amino acid composition ratios

### Fire Amoeba
- **Source**: Original publication (when available)
- **Search**: "extreme temperature amoeba 2025"
- **Data Needed**: Genome sequence, codon usage

### Long COVID Blood Structures
- **Source**: Pretorius et al. publications
- **Search**: "long COVID microclots fibrin"
- **Data Needed**: Protein composition, PTM data

### Medieval Plague
- **Source**: Ancient DNA studies
- **URL**: https://www.ebi.ac.uk/ena/browser/home (European Nucleotide Archive)
- **Data Needed**: Yersinia pestis ancient sequences

---

## 10. Tools and Software

### Bioinformatics Tools

| Tool | Purpose | URL |
|------|---------|-----|
| **Biopython** | Sequence analysis | https://biopython.org/ |
| **HMMER** | Sequence search | http://hmmer.org/ |
| **BLAST** | Sequence alignment | https://blast.ncbi.nlm.nih.gov/ |
| **Clustal Omega** | Multiple alignment | https://www.ebi.ac.uk/Tools/msa/clustalo/ |

### Structural Tools

| Tool | Purpose | URL |
|------|---------|-----|
| **PyMOL** | Structure visualization | https://pymol.org/ |
| **ChimeraX** | Molecular graphics | https://www.cgl.ucsf.edu/chimerax/ |
| **FoldSeek** | Structure search | https://search.foldseek.com/ |

### Machine Learning

| Tool | Purpose | URL |
|------|---------|-----|
| **PyTorch Geometric** | Graph neural networks | https://pytorch-geometric.readthedocs.io/ |
| **geoopt** | Riemannian optimization | https://geoopt.readthedocs.io/ |
| **Weights & Biases** | Experiment tracking | https://wandb.ai/ |

---

## 11. API Access

### Programmatic Data Access

```python
# Example: Fetching data from various sources

# UniProt API
import requests
def get_uniprot_ptms(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    return requests.get(url).json()

# NCBI E-utilities
def get_ncbi_sequence(accession):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    url = f"{base}efetch.fcgi?db=nucleotide&id={accession}&rettype=fasta"
    return requests.get(url).text

# Stanford HIVDB
def get_drug_resistance(sequence):
    url = "https://hivdb.stanford.edu/graphql"
    # GraphQL query for resistance analysis
    pass
```

---

## 12. Download Checklist

### Priority 1 (Immediate)
- [ ] Nobel Prize 2025 Medicine papers
- [ ] Bennu amino acid data
- [ ] Long COVID protein data
- [ ] Stanford HIVDB mutation data

### Priority 2 (Short-term)
- [ ] Thermophile genomes (5+ species)
- [ ] AlphaFold HIV glycan structures
- [ ] GISAID SARS-CoV-2 spike sequences
- [ ] Citrullinome dataset for RA

### Priority 3 (Medium-term)
- [ ] Tardigrade genome + annotations
- [ ] Ancient plague genomes
- [ ] Urban evolution datasets
- [ ] Repeat expansion disease data

---

## 13. Data Storage Plan

Recommended directory structure:
```
data/
├── external/
│   ├── genomes/
│   │   ├── extremophiles/
│   │   ├── viruses/
│   │   └── reference/
│   ├── structures/
│   │   ├── alphafold/
│   │   └── experimental/
│   ├── ptms/
│   └── papers/
└── processed/
    ├── codon_tables/
    ├── padic_embeddings/
    └── goldilocks_predictions/
```
