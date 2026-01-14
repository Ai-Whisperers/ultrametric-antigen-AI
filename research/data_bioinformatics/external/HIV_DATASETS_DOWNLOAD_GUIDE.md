# HIV Datasets Download Guide

This guide lists all freely available HIV datasets for the p-adic hyperbolic codon analysis project.

**Note:** Run the automated script when network is available:
```bash
python scripts/download_hiv_datasets.py --all
```

---

## Quick Download Commands

### 1. GitHub Repositories (git clone)

```bash
# Navigate to data directory
cd data/external/github

# HIV Sequence Data by Length
git clone --depth 1 https://github.com/malabz/HIV-data.git

# Drug Resistance ML Data (African & UK datasets)
git clone --depth 1 https://github.com/lucblassel/HIV-DRM-machine-learning.git

# HIV-1 Nigeria Drug Resistance
git clone --depth 1 https://github.com/pauloluniyi/HIV-1_Paper.git

# HIV Intact Pipeline
git clone --depth 1 https://github.com/ramics/HIVIntact.git

# HIV Sequence Processing (shiver)
git clone --depth 1 https://github.com/ChrisHIV/shiver.git

# HIVSeqinR - R pipeline for HIV sequences
git clone --depth 1 https://github.com/cfe-lab/HIVSeqinR.git
```

---

### 2. Kaggle Datasets (kaggle CLI)

First install and configure Kaggle:
```bash
pip install kaggle
# Get API key from: https://www.kaggle.com/settings → API → Create New Token
# Place kaggle.json in ~/.kaggle/
```

Then download:
```bash
cd data/external/kaggle

# HIV-1 and HIV-2 RNA Sequences (FASTA/GenBank)
kaggle datasets download -d protobioengineering/hiv-1-and-hiv-2-rna-sequences --unzip

# HIV AIDS Statistics (WHO/UNESCO)
kaggle datasets download -d imdevskp/hiv-aids-dataset --unzip

# HIV Surveillance Annual Report
kaggle datasets download -d mostafafaramin/hivaids-annual-report --unzip

# HIV Progression Competition Data
kaggle competitions download -c hivprogression
```

**Manual Download URLs:**
- https://www.kaggle.com/datasets/protobioengineering/hiv-1-and-hiv-2-rna-sequences
- https://www.kaggle.com/datasets/imdevskp/hiv-aids-dataset
- https://www.kaggle.com/datasets/mostafafaramin/hivaids-annual-report
- https://www.kaggle.com/c/hivprogression/data

---

### 3. Hugging Face Datasets (Python)

```python
# Install huggingface_hub
pip install huggingface_hub datasets

# Download via Python
from huggingface_hub import snapshot_download

# Human-HIV Protein-Protein Interactions (16k+ pairs)
snapshot_download("damlab/human_hiv_ppi", repo_type="dataset",
                  local_dir="data/external/huggingface/human_hiv_ppi")

# HIV V3 Coreceptor Usage
snapshot_download("damlab/HIV_V3_coreceptor", repo_type="dataset",
                  local_dir="data/external/huggingface/HIV_V3_coreceptor")

# HIV Protease Drug Resistance
snapshot_download("rebe121314/Protease_Hiv_drug", repo_type="dataset",
                  local_dir="data/external/huggingface/Protease_Hiv_drug")
```

**Manual Download URLs:**
- https://huggingface.co/datasets/damlab/human_hiv_ppi
- https://huggingface.co/datasets/damlab/HIV_V3_coreceptor
- https://huggingface.co/datasets/rebe121314/Protease_Hiv_drug

---

### 4. Zenodo Datasets (direct download)

```bash
cd data/external/zenodo

# HIV Genome-to-Genome Study (Record 7139)
curl -LO https://zenodo.org/record/7139/files/genome_to_genome_data.zip

# CView gp120 Sequences - CCR5/CXCR4 (Record 6475667)
curl -LO https://zenodo.org/record/6475667/files/CCR5_sequences.fasta
curl -LO https://zenodo.org/record/6475667/files/CXCR4_sequences.fasta

# HIV-1 Virion Morphology TEM Images (Record 5149062)
curl -LO https://zenodo.org/record/5149062/files/HIV_TEM_dataset.zip
```

**Manual Download URLs:**
- https://zenodo.org/record/7139
- https://zenodo.org/records/6475667
- https://zenodo.org/records/5149062

---

### 5. Direct CSV Downloads

```bash
cd data/external/csv

# CORGIS AIDS Dataset (UNAIDS global statistics)
curl -o corgis_aids.csv https://corgis-edu.github.io/corgis/datasets/csv/aids/aids.csv

# UNICEF HIV/AIDS Data
# Download from: https://data.unicef.org/resources/dataset/hiv-aids-statistical-tables/
```

---

### 6. Los Alamos HIV Database (LANL)

**Website:** https://www.hiv.lanl.gov/

#### CTL Epitopes (CSV/JSON):
1. Go to: https://www.hiv.lanl.gov/content/immunology/tables/ctl_summary.html
2. Click "Download CSV" or "Download JSON" at top of page
3. Save to: `data/external/lanl/ctl_epitopes.csv`

#### Antibody Data:
1. Go to: https://www.hiv.lanl.gov/content/immunology/tables/tables.html
2. Download antibody tables
3. Save to: `data/external/lanl/`

#### Sequence Alignments:
1. Go to: https://www.hiv.lanl.gov/content/sequence/NEWALIGN/align.html
2. Select organism, region, subtype
3. Download FASTA alignments

#### CATNAP (Neutralization Data):
1. Go to: https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/
2. Export IC50/IC80 data
3. Save to: `data/external/lanl/catnap/`

---

### 7. Stanford HIV Drug Resistance Database

**Website:** https://hivdb.stanford.edu/

#### Drug Resistance Mutations:
1. Go to: https://hivdb.stanford.edu/dr-summary/resistance-notes/
2. Download mutation tables for each drug class

#### Mutation Scores:
1. Go to: https://hivdb.stanford.edu/dr-summary/comments/
2. Export penalty scores for mutations

#### Sequence Data:
1. Use HIVdb Sequence Analysis tool
2. Export genotype-phenotype correlations

**PDF Reference:** https://cms.hivdb.org/prod/downloads/resistance-mutation-handout/resistance-mutation-handout.pdf

---

## Dataset Descriptions

| Source | Dataset | Description | Size Est. |
|--------|---------|-------------|-----------|
| **GitHub** | HIV-data | HIV sequences by length | ~50 MB |
| **GitHub** | HIV-DRM-ML | Drug resistance ML data | ~20 MB |
| **Kaggle** | HIV RNA Sequences | HIV-1/HIV-2 FASTA | ~10 MB |
| **HuggingFace** | human_hiv_ppi | 16k+ protein interactions | ~5 MB |
| **HuggingFace** | HIV_V3_coreceptor | V3 loop sequences | ~2 MB |
| **Zenodo** | gp120 sequences | CCR5/CXCR4 FASTA | ~1 MB |
| **Zenodo** | TEM images | Virion morphology | ~500 MB |
| **LANL** | CTL epitopes | ~350 epitopes + HLA | ~1 MB |
| **LANL** | CATNAP | Antibody IC50/IC80 | ~10 MB |
| **Stanford** | Drug resistance | Mutation tables | ~5 MB |

---

## Integration with Codon Encoder

After downloading, these datasets can be processed for the p-adic hyperbolic analysis:

```python
# Example: Load LANL CTL epitopes
import json
from pathlib import Path

lanl_dir = Path("data/external/lanl")
with open(lanl_dir / "ctl_epitopes.json") as f:
    epitopes = json.load(f)

# Process for codon encoder analysis
for epitope in epitopes:
    sequence = epitope["optimal_peptide"]
    hla = epitope["hla_restriction"]
    # ... analyze with hyperbolic encoder
```

---

## Recommended Download Order

1. **LANL CTL Epitopes** - Essential for expanding escape analysis
2. **Stanford Drug Resistance** - Essential for expanding resistance analysis
3. **Kaggle HIV Sequences** - Full FASTA for training
4. **HuggingFace PPI** - Protein interactions
5. **GitHub ML Data** - Pre-processed ML datasets
6. **Zenodo** - Supplementary/specialized data

---

## Network Troubleshooting

If you're experiencing DNS issues:

```bash
# Windows - Flush DNS cache
ipconfig /flushdns

# Try Google DNS
netsh interface ip set dns "Wi-Fi" static 8.8.8.8

# Test connectivity
ping github.com
curl -I https://github.com
```

Once network is restored:
```bash
python scripts/download_hiv_datasets.py --all
```
