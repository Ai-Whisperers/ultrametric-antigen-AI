# APIs and Data Sources for HIV Drug Resistance Prediction

**Date**: December 28, 2024
**Status**: Research completed

---

## Executive Summary

This document catalogs APIs and data sources that can enhance our HIV drug resistance prediction models. We've identified **15+ APIs** across 5 categories that could provide:

1. **Better embeddings**: ESM2, ProtTrans, EVE
2. **Structural context**: AlphaFold, PDB
3. **More training data**: Los Alamos, MaveDB, ChEMBL
4. **Drug information**: DrugBank, BindingDB
5. **Clinical validation**: ClinicalTrials.gov, Stanford HIVDB

---

## 1. Protein Language Model APIs (High Priority)

### 1.1 ESM-2 (Facebook/Meta Research)

**What it provides**: Pre-trained protein embeddings that capture evolutionary and structural information.

| Property | Details |
|----------|---------|
| **URL** | https://github.com/facebookresearch/esm |
| **Models** | esm2_t33_650M_UR50D (1024-dim), esm2_t36_3B |
| **API** | Python library + Hugging Face |
| **Cost** | Free (open source) |
| **Embedding dim** | 1280 (650M model) |

**Integration approach**:
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Get embeddings for HIV protease sequence
sequence = "PQVTLWQRPL..."
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # [1, seq_len, 1280]
```

**Why use it**:
- Replaces one-hot encoding with learned representations
- Captures evolutionary constraints
- State-of-the-art on mutation effect prediction

### 1.2 ESM-C (EvolutionaryScale - Newer)

**What it provides**: Drop-in replacement for ESM2 with better efficiency.

| Property | Details |
|----------|---------|
| **URL** | https://github.com/evolutionaryscale/esm |
| **Advantage** | 300M ESM-C ≈ 650M ESM2 performance |
| **Memory** | 50% reduction vs ESM2 |

### 1.3 ProtTrans (Rostlab)

**What it provides**: Multiple protein transformer models on Hugging Face.

| Property | Details |
|----------|---------|
| **URL** | https://huggingface.co/Rostlab |
| **Models** | prot_bert, prot_t5_xl_bfd, prot_t5_xl_half_uniref50-enc |
| **Best for** | Low GPU memory (8GB compatible) |
| **Validation** | Q3=84, Q8=73, Q10=74 |

**Integration approach**:
```python
from transformers import T5Tokenizer, T5EncoderModel

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

# Space-separated sequence for ProtTrans
sequence = " ".join(list("PQVTLWQRPL..."))
inputs = tokenizer(sequence, return_tensors="pt")
embeddings = model(**inputs).last_hidden_state
```

### 1.4 BioLM API (Commercial)

**What it provides**: Hosted ESM2 inference endpoints.

| Property | Details |
|----------|---------|
| **URL** | https://biolm.ai/models/esm2-35m/ |
| **Endpoint** | `/api/v3/esm2-35m/encode/` |
| **Returns** | Mean and per-token embeddings |
| **Cost** | Paid API |

---

## 2. Protein Structure APIs

### 2.1 AlphaFold Database

**What it provides**: Pre-computed protein structure predictions.

| Property | Details |
|----------|---------|
| **URL** | https://alphafold.ebi.ac.uk/ |
| **API** | REST + GraphQL |
| **Coverage** | 214+ million structures |
| **Data access** | FTP, API, Google Cloud |

**HIV-specific structures available**:
- HIV-1 Protease: Multiple variants
- Reverse Transcriptase: Full-length + RNase H domain
- Integrase: Core domain + full-length predictions

**Integration**:
```python
import requests

# Get AlphaFold structure for UniProt ID
uniprot_id = "P04585"  # HIV-1 Protease
url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
response = requests.get(url)
structure_url = response.json()["pdbUrl"]
```

### 2.2 RCSB PDB Data API

**What it provides**: Experimental protein structures and binding site annotations.

| Property | Details |
|----------|---------|
| **URL** | https://data.rcsb.org/ |
| **API Types** | REST + GraphQL |
| **HIV Structures** | 1000+ entries |

**Key HIV PDB entries**:
| Enzyme | Representative PDBs |
|--------|---------------------|
| Protease | 3PJ6, 1HXB, 2AZ9, 2HB4 |
| RT | 3HVT, 3V4I, 1DLO, 3DLK |
| Integrase | 3OYA, 3OS1, 1QS4 |

**Use cases**:
- Extract binding site residues for feature engineering
- Identify drug contact positions
- Map mutations to structural context

---

## 3. HIV-Specific Databases

### 3.1 Stanford HIV Drug Resistance Database (Primary)

**What it provides**: Curated genotype-phenotype data, resistance interpretation.

| Property | Details |
|----------|---------|
| **URL** | https://hivdb.stanford.edu/ |
| **API** | Sierra Web Service 2.0 |
| **Data** | 119K PR, 128K RT, 13K IN sequences |
| **Countries** | 143 countries |

**API endpoint**:
```
https://hivdb.stanford.edu/page/webservice
```

**Data available**:
- Genotype-phenotype correlations
- Resistance mutation lists
- Drug susceptibility scores
- Treatment history correlations

**Direct dataset download**:
- https://hivdb.stanford.edu/pages/genopheno.dataset.html

### 3.2 Los Alamos HIV Database

**What it provides**: Comprehensive HIV sequence repository with immunology data.

| Property | Details |
|----------|---------|
| **URL** | https://www.hiv.lanl.gov/ |
| **Sequences** | 1,240,044 |
| **CTL epitopes** | 12,670 records |
| **Antibodies** | 4,579 |

**New in 2025**: CSV and JSON downloads for immunology data (updated weekly).

**Tools available**:
- Sequence alignment
- Phylogenetics
- Quality control

### 3.3 NCBI GenBank (via Biopython)

**What it provides**: Raw HIV sequences from publications.

| Property | Details |
|----------|---------|
| **API** | E-utilities (Entrez) |
| **Python** | Bio.Entrez module |
| **HIV sequences** | Millions |

**Integration**:
```python
from Bio import Entrez, SeqIO

Entrez.email = "your@email.com"
handle = Entrez.esearch(db="nucleotide", term="HIV-1[Organism] AND pol[Gene]")
records = Entrez.read(handle)
```

---

## 4. Mutation Effect Databases

### 4.1 MaveDB (2024 Update)

**What it provides**: Deep mutational scanning variant effects.

| Property | Details |
|----------|---------|
| **URL** | https://www.mavedb.org/ |
| **Variants** | 7+ million |
| **Datasets** | 1,884 |
| **API** | REST API + Python package |
| **Bulk download** | Zenodo (CC0 license) |

**Why use it**:
- Ground truth mutation effects from experiments
- Can train/validate mutation predictors
- HIV-related DMS datasets may exist

**API**:
```python
# Install: pip install mavedb-api
from mavedb_api import MaveDBAPI

api = MaveDBAPI()
datasets = api.search("HIV")
```

### 4.2 EVE (Evolutionary model of Variant Effect)

**What it provides**: Pre-computed pathogenicity predictions for 36M variants.

| Property | Details |
|----------|---------|
| **URL** | https://evemodel.org/ |
| **GitHub** | https://github.com/OATML-Markslab/EVE |
| **Performance** | AUC = 0.91 on ClinVar |
| **Genes covered** | 3,219 disease genes |

**Approach**: Bayesian VAE on evolutionary sequences.

### 4.3 ProteinGym Benchmark

**What it provides**: Standardized DMS benchmarks for evaluating models.

| Property | Details |
|----------|---------|
| **URL** | https://github.com/OATML-Markslab/ProteinGym |
| **Substitutions** | 2.7M across 217 assays |
| **Indels** | 300K across 74 assays |
| **Baselines** | ESM3, ESM-C, ProGen3 |

---

## 5. Drug & Bioactivity Databases

### 5.1 ChEMBL

**What it provides**: Drug bioactivity measurements and targets.

| Property | Details |
|----------|---------|
| **URL** | https://www.ebi.ac.uk/chembl/ |
| **API** | REST API |
| **Activities** | 13+ million |
| **Drug warnings** | Safety/withdrawal data |

**New in ChEMBL 35 (Dec 2024)**:
- Antimicrobial resistance data (CO-ADD)
- 100K+ new bioactivity measurements
- Natural product annotations

**API endpoint**:
```
https://www.ebi.ac.uk/chembl/api/data/activity
https://www.ebi.ac.uk/chembl/api/data/drug_warning
```

### 5.2 DrugBank 6.0

**What it provides**: Comprehensive drug information and pharmacogenomics.

| Property | Details |
|----------|---------|
| **URL** | https://go.drugbank.com/ |
| **API** | REST API |
| **Drugs** | 11,891 |
| **DDIs** | 1.4M interactions |
| **SNP data** | Pharmacogenomics |

**Relevant endpoints**:
- Drug metabolism pathways
- SNP effects on drug activity
- ADMET predictions

**License**: Free for academic use.

### 5.3 BindingDB

**What it provides**: Protein-small molecule binding data.

| Property | Details |
|----------|---------|
| **URL** | https://www.bindingdb.org/ |
| **Binding data** | 2.9 million points |
| **Compounds** | 1.3 million |
| **PDB mappings** | 15,328 structures |

---

## 6. UniProt (Universal Protein Resource)

**What it provides**: Comprehensive protein annotations.

| Property | Details |
|----------|---------|
| **URL** | https://www.uniprot.org/ |
| **API** | REST API (303M requests/month) |
| **Base URL** | https://rest.uniprot.org/uniprotkb/ |

**HIV-relevant entries**:
| Protein | UniProt ID |
|---------|------------|
| Gag-Pol polyprotein | P04585 |
| Protease | Cleaved from P04585 |
| RT | Cleaved from P04585 |
| Integrase | Cleaved from P04585 |

**API features**:
- Sequence retrieval
- Feature annotations
- ID mapping (up to 100K IDs)
- Variation data

---

## 7. Clinical Data Sources

### 7.1 ClinicalTrials.gov

**What it provides**: HIV treatment trial data and outcomes.

| Property | Details |
|----------|---------|
| **URL** | https://clinicaltrials.gov/ |
| **API** | REST API |
| **HIV trials** | Thousands |

### 7.2 AACT Database

**What it provides**: Relational database of ClinicalTrials.gov data.

| Property | Details |
|----------|---------|
| **URL** | https://aact.ctti-clinicaltrials.org/ |
| **Updates** | Daily |
| **Access** | PostgreSQL database |

---

## Implementation Priority Matrix

| API/Data Source | Impact | Effort | Priority |
|-----------------|--------|--------|----------|
| **ESM-2 embeddings** | Very High | Low | **P0** |
| **ProtTrans (T5)** | High | Low | **P0** |
| **Stanford HIVDB API** | High | Low | **P1** |
| **AlphaFold structures** | High | Medium | **P1** |
| **MaveDB data** | Medium | Medium | **P2** |
| **PDB binding sites** | Medium | Medium | **P2** |
| **ChEMBL bioactivity** | Medium | High | **P3** |
| **Los Alamos sequences** | Medium | Medium | **P2** |
| **EVE predictions** | Medium | Low | **P2** |
| **DrugBank** | Low | Medium | **P3** |

---

## Recommended Implementation Order

### Phase 1: Replace One-Hot with PLM Embeddings (Week 1-2)

```python
# 1. Install dependencies
pip install transformers torch sentencepiece

# 2. Create embedding extractor
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # [1, 1280]
```

### Phase 2: Add Structural Features (Week 3-4)

1. Download AlphaFold structures for HIV enzymes
2. Extract binding site residues from PDB
3. Create distance-to-binding-site features
4. Add secondary structure annotations

### Phase 3: Expand Training Data (Week 5-6)

1. Query Los Alamos for additional sequences
2. Download MaveDB HIV-related datasets
3. Integrate phenotypic data from Stanford
4. Create unified data pipeline

### Phase 4: Validate with External Data (Week 7-8)

1. Use ChEMBL bioactivity as external validation
2. Compare predictions with EVE scores
3. Cross-reference with clinical trial outcomes

---

## Code Templates

### ESM-2 Integration

```python
# scripts/embeddings/esm2_embedder.py

import torch
from transformers import EsmTokenizer, EsmModel
from typing import List, Dict
import numpy as np

class ESM2Embedder:
    """Extract ESM-2 embeddings for HIV sequences."""

    MODELS = {
        "small": "facebook/esm2_t6_8M_UR50D",      # 8M params
        "medium": "facebook/esm2_t12_35M_UR50D",   # 35M params
        "large": "facebook/esm2_t33_650M_UR50D",   # 650M params
        "xlarge": "facebook/esm2_t36_3B_UR50D",    # 3B params
    }

    def __init__(self, size: str = "large", device: str = "cuda"):
        model_name = self.MODELS[size]
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.device = device

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Get mean-pooled embedding for a sequence."""
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def embed_mutations(self, wt_seq: str, mutations: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for wild-type and mutant sequences."""
        results = {"wt": self.embed_sequence(wt_seq)}
        for mut in mutations:
            mut_seq = self._apply_mutation(wt_seq, mut)
            results[mut] = self.embed_sequence(mut_seq)
        return results
```

### Stanford HIVDB API Integration

```python
# scripts/data/stanford_api.py

import requests
from typing import List, Dict

class StanfordHIVDB:
    """Query Stanford HIV Drug Resistance Database."""

    BASE_URL = "https://hivdb.stanford.edu"

    def query_mutations(self, mutations: List[str], gene: str = "PR") -> Dict:
        """Get resistance interpretation for mutations."""
        url = f"{self.BASE_URL}/page/webservice"
        payload = {
            "gene": gene,
            "mutations": ",".join(mutations)
        }
        response = requests.post(url, json=payload)
        return response.json()

    def get_phenotype_data(self, drug: str) -> Dict:
        """Get phenotype data for a drug."""
        url = f"{self.BASE_URL}/cgi-bin/RT_Phenotype.cgi"
        params = {"drug": drug}
        response = requests.get(url, params=params)
        return self._parse_response(response.text)
```

### AlphaFold Structure Fetcher

```python
# scripts/structures/alphafold_fetcher.py

import requests
from pathlib import Path

class AlphaFoldFetcher:
    """Fetch AlphaFold structures for HIV proteins."""

    BASE_URL = "https://alphafold.ebi.ac.uk/api"

    HIV_PROTEINS = {
        "gag_pol": "P04585",
        "env": "P04578",
        "tat": "P04608",
    }

    def fetch_structure(self, uniprot_id: str, output_dir: Path) -> Path:
        """Download AlphaFold structure as PDB file."""
        url = f"{self.BASE_URL}/prediction/{uniprot_id}"
        response = requests.get(url)
        data = response.json()

        pdb_url = data["pdbUrl"]
        pdb_content = requests.get(pdb_url).text

        output_path = output_dir / f"{uniprot_id}.pdb"
        output_path.write_text(pdb_content)
        return output_path
```

---

## Data Volume Estimates

| Source | Records | Size | Format |
|--------|---------|------|--------|
| Stanford HIVDB | 260K sequences | ~500 MB | TSV |
| Los Alamos | 1.2M sequences | ~2 GB | FASTA |
| MaveDB (all) | 7M variants | ~5 GB | JSON |
| AlphaFold (HIV) | ~50 structures | ~100 MB | PDB |
| ChEMBL (HIV targets) | ~100K activities | ~200 MB | JSON |
| ESM-2 embeddings | Pre-computed | ~10 GB | Tensors |

---

## Sources

### Protein Language Models
- [ESM-2 GitHub](https://github.com/facebookresearch/esm)
- [ESM-C GitHub](https://github.com/evolutionaryscale/esm)
- [ProtTrans on Hugging Face](https://huggingface.co/Rostlab)
- [BioLM API](https://biolm.ai/models/esm2-35m/)

### Protein Structure
- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [RCSB PDB Data API](https://data.rcsb.org/)

### HIV Databases
- [Stanford HIVDB](https://hivdb.stanford.edu/)
- [Los Alamos HIV Database](https://www.hiv.lanl.gov/)
- [NCBI Entrez](https://www.ncbi.nlm.nih.gov/home/develop/api/)

### Mutation Effect
- [MaveDB](https://www.mavedb.org/)
- [EVE Model](https://evemodel.org/)
- [ProteinGym](https://github.com/OATML-Markslab/ProteinGym)

### Drug Data
- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [DrugBank 6.0](https://go.drugbank.com/)
- [BindingDB](https://www.bindingdb.org/)

### Protein Annotations
- [UniProt REST API](https://www.uniprot.org/help/programmatic_access)

### Clinical Data
- [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api)
- [AACT Database](https://aact.ctti-clinicaltrials.org/)
