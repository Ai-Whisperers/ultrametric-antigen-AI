# AlphaFold3 Integration

**Doc-Type:** Setup Guide · Version 1.0 · Updated 2025-12-24 · Author AI Whisperers

---

## Overview

This directory contains AlphaFold3 resources for structural validation of HIV integrase reveal mutations. **We use a hybrid approach** that eliminates the need for massive hardware requirements.

---

## Recommended: Hybrid Approach

**See [HYBRID_APPROACH.md](./HYBRID_APPROACH.md)** for our strategy.

Instead of replicating AF3's full pipeline (A100 GPU, 630GB databases, proprietary weights), we leverage AF3's open-source utilities for structure parsing combined with our **ternary VAE model (v5_11_11_production/best.pt)** which already captures evolutionary constraints through 3-adic geometry. The discrete-continuum bridge in our model (binary↔ternary↔continuous) encodes the same information that AF3 extracts from MSAs through sequence statistics, but geometrically - meaning we only need the PDB structural database (~50GB) rather than redundant MSA databases (580GB).

---

## Directory Structure

```
alphafold3/
├── README.md                    # This file
├── HYBRID_APPROACH.md          # Our lightweight strategy
├── repo/                        # Cloned AF3 repository (utilities only)
├── utils/                       # Extracted useful AF3 modules
├── inputs/integrase/           # JSON inputs for predictions
├── outputs/                     # Prediction results
└── scripts/                     # Input generation utilities
```

---

## Three Paths Forward

### Path 1: Hybrid Approach (RECOMMENDED)

| Aspect | Requirement |
|:-------|:------------|
| Storage | ~50GB (PDB only) |
| GPU | Any (or CPU) |
| Model | Our ternary VAE |
| Setup | Hours |

### Path 2: AlphaFold Server

| Aspect | Requirement |
|:-------|:------------|
| URL | https://alphafoldserver.com |
| Limit | 20 jobs/day |
| Cost | Free (non-commercial) |

### Path 3: Full Local AF3 (If Hardware Available)

| Component | Requirement |
|:----------|:------------|
| GPU | NVIDIA A100 80GB or H100 80GB |
| RAM | 64 GB minimum |
| Storage | 1 TB SSD (630GB databases + models) |
| OS | Linux (Ubuntu 22.04 recommended) |

---

## Installation Steps

### 1. Request Model Weights

**Critical:** Model weights require Google approval.

1. Submit form: https://forms.gle/svvpY4u2jsHEwWYS6
2. Wait 2-3 business days
3. Download weights when approved

### 2. Install Docker + NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Fetch Databases (630GB)

```bash
cd repo
./fetch_databases.sh /path/to/databases
```

Expected time: ~45 minutes with fast connection.

### 4. Build Docker Image

```bash
cd repo
docker build -t alphafold3 -f docker/Dockerfile .
```

### 5. Run Prediction

```bash
docker run -it \
  --volume $HOME/af_input:/root/af_input \
  --volume $HOME/af_output:/root/af_output \
  --volume /path/to/models:/root/models \
  --volume /path/to/databases:/root/public_databases \
  --gpus all \
  alphafold3 \
  python run_alphafold.py \
  --json_path=/root/af_input/fold_input.json \
  --model_dir=/root/models \
  --output_dir=/root/af_output
```

---

## Input Format

AlphaFold3 uses JSON input with this structure:

```json
{
  "name": "job_name",
  "modelSeeds": [1, 2, 3],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL..."
      }
    }
  ],
  "dialect": "alphafold3",
  "version": 1
}
```

### For Mutations

Include the mutated sequence directly - AlphaFold3 will predict the structure with the mutation.

---

## HIV Integrase Validation Jobs

### Priority 1: Wild-Type Integrase

Baseline structure for comparison.

### Priority 2: E166K Mutation

Top reveal candidate (score=34.93). Salt bridge reversal at LEDGF interface.

### Priority 3: K175E Mutation

Second top candidate (score=34.93). Charge reversal at key contact.

### Priority 4: W131A Mutation

Third candidate (score=33.03). Aromatic cap removal.

---

## Expected Outputs

| File | Description |
|:-----|:------------|
| `*_model.cif` | Predicted structure in mmCIF format |
| `*_confidences.json` | Per-residue confidence scores (pLDDT) |
| `*_summary.json` | Overall prediction summary |

### Analysis Metrics

1. **RMSD** - Compare WT vs mutant backbone
2. **pLDDT at LEDGF interface** - Confidence changes indicate exposure
3. **Surface accessibility** - Calculate epitope exposure

---

## Immediate Validation Path

While setting up local AlphaFold3:

1. **Use AlphaFold Server** (https://alphafoldserver.com)
2. Upload JSON from `inputs/integrase/`
3. Download results to `outputs/`
4. Run analysis scripts

### JSON Generator

Use our script to generate AlphaFold-compatible inputs:

```bash
python scripts/generate_integrase_inputs.py
```

---

## Licensing Notes

- **Code**: CC-BY-NC-SA 4.0 (non-commercial only)
- **Weights**: Separate terms, direct from Google only
- **Outputs**: "Theoretical modeling only, not for clinical use"

---

## References

- AlphaFold3 Paper: Abramson et al., Nature 630:493-500 (2024)
- DOI: 10.1038/s41586-024-07487-w
- Repository: https://github.com/google-deepmind/alphafold3

---

## Version History

| Version | Date | Changes |
|:--------|:-----|:--------|
| 1.0 | 2025-12-24 | Initial setup documentation |
