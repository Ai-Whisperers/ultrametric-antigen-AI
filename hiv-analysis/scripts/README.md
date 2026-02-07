# HIV Analysis Scripts

This directory contains scripts for HIV sequence analysis pipeline.

## Scripts Overview

### 1. sequence_pipeline.py
- **Purpose**: Sets up the basic data processing pipeline
- **Features**: Data structure initialization, logging, manifest generation
- **Status**: ✅ Complete

### 2. setup_hiv_data.py  
- **Purpose**: Creates HIV data structure and provides download guidance
- **Features**: Directory setup, sample data, download guide
- **Status**: ✅ Complete

### 3. mafft_wrapper.py
- **Purpose**: Python wrapper for MAFFT multiple sequence alignment
- **Features**: 
  - Multiple alignment algorithms (auto, linsi, ginsi, einsi, fftns, fftnsi)
  - Batch processing capabilities
  - Statistical parsing and reporting
  - Mock alignment for development environments
  - Configurable parameters
- **Status**: ✅ Complete
- **Dependencies**: MAFFT (install with `sudo apt-get install mafft`)

### 4. los_alamos_downloader.py
- **Purpose**: Downloads HIV sequences from Los Alamos database
- **Features**: Respectful crawling, retry logic, manifest generation
- **Status**: ⚠️ Manual download required (anti-scraping measures)
- **Alternative**: Use setup_hiv_data.py for manual download guidance

## Usage Examples

### Running MAFFT Alignment

```bash
# Basic usage (processes sample data)
python3 scripts/mafft_wrapper.py

# In Python code
from scripts.mafft_wrapper import MAFFTWrapper, MAFFTConfig

config = MAFFTConfig(algorithm="linsi", threads=4)
mafft = MAFFTWrapper(config)
result = mafft.align_sequences("input.fasta", "output_aligned.fasta")
```

### Setting Up Data Structure

```bash
# Create full HIV data structure
python3 scripts/setup_hiv_data.py

# Follow manual download guide
cat data/DOWNLOAD_GUIDE.md
```

## Installation Requirements

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mafft

# For more advanced features (optional)
sudo apt-get install clustalw muscle t-coffee
```

### Python Dependencies
```bash
pip install biopython requests beautifulsoup4
```

## Pipeline Workflow

1. **Setup**: Run `setup_hiv_data.py` to create directory structure
2. **Download**: Manually download HIV sequences following `DOWNLOAD_GUIDE.md`
3. **Align**: Use `mafft_wrapper.py` for multiple sequence alignment
4. **Analyze**: Ready for downstream analysis (conservation scoring, visualization)

## Development Notes

- All scripts include comprehensive error handling and logging
- Mock data and mock alignment capabilities for development without dependencies
- Modular design allows for easy integration into larger pipelines
- Full documentation strings for all functions and classes

## Next Steps

- T004: Create alignment viewer
- T005: Calculate conservation scores
- T006: Add FASTA/Clustal export
- T007: Document workflow