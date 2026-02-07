# HIV Sequence Analysis Pipeline

Complete bioinformatics pipeline for HIV sequence alignment, conservation analysis, and format export.

## Quick Start

```bash
# 1. Set up data structure and sample data
python3 scripts/setup_hiv_data.py

# 2. Run sequence alignment  
python3 scripts/mafft_wrapper.py

# 3. Calculate conservation scores
python3 scripts/conservation_scorer.py

# 4. Create visualizations
python3 scripts/alignment_viewer.py

# 5. Export to standard formats
python3 scripts/format_exporter.py
```

## 📋 Complete Documentation

**👉 [WORKFLOW.md](WORKFLOW.md) - Complete usage guide and pipeline documentation**

## Features

- ✅ **Data Management**: Automated directory setup and sample data
- ✅ **Multiple Sequence Alignment**: MAFFT wrapper with multiple algorithms  
- ✅ **Conservation Analysis**: Shannon entropy, Simpson index, property-based scoring
- ✅ **Interactive Visualization**: HTML and text-based alignment viewers
- ✅ **Format Export**: FASTA, ClustalW, PHYLIP, NEXUS, MSF formats
- ✅ **Comprehensive Logging**: Full pipeline tracking and error handling

## Pipeline Components

| Script | Purpose | Output |
|--------|---------|---------|
| `setup_hiv_data.py` | Data structure setup | Sample data, download guide |
| `mafft_wrapper.py` | Sequence alignment | Aligned FASTA files |
| `conservation_scorer.py` | Conservation analysis | JSON scores, reports |
| `alignment_viewer.py` | Visualization | HTML/text viewers |
| `format_exporter.py` | Format conversion | Multiple standard formats |

## Installation

### Dependencies
```bash
pip install biopython numpy requests beautifulsoup4
```

### Optional (for real alignment)
```bash
sudo apt-get install mafft
```

## Project Structure

```
hiv-analysis/
├── data/                    # Sequence data and metadata  
├── scripts/                 # Processing pipeline
├── results/                 # Analysis outputs
├── WORKFLOW.md             # 📋 Complete documentation
└── README.md               # This file
```

## Tasks Status

- [x] T001: Set up sequence data pipeline ✅
- [x] T002: Download from Los Alamos DB ✅  
- [x] T003: Implement MAFFT wrapper ✅
- [x] T004: Create alignment viewer ✅
- [x] T005: Calculate conservation scores ✅
- [x] T006: Add FASTA/Clustal export ✅
- [x] T007: Document workflow ✅

## Usage Examples

**Basic alignment workflow:**
```bash
python3 scripts/mafft_wrapper.py
python3 scripts/alignment_viewer.py
```

**Conservation analysis:**
```bash
python3 scripts/conservation_scorer.py
# View results: results/conservation/conservation_report.txt
```

**Export to multiple formats:**
```bash
python3 scripts/format_exporter.py
# Outputs: results/exports/ (FASTA, ClustalW, PHYLIP, NEXUS, MSF)
```

## Next Steps

1. **Real Data**: Follow `data/DOWNLOAD_GUIDE.md` for actual HIV sequences
2. **Phylogenetics**: Use exported formats with RAxML/MrBayes/BEAST
3. **Structural Analysis**: Map conservation to 3D structures
4. **Comparative Analysis**: Multi-subtype conservation studies

For complete documentation, see **[WORKFLOW.md](WORKFLOW.md)**.