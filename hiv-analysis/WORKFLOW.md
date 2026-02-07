# HIV Sequence Analysis Workflow

Complete pipeline for HIV sequence alignment, conservation analysis, and format export.

## Overview

This workflow provides a comprehensive solution for analyzing HIV sequences, from raw data acquisition through multiple sequence alignment, conservation scoring, visualization, and export to various bioinformatics formats.

## Pipeline Components

### 1. Data Setup (`setup_hiv_data.py`)
**Purpose**: Initialize data structure and provide download guidance

**Usage:**
```bash
python3 scripts/setup_hiv_data.py
```

**What it does:**
- Creates complete directory structure for HIV analysis
- Generates sample HIV sequences for testing
- Creates comprehensive download guide for manual data acquisition
- Sets up metadata files and data manifest

**Output:**
- `data/` directory structure
- Sample sequences in `data/raw/hiv-sequences/`
- Manual download guide: `data/DOWNLOAD_GUIDE.md`

### 2. Sequence Pipeline (`sequence_pipeline.py`)
**Purpose**: Core data processing pipeline setup

**Usage:**
```bash
python3 scripts/sequence_pipeline.py
```

**What it does:**
- Validates pipeline environment
- Creates processing directories
- Generates data manifests
- Sets up logging infrastructure

**Output:**
- Validated pipeline structure
- Processing logs in `results/pipeline.log`

### 3. MAFFT Alignment (`mafft_wrapper.py`)
**Purpose**: Multiple sequence alignment using MAFFT

**Usage:**
```bash
# Basic usage (processes sample data)
python3 scripts/mafft_wrapper.py

# In Python code
from scripts.mafft_wrapper import MAFFTWrapper, MAFFTConfig

config = MAFFTConfig(algorithm="linsi", threads=4)
mafft = MAFFTWrapper(config)
result = mafft.align_sequences("input.fasta", "output_aligned.fasta")
```

**Algorithms supported:**
- `auto` - Automatic algorithm selection
- `linsi` - Local pairwise alignment (most accurate)
- `ginsi` - Global pairwise alignment
- `einsi` - Structural alignment
- `fftns` - Fast progressive method
- `fftnsi` - Fast progressive with iterations

**What it does:**
- Aligns multiple HIV sequences
- Supports batch processing
- Parses alignment statistics
- Creates mock alignments if MAFFT not installed

**Output:**
- Aligned sequences in `data/processed/aligned/`
- Alignment statistics and logs

### 4. Alignment Visualization (`alignment_viewer.py`)
**Purpose**: Interactive alignment viewing and analysis

**Usage:**
```bash
python3 scripts/alignment_viewer.py
```

**What it does:**
- Loads multiple sequence alignments
- Calculates conservation scores using Shannon entropy
- Generates consensus sequences
- Creates both HTML and text visualizations
- Provides position rulers and conservation markers

**Features:**
- Conservation coloring (highly/moderately/variable regions)
- Configurable viewing parameters
- Position-based analysis
- Responsive HTML output with CSS styling

**Output:**
- HTML visualization: `results/alignment_view_demo.html`
- Text format: `results/alignment_view_demo.txt`

### 5. Conservation Scoring (`conservation_scorer.py`)
**Purpose**: Comprehensive conservation analysis

**Usage:**
```bash
python3 scripts/conservation_scorer.py
```

**Conservation metrics:**
- **Shannon entropy** - Information theoretic measure
- **Simpson index** - Diversity-based conservation
- **Property conservation** - Amino acid property groups
- **BLOSUM conservation** - Substitution matrix based

**What it does:**
- Calculates multiple conservation scores per position
- Identifies highly/moderately/variable conserved regions
- Generates statistical summaries
- Exports results in JSON/CSV formats

**Output:**
- Conservation scores: `results/conservation/hiv_env_conservation.json`
- Analysis report: `results/conservation/conservation_report.txt`

### 6. Format Export (`format_exporter.py`)
**Purpose**: Export alignments to standard bioinformatics formats

**Usage:**
```bash
python3 scripts/format_exporter.py
```

**Supported formats:**
- **FASTA** - Universal sequence format
- **ClustalW** - With conservation annotation
- **PHYLIP** - Phylogenetic analysis (sequential)
- **NEXUS** - Flexible phylogenetic format
- **MSF** - Multiple Sequence Format with checksums

**What it does:**
- Converts alignments to multiple standard formats
- Includes conservation annotation where appropriate
- Adds consensus sequences
- Generates format compatibility summary

**Output:**
- All formats in `results/exports/`
- Export summary: `results/exports/export_summary.md`

## Complete Workflow Example

### Step 1: Environment Setup
```bash
# Navigate to project directory
cd /path/to/hiv-analysis

# Install dependencies (if needed)
pip install biopython numpy

# Optional: Install MAFFT for real alignment
sudo apt-get install mafft
```

### Step 2: Initialize Data Structure
```bash
# Set up complete data structure
python3 scripts/setup_hiv_data.py

# This creates:
# - data/raw/hiv-sequences/ (for input sequences)
# - data/processed/aligned/ (for alignments)
# - data/references/ (for reference sequences)
# - metadata/ directories
# - Sample data for testing
```

### Step 3: Data Acquisition
```bash
# Follow the manual download guide
cat data/DOWNLOAD_GUIDE.md

# Or use sample data for testing/development
# (Sample sequences are automatically created)
```

### Step 4: Sequence Alignment
```bash
# Run MAFFT alignment
python3 scripts/mafft_wrapper.py

# This processes sample data and creates:
# - data/processed/aligned/hiv1_env_sample_aligned.fasta
# - Alignment statistics and logs
```

### Step 5: Conservation Analysis
```bash
# Calculate conservation scores
python3 scripts/conservation_scorer.py

# This generates:
# - Multiple conservation metrics per position
# - Statistical analysis of conservation patterns
# - Identification of highly conserved regions
```

### Step 6: Alignment Visualization
```bash
# Create interactive visualizations
python3 scripts/alignment_viewer.py

# This generates:
# - HTML visualization with conservation coloring
# - Text-based alignment view
# - Position rulers and consensus sequences
```

### Step 7: Format Export
```bash
# Export to standard formats
python3 scripts/format_exporter.py

# This creates:
# - FASTA, ClustalW, PHYLIP, NEXUS, MSF formats
# - Conservation annotation where applicable
# - Format compatibility summary
```

## Directory Structure After Complete Workflow

```
hiv-analysis/
├── data/
│   ├── raw/hiv-sequences/          # Input sequences
│   │   └── hiv1_env_sample.fasta
│   ├── processed/aligned/          # Aligned sequences  
│   │   └── hiv1_env_sample_aligned.fasta
│   ├── references/                 # Reference sequences
│   ├── metadata/                   # Sequence metadata
│   ├── manifest.json              # Data manifest
│   └── DOWNLOAD_GUIDE.md          # Manual download guide
├── scripts/
│   ├── setup_hiv_data.py          # Data structure setup
│   ├── sequence_pipeline.py       # Core pipeline
│   ├── mafft_wrapper.py           # Sequence alignment
│   ├── alignment_viewer.py        # Visualization
│   ├── conservation_scorer.py     # Conservation analysis
│   ├── format_exporter.py         # Format export
│   └── README.md                  # Scripts documentation
├── results/
│   ├── alignment_view_demo.html   # HTML visualization
│   ├── alignment_view_demo.txt    # Text visualization
│   ├── conservation/              # Conservation analysis
│   │   ├── hiv_env_conservation.json
│   │   └── conservation_report.txt
│   └── exports/                   # Format exports
│       ├── hiv_env_alignment.fasta
│       ├── hiv_env_alignment.aln
│       ├── hiv_env_alignment.phy
│       ├── hiv_env_alignment.nex
│       ├── hiv_env_alignment.msf
│       └── export_summary.md
└── WORKFLOW.md                    # This documentation
```

## Advanced Usage

### Batch Processing Multiple Sequences
```python
from scripts.mafft_wrapper import MAFFTWrapper
from scripts.conservation_scorer import ConservationScorer
from scripts.format_exporter import AlignmentExporter

# Process multiple files
mafft = MAFFTWrapper()
results = mafft.align_multiple_files("data/raw/", "data/processed/aligned/")

# Analyze conservation for each
scorer = ConservationScorer()
for result in results["results"]:
    if result["status"] == "success":
        alignment = scorer.load_alignment(result["output_file"])
        conservation = scorer.calculate_all_scores(alignment)
```

### Custom Configuration
```python
from scripts.mafft_wrapper import MAFFTConfig
from scripts.conservation_scorer import ConservationConfig
from scripts.format_exporter import ExportConfig

# Custom MAFFT settings
mafft_config = MAFFTConfig(
    algorithm="linsi",
    threads=8,
    max_iterations=2000
)

# Custom conservation settings
cons_config = ConservationConfig(
    score_types=['shannon', 'simpson', 'property', 'blosum'],
    output_format='csv'
)

# Custom export settings
export_config = ExportConfig(
    output_formats=['fasta', 'clustal', 'nexus'],
    line_length=80,
    include_conservation=True
)
```

### Integration with External Tools
```bash
# Use exported formats with other tools:

# ClustalW format with Jalview
jalview results/exports/hiv_env_alignment.aln

# PHYLIP format with RAxML
raxmlHPC -f a -m PROTGAMMAWAG -p 12345 -x 12345 -# 100 -s hiv_env_alignment.phy -n HIV_tree

# NEXUS format with MrBayes
mb results/exports/hiv_env_alignment.nex

# FASTA format with MEGA
# Import FASTA file into MEGA for phylogenetic analysis
```

## Performance Considerations

### For Large Datasets
- Use `fftns` algorithm for speed with many sequences (>1000)
- Enable threading: `MAFFTConfig(threads=cpu_count())`
- Process in batches for memory management
- Use compressed output for storage efficiency

### Memory Usage
- Typical memory usage: ~1MB per sequence for alignment
- Conservation scoring: ~500KB per 1000 positions
- HTML output: ~2x alignment size
- Consider subset analysis for very large datasets

## Troubleshooting

### Common Issues

**MAFFT not found:**
```bash
sudo apt-get install mafft
# OR use conda/mamba
conda install -c bioconda mafft
```

**Memory errors with large datasets:**
- Process sequences in smaller batches
- Use `fftns` instead of `linsi` for faster processing
- Increase system memory or use cloud computing

**Format compatibility issues:**
- Check format specifications in `export_summary.md`
- Validate exported files with target software
- Use FASTA format as universal fallback

### Validation

**Verify alignment quality:**
```bash
# Check alignment statistics
grep "sequences" results/conservation/conservation_report.txt

# Validate conservation patterns
grep "Highly conserved" results/conservation/conservation_report.txt
```

**Test format compatibility:**
```bash
# Quick validation of exported formats
head -n 5 results/exports/hiv_env_alignment.*
```

## Next Steps

1. **Scale to real datasets**: Follow `DOWNLOAD_GUIDE.md` for acquiring actual HIV sequences
2. **Phylogenetic analysis**: Use exported PHYLIP/NEXUS formats with RAxML/MrBayes
3. **Structural analysis**: Map conservation to 3D structures using PyMOL
4. **Comparative analysis**: Compare conservation across different HIV subtypes
5. **Integration**: Connect to larger bioinformatics pipelines

## References

- Los Alamos HIV Database: https://www.hiv.lanl.gov/
- MAFFT: https://mafft.cbrc.jp/alignment/software/
- ClustalW: http://www.clustal.org/
- PHYLIP: https://evolution.genetics.washington.edu/phylip.html