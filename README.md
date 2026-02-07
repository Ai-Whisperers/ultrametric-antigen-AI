# Ultrametric Antigen AI

**p-adic Geometry and Hyperbolic Manifolds for Bioinformatics Research**

A cutting-edge bioinformatics package leveraging p-adic mathematics and hyperbolic geometry for advanced sequence analysis, with a focus on HIV research and viral evolution.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

```bash
# Install the package
pip install -e .

# Set up HIV analysis environment
python3 -m hiv_analysis.scripts.setup_hiv_data

# Run complete pipeline
python3 -m hiv_analysis.scripts.mafft_wrapper      # Align sequences
python3 -m hiv_analysis.scripts.conservation_scorer # Score conservation  
python3 -m hiv_analysis.scripts.alignment_viewer    # Visualize results
python3 -m hiv_analysis.scripts.format_exporter     # Export formats
```

## 📋 Features

### 🧬 HIV Sequence Analysis
- **Multiple Sequence Alignment**: MAFFT integration with multiple algorithms
- **Conservation Analysis**: Shannon entropy, Simpson index, property-based scoring
- **Interactive Visualization**: HTML and text-based alignment viewers
- **Format Export**: FASTA, ClustalW, PHYLIP, NEXUS, MSF support
- **Comprehensive Pipeline**: End-to-end workflow for viral sequence analysis

### 🔬 Mathematical Foundation
- **p-adic Geometry**: Ultrametric spaces for biological sequence analysis
- **Hyperbolic Manifolds**: Non-Euclidean geometry for evolutionary modeling
- **Ternary VAE**: Advanced variational autoencoders for sequence generation
- **Conservation Metrics**: Information-theoretic measures of sequence conservation

## 📖 Documentation

- **[HIV Analysis Pipeline](hiv-analysis/README.md)** - Complete HIV sequence analysis workflow
- **[Workflow Guide](hiv-analysis/WORKFLOW.md)** - Step-by-step usage documentation
- **[Scripts Documentation](hiv-analysis/scripts/README.md)** - Individual script details

## 🛠 Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install mafft python3-pip python3-dev

# Optional: Install additional bioinformatics tools
sudo apt-get install clustalw muscle t-coffee
```

### Package Installation
```bash
# Clone repository
git clone https://github.com/Ai-Whisperers/ultrametric-antigen-AI.git
cd ultrametric-antigen-AI

# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e .[dev,docs,analysis,visualization]
```

## 📚 Examples

### Basic HIV Analysis

```python
from hiv_analysis.scripts import mafft_wrapper, conservation_scorer, alignment_viewer

# 1. Align HIV sequences
config = mafft_wrapper.MAFFTConfig(algorithm="linsi", threads=4)
mafft = mafft_wrapper.MAFFTWrapper(config)
result = mafft.align_sequences("hiv_sequences.fasta", "aligned.fasta")

# 2. Calculate conservation scores
scorer = conservation_scorer.ConservationScorer()
alignment = scorer.load_alignment("aligned.fasta")
scores = scorer.calculate_all_scores(alignment)

# 3. Create visualization
viewer = alignment_viewer.AlignmentViewer()
viewer.view_alignment(alignment, "visualization.html")
```

### Advanced Conservation Analysis

```python
from hiv_analysis.scripts.conservation_scorer import ConservationScorer, ConservationConfig

# Configure multiple conservation metrics
config = ConservationConfig(
    score_types=['shannon', 'simpson', 'property', 'blosum'],
    property_groups={
        'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'],
        'polar': ['S', 'T', 'N', 'Q'],
        'charged': ['R', 'H', 'K', 'D', 'E']
    },
    output_format='csv'
)

scorer = ConservationScorer(config)
results = scorer.calculate_all_scores(alignment)

# Export detailed analysis
scorer.export_scores(results, "conservation_analysis")
report = scorer.generate_conservation_report(results, "report.txt")
```

### Batch Processing Pipeline

```python
from pathlib import Path
from hiv_analysis.scripts import mafft_wrapper, format_exporter

# Process multiple sequence files
input_dir = Path("sequences/")
output_dir = Path("results/")

# Batch alignment
mafft = mafft_wrapper.MAFFTWrapper()
batch_results = mafft.align_multiple_files(input_dir, output_dir, "*.fasta")

# Export to multiple formats for each alignment
exporter = format_exporter.AlignmentExporter()
for result in batch_results["results"]:
    if result["status"] == "success":
        alignment = format_exporter.SequenceAlignment.from_fasta(result["output_file"])
        exporter.export_alignment(alignment, result["output_file"].replace(".fasta", ""))
```

### Command Line Interface

```bash
# Initialize project structure
hiv-analysis setup

# Basic workflow
hiv-analysis align sequences.fasta
hiv-analysis score aligned_sequences.fasta  
hiv-analysis view aligned_sequences.fasta
hiv-analysis export aligned_sequences.fasta

# Advanced options
hiv-analysis align sequences.fasta --algorithm linsi --threads 8
hiv-analysis export aligned.fasta --format fasta,clustal,phylip --output results/
```

## 🔬 Scientific Applications

### HIV Drug Resistance Analysis
```python
# Analyze resistance mutations with conservation context
alignment = load_hiv_pol_alignment("hiv_pol.fasta")
conservation = calculate_conservation(alignment)

# Identify highly conserved regions that might be drug targets
highly_conserved = [i for i, score in enumerate(conservation) if score > 0.9]
print(f"Potential drug targets at positions: {highly_conserved}")
```

### Evolutionary Pressure Mapping
```python
# Map evolutionary pressure across protein regions
from hiv_analysis.scripts.conservation_scorer import ConservationScorer

scorer = ConservationScorer()
scores = scorer.calculate_property_conservation(alignment)

# Identify regions under different selective pressures
variable_regions = [i for i, s in enumerate(scores) if s < 0.5]
conserved_regions = [i for i, s in enumerate(scores) if s > 0.8]
```

### Multi-Format Export for Phylogenetics
```python
# Prepare data for phylogenetic analysis
from hiv_analysis.scripts.format_exporter import AlignmentExporter, ExportConfig

config = ExportConfig(
    output_formats=['phylip', 'nexus'],  # For RAxML and MrBayes
    include_conservation=True,
    line_length=80
)

exporter = AlignmentExporter(config)
files = exporter.export_alignment(alignment, "phylo_analysis")

# Use with external tools:
# raxmlHPC -f a -m PROTGAMMAWAG -p 12345 -x 12345 -# 100 -s phylo_analysis.phy -n HIV_tree
# mb phylo_analysis.nex
```

## 🏗 Project Structure

```
ultrametric-antigen-AI/
├── hiv_analysis/              # Main Python package
│   ├── scripts/              # Analysis pipeline scripts
│   ├── core/                 # Core algorithms  
│   ├── data/                 # Data handling utilities
│   ├── utils/                # Common utilities
│   └── cli.py                # Command-line interface
├── hiv-analysis/             # Working directory
│   ├── data/                 # Sequence data
│   ├── results/              # Analysis outputs
│   ├── scripts/              # Standalone scripts
│   ├── WORKFLOW.md          # Complete workflow documentation
│   └── README.md            # Pipeline-specific documentation
├── pyproject.toml           # Package configuration
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🧮 Mathematical Background

This package implements novel applications of p-adic mathematics and hyperbolic geometry to biological sequence analysis:

- **p-adic Ultrametrics**: Used for measuring evolutionary distances with non-Archimedean properties
- **Hyperbolic Embeddings**: Sequences mapped to hyperbolic spaces for hierarchical analysis  
- **Ternary Structures**: Three-state mathematical models for codon analysis
- **Conservation Geometry**: Information-theoretic measures in geometric frameworks

## 🔬 Research Applications

### Published & In-Progress Research
- HIV drug resistance prediction using p-adic sequence embeddings
- Conservation landscape analysis with hyperbolic geometry
- Viral evolution modeling in ultrametric spaces
- Multi-scale sequence analysis using ternary variational autoencoders

### Collaboration Opportunities
We welcome collaborations in:
- Computational virology and epidemiology
- Mathematical biology and bioinformatics
- Drug resistance prediction and vaccine design
- Novel applications of p-adic methods in biology

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/Ai-Whisperers/ultrametric-antigen-AI.git
cd ultrametric-antigen-AI
pip install -e .[dev]

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy hiv_analysis/
```

## 📊 Performance Benchmarks

### Typical Performance (HIV Env sequences, ~500 residues)
- **Alignment (MAFFT)**: ~30 seconds for 100 sequences
- **Conservation Analysis**: ~5 seconds for Shannon entropy
- **Visualization Generation**: ~2 seconds for HTML output
- **Format Export**: ~1 second for all 5 formats

### Scalability
- **Memory Usage**: ~1MB per sequence for alignment
- **Recommended Limits**: <10,000 sequences per batch
- **Threading**: Full multi-core support for alignment

## 🔍 Troubleshooting

### Common Issues

**MAFFT not found:**
```bash
# Ubuntu/Debian
sudo apt-get install mafft

# macOS
brew install mafft

# Conda
conda install -c bioconda mafft
```

**Import errors:**
```bash
# Reinstall in development mode
pip uninstall ultrametric-antigen-ai
pip install -e .
```

**Memory issues with large datasets:**
- Process sequences in batches using `align_multiple_files()`
- Use `fftns` algorithm instead of `linsi` for speed
- Increase system memory or use cloud computing

## 📚 References

### Mathematical Foundations
- Koblitz, N. (1984). p-adic Numbers, p-adic Analysis, and Zeta-Functions
- Ratcliffe, J. (2019). Foundations of Hyperbolic Manifolds
- Vladimirov, V.S. (1994). p-adic Analysis and Mathematical Physics

### Bioinformatics Applications
- Katoh, K. & Standley, D.M. (2013). MAFFT: Multiple sequence alignment software
- Shannon, C.E. (1948). A mathematical theory of communication
- Los Alamos HIV Database: https://www.hiv.lanl.gov/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Contact

- **AI Whisperers Research Team**: research@ai-whisperers.com
- **GitHub**: https://github.com/Ai-Whisperers/ultrametric-antigen-AI
- **Issues**: https://github.com/Ai-Whisperers/ultrametric-antigen-AI/issues

---

**Keywords**: bioinformatics, p-adic geometry, hyperbolic manifolds, HIV analysis, sequence alignment, conservation analysis, viral evolution, mathematical biology