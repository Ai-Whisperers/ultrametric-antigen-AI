# HIV Sequence Analysis

This project implements a pipeline for HIV sequence analysis including alignment, conservation scoring, and evolutionary pattern analysis.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize the data pipeline:
   ```bash
   python3 scripts/sequence_pipeline.py
   ```

## Structure

- `data/` - Sequence data and metadata
  - `raw/hiv-sequences/` - Original sequences from databases
  - `processed/aligned/` - Multiple sequence alignments
  - `processed/filtered/` - Quality-filtered sequences
  - `metadata/` - Sample annotations
  - `references/` - Reference sequences
- `scripts/` - Processing scripts
- `notebooks/` - Analysis notebooks
- `results/` - Output files and figures

## Tasks

- [x] T001: Set up sequence data pipeline ✅
- [ ] T002: Download from Los Alamos DB
- [ ] T003: Implement MAFFT wrapper
- [ ] T004: Create alignment viewer
- [ ] T005: Calculate conservation scores
- [ ] T006: Add FASTA/Clustal export
- [ ] T007: Document workflow

## Usage

The pipeline is designed to process HIV sequences from major databases and perform multiple sequence alignment for evolutionary analysis.