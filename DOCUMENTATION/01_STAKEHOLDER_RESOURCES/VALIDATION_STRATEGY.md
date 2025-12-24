# Validation Strategy & Benchmark Suite

> **Overview**: This suite defines the comprehensive validation protocol for the Ternary VAE. It is broken down into four distinct pillars.

## 🗂️ Validation Pillars

### 1. [Biological Benchmarks](./validation_suite/01_BIOLOGICAL_BENCHMARKS.md)

- **Scope**: 40+ Viruses & 40+ Protein Assays.
- **Goal**: Prove predictive accuracy on real-world biological data (Viral Escape, Protein Fitness).

### 2. [Mathematical Stress Tests](./validation_suite/02_MATHEMATICAL_STRESS_TESTS.md)

- **Scope**: 40+ Synthetic Scenarios.
- **Goal**: Prove geometric consistency (Hyperbolicity, Ultrametricity) across dimensions 8 to 1024.

### 3. [Computational Scalability](./validation_suite/03_COMPUTATIONAL_SCALABILITY.md)

- **Scope**: 40+ Hardware Configurations.
- **Goal**: Prove "Exascale on a Laptop" performance (Speed, VRAM, Cost).

### 4. [Competitive Landscape](./validation_suite/04_COMPETITIVE_LANDSCAPE.md)

- **Scope**: vs. EVE, ESM, AlphaFold.
- **Goal**: The "Kill Sheet" - direct head-to-head comparisons.

---

## 🏗️ Execution Checklist

- [ ] **Download**: `benchmarks/download_proteingym.sh`
- [ ] **Download**: `benchmarks/get_viral_datasets.py` (GISAID/NCBI API)
- [ ] **Run**: `verify_40_pathogens.sh`
- [ ] **Report**: `generate_massive_report.ipynb`
