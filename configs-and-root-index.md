# Configs & Project Root Files Separation Index

**Generated:** 2026-01-14
**Purpose:** Checklist for staging configs and root files to training-opt branch
**Focus:** Mathematical ML infrastructure vs bioinformatics separation

---

## Mission: STAGE PURE ML/INFRASTRUCTURE FILES TO "training-opt" BRANCH

**Goal:** Extract mathematical ML configuration and infrastructure files that are completely domain-agnostic. All configs analyzed are focused on ternary VAE mathematical foundations (p-adic theory, hyperbolic geometry) rather than bioinformatics applications.

**Success Criteria:**
- ✅ Mathematical VAE configurations and infrastructure preserved
- ✅ Domain-agnostic build and project management files included
- ✅ Zero biological domain dependencies in configurations
- ✅ Self-contained mathematical ML pipeline configuration

---

## Analysis Summary

### 📋 **configs/ Directory Analysis**
All configuration files examined are focused on **mathematical ML foundations**:
- Ternary VAE architectures and training parameters
- Hyperbolic geometry and Poincaré ball operations
- P-adic mathematical structures and geodesic losses
- Frozen encoder strategies and curriculum learning
- Meta-learning and homeostatic control systems

**Key Finding:** Despite the project's bioinformatics applications, the actual VAE configurations are **pure mathematical** and domain-agnostic.

### 📋 **Project Root Files Analysis**
Project root contains standard ML infrastructure files:
- Build and dependency management (pyproject.toml, requirements.txt)
- Documentation and licensing files
- Git configuration and development tools
- General ML training environment configuration

---

## ✅ **STAGE TO training-opt - All Configs & Infrastructure Files**

### 🧮 **configs/ Directory (KEEP ALL - 10 files)**
- [x] `configs/env.example` - **INFRASTRUCTURE**: ML training environment template
- [x] `configs/project-words.txt` - **INFRASTRUCTURE**: Spell-check dictionary (contains both domains)
- [x] `configs/ternary.yaml` - **PURE MATH**: V5.11 ternary VAE with frozen encoders & hyperbolic geometry
- [x] `configs/ternary_fast_test.yaml` - **PURE MATH**: Fast testing configuration for ternary VAE
- [x] `configs/v5_12.yaml` - **PURE MATH**: V5.12 architecture with optimizations
- [x] `configs/v5_12_1.yaml` - **PURE MATH**: V5.12.1 enhanced architecture
- [x] `configs/v5_12_3.yaml` - **PURE MATH**: V5.12.3 configuration
- [x] `configs/v5_12_4.yaml` - **PURE MATH**: V5.12.4 improved components architecture
- [x] `configs/v5_12_4_extended_grokking.yaml` - **PURE MATH**: Extended grokking detection

### 🧮 **configs/archive/ Directory (KEEP ALL - 10 files)**
- [x] `configs/archive/README.md` - **INFRASTRUCTURE**: Archive documentation
- [x] `configs/archive/appetitive_vae.yaml` - **PURE MATH**: Bio-inspired VAE (mathematical metaphors)
- [x] `configs/archive/ternary_v5_10.yaml` - **PURE MATH**: V5.10 ternary VAE configuration
- [x] `configs/archive/ternary_v5_6.yaml` - **PURE MATH**: V5.6 ternary VAE configuration
- [x] `configs/archive/ternary_v5_7.yaml` - **PURE MATH**: V5.7 ternary VAE configuration
- [x] `configs/archive/ternary_v5_8.yaml` - **PURE MATH**: V5.8 ternary VAE configuration
- [x] `configs/archive/ternary_v5_9.yaml` - **PURE MATH**: V5.9 ternary VAE configuration
- [x] `configs/archive/ternary_v5_9_2.yaml` - **PURE MATH**: V5.9.2 ternary VAE configuration
- [x] `configs/archive/v5_11_11_homeostatic_ale_device.yaml` - **PURE MATH**: V5.11.11 homeostatic (ALE device)
- [x] `configs/archive/v5_11_11_homeostatic_rtx2060s.yaml` - **PURE MATH**: V5.11.11 homeostatic (RTX2060s)

### 🏗️ **Project Root Infrastructure Files (KEEP ALL - 13 files)**
- [x] `.gitattributes` - **INFRASTRUCTURE**: Git line ending and file handling configuration
- [x] `.gitignore` - **INFRASTRUCTURE**: Git ignore patterns for Python/ML projects
- [x] `.pre-commit-config.yaml` - **INFRASTRUCTURE**: Code quality and formatting hooks
- [x] `CHANGELOG.md` - **INFRASTRUCTURE**: Version history and release notes
- [x] `CITATION.cff` - **INFRASTRUCTURE**: Academic citation format metadata
- [x] `LICENSE` - **INFRASTRUCTURE**: Software license (PolyForm Non-Commercial 1.0.0)
- [x] `NOTICE` - **INFRASTRUCTURE**: Legal notices and attribution
- [x] `README.md` - **INFRASTRUCTURE**: Project documentation and quick start guide
- [x] `cspell.json` - **INFRASTRUCTURE**: Spell-checker configuration
- [x] `pyproject.toml` - **INFRASTRUCTURE**: Python project build configuration and metadata
- [x] `requirements-extensions.txt` - **INFRASTRUCTURE**: Extended Python dependencies
- [x] `requirements.txt` - **INFRASTRUCTURE**: Core Python dependencies (mathematical ML only)
- [x] `scripts-index.md` - **INFRASTRUCTURE**: Scripts separation documentation (created in this session)

---

## ❌ **DO NOT STAGE - No Bioinformatics-Specific Config Files Found**

**Analysis Result:** All configuration files in the configs/ directory are focused on mathematical VAE architectures rather than bioinformatics applications. The mathematical foundations (ternary operations, p-adic theory, hyperbolic geometry) are domain-agnostic and suitable for any field requiring advanced geometric machine learning.

**No Exclusions Needed:** Unlike the src/ and scripts/ directories, the configs/ and project root contain no bioinformatics-specific files that need to be excluded.

---

## Implementation Checklist

### Phase 1: Stage All Config Files (20 files)
- [ ] Copy entire `configs/` directory to training-opt branch
- [ ] All 9 main config files (ternary VAE architectures V5.6-V5.12.4)
- [ ] All 10 archive config files (historical configurations)
- [ ] 1 environment template and 1 project dictionary

### Phase 2: Stage Project Infrastructure (13 files)
- [ ] Copy all Git configuration files (.gitignore, .gitattributes, .pre-commit-config.yaml)
- [ ] Copy all build files (pyproject.toml, requirements.txt, requirements-extensions.txt)
- [ ] Copy all documentation (README.md, CHANGELOG.md, LICENSE, NOTICE, CITATION.cff)
- [ ] Copy all development tools (cspell.json, scripts-index.md)

### Phase 3: Verification
- [ ] Verify all 33 files copied successfully
- [ ] Test configuration loading with mathematical data only
- [ ] Confirm no bioinformatics dependencies in configs
- [ ] Validate build system works with pure mathematical pipeline

---

## Expected Results

**Before:** Mixed mathematical + bioinformatics project with shared infrastructure
**After:** Self-contained mathematical ML pipeline with complete configuration support

**Resulting Configuration Capabilities:**
1. **Complete VAE Training** - All ternary VAE architectures (V5.6-V5.12.4)
2. **Mathematical Foundations** - P-adic theory, hyperbolic geometry, category theory
3. **Advanced Techniques** - Meta-learning, homeostatic control, curriculum training
4. **Development Infrastructure** - Build system, dependencies, code quality tools
5. **Documentation** - README, changelog, licensing, citation support

**Use Cases for Pure Config Pipeline:**
- Mathematical machine learning research projects
- Geometric deep learning applications
- P-adic and tropical geometry experiments
- Category-theoretic neural network architectures
- Any domain requiring advanced mathematical ML foundations

**Key Mathematical Techniques Preserved:**
- ✅ Ternary VAE architectures with dual encoders
- ✅ Hyperbolic Poincaré ball geometry and geodesic losses
- ✅ P-adic number theory and 3-adic operations
- ✅ Homeostatic control and meta-learning systems
- ✅ Curriculum learning and adaptive scheduling
- ✅ Grokking detection and optimization techniques

---

## Implementation Commands

### Copy configs/ directory:
```bash
git checkout training-opt
git checkout main -- configs/
git add configs/
```

### Copy project root files:
```bash
git checkout main -- .gitignore .gitattributes .pre-commit-config.yaml
git checkout main -- CHANGELOG.md CITATION.cff LICENSE NOTICE README.md
git checkout main -- cspell.json pyproject.toml requirements.txt requirements-extensions.txt
git checkout main -- scripts-index.md
git add .gitignore .gitattributes .pre-commit-config.yaml
git add CHANGELOG.md CITATION.cff LICENSE NOTICE README.md
git add cspell.json pyproject.toml requirements.txt requirements-extensions.txt scripts-index.md
```

### Commit and push:
```bash
git commit -m "feat(training-opt): Add complete configs and infrastructure for pure mathematical ML pipeline"
git push origin training-opt
```

---

## References

- **Scripts Analysis:** `scripts-index.md` - Scripts directory separation (156 mathematical files)
- **Source Analysis:** `src/index.md` - Source code separation (290 mathematical files)
- **Training-Opt Branch:** Contains pure mathematical ML pipeline with no biological coupling

**Note:** This index provides complete configuration and infrastructure support for the pure mathematical ML pipeline, enabling self-contained development and deployment of advanced geometric machine learning systems.