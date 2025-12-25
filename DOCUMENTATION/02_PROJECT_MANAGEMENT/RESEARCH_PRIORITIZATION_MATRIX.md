# Research Projects Prioritization Matrix

**Branch**: `feature/research-prioritization`
**Last Updated**: 2025-12-25
**Purpose**: Comprehensive analysis of all research projects, grouped by complexity and priority

---

## Executive Summary

This document consolidates **50+ research projects, proposals, and tasks** into a structured prioritization matrix. Projects are grouped by:

1. **Complexity** (Trivial → Expert)
2. **Priority** (P0 Critical → P3 Nice-to-have)
3. **Domain** (Infrastructure, Bioinformatics, Theory, Visualization)

---

## Quick Reference: What to Work On First

### Immediate Priority (This Week)

| Project                | Complexity | Time | Why First                      |
| :--------------------- | :--------- | :--- | :----------------------------- |
| P0: Critical Bug Fixes | Easy       | 2-4h | Blocking all training          |
| P0: Security Fixes     | Easy       | 1-2h | Syntax errors breaking scripts |
| P1: Unit Testing Setup | Medium     | 4-8h | Enables safe development       |

### Short-Term (This Month)

| Project                | Complexity | Time   | Why Now                     |
| :--------------------- | :--------- | :----- | :-------------------------- |
| RA Analysis Completion | Medium     | 8-16h  | 80% done, high-value output |
| HIV Glycan Analysis    | Medium     | 8-16h  | Foundation for PTM encoder  |
| PTM-Goldilocks Encoder | Medium     | 16-24h | Unifies all disease models  |

### Medium-Term (Next Quarter)

| Project                      | Complexity | Time  | Why This Quarter             |
| :--------------------------- | :--------- | :---- | :--------------------------- |
| AlphaFold3 Integration       | Hard       | 40h+  | Structural validation needed |
| Disease Roadmap (5 diseases) | Hard       | 80h+  | Publication material         |
| Accessible Therapeutics      | Expert     | 100h+ | Regulatory pathway work      |

---

## Complexity Tiers Explained

| Tier        | Description                                        | Typical Time | Skills Needed         |
| :---------- | :------------------------------------------------- | :----------- | :-------------------- |
| **Trivial** | Config changes, typo fixes, simple refactors       | < 1 hour     | Basic Python          |
| **Easy**    | Bug fixes, small features, documentation           | 1-4 hours    | Python, Git           |
| **Medium**  | New modules, analysis scripts, integrations        | 4-24 hours   | ML, Bioinformatics    |
| **Hard**    | Architecture changes, new models, validations      | 1-2 weeks    | Deep ML/Bio expertise |
| **Expert**  | Research proposals, novel algorithms, publications | Weeks-months | Domain expertise      |

---

## TIER 1: TRIVIAL COMPLEXITY (< 1 hour each)

### Infrastructure

| ID  | Project                               | Status | Dependencies        | Location             |
| :-- | :------------------------------------ | :----- | :------------------ | :------------------- |
| T1  | Remove unused imports across codebase | Open   | None                | `src/**/*.py`        |
| T2  | Update CITATION.cff with Zenodo DOI   | Open   | Zenodo registration | `CITATION.cff`       |
| T3  | Add missing `__all__` exports         | Open   | None                | `src/**/__init__.py` |
| T4  | Fix hardcoded paths in scripts        | Open   | None                | `research/**/*.py`   |
| T5  | Clean up test output files            | Open   | None                | `outputs/test_viz/`  |

### Documentation

| ID  | Project                                | Status  | Dependencies      | Location             |
| :-- | :------------------------------------- | :------ | :---------------- | :------------------- |
| T6  | Update README quick-start paths        | Done    | None              | `README.md`          |
| T7  | Fix remaining TBD placeholders         | Partial | Content decisions | Various `.md` files  |
| T8  | Add docstrings to visualization module | Open    | None              | `src/visualization/` |

---

## TIER 2: EASY COMPLEXITY (1-4 hours each)

### P0 Critical Infrastructure (MUST DO FIRST)

| ID  | Project                                  | Est. Time | Status | Location                                                         |
| :-- | :--------------------------------------- | :-------- | :----- | :--------------------------------------------------------------- |
| E1  | **Fix division by zero in trainer**      | 30 min    | Open   | `src/training/trainer.py:362-363`                                |
| E2  | **Fix syntax error in alphafold script** | 15 min    | Open   | `rheumatoid_arthritis/scripts/19_alphafold_structure_mapping.py` |
| E3  | **Fix broken operation composition**     | 2h        | Open   | `src/losses/appetitive_losses.py:428-486`                        |
| E4  | **Fix unconditional val_loader crash**   | 30 min    | Open   | `src/training/appetitive_trainer.py:541`                         |
| E5  | **Remove trivial addition test**         | 15 min    | Open   | `src/losses/consequence_predictor.py:219-229`                    |

### P1 Testing & Quality

| ID  | Project                                  | Est. Time | Status | Dependencies | Location                  |
| :-- | :--------------------------------------- | :-------- | :----- | :----------- | :------------------------ |
| E6  | Add pytest fixtures for common test data | 2h        | Open   | None         | `tests/conftest.py`       |
| E7  | Create geometry test suite               | 3h        | Open   | E6           | `tests/unit/geometry/`    |
| E8  | Add CI GitHub Action for linting         | 2h        | Open   | None         | `.github/workflows/`      |
| E9  | Configure pre-commit hooks               | 1h        | Open   | None         | `.pre-commit-config.yaml` |

### Visualization Enhancements

| ID  | Project                      | Est. Time | Status | Dependencies | Location                                    |
| :-- | :--------------------------- | :-------- | :----- | :----------- | :------------------------------------------ |
| E10 | Add manifold plotting module | 4h        | Open   | viz package  | `src/visualization/plots/manifold.py`       |
| E11 | Add training metrics plots   | 3h        | Open   | viz package  | `src/visualization/plots/training.py`       |
| E12 | Add Poincare ball projection | 4h        | Open   | viz package  | `src/visualization/projections/poincare.py` |

---

## TIER 3: MEDIUM COMPLEXITY (4-24 hours each)

### P1 Bioinformatics Analysis (High Priority)

| ID  | Project                                 | Est. Time | Status   | Dependencies | Location                             |
| :-- | :-------------------------------------- | :-------- | :------- | :----------- | :----------------------------------- |
| M1  | **Complete RA Citrullination Analysis** | 8h        | 80% done | E2 fix       | `research/.../rheumatoid_arthritis/` |
| M2  | **Complete HIV Glycan Shield Analysis** | 8h        | 70% done | None         | `research/.../hiv/glycan_shield/`    |
| M3  | **Generate RA Pitch Deck Figures**      | 4h        | Open     | M1           | `research/.../visualizations/pitch/` |
| M4  | **Run AlphaFold3 Validation Batch**     | 6h        | Open     | M1, M2       | `research/.../VALIDATION_PROTOCOLS/` |
| M5  | SARS-CoV-2 Handshake Complete Analysis  | 8h        | 60% done | None         | `research/.../sars_cov_2/`           |

### P2 Model & Training

| ID  | Project                                | Est. Time | Status | Dependencies | Location                       |
| :-- | :------------------------------------- | :-------- | :----- | :----------- | :----------------------------- |
| M6  | Implement Riemannian optimizer wrapper | 8h        | Open   | geoopt       | `src/optimizers/riemannian.py` |
| M7  | Add mixed-precision training support   | 6h        | Open   | M6           | `src/training/trainer.py`      |
| M8  | Create unified trainer base class      | 12h       | Open   | E1-E5        | `src/training/base.py`         |
| M9  | Benchmark memory usage and profile     | 4h        | Open   | None         | `scripts/benchmark/`           |

### P2 Data & Integration

| ID  | Project                            | Est. Time | Status | Dependencies | Location                 |
| :-- | :--------------------------------- | :-------- | :----- | :----------- | :----------------------- |
| M10 | Add AnnData/ScanPy integration     | 8h        | Open   | None         | `src/data/bio_loader.py` |
| M11 | Add scVelo fitness gradient metric | 6h        | Open   | M10          | `src/metrics/`           |
| M12 | Create minimal PDB subset (~100MB) | 4h        | Open   | None         | `data/pdb_subset/`       |

### P2 Code Quality

| ID  | Project                        | Est. Time | Status | Dependencies | Location                      |
| :-- | :----------------------------- | :-------- | :----- | :----------- | :---------------------------- |
| M13 | Extract benchmark utilities    | 4h        | Open   | None         | `src/benchmark/utils.py`      |
| M14 | Consolidate visualization math | 6h        | Open   | viz package  | `src/geometry/projections.py` |
| M15 | Remove dead code modules       | 4h        | Open   | None         | Various                       |
| M16 | Add comprehensive type hints   | 8h        | Open   | None         | `src/**/*.py`                 |

---

## TIER 4: HARD COMPLEXITY (1-2 weeks each)

### P1 Core Architecture

| ID  | Project                             | Est. Time | Status | Dependencies | Location                         |
| :-- | :---------------------------------- | :-------- | :----- | :----------- | :------------------------------- |
| H1  | **PTM-Goldilocks Encoder Training** | 40h       | Ready  | M1, M2       | New: `src/models/ptm_encoder.py` |
| H2  | Enforce geoopt as hard dependency   | 16h       | Open   | None         | `src/geometry/`, `src/models/`   |
| H3  | Unit test coverage to 40%           | 40h       | Open   | E6-E9        | `tests/`                         |

### P2 AlphaFold Integration

| ID  | Project                        | Est. Time | Status   | Dependencies | Location                      |
| :-- | :----------------------------- | :-------- | :------- | :----------- | :---------------------------- |
| H4  | AlphaFold3 Hybrid Architecture | 40h       | Planning | M12          | `research/alphafold3/hybrid/` |
| H5  | Structural validation pipeline | 24h       | Open     | H4           | `src/validation/`             |

### P3 Advanced Features

| ID  | Project                         | Est. Time | Status      | Dependencies | Location                        |
| :-- | :------------------------------ | :-------- | :---------- | :----------- | :------------------------------ |
| H6  | Spectral analysis operators     | 32h       | Exploratory | None         | `src/analysis/spectral.py`      |
| H7  | Interactive Streamlit dashboard | 40h       | Concept     | viz package  | `app/dashboard/`                |
| H8  | Swarm VAE distributed training  | 40h       | Concept     | H2           | `src/training/swarm_trainer.py` |

---

## TIER 5: EXPERT COMPLEXITY (Weeks-Months)

### Research Proposals (12 Major)

| ID  | Proposal                            | Est. Time | Status      | Dependencies  | Priority |
| :-- | :---------------------------------- | :-------- | :---------- | :------------ | :------- |
| X1  | Geometric Vaccine Design            | 3 months  | Documented  | H1, H4        | High     |
| X2  | Drug-Interaction Modeling           | 2 months  | Documented  | M6            | High     |
| X3  | Codon-Space Exploration             | 2 months  | In-progress | None          | High     |
| X4  | Extraterrestrial Genetic Code       | 6 months  | Concept     | NASA data     | Low      |
| X5  | Spectral Bio-ML & Holography        | 3 months  | Concept     | H6            | Medium   |
| X6  | Autoimmunity & Codon Adaptation     | 2 months  | Partial     | M1            | High     |
| X7  | Multi-Objective Evolutionary Opt    | 2 months  | Concept     | None          | Medium   |
| X8  | Quantum-Biology Signatures          | 6 months  | Concept     | DFT tools     | Low      |
| X9  | Long-COVID Microclots               | 3 months  | Concept     | Clinical data | Medium   |
| X10 | Huntington's Repeat Expansion       | 3 months  | Concept     | Sequence data | Medium   |
| X11 | Swarm VAE Architecture              | 2 months  | Concept     | H8            | Low      |
| X12 | Cross-Disease Biomarker Integration | 4 months  | Concept     | M10           | Medium   |

### Disease Applications (Priority 1)

| ID  | Disease                         | Est. Time | Status | Dependencies |
| :-- | :------------------------------ | :-------- | :----- | :----------- |
| D1  | HIV (Complete Analysis)         | 2 weeks   | 70%    | M2           |
| D2  | SARS-CoV-2 (Pan-Coronavirus)    | 3 weeks   | 60%    | M5           |
| D3  | Rheumatoid Arthritis            | 2 weeks   | 80%    | M1           |
| D4  | Alzheimer's Tau Phosphorylation | 4 weeks   | 30%    | H1           |
| D5  | Cancer Neoantigen Prediction    | 4 weeks   | 10%    | H1           |

### Disease Applications (Priority 2-3)

| ID  | Disease                 | Est. Time | Status      | Dependencies |
| :-- | :---------------------- | :-------- | :---------- | :----------- |
| D6  | Influenza Hemagglutinin | 3 weeks   | Planned     | D2           |
| D7  | Dengue/Zika Envelope    | 3 weeks   | Planned     | D2           |
| D8  | Parkinson's α-synuclein | 4 weeks   | Exploratory | D4           |
| D9  | ALS TDP-43              | 4 weeks   | Exploratory | D4           |
| D10 | Type 1 Diabetes         | 4 weeks   | Planned     | D3           |

---

## Recommended Execution Order

### Phase 1: Foundation (Week 1-2)

```text
Day 1-2:   E1, E2, E3, E4, E5 (Critical bug fixes)
Day 3-4:   E6, E7 (Testing foundation)
Day 5-7:   E8, E9, E10 (CI + basic visualization)
Week 2:    M1, M2 (Complete RA & HIV analyses)
```

### Phase 2: Core Deliverables (Week 3-4)

```text
Week 3:    M3, M4 (Figures + AlphaFold validation)
           M5 (SARS-CoV-2 completion)
Week 4:    H1 (PTM-Goldilocks Encoder training)
           M6 (Riemannian optimizer)
```

### Phase 3: Publication Prep (Month 2)

```text
Week 5-6:  H3 (Test coverage), H2 (geoopt enforcement)
Week 7-8:  H4, H5 (AlphaFold hybrid + validation)
           X1, X6 (Begin vaccine & autoimmunity papers)
```

### Phase 4: Expansion (Month 3+)

```text
Month 3:   D4, D5 (Alzheimer's, Cancer applications)
           H7 (Dashboard), X3 (Codon-space paper)
Month 4+:  D6-D10 (Additional diseases)
           X2, X5, X7 (Drug interactions, Spectral, MOEA)
```

---

## Dependencies Graph

```text
                    ┌─────────────────┐
                    │ E1-E5: Bug Fixes │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │E6-E9:   │    │M1: RA   │    │M2: HIV  │
        │Testing  │    │Analysis │    │Analysis │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             │              └──────┬───────┘
             │                     │
             ▼                     ▼
        ┌─────────┐         ┌───────────┐
        │H3: 40%  │         │H1: PTM    │
        │Coverage │         │Encoder    │
        └─────────┘         └─────┬─────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │D4:      │   │X1:      │   │X6:      │
              │Alzheimer│   │Vaccine  │   │Autoimmun│
              └─────────┘   └─────────┘   └─────────┘
```

---

## Resource Requirements

### Minimum Viable Product (MVP)

- **Time**: 4 weeks full-time
- **Deliverables**: Bug-free codebase, RA/HIV analyses, PTM encoder
- **Skills**: Python, PyTorch, basic bioinformatics

### Publication-Ready

- **Time**: 3 months
- **Deliverables**: MVP + AlphaFold validation + 2 disease papers
- **Skills**: + AlphaFold, manuscript writing

### Full Platform

- **Time**: 6-12 months
- **Deliverables**: All 12 proposals, 10+ disease applications
- **Skills**: + Clinical partnerships, regulatory knowledge

---

## Risk Assessment

| Risk                            | Probability | Impact | Mitigation                        |
| :------------------------------ | :---------- | :----- | :-------------------------------- |
| AlphaFold3 access restrictions  | Medium      | High   | Use OSS utilities only            |
| Training instability (NaN)      | High        | Medium | Enforce geoopt, clip gradients    |
| Insufficient test coverage      | High        | High   | Prioritize H3                     |
| Clinical validation delays      | High        | Medium | Focus on computational validation |
| Scope creep across 12 proposals | High        | High   | Strict phase gates                |

---

## Decision Points

### Must Decide Now

1. **Which 3 diseases to focus on first?** (Recommend: HIV, RA, Alzheimer's)
2. **Publication target?** (arXiv pre-print vs peer-reviewed)
3. **AlphaFold approach?** (Full integration vs minimal utilities)

### Decide by End of Phase 1

1. **Citizen science pathway?** (Accessible therapeutics proposal)
2. **Commercial licensing strategy?** (PolyForm terms)
3. **External validation partners?** (Academic collaborations)

---

## Summary Statistics

| Category           | Count  | Trivial | Easy   | Medium | Hard  | Expert |
| :----------------- | :----- | :------ | :----- | :----- | :---- | :----- |
| Infrastructure     | 18     | 5       | 9      | 4      | 0     | 0      |
| Bioinformatics     | 22     | 0       | 0      | 5      | 2     | 15     |
| Model/Training     | 12     | 0       | 5      | 4      | 3     | 0      |
| Visualization      | 6      | 2       | 3      | 1      | 0     | 0      |
| Research Proposals | 12     | 0       | 0      | 0      | 0     | 12     |
| **Total**          | **70** | **7**   | **17** | **14** | **5** | **27** |

**Quick Win Potential**: 24 tasks (Trivial + Easy) can be completed in < 1 week
**Core Value**: 14 Medium tasks deliver 80% of research value
**Moon Shots**: 27 Expert tasks represent long-term vision
