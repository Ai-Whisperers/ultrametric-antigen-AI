# Master Documentation Index

> **Single source of truth for navigating the Ternary VAE Bioinformatics documentation.**

---

## Quick Navigation

| I need to...                        | Go to                                                                                                        |
| :---------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| Understand the project in 5 minutes | [QUICK_START.md](QUICK_START.md)                                                                             |
| Find documents for my role          | [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)                                                                   |
| See project vision and pitch        | [01_PROJECT_KNOWLEDGE_BASE/00_STRATEGY_AND_VISION/](01_PROJECT_KNOWLEDGE_BASE/00_STRATEGY_AND_VISION/)       |
| Understand the math/theory          | [01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/) |
| Review research findings            | [01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/](01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/)     |
| Check project roadmap               | [02_PROJECT_MANAGEMENT/01_ROADMAPS_AND_PLANS/](02_PROJECT_MANAGEMENT/01_ROADMAPS_AND_PLANS/)                 |
| Access stakeholder packages         | [03_PRESENTATION_TIERS/](03_PRESENTATION_TIERS/)                                                             |

---

## Documentation Structure

### 01 PROJECT KNOWLEDGE BASE

**Purpose**: Immutable concepts, standards, and foundational assets.

| Section                       | Description                             | Key Documents                                         |
| :---------------------------- | :-------------------------------------- | :---------------------------------------------------- |
| **02_THEORY_AND_FOUNDATIONS** | Mathematical and biological foundations | Hyperbolic geometry, 3-adic theory, validation suites |
| **03_EXPERIMENTS_AND_LABS**   | Active research notebooks and findings  | HIV, SARS-CoV-2, RA, neurodegeneration research       |
| **04_SCIENTIFIC_HISTORY**     | Archive of discoveries and reports      | Discovery records, academic outputs                   |
| **05_LEGAL_AND_IP**           | IP protection, licensing, timestamps    | IP manifests, legal documentation                     |
| **insights/**                 | External research and connection ideas  | Video analysis, research papers, datasets             |

### 02 PROJECT MANAGEMENT

**Purpose**: Mutable plans, tasks, and health checks.

| Section                    | Description                       | Key Documents                                      |
| :------------------------- | :-------------------------------- | :------------------------------------------------- |
| **00_TASKS**               | Actionable items (P0-P3 priority) | Bioinformatics, architecture, infrastructure tasks |
| **01_ROADMAPS_AND_PLANS**  | Strategic direction               | `00_MASTER_ROADMAP_JONA.md`, quarterly plans       |
| **02_CODE_HEALTH_METRICS** | Code quality audits               | Technical debt reports, health dashboards          |
| **03_ARCHIVE**             | Deprecated plans                  | Historical planning documents                      |

### 03 PRESENTATION TIERS

**Purpose**: Tailored packages for specific stakeholders.

| Tier                    | Audience                          | Content Type                                        |
| :---------------------- | :-------------------------------- | :-------------------------------------------------- |
| **01_TIER_1_PUBLIC**    | Scientists, clinicians, media     | Validated predictions, domain explainers, protocols |
| **02_TIER_2_INVESTOR**  | VCs, grant committees             | Pitch materials, market analysis, case studies      |
| **03_TIER_3_TECHNICAL** | Due diligence, academic reviewers | Technical specs, IP inventory, theory deep dives    |

### 06 VALIDATION (under 02_THEORY_AND_FOUNDATIONS)

**Purpose**: Test strategies and verification frameworks.

| Section                    | Description                                      |
| :------------------------- | :----------------------------------------------- |
| **01_PLANS**               | CI/CD plans, master test strategy                |
| **02_SUITES**              | Unit, integration, scientific test documentation |
| **03_TEST_FRAMEWORK_DOCS** | Test architecture and strategy guides            |
| **07_VALIDATION_SUITE**    | Biological benchmarks, mathematical stress tests |

### 06 DIAGRAMS

**Purpose**: Visual documentation (100+ Mermaid diagrams).

| Section                  | Description                           |
| :----------------------- | :------------------------------------ |
| **01_ARCHITECTURE**      | Component and model diagrams          |
| **02_SCIENTIFIC_THEORY** | Algebra and geometry visualizations   |
| **03_WORKFLOWS**         | Training and evaluation flow diagrams |
| **04_INFRASTRUCTURE**    | Testing and config diagrams           |

---

## External Documentation

Documentation that exists outside this folder but is part of the project:

| Location                | Purpose                                      | Link                                     |
| :---------------------- | :------------------------------------------- | :--------------------------------------- |
| `/src/README.md`        | Source code architecture overview            | [View](../src/README.md)                 |
| `/research/alphafold3/` | AlphaFold3 integration guides                | [View](../research/alphafold3/README.md) |
| `/results/discoveries/` | Research discovery reports                   | [View](../results/discoveries/)          |
| `/conductor/`           | Project management tooling (separate system) | [View](../conductor/)                    |

See [EXTERNAL_DOCS_INDEX.md](EXTERNAL_DOCS_INDEX.md) for the complete registry.

---

## Stakeholder Quick Links

### For Scientists & Researchers

- [Technical Whitepaper](03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/04_THEORY_DEEP_DIVES/THEORY_Technical_Whitepaper.md)
- [Mathematical Foundations](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/)
- [Validation Benchmarks](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/06_VALIDATION/07_VALIDATION_SUITE/)
- [HIV Research Findings](01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/BIOINFORMATICS/CODON_ENCODER_RESEARCH/HIV/)

### For Investors & Partners

- [Pitch Deck Script](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/01_VENTURE_CAPITAL/PITCH_DECK_SCRIPT.md)
- [Market Opportunity](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/03_MARKET_ANALYSIS/MARKET_OPPORTUNITY.md)
- [IP & Defensibility](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/04_IP_SUMMARY/IP_AND_DEFENSIBILITY.md)
- [Claims & Validation](03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/01_VENTURE_CAPITAL/CLAIMS.md)

### For Developers

- [Source Architecture](../src/README.md)
- [Setup Guide](03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/05_SPECS_AND_GUIDES/SETUP.md)
- [Testing Documentation](01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/06_VALIDATION/)
- [Code Health Metrics](02_PROJECT_MANAGEMENT/02_CODE_HEALTH_METRICS/)

---

## Content Standards

All documentation follows standards defined in:

- [STANDARDS/CONTENT_STANDARDS.md](STANDARDS/CONTENT_STANDARDS.md) - Writing guidelines
- [STANDARDS/TEMPLATE_LIBRARY.md](STANDARDS/TEMPLATE_LIBRARY.md) - Document templates
- [STANDARDS/REVIEW_CHECKLIST.md](STANDARDS/REVIEW_CHECKLIST.md) - Pre-publication checklist

---

## Status Legend

| Icon        | Meaning                           |
| :---------- | :-------------------------------- |
| Complete    | Section is fully documented       |
| In Progress | Active development                |
| Placeholder | Structure exists, content planned |

---

_Last updated: 2025-12-25_
