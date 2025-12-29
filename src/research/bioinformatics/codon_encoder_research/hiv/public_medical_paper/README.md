# HIV Evolution Under Multi-Pressure Selection: A Computational Geometry Approach

**Audience**: Clinicians, Virologists, Immunologists, Vaccine Researchers  
**Format**: Medical-oriented findings without implementation details

---

## About This Document Collection

This collection presents findings from a large-scale computational analysis of HIV evolution under multiple selective pressures. The research integrates five major HIV databases (202,085 total records) to understand how the virus evolves in response to:

- **Antiretroviral therapy** (drug resistance pressure)
- **Cellular immunity** (CTL/CD8+ T-cell pressure)
- **Humoral immunity** (antibody pressure)
- **Tropism requirements** (coreceptor usage)

Our computational approach models genetic changes as movements in a geometric space, allowing us to quantify "evolutionary distance" between mutations and predict viral behavior.

---

## Document Index

| Document | Description |
|:---------|:------------|
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | Overview of key findings and clinical significance |
| **Findings/** | |
| [DRUG_RESISTANCE.md](findings/DRUG_RESISTANCE.md) | Why some resistance mutations emerge faster than others |
| [IMMUNE_ESCAPE.md](findings/IMMUNE_ESCAPE.md) | CTL escape patterns and protective HLA alleles |
| [ANTIBODY_NEUTRALIZATION.md](findings/ANTIBODY_NEUTRALIZATION.md) | Broadly neutralizing antibody insights |
| [TROPISM_ANALYSIS.md](findings/TROPISM_ANALYSIS.md) | Coreceptor tropism determinants (new Position 22 finding) |
| [VACCINE_TARGETS.md](findings/VACCINE_TARGETS.md) | 328 safe epitopes for vaccine development |
| **Supplementary/** | |
| [GLOSSARY.md](supplementary/GLOSSARY.md) | Medical terminology and key concepts |
| [DATA_SOURCES.md](supplementary/DATA_SOURCES.md) | Database descriptions and record counts |
| **Image Specifications/** | |
| [IMAGE_SPECIFICATIONS.md](image_specifications/IMAGE_SPECIFICATIONS.md) | Detailed specs for 30 figures |

---

## Key Discoveries

### Novel Findings

1. **Position 22 in V3 Loop**: We identified position 22 as the strongest tropism determinant, exceeding the classic 11/25 rule
2. **328 Safe Vaccine Targets**: Epitopes with broad HLA coverage that avoid drug resistance conflict
3. **Geometric Resistance Prediction**: 78% accuracy classifying primary vs. accessory mutations using geometric features alone
4. **Quantified Trade-offs**: 16,054 positions where drug resistance and immune escape pressures conflict

### Confirmed (Validates Methodology)

- B*57 and B*27 as protective HLA alleles
- The 11/25 tropism rule (independently recovered)
- Known bnAb breadth and potency profiles
- Protein conservation hierarchy (Gag > Pol > Env > Nef)

---

## Datasets Analyzed

| Database | Records | Focus |
|:---------|--------:|:------|
| Stanford HIV Drug Resistance | 7,154 sequences | Drug resistance mutations |
| LANL CTL Epitope Database | 2,115 epitopes | Cellular immune targets |
| CATNAP Neutralization | 189,879 records | Antibody potency/breadth |
| V3 Tropism Dataset | 2,932 sequences | Coreceptor preference |
| **Total** | **202,085** | Multi-pressure integration |

---

## Clinical Relevance

This research provides:

- **Treatment optimization**: Understanding which resistance mutations require larger genetic changes helps predict durability of ART regimens
- **Vaccine design**: Identifying epitopes that are both immunogenic and resistant to viral escape
- **Prognostic markers**: Geometric signatures that correlate with disease progression
- **Cure research**: Mapping evolutionary constraints that limit viral adaptation

---

## Contact

For validation protocols, data access, or collaboration inquiries, please contact the research team.

---

*This document contains findings only. Implementation details and source code are maintained separately.*
