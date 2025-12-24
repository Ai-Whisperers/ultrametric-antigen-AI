# How to Report and Make Sense of a New HIV-1 Circulating Recombinant Form

**ID:** EPI-003
**Year:** 2024
**Journal:** Frontiers in Microbiology
**DOI:** [10.3389/fmicb.2024.1343143](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1343143/full)

---

## Abstract

This 2024 Frontiers review provides standardized criteria for classification, nomenclature, and reporting of new HIV-1 circulating recombinant forms (CRFs). With 157+ CRFs now identified globally, proper documentation and analysis of recombinant strains is critical for understanding HIV evolution, tracking epidemics, and developing broadly effective vaccines and diagnostics.

---

## Key Concepts

- **CRF (Circulating Recombinant Form)**: Recombinant HIV found in ≥3 epidemiologically unlinked individuals
- **URF (Unique Recombinant Form)**: Recombinant found in single individual/cluster
- **Mosaic Structure**: Alternating segments from different HIV subtypes
- **Breakpoint**: Junction between parental subtype segments

---

## CRF Classification Criteria

### Requirements for CRF Designation
| Criterion | Requirement |
|:----------|:------------|
| Minimum cases | ≥3 epidemiologically unlinked individuals |
| Sequence confirmation | Full-genome or near-full-genome |
| Recombination evidence | Clear mosaic structure with defined breakpoints |
| Phylogenetic support | Bootstrap/posterior probability thresholds met |

### Nomenclature System
```
CRF[number]_[parental subtypes]
Example: CRF01_AE (from subtypes A and E)
         CRF07_BC (from subtypes B and C)
         CRF159_01103 (from CRF103_01B and CRF01_AE)
```

---

## Global CRF Landscape (2024)

### Current Count
| Metric | Value |
|:-------|:------|
| Total CRFs described | **157+** |
| Previously reported | 140 (as of 2023) |
| New in 2024 | ~17+ |

### Recent CRF Discoveries (2024)
| CRF | Location | Parental Forms | Origin Date |
|:----|:---------|:---------------|:------------|
| CRF139_02B | Japan | 02_AG + B | UK origin |
| CRF159_01103 | China (Hebei) | CRF103_01B + CRF01_AE | 2018-2019 |
| CRF168_0107 | China (Beijing) | CRF01_AE + CRF07_BC | Recent |
| CRF126_0755 | China (Guangdong) | CRF55_01B + CRF07_BC | 2005-2007 |
| CRF150_Cpx | China | Complex | MSM population |

---

## Why CRFs Matter

### Public Health Implications
| Concern | Impact |
|:--------|:-------|
| Vaccine coverage | May escape strain-specific immunity |
| Diagnostic detection | Assays must recognize all variants |
| Drug resistance | Novel resistance patterns possible |
| Transmission dynamics | Indicate mixing populations |

> "Identification and prompt reporting of new CRFs will provide not only new insights into the understanding of genetic diversity and evolution of HIV-1, but also an early warning of potential prevalence of these variants."

---

## Analysis Methods

### Bioinformatic Pipeline
| Step | Tools |
|:-----|:------|
| Sequence alignment | MAFFT, MUSCLE |
| Recombination detection | RDP4, jpHMM, SimPlot |
| Phylogenetic analysis | RAxML, IQ-TREE, BEAST |
| Breakpoint mapping | bootscan, RIP |

### Quality Control
| Check | Requirement |
|:------|:------------|
| Sequence quality | Q30+ for NGS data |
| Coverage | ≥95% of reference genome |
| Contamination | Rule out lab artifacts |
| Epidemiological independence | No direct transmission links |

---

## Regional Hotspots

### High CRF Diversity Regions
| Region | Dominant CRFs | Driving Factor |
|:-------|:--------------|:---------------|
| Southeast Asia | CRF01_AE, CRF07_BC, novel CRFs | MSM + IDU networks |
| Central Africa | Multiple pure subtypes + CRFs | Subtype co-circulation |
| Eastern Europe | CRF03_AB, pure subtypes | IDU epidemics |
| China | CRF07_BC, CRF01_AE, new CRFs | Epidemic expansion |

### China Surveillance
> "Over the past decade, CRF07_BC and CRF01_AE have become the predominant circulating strains, especially among the MSM population in China, along with the frequent recombination between the two strains resulting in serial novel CRFs."

---

## Reporting Best Practices

### Publication Requirements
| Element | Details |
|:--------|:--------|
| Sequence deposition | GenBank with accession numbers |
| Phylogenetic trees | Publication-quality figures |
| Breakpoint maps | Visual recombination structure |
| Epidemiological data | De-identified patient info |

### Database Submission
| Database | Purpose |
|:---------|:--------|
| Los Alamos HIV Database | Official CRF registration |
| GenBank | Sequence deposition |
| GISAID (if applicable) | Outbreak sequences |

---

## Vaccine and Diagnostic Implications

### Challenges Posed by CRFs
| Challenge | Consequence |
|:----------|:------------|
| Antigenic diversity | Broader vaccine coverage needed |
| Recombination hotspots | Unpredictable epitope changes |
| Detection gaps | Diagnostic kit updates required |
| Drug target variation | Potential novel resistance |

> "Among the hurdles facing HIV vaccine development is the global HIV-1 genetic diversity, where recombinants now constitute the highest proportion of circulating HIV while continuing to increase in prevalence."

---

## Relevance to Project

CRF research directly informs the Ternary VAE project:
- **Recombination modeling**: Sequence evolution beyond point mutations
- **Breakpoint prediction**: Structural constraints on recombination
- **Fitness landscapes**: Hybrid genomes and viability
- **Global coverage**: Training on diverse recombinant sequences

---

*Added: 2025-12-24*
