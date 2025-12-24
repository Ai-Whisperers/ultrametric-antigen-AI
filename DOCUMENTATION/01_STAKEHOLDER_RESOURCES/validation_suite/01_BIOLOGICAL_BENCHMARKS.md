# 01 Validation Suite: Biological Benchmarks

> **Objective**: Prove the model predicts "Escape" and fitness across the virosphere and proteome.
> **Target Scale**: 40+ Datasets per category.

## A. Viral Evolution (40+ Pathogens)

_We prove the model predicts "Escape" across the entire virosphere._

**Tier 1: Global Pandemics (Deep Analysis)**

1.  **SARS-CoV-2 (Omicron BA.5)** - Spike RBD escape.
2.  **SARS-CoV-2 (XBB.1.5)** - Antibody evasion.
3.  **Influenza A (H3N2)** - Hemagglutinin drift.
4.  **Influenza A (H1N1)** - Neuraminidase shift.
5.  **HIV-1 (Clade B)** - Envelope Glycan Shield.
6.  **HIV-1 (Clade C)** - Drug resistance (Reverse Transcriptase).
7.  **Ebola Zaire** - Glycoprotein stability.
8.  **Zika Virus** - Envelope protein.
9.  **Dengue Virus (Serotype 2)** - E protein.
10. **Respiratory Syncytial Virus (RSV)** - F protein.

**Tier 2: High-Risk Pathogens (Generalization Test)** 11. **Lassa Virus** (GPC) 12. **Nipah Virus** (G) 13. **Hendra Virus** 14. **Marburg Virus** 15. **MERS-CoV** 16. **Hepatitis C (HCV)** - E1/E2 hypervariable types. 17. **Hepatitis B (HBV)** - Polymerase. 18. **Chikungunya Virus** 19. **West Nile Virus** 20. **Yellow Fever Virus** 21. **Rabies Virus** (G) 22. **Measles Virus** (H) 23. **Mumps Virus** 24. **Rubella Virus** 25. **Poliovirus 1** 26. **Enterovirus D68** 27. **Rhinovirus C** 28. **Human Cytomegalovirus (CMV)** 29. **Epstein-Barr Virus (EBV)** 30. **Varicella-Zoster Virus (VZV)**

**Tier 3: Emerging & Animal Reservoirs (Zero-Shot)** 31. **Avian Influenza H5N1** 32. **Canine Distemper Virus** 33. **Feline Coronavirus** 34. **Porcine Epidemic Diarrhea Virus** 35. **Bat Coronavirus RaTG13** 36. **Pangolin Coronavirus** 37. **Simian Immunodeficiency Virus (SIV)** 38. **Monkeypox (Mpox)** 39. **Crimean-Congo Hemorrhagic Fever** 40. **Rift Valley Fever**

## B. Protein Fitness (ProteinGym - 40+ Selected Assays)

_We benchmark against the industry standard (ProteinGym)._

**Substitutions (Deep Mutational Scanning)**:

1.  **P53_HUMAN** (Tumor Suppressor)
2.  **BRCA1_HUMAN** (Breast Cancer)
3.  **PTEN_HUMAN**
4.  **CFTR_HUMAN** (Cystic Fibrosis)
5.  **MSH2_HUMAN**
6.  **CALM1_HUMAN**
7.  **TPK1_HUMAN**
8.  **SYUA_HUMAN** (Alpha-synuclein)
9.  **TDP43_HUMAN** (ALS)
10. **SOD1_HUMAN**
11. **BETA-LACTAMASE** (Antibiotic resistance)
12. **GFP** (Fluorescence)
13. **AAV2** (Gene Therapy Capsid)
14. **HSP90** (Heat Shock Protein)
15. **KKA2** (Antibiotic resistance)
16. **MATA_YEAST**
17. **SUMO1_HUMAN**
18. **UBC9_HUMAN**
19. **TPK1_HUMAN**
20. **ESTA_BAC** (Esterase)
    ... [Full list in ProteinGym manifest]
