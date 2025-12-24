# Dataset Overview for Ternary‑VAE Bioinformatics Project

**Date:** 2025‑12‑24

---

## 1. HIV‑related datasets

| Dataset                                                         | Content                                                                                    | Typical size        | How we can use it                                                                                                                    |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Los Alamos HIV Database**                                     | Full‑length HIV‑1/2 genomic sequences, protein structures, drug‑resistance mutations.      | >10 GB (FASTA, CSV) | Import sequences → `CodonEncoder` → compute p‑adic shift per strain → label resistant vs. sensitive for VAE training.                |
| **Stanford HIV Drug Resistance Database (HIVDB)**               | Annotated resistance mutations for protease, RT, integrase, capsid; phenotype data (IC₅₀). | ~500 MB             | Use mutation lists as categorical features; correlate specific mutations (e.g., N332) with geometric shift metrics.                  |
| **Retrovirus Integration Database (RID)**                       | Integration site coordinates for HIV and other retroviruses.                               | ~200 MB             | Encode integration loci as positional embeddings; test whether integration hotspots align with p‑adic “invisible zones”.             |
| **ClinicalTrials.gov – HIV vaccine trials (e.g., RV144, HVTN)** | Immunogenicity read‑outs, antibody titers, epitope maps.                                   | ~100 MB (CSV/JSON)  | Treat antibody titers as target variables; evaluate whether engineered immunogens (e.g., N332‑deleted) improve predicted visibility. |

## 2. Antibiotics & Retrovirals (bacterial infections)

| Dataset                                                        | Content                                                                | How we can use it                                                                                                       |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **NCBI GEO – Antibiotic stress series (e.g., GSE72220)**       | RNA‑seq of _E. coli_ under different antibiotics, expression matrices. | Convert raw counts → codon usage per gene → feed into VAE to learn how antibiotic pressure reshapes codon geometry.     |
| **PATRIC – Antimicrobial Resistance (AMR) Data**               | Whole‑genome assemblies with phenotypic MIC values.                    | Encode genomes → compute p‑adic shift → train classifier that predicts MIC from geometric features.                     |
| **DrugBank – Retroviral drugs (tenofovir, lenacapavir, etc.)** | SMILES, pharmacodynamics (IC₅₀, PK/PD).                                | Generate molecular fingerprints → concatenate with viral p‑adic vectors for multimodal VAE that predicts drug efficacy. |

## 3. Syphilis (Treponema pallidum)

| Dataset                               | Content                                        | How we can use it                                                                                     |
| ------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **CDC Syphilis Surveillance**         | Case counts, RPR titers, demographic metadata. | Link serology to available _T. pallidum_ genomes (NCBI SRA) → study codon‑shift vs. disease severity. |
| **NCBI SRA – BioProject PRJNA437123** | Raw reads for _T. pallidum_ isolates.          | Assemble genomes → feed into codon encoder; compare p‑adic profiles across strains.                   |

## 4. Hepatitis B & C

| Dataset                         | Content                                                                           | How we can use it                                                                             |
| ------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **HBVdb**                       | Complete HBV genomes (genotypes A‑H), annotation of surface/ polymerase proteins. | Encode each genotype → evaluate whether certain genotypes sit closer to the “invisible zone”. |
| **HCVdb**                       | Full‑length HCV genomes, subtype information.                                     | Same workflow as HBV; compare p‑adic shifts across flaviviridae.                              |
| **GISAID – Hepatitis datasets** | Sequences with clinical metadata (treatment, viral load).                         | Use treatment labels to train supervised VAE that predicts response to antivirals.            |

## 5. Tuberculosis (Mycobacterium tuberculosis)

| Dataset                                                    | Content                                                            | How we can use it                                                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **TB Portals Program**                                     | Clinical data, chest X‑ray DICOM, PPD results, treatment outcomes. | Extract PPD status → map to bacterial genomic data (SRA) → test whether p‑adic shift correlates with latent vs. active disease. |
| **NCBI SRA – M. tuberculosis RNA‑seq under drug pressure** | Transcriptomes for drug‑sensitive and resistant strains.           | Encode expression‑derived codon usage; train VAE to separate resistant phenotypes.                                              |

## 6. Immunosuppression (CD4 / CD8 counts)

| Dataset                               | Content                                                      | How we can use it                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **ImmPort – HIV cohort studies**      | Longitudinal CD4/CD8 counts, viral load, treatment regimens. | Use counts as continuous labels for regression on p‑adic shift; assess whether higher shift predicts CD4 decline. |
| **FlowRepository – Cytometry panels** | .fcs files with CD4/CD8 gating.                              | Convert to numeric summaries → integrate as additional features in multimodal VAE.                                |

## 7. Antiretroviral (AR) drug data

| Dataset                                      | Content                                                        | How we can use it                                                                              |
| -------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **ClinicalTrials.gov – AR efficacy studies** | Protocols, outcome measures (viral suppression, side‑effects). | Map outcomes to viral genotype‑shift vectors; evaluate predictive power of geometric metrics.  |
| **DrugBank – AR compounds**                  | Chemical structures, mechanism of action.                      | Create drug embeddings; combine with viral p‑adic vectors for drug‑virus interaction modeling. |

## 8. HIV Vaccine research

| Dataset                      | Content                                                 | How we can use it                                                                                    |
| ---------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **RV144 trial data**         | Antibody breadth, neutralization titers, Env sequences. | Align Env sequences → compute p‑adic shift after in‑silico removal of sentinel glycans (e.g., N332). |
| **HVTN – Immunogen designs** | Engineered Env variants, structural models.             | Use structural models to validate AlphaFold‑predicted stability changes after glycan deletion.       |

## 9. Lenacapavir & Sertraline interaction

| Dataset                                   | Content                                                    | How we can use it                                                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **PubMed – Lenacapavir pharmacodynamics** | IC₅₀, PK/PD, reported drug‑drug interactions (sertraline). | Encode pharmacodynamic curves as numeric vectors; test whether lenacapavir‑induced p‑adic shift reduces sertraline re‑uptake in silico. |
| **ClinicalTrials.gov – NCT05364044**      | Study of lenacapavir + antidepressants.                    | Extract adverse‑event frequencies; correlate with predicted geometric changes in capsid protein.                                        |

---

## How to integrate these datasets into the Ternary‑VAE pipeline

1. **Ingestion** – Write a small Python script (`src/data_ingest/ingest_<source>.py`) that downloads the raw files (using `requests`, `sra-tools`, or `pandas`). Store the cleaned files under `data/raw/<source>/`.
2. **Pre‑processing** – Convert sequences to codon matrices with the existing `codon_encoder_research` module. For expression data, compute TPM and then derive codon usage per gene.
3. **p‑adic Scaler** – Feed the codon matrices into `p_adic_scaler.py` to obtain a **percent shift** for each sample. Append this value to a metadata CSV (`data/processed/<source>_metadata.csv`).
4. **VAE training** – Extend `train_vae.py` to accept multiple `DATA_SOURCES`. Use the p‑adic shift as an auxiliary label (supervised or semi‑supervised) to guide the latent space.
5. **Validation & Visualization** – In `notebooks/validation.ipynb` plot `p_adic_shift` vs. clinical outcomes (CD4 count, MIC, vaccine titer). Generate ROC curves for classification tasks (resistant vs. susceptible).
6. **Documentation** – For each new source, create a **Research Digest** in `DOCUMENTATION/.../RESEARCH_LIBRARY/` following the standard template (author, year, abstract, key data, relevance). Link the digest from this overview file.

---

**Result:** This markdown file (`2025_DATASET_OVERVIEW.md`) provides a single reference for all public resources we plan to exploit, describes their contents, and outlines a concrete integration path into our codon‑p‑adic VAE workflow.
