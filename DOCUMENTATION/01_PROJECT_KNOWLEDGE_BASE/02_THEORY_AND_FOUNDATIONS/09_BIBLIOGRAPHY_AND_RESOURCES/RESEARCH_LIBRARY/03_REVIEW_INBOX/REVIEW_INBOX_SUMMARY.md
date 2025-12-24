# REVIEW_INBOX – Paper‑to‑Repo Mapping

**Date:** 2025‑12‑24

---

## 1. 01_AUTOIMMUNITY_AND_CODONS

| Paper                                    | Core idea                                              | How it connects to our code base                                                                             | Implementation tip                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `01_NOBEL_PRIZE_IMMUNE_SYSTEM.md`        | Historical perspective on immune‑system breakthroughs. | Provides motivation for the **Autoimmunity pillar** of the project.                                          | Use the narrative in `README.md` of `autoimmunity/` to frame experiments.                                                |
| `02_VIRAL_EVOLUTION.md`                  | Evolutionary pressures shaping viral genomes.          | Reinforces the **p‑adic evolution hypothesis** used in `p_adic_scaler.py`.                                   | Add a comment in the scaler explaining the evolutionary rationale.                                                       |
| `04_LONG_COVID_AND_VIRAL_DISEASE.md`     | Long‑COVID mechanisms and viral persistence.           | Highlights the importance of **latent‑state detection** (CD4/CD8 tracking).                                  | Extend `src/metrics/latent_state.py` to include long‑COVID biomarkers.                                                   |
| `08_AUTOIMMUNE_DISEASE.md`               | Mechanisms of auto‑immune pathology.                   | Directly relevant to the **Autoimmunity & Codons** experiments.                                              | Create a new notebook `autoimmunity_analysis.ipynb` that loads the P‑adic shift and correlates with auto‑immune markers. |
| `1995_Wucherpfennig_EBV_MBP_Mimicry.md`  | Molecular mimicry between EBV and host proteins.       | Shows a **cross‑reactivity** pattern that can be modeled as a **geometric similarity** in p‑adic space.      | Add a similarity‑matrix feature to `src/analysis/similarity.py`.                                                         |
| `2003_Suzuki_PADI4_Haplotypes.md`        | PADI4 haplotypes modulate citrullination.              | Provides a **genetic‑variant** case study for codon‑bias analysis.                                           | Encode the haplotype SNPs as additional dimensions in the VAE latent vector.                                             |
| `2005_Matsumoto_PADI4_Localization.md`   | Sub‑cellular localisation of PADI4.                    | Gives a **structural** angle that can be linked to our **protein‑geometry** module.                          | Use AlphaFold‑predicted structures to compute p‑adic shift of localisation motifs.                                       |
| `2006_Poole_EBV_SLE_Paradox.md`          | EBV involvement in systemic lupus.                     | Another **viral‑autoimmune link** that justifies the combined pillar.                                        | Add a case‑study entry in `docs/autoimmunity_cases.md`.                                                                  |
| `2009_Vossenaar_PADI4_Review.md`         | Review of PADI4 biology.                               | Serves as a **reference compendium** for codon‑bias experiments.                                             | Cite this file in the `methods` section of the VAE paper.                                                                |
| `2011_Pandit_HIV_Codon_Trends.md`        | Temporal trends in HIV codon usage.                    | Directly supports our **CodonEncoder** validation.                                                           | Use the reported trends as a benchmark dataset in `tests/codon_trends_test.py`.                                          |
| `2014_AlHarthi_HIV_Latency_Evolution.md` | Evolution of HIV latency mechanisms.                   | Provides a **biological justification** for the “Invisible Zone” concept.                                    | Add a paragraph in `docs/theory/padic_biology.md` referencing this work.                                                 |
| `2014_Halvorsen_AntiPAD4_Severity.md`    | Anti‑PAD4 therapeutic effects.                         | Shows a **potential drug target** that could be modeled with our VAE loss functions.                         |
| `2017_Roy_HIV_Env_Codon.md`              | Env‑gene codon optimisation.                           | Supplies real‑world examples of **codon optimisation** for vaccine design.                                   | Feed the Env sequences into `src/codon_encoder/` and compare with our synthetic designs.                                 |
| `2022_Lanz_EBNA1_GlialCAM.md`            | EBNA‑1 interaction with GlialCAM.                      | Highlights a **viral‑host protein interaction** that can be explored with our **protein‑geometry** pipeline. |
| `README.md`                              | Index of the folder.                                   | N/A                                                                                                          |

---

## 2. 02_GENETIC_CODE_THEORY

| Paper                                         | Core idea                                                     | Repo relevance                                                                                | Implementation tip                                                                  |
| --------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `1965_Woese_Stereochemical_Origin.md`         | Stereochemical hypothesis for the origin of the genetic code. | Provides a **theoretical foundation** for our p‑adic encoding of codons.                      | Add a `theory` section in `docs/genetic_code_theory.md` linking to this hypothesis. |
| `1969_AlffSteinberger_Code_Simulation.md`     | Computational simulation of early genetic code evolution.     | Mirrors our **simulation engine** (`src/simulation/`).                                        | Use the simulation parameters as a test case for the VAE’s generative capacity.     |
| `1975_Wong_CoEvolution_Theory.md`             | Co‑evolution of codon usage and amino‑acid biosynthesis.      | Directly informs the **co‑evolution loss** we plan to add to the VAE.                         | Implement a custom loss term `co_evolution_loss` in `src/models/vae.py`.            |
| `1991_Haig_Error_Minimization.md`             | Error‑minimization principle for the genetic code.            | Aligns with our **reconstruction loss** weighting scheme.                                     | Adjust loss weighting to reflect error‑minimization priorities.                     |
| `1998_Freeland_One_In_A_Million.md`           | Rarity of the canonical code among alternatives.              | Justifies the **regularisation** we apply to latent space.                                    |
| `2010_Johnson_Stereochemical_Evidence.md`     | Experimental evidence for stereochemical interactions.        | Supplies **ground‑truth data** for validating our codon‑geometry mapping.                     |
| `2018_Wnetrzak_Eight_Objective_Optimality.md` | Multi‑objective optimisation of the code.                     | Provides a **framework** for multi‑task VAE training (e.g., reconstruction + classification). |
| `2020_Shenhav_Resource_Conservation.md`       | Resource‑conservation principle in code evolution.            | Mirrors our **resource‑aware loss** (e.g., penalising high‑energy codon choices).             |
| `2021_Xu_Resource_Conservation_Rebuttal.md`   | Counter‑argument to resource‑conservation.                    | Useful for **discussion** of model limitations.                                               |
| `2024_Wehbi_Code_Evolution_Phylogenetics.md`  | Phylogenetic analysis of code evolution.                      | Gives a **benchmark phylogeny** we can compare against VAE‑generated trees.                   |
| `README.md`                                   | Index.                                                        |

---

## 3. 03_PADIC_BIOLOGY

| Paper                                                                                                                                                                                    | Core idea                                           | Repo relevance                                                                               | Implementation tip                                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `2012_Scalco_Protein_Ultrametricity.md`                                                                                                                                                  | Ultrametric geometry of protein folding landscapes. | Directly supports our **p‑adic metric** for protein structures.                              | Use the ultrametric distance matrix as an additional input to the VAE encoder. |
| `1997_Onuchic_Protein_Landscapes.md`                                                                                                                                                     | Energy landscape theory of proteins.                | Provides a **theoretical backdrop** for the “energy‑basin” interpretation of latent vectors. |
| `2010_Khrennikov_Ultrametric_Disease_Spread.md`                                                                                                                                          | Ultrametric models of disease transmission.         | Justifies the **geometric invisibility** concept used for HIV.                               |
| `2006_Kozyrev_Padic_Analysis_Methods.md`                                                                                                                                                 | Methods for p‑adic analysis of biological data.     | Supplies **algorithmic recipes** that can be ported to `src/p_adic/`.                        |
| `...` (other files in this folder follow the same pattern – each introduces a mathematical tool or case study that can be wrapped as a Python module and called from the main pipeline). |

---

## 4. 04_SPECTRAL_BIO_ML

| Paper                                                                       | Core idea                                                | Repo relevance                                                                      | Implementation tip |
| --------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------ |
| `2020_Jing_Geometric_Vector_Perceptrons.md`                                 | Geometric vector perceptrons for protein representation. | Aligns with our **spectral embedding** layer in the VAE.                            |
| `2006_Coifman_Diffusion_Maps.md`                                            | Diffusion‑maps for manifold learning.                    | Can replace the current PCA step with a diffusion‑map encoder.                      |
| `2020_Gainza_MaSIF_Surfaces.md`                                             | Surface‑based deep learning for proteins.                | Provides a **surface‑feature extractor** that can be added to the encoder pipeline. |
| (additional papers in this folder similarly map to spectral‑ML components). |

---

## 5. HIV_RESEARCH_2024 (selected highlights)

Given the large number of files (>100) we summarise the most relevant groups:

### 5.1 Glycan‑Shield & Structural Studies

- Papers describing **BG505 SOSIP**, **N332 sentinel glycan**, and AlphaFold‑predicted mutants.
- **Repo link:** `research/bioinformatics/codon_encoder_research/hiv/` – the structural files are already stored here.
- **Implementation:** Use the `alpha_fold3` validation results (already in `research/alphafold3/`) to train a **geometry‑aware loss** that penalises large p‑adic shifts when glycans are removed.

### 5.2 Codon‑Bias & Evolutionary Trends

- `2011_Pandit_HIV_Codon_Trends.md`, `2017_Roy_HIV_Env_Codon.md` – provide time‑series codon‑usage data.
- **Repo link:** `src/codon_encoder/` – extend the encoder to output a **trend vector** that can be compared with the historical data.

### 5.3 Vaccine & Immunogen Design

- Papers on **broadly neutralising antibodies**, **N332 deletion strategies**, and **immunogen engineering**.
- **Implementation:** Create a new notebook `hiv_vaccine_design.ipynb` that loads the Env sequences, applies the sentinel‑glycan removal, runs AlphaFold, and evaluates the p‑adic shift.

### 5.4 Drug‑Resistance & Lenacapavir

- `2024_Wehbi_Code_Evolution_Phylogenetics.md` (also appears in Genetic‑Code folder) and any **lenacapavir**‑specific studies.
- **Repo link:** `research/drug_resistance/` – currently empty; we will add a new module `src/drug_resistance/` that parses resistance‑mutation tables and couples them to p‑adic shift scores.

---

## 6. How to turn the insights into code

1. **Create a “paper‑to‑module” mapping table** (`docs/paper_module_map.csv`). Each row contains `paper_file, related_module, implementation_idea`.
2. **Add a thin wrapper** for each idea:
   - _Metric wrapper_ – e.g., `p_adic_shift(seq)` → returns a scalar.
   - _Loss wrapper_ – e.g., `sentinel_glycan_loss(recon, target_shift)`.
   - _Dataset wrapper_ – e.g., `load_hiv_glycan_dataset()` that reads the FASTA files from the research folder.
3. **Update the training script** (`train_vae.py`):
   - Load the new wrappers via a config file (`config/papers.yaml`).
   - Enable multi‑task learning: reconstruction + `sentinel_glycan_loss` + `co_evolution_loss`.
4. **Document each step** in the corresponding markdown file under `docs/implementation/` so future collaborators can trace the provenance from paper → code.

---

## 7. Final checklist (to be added to `task.md`)

- [ ] Generate `paper_module_map.csv`.
- [ ] Implement `p_adic_shift` wrapper (if not already present).
- [ ] Add `sentinel_glycan_loss` to VAE loss module.
- [ ] Create notebooks `hiv_vaccine_design.ipynb` and `autoimmunity_analysis.ipynb`.
- [ ] Update `README.md` of each sub‑folder with the new implementation notes.
- [ ] Run a quick sanity‑check on a subset of papers (e.g., `2011_Pandit_HIV_Codon_Trends.md`).

---

_End of document._
