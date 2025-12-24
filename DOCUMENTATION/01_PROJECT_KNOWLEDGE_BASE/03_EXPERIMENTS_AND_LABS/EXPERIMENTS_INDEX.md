# Experiments & Labs Index

> **Purpose**: This interactive index maps our **theoretical research** (from `ACADEMIC_DATABASE.md`) to our **active code labs**. Use this to find where to test specific hypotheses.

## 🧬 Domain: Bioinformatics

| Research Area         | Key Theory (Papers)                                     | Active Lab Directory                                                                                                           | Status              |
| :-------------------- | :------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------- | :------------------ |
| **Genetic Code**      | _Dragovich et al. (2010)_ "p-Adic Modelling"            | [`bioinformatics/genetic_code/`](./bioinformatics/genetic_code/)                                                               | 🟢 **Stable**       |
| **Viral Evolution**   | _Frazer et al. (2021)_ "EVE"; _Obermeyer (2022)_ "PyR0" | [`bioinformatics/codon_encoder_research/sars_cov_2/`](./bioinformatics/codon_encoder_research/sars_cov_2/)                     | 🟡 **Active**       |
| **Glycan Shielding**  | _PeSTo-Carbs_; _Torres et al._                          | [`bioinformatics/codon_encoder_research/hiv/`](./bioinformatics/codon_encoder_research/hiv/)                                   | 🟡 **Active**       |
| **Autoimmunity**      | _MHC-II HLA interactions_                               | [`bioinformatics/codon_encoder_research/rheumatoid_arthritis/`](./bioinformatics/codon_encoder_research/rheumatoid_arthritis/) | 🔴 **Experimental** |
| **Neurodegeneration** | _Ternary Logic in Cognition_                            | [`bioinformatics/codon_encoder_research/neurodegeneration/`](./bioinformatics/codon_encoder_research/neurodegeneration/)       | ⚪ **Planned**      |

## 📐 Domain: Mathematics & Geometry

| Research Area             | Key Theory (Papers)                           | Active Lab Directory                                                                               | Status        |
| :------------------------ | :-------------------------------------------- | :------------------------------------------------------------------------------------------------- | :------------ |
| **Hyperbolic Embeddings** | _Nickel & Kiela (2017)_ "Poincaré Embeddings" | [`mathematics/hyperbolic_geodesics/`](./mathematics/)                                              | 🟢 **Stable** |
| **3-adic Numbers**        | _Khrennikov (2004)_ "p-adic Information"      | [`bioinformatics/genetic_code/scripts/`](./bioinformatics/genetic_code/scripts/)                   | 🟢 **Stable** |
| **Spectral Analysis**     | _Smita Krishnaswamy_ "Geometric Scattering"   | [`bioinformatics/spectral_analysis_over_models/`](./bioinformatics/spectral_analysis_over_models/) | 🟡 **Active** |

## 🧪 Workflow Guides

- **To run a new biological experiment**: Copy the template from [`bioinformatics/genetic_code/scripts/`](./bioinformatics/genetic_code/scripts/) and adapt the `PROJECT_ROOT` path.
- **To test a new theory**: Create a subfolder in `mathematics/` if it's pure theory, or add a domain folder in `codon_encoder_research/` if it requires biological data.

## 🔗 Connection to Academic Database

- **Hub A (Geometric DL)** -> Testing Grounds: `mathematics/` & `spectral_analysis_over_models/`
- **Hub B (Algebraic Bio)** -> Testing Grounds: `genetic_code/` (3-adic logic)
- **Hub C (P-adic Physics)** -> Testing Grounds: `genetic_code/`
- **Hub D (Viral Evolution)** -> Testing Grounds: `sars_cov_2/` & `hiv/`
