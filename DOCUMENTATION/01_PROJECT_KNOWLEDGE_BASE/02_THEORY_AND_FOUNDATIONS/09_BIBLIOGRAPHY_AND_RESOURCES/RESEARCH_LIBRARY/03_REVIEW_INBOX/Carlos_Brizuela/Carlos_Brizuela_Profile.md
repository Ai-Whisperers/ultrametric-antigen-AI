# Carlos Brizuela
**Affiliation:** Researcher, Computer Science Department, CICESE (Center for Scientific Research and Higher Education of Ensenada), Mexico.
**Email:** (Verified at cicese.mx)
**Profile:** [Google Scholar](https://scholar.google.com/citations?user=cicese) (Implied)

## Biography
Carlos Brizuela is a Researcher in the Department of Computer Science at CICESE. He holds a Ph.D. from the Institut National Polytechnique de Grenoble (INPG), France (2002), a Master's from CICESE (1998), and a Bachelor's in Computer Science from UABC.

## Research Interests
His primary research focus lies at the intersection of **Computational Biology** and **Machine Learning**, with specific interests in:
*   **Antimicrobial Peptides (AMPs):** Design and identification using computational methods.
*   **Algorithms for Bioinformatics:** Structural modeling and molecular encoding.
*   **Multi-objective Optimization:** Evolutionary algorithms applied to complex problems.
*   **Ubiquitous Computing & IoT:** Social aspects of technology and HCI.

## Selected Publications (2020-2025)

### 2025
*   **A Genetic Algorithm for the Shortest Common Superstring Problem**
    *   *Authors:* Brizuela, C. A., Gonzalez Gurrola, L. C., Tchernykh, A., & Trystram, D.
    *   *Expected:* August 2025

### 2023
*   **Overproduce and select, or determine optimal molecular descriptor subset via configuration space optimization? Application to the prediction of ecotoxicological endpoints**
    *   *Authors:* García-González, L. A., Marrero-Ponce, Y., Brizuela, C. A., & García-Jacas, C. R.
    *   *Journal:* Molecular Informatics, 42(6):e2200227.

### 2022
*   **Antimicrobial peptides with cell-penetrating activity as prophylactic and treatment drugs**
    *   *Authors:* Brizuela, C. A., et al.

### 2021
*   **Alignment-Free Antimicrobial Peptide Predictors: Improving Performance by a Thorough Analysis of the Largest Available Data Set**
    *   *Authors:* Pinacho Castellanos, S. A., García Jacas, C. R., Gilson, M. K., & Brizuela Rodríguez, C. A.
    *   *Journal:* Journal of Chemical Information and Modeling, 61(6), 3141-3157.
*   **A Comprehensive Review and a Taxonomy Proposal of Team Formation Problems**
    *   *Authors:* Juárez Jiménez, J. A., Santos, C., & Brizuela Rodríguez, C. A.
    *   *Journal:* ACM Computing Surveys, 54(7), 1-33.

### 2020
*   **Automatic construction of molecular similarity networks for visual graph mining in chemical space of bioactive peptides: an unsupervised learning approach**
    *   *Authors:* Aguilera Mendoza, L., et al.
    *   *Journal:* Scientific Reports, 10, 18074.
*   **Relevant Features of Polypharmacologic Human-Target Antimicrobials Discovered by Machine-Learning Techniques**
    *   *Authors:* Nava Lara, R., Beltrán Verdugo, J. A., Brizuela Rodríguez, C. A., & Del Río, G.
    *   *Journal:* Pharmaceuticals, 13(9), 204.
*   **An automatic representation of peptides for effective antimicrobial activity classification**
    *   *Authors:* Beltrán Verdugo, J. A., Del Río, G., & Brizuela Rodríguez, C. A.
    *   *Journal:* Computational and Structural Biotechnology Journal, 18, 455-463.
*   **Smoothed Spherical Truncation based on Fuzzy Membership Functions: Application to the Molecular Encoding**
    *   *Authors:* García-Jacas CR, Marrero-Ponce Y, Brizuela CA, et al.
    *   *Journal:* Journal of Computational Chemistry, 41(3):203-217.

## Key Projects & Contributions
*   **Machine Learning for AMPs:** Development of alignment-free predictors and automatic representation methods for peptides to improve antimicrobial activity classification.
*   **Visual Graph Mining:** Construction of molecular similarity networks to explore the chemical space of bioactive peptides.
*   **Evolutionary Algorithms:** Application of genetic algorithms to problems like the Shortest Common Superstring Problem and multi-objective optimization tasks.
*   **Polypharmacology:** Identifying features of human-target antimicrobials using ML techniques.

## Codebase Integration
*   **Ingestion Pipeline:** `scripts/ingest/ingest_starpep.py` implements the loading of peptide datasets (like StarPepDB) used in Brizuela's work.
*   **Geometric Encoding:** We utilize the `encode_sequence_hyperbolic` function (in `hyperbolic_utils.py`) to map his alignment-free peptide sequences into our project's 3-adic hyperbolic space, effectively translating his "molecular similarity" concepts into our "ternary geometric" framework.
