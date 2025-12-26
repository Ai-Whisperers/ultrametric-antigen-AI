# Glossary of Terms

**Purpose**: Define key terms used throughout this research for clinicians and medical professionals.

---

## HIV Biology

### Structural Proteins

| Term | Definition |
|:-----|:-----------|
| **Gag** | Group-specific antigen. Precursor polyprotein that is cleaved into matrix (MA), capsid (CA), nucleocapsid (NC), and p6 proteins. Essential for viral particle assembly. |
| **Pol** | Polymerase. Precursor polyprotein containing protease (PR), reverse transcriptase (RT), and integrase (IN). Essential enzymes for viral replication. |
| **Env** | Envelope. Precursor protein cleaved into gp120 (surface) and gp41 (transmembrane). Mediates viral entry into cells. |

### Accessory Proteins

| Term | Definition |
|:-----|:-----------|
| **Nef** | Negative regulatory factor. Enhances viral replication, downregulates CD4 and MHC-I. Not essential in vitro but important in vivo. |
| **Vif** | Viral infectivity factor. Counteracts APOBEC3G restriction factor. |
| **Vpr** | Viral protein R. Causes G2 cell cycle arrest, enhances viral replication. |
| **Vpu** | Viral protein U. Enhances CD4 degradation and viral release. |

### Viral Regions

| Term | Definition |
|:-----|:-----------|
| **V3 Loop** | Third variable region of gp120. Determines coreceptor tropism (CCR5 vs. CXCR4). Key target for antibodies. |
| **MPER** | Membrane-proximal external region of gp41. Highly conserved, target of broadly neutralizing antibodies like 10E8. |
| **CD4 Binding Site** | Region of gp120 that contacts CD4 receptor. Functionally constrained, target of bnAbs like VRC01. |

---

## Drug Resistance

### Drug Classes

| Term | Definition | Examples |
|:-----|:-----------|:---------|
| **NRTI** | Nucleoside/nucleotide reverse transcriptase inhibitor. Incorporates into DNA and terminates chain. | Tenofovir, emtricitabine, abacavir |
| **NNRTI** | Non-nucleoside reverse transcriptase inhibitor. Binds allosteric site, changes RT conformation. | Efavirenz, rilpivirine, doravirine |
| **PI** | Protease inhibitor. Blocks cleavage of Gag-Pol polyprotein. | Darunavir, atazanavir, lopinavir |
| **INSTI** | Integrase strand transfer inhibitor. Blocks integration of viral DNA into host genome. | Dolutegravir, bictegravir, raltegravir |
| **CCR5 Antagonist** | Blocks CCR5 coreceptor, preventing R5-tropic virus entry. | Maraviroc |

### Resistance Terminology

| Term | Definition |
|:-----|:-----------|
| **Primary Mutation** | Mutation that directly reduces drug binding or efficacy. Typically first to emerge. |
| **Accessory Mutation** | Mutation that compensates for fitness cost of primary mutation. Enhances resistance level. |
| **Fold-Change** | Ratio of IC50 for mutant vs. wild-type virus. Higher = more resistant. |
| **Genetic Barrier** | Number and difficulty of mutations required for resistance. High barrier = more durable regimen. |
| **Cross-Resistance** | When a mutation confers resistance to multiple drugs. Common within drug classes. |

---

## Immunology

### Cellular Immunity (CTL)

| Term | Definition |
|:-----|:-----------|
| **CTL** | Cytotoxic T lymphocyte (CD8+ T cell). Kills virus-infected cells by recognizing viral peptides on MHC-I. |
| **Epitope** | Short peptide (8-11 amino acids) presented by MHC molecules and recognized by T cells. |
| **HLA** | Human leukocyte antigen. Human MHC molecules that present peptides to T cells. |
| **HLA Restriction** | Each T cell receptor recognizes a specific epitope only when presented by a specific HLA allele. |
| **Escape Mutation** | Viral mutation that reduces recognition by CTL, allowing immune evasion. |
| **Elite Controller** | HIV-infected individual who maintains undetectable viral load without ART. Often carry B*57 or B*27. |

### Humoral Immunity (Antibodies)

| Term | Definition |
|:-----|:-----------|
| **bnAb** | Broadly neutralizing antibody. Neutralizes diverse HIV strains by targeting conserved epitopes. |
| **IC50** | Inhibitory concentration 50%. Antibody concentration that neutralizes 50% of virus. Lower = more potent. |
| **Breadth** | Percentage of viral strains neutralized by an antibody. Higher = more broadly active. |
| **Neutralization** | Antibody-mediated blocking of viral entry into cells. |
| **Epitope Class** | Category of antibody binding site: CD4bs, V2-apex, V3-glycan, MPER, or interface. |

---

## Tropism

| Term | Definition |
|:-----|:-----------|
| **Coreceptor** | Secondary receptor required for HIV entry (CCR5 or CXCR4). |
| **CCR5 (R5)** | Chemokine receptor used by most transmitted HIV strains. Target of maraviroc. |
| **CXCR4 (X4)** | Chemokine receptor. X4-tropic virus associated with faster disease progression. |
| **Dual-Tropic (R5X4)** | Virus capable of using either coreceptor. |
| **Tropism Testing** | Laboratory test to determine whether a patient's virus uses CCR5, CXCR4, or both. |
| **11/25 Rule** | Classic prediction that basic amino acids at V3 positions 11 or 25 predict X4 tropism. |

---

## Computational Terms

| Term | Definition |
|:-----|:-----------|
| **Genetic Distance** | Quantitative measure of how different two sequences are at the codon level. In our framework, this correlates with evolutionary cost. |
| **Escape Velocity** | Rate at which escape mutations emerge for a given epitope. Lower = more durable immunity. |
| **Escape Barrier** | Magnitude of genetic change required for successful immune escape. Higher = more difficult. |
| **Conservation** | Degree to which a sequence position remains unchanged across viral populations. High conservation indicates functional constraint. |
| **Centrality** | Position within the evolutionary landscape. Central positions are more conserved; peripheral positions are more variable. |
| **Trade-off Score** | Quantifies conflict between drug resistance and immune escape at overlapping positions. |

---

## Clinical Terms

| Term | Definition |
|:-----|:-----------|
| **ART** | Antiretroviral therapy. Combination drug treatment for HIV. |
| **Viral Load** | Amount of HIV RNA in blood, typically copies/mL. |
| **CD4 Count** | Number of CD4+ T cells per microliter of blood. Marker of immune function. |
| **Virologic Failure** | Inability to achieve or maintain viral suppression on ART. |
| **Drug-Drug Interaction** | When one drug affects the metabolism or efficacy of another. |
| **Therapeutic Drug Monitoring** | Measuring drug levels in blood to optimize dosing. |

---

## Database Abbreviations

| Abbreviation | Full Name | Content |
|:-------------|:----------|:--------|
| **LANL** | Los Alamos National Laboratory | HIV sequence database, CTL epitope database |
| **CATNAP** | Compile, Analyze, and Tally NAb Panels | Neutralization data for bnAbs |
| **Stanford HIVDB** | Stanford HIV Drug Resistance Database | Resistance mutations and interpretations |
| **IEDB** | Immune Epitope Database | Epitope data across pathogens |

---

## HLA Nomenclature

### Format

HLA alleles are named: **Locus*XX:YY**

- **Locus**: A, B, or C for class I
- **XX**: Allele group (historically serotype)
- **YY**: Specific protein sequence

### Examples

| Allele | Interpretation |
|:-------|:---------------|
| B*57:01 | HLA-B locus, allele group 57, protein variant 01 |
| A*02:01 | HLA-A locus, allele group 02, protein variant 01 |
| B*27:05 | HLA-B locus, allele group 27, protein variant 05 |

### Clinical Significance

| Allele | Clinical Association |
|:-------|:--------------------|
| B*57:01 | Elite controller, abacavir hypersensitivity |
| B*27:05 | Elite controller, ankylosing spondylitis |
| A*02:01 | Common worldwide, moderate HIV control |
| B*35:01 | Faster HIV progression |

---

*This glossary covers terminology used in the public medical paper. For additional technical terms, contact the research team.*
