# Glossary of Terms

## Technical Terminology Reference

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## A

**Accessory Mutation**
A drug resistance mutation that does not directly confer resistance but compensates for fitness costs of primary mutations. Typically shows lower fold-change values and appears in combination with primary mutations.

**AUC-ROC (Area Under Receiver Operating Characteristic Curve)**
A measure of classifier performance ranging from 0.5 (random) to 1.0 (perfect). Represents the probability that the classifier ranks a randomly chosen positive instance higher than a randomly chosen negative one.

---

## B

**bnAb (Broadly Neutralizing Antibody)**
An antibody capable of neutralizing a diverse range of HIV-1 strains, typically defined as neutralizing >50% of tested viruses at IC50 < 50 μg/mL.

**Bonferroni Correction**
A multiple testing correction that divides the significance threshold by the number of tests performed. Very conservative; controls family-wise error rate.

**Boundary Crossing**
In our geometric framework, the number of codon cluster boundaries a mutation traverses. Higher crossing indicates larger evolutionary "jump."

**Breadth**
The percentage of HIV strains neutralized by an antibody at a given IC50 threshold (typically <50 μg/mL).

---

## C

**CATNAP (Compile, Analyze and Tally NAb Panels)**
A Los Alamos database containing neutralization data for HIV-1 viruses tested against various antibodies.

**CD4bs (CD4 Binding Site)**
The region on gp120 that binds the host CD4 receptor. A major target for broadly neutralizing antibodies.

**Centrality**
In our geometric framework, the inverse of radial position. Central positions are closer to the origin and typically more functionally constrained.

**Cohen's d**
An effect size measure for comparing two groups, calculated as the difference in means divided by pooled standard deviation.

**Constraint Score**
A composite metric reflecting the evolutionary pressure at a position, incorporating drug resistance, immune recognition, and functional importance.

**Coreceptor**
A secondary receptor (CCR5 or CXCR4) required for HIV entry after CD4 binding.

**Cross-Resistance**
When a mutation conferring resistance to one drug also reduces susceptibility to other drugs.

**CTL (Cytotoxic T Lymphocyte)**
A type of T cell that kills virus-infected cells. Recognizes viral peptides presented by HLA molecules.

**CV (Cross-Validation)**
A resampling method to assess model performance by training on subsets of data and testing on held-out portions.

---

## D

**Dual-Tropic (R5X4)**
HIV strains capable of using both CCR5 and CXCR4 coreceptors for cell entry.

---

## E

**Effect Size**
A quantitative measure of the magnitude of a phenomenon, independent of sample size.

**Embedding**
A mapping of discrete objects (like codons) into a continuous vector space while preserving relevant properties.

**Epitope**
The specific portion of an antigen recognized by the immune system, typically 8-11 amino acids for CTL epitopes.

**Escape Mutation**
A viral mutation that reduces recognition by the immune system while maintaining viral fitness.

**Escape Velocity**
In our framework, a measure of how easily a virus can evolve to escape immune recognition, calculated from the geometric spread of possible escape mutations.

**Eta-squared (η²)**
An effect size for ANOVA, representing the proportion of variance explained by group membership.

---

## F

**FDR (False Discovery Rate)**
The expected proportion of false positives among all significant results. Controlled by methods like Benjamini-Hochberg.

**Fold-Change**
The ratio of IC50 (or other measure) for a mutant versus wild-type virus. A fold-change of 10 means 10× reduced susceptibility.

---

## G

**gp120**
The HIV envelope glycoprotein that binds CD4. Contains the V3 loop and is the primary target for neutralizing antibodies.

**gp41**
The transmembrane HIV envelope glycoprotein. Contains the MPER region targeted by some bnAbs.

---

## H

**HLA (Human Leukocyte Antigen)**
The human version of MHC molecules. Present viral peptides to T cells. Highly polymorphic across individuals.

**HLA Supertype**
A grouping of HLA alleles with similar peptide binding properties. Simplifies analysis across the diverse HLA repertoire.

**HXB2**
The HIV-1 reference genome (GenBank K03455). Used as the standard coordinate system for position mapping.

**Hyperbolic Distance**
Distance measured using the Poincaré ball metric, which gives exponentially more "space" toward the boundary.

**Hyperbolic Space**
A non-Euclidean geometry with constant negative curvature. Naturally accommodates tree-like and hierarchical structures.

---

## I

**IC50**
The antibody concentration required to neutralize 50% of virus. Lower values indicate higher potency.

**IC80**
The antibody concentration required to neutralize 80% of virus.

**INSTI (Integrase Strand Transfer Inhibitor)**
A class of antiretroviral drugs that block HIV integrase (e.g., raltegravir, dolutegravir).

---

## J

**Jaccard Similarity**
A measure of similarity between two sets, calculated as the size of their intersection divided by the size of their union.

---

## L

**LANL (Los Alamos National Laboratory)**
Maintains the HIV Sequence Database and Immunology Database, major resources for HIV research.

---

## M

**Mann-Whitney U Test**
A non-parametric test comparing two groups. Tests whether one group tends to have larger values than the other.

**MPER (Membrane Proximal External Region)**
A conserved region of gp41 near the viral membrane. Target of bnAbs like 10E8 and 4E10.

**Multiple Testing Correction**
Statistical procedures that adjust p-values or significance thresholds to account for performing many tests.

---

## N

**NNRTI (Non-Nucleoside Reverse Transcriptase Inhibitor)**
A class of antiretroviral drugs that bind to an allosteric site on RT (e.g., efavirenz, nevirapine).

**NRTI (Nucleoside/Nucleotide Reverse Transcriptase Inhibitor)**
A class of antiretroviral drugs that mimic nucleosides and terminate DNA synthesis (e.g., AZT, tenofovir).

---

## P

**P-adic Valuation**
For a prime p, the largest power of p that divides an integer. Forms the basis of p-adic number theory.

**Pearson Correlation (r)**
A measure of linear correlation between two variables, ranging from -1 to +1.

**PI (Protease Inhibitor)**
A class of antiretroviral drugs that block HIV protease (e.g., darunavir, atazanavir).

**Poincaré Ball**
A model of hyperbolic space where the space is represented as the interior of a unit ball. Distance increases exponentially toward the boundary.

**Potency**
The effectiveness of an antibody at neutralizing virus, typically measured as geometric mean IC50.

**Primary Mutation**
A drug resistance mutation that directly affects drug binding and confers measurable resistance. Shows high fold-change values.

---

## R

**R5 (CCR5-tropic)**
HIV strains that use the CCR5 coreceptor for cell entry. Predominant during most of infection.

**Radial Position**
The distance from the origin in the Poincaré ball embedding. Central = constrained; peripheral = variable.

**ROC (Receiver Operating Characteristic)**
A curve plotting true positive rate vs. false positive rate at various classification thresholds.

---

## S

**Sensitivity (Recall)**
The proportion of actual positives correctly identified by a classifier. TP/(TP+FN).

**Separation Score**
In tropism analysis, the hyperbolic distance between R5 and X4 centroids at a specific position.

**Specificity**
The proportion of actual negatives correctly identified. TN/(TN+FP).

**Stanford HIVDB**
Stanford HIV Drug Resistance Database. Contains genotype-phenotype data for drug resistance.

---

## T

**3-adic**
Relating to the prime p=3 in p-adic number theory. Our codon encoding uses 3-adic valuations due to the triplet nature of codons.

**Tier (Neutralization)**
Classification of HIV strains by neutralization sensitivity:
- Tier 1A: Highly sensitive
- Tier 1B: Sensitive
- Tier 2: Moderately resistant
- Tier 3: Highly resistant

**Trade-off Score**
A measure of positions under dual selective pressure from drugs and immune system.

**Tropism**
The preference of HIV for using CCR5 vs. CXCR4 as coreceptor.

---

## U

**UMAP (Uniform Manifold Approximation and Projection)**
A dimensionality reduction technique for visualizing high-dimensional data.

---

## V

**V3 Loop**
A 35-amino acid region of gp120 that determines coreceptor tropism and is a major target for antibodies.

**V2-glycan**
Epitope class involving glycans at positions 156/160 in the V1/V2 region. Target of PG9/PG16.

**V3-glycan**
Epitope class involving the glycan at position 332 near the V3 loop. Target of PGT121/10-1074.

**Vaccine Score**
A composite metric for prioritizing CTL epitopes as vaccine targets, incorporating HLA breadth, constraint, and safety.

---

## W

**Wild-Type**
The reference or consensus sequence, typically representing the most common variant in the population.

---

## X

**X4 (CXCR4-tropic)**
HIV strains that use the CXCR4 coreceptor for cell entry. Associated with disease progression.

---

## Symbols and Abbreviations

| Symbol | Meaning |
|--------|---------|
| α | Significance level (typically 0.05) |
| μ | Population mean |
| σ | Standard deviation |
| r | Pearson correlation coefficient |
| ρ | Spearman rank correlation |
| χ² | Chi-squared statistic |
| η² | Eta-squared effect size |
| CI | Confidence interval |
| n | Sample size |
| p | P-value |
| H₀ | Null hypothesis |
| H₁ | Alternative hypothesis |
| FC | Fold-change |
| Geo | Geometric (as in geometric mean) |

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
