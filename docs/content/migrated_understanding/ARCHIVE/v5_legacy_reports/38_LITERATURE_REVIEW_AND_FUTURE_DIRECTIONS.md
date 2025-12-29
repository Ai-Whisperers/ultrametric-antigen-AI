# Literature Review: State-of-the-Art Research and Future Directions

**Date**: December 28, 2024
**Scope**: HIV Drug Resistance, VAEs for Proteins, Ranking Losses, Graph Neural Networks, and Related Topics

---

## Executive Summary

This document synthesizes research findings from recent literature (2024-2025) to identify:
1. State-of-the-art methods we should validate against
2. New approaches we could integrate into our framework
3. Research gaps our p-adic VAE approach could address

---

## 1. HIV Drug Resistance Prediction

### Current State-of-the-Art

| Method | Architecture | Performance | Reference |
|--------|-------------|-------------|-----------|
| GDL (Geometric DL) | Message Passing Neural Networks | 93.3% accuracy | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169743922001873) |
| Logistic Regression | Feature-based (binding site, structure) | Good RAM detection | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.25.650610v1.full) |
| CNN/RNN/MLP | Standard DL architectures | Baseline approaches | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/) |

### Key Insights from Literature

1. **Graph representations outperform sequence-only methods** - Converting drug SMILES to molecular graphs enables better drug-virus interaction modeling

2. **Biological features matter** - Recent work shows adding binding site info, secondary structure, and solvent accessibility improves predictions

3. **Limited training data is the main bottleneck** - Phenotypic assays are expensive, so most datasets are small

4. **Our Approach Fills a Gap**: Our p-adic VAE leverages mathematical structure (hierarchical embeddings) rather than requiring expensive biological annotations

### Validation Tasks for Our Model

- [ ] Compare against GDL baseline (93.3% accuracy)
- [ ] Compare against logistic regression with biological features
- [ ] Evaluate on held-out drug classes (leave-one-drug-out)
- [ ] Test on non-subtype-B sequences

---

## 2. Protein Language Models (PLMs)

### State-of-the-Art Models (2024-2025)

| Model | Parameters | Best Use Case | Reference |
|-------|------------|---------------|-----------|
| ESM2-15B | 15B | Large proteins | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.25.650688v1.full) |
| ESM2-3B | 3B | General purpose | Meta AI |
| ProtT5-XL | 3B | Mutation effects | [ProtTrans](https://github.com/agemagician/ProtTrans) |
| ESM-C (Cambrian) | Various | Efficient inference | [Cell iScience](https://www.cell.com/iscience/fulltext/S2589-0042(25)01756-0) |

### Critical Finding: Size Isn't Everything

From [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.25.650688v1.full):

> "Performance of ESM2 peaks when the predicted perplexity for a given protein falls within the range of 3–6. Models that yield excessively high or low perplexity tend to predict uniformly near-zero or large negative log-likelihood ratios for all mutations."

**Implication for our work**: Medium-sized models (ESM2-650M) may be optimal for HIV proteins.

### Integration Opportunities

1. **ESM2 Embeddings as Features**
   - Replace OneHot with ESM2 embeddings
   - Use frozen embeddings to avoid overfitting
   - Expected benefit: Better mutation effect prediction

2. **ProtTrans for Mutation Scoring**
   - Pre-compute mutation effects with ProtT5
   - Use as auxiliary labels for multi-task learning

3. **Efficiency Techniques**
   - Flash Attention: 4-9x faster inference
   - 4-bit quantization: 2-3x lower memory
   - Sequence packing: Handle variable lengths

### Validation Tasks

- [ ] Compare OneHot vs ESM2-650M embeddings
- [ ] Test perplexity-guided model selection
- [ ] Benchmark inference speed with Flash Attention

---

## 3. Variational Autoencoders for Proteins

### State-of-the-Art Models (2024-2025)

| Model | Innovation | Performance | Reference |
|-------|-----------|-------------|-----------|
| ProT-VAE | Transformer + VAE | 2.5x catalytic activity | [PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2408737122) |
| TP-VWGAN | VAE + WGAN-GP | Realistic structure generation | [Nature 2025](https://www.nature.com/articles/s41598-025-94747-y) |
| T-VAE | Dilated causal conv | Better long sequences | [2023](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05415-9) |

### Key Advantages of VAEs

From [PLOS CompBio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008736):

> "Unlike the embedding of models such as LSTM, Transformer, and Resnet, the variational autoencoder (VAE) can clearly show phylogenetic separation in 2-dimensional latent space."

**This validates our approach**: VAEs are particularly suited for capturing evolutionary/phylogenetic structure.

### Integration Opportunities

1. **ProT-VAE Architecture**
   - Replace MLP encoder with Transformer
   - Pre-train on UniRef, fine-tune on HIV

2. **Hybrid VAE-GAN**
   - Add discriminator for more realistic reconstructions
   - May help with minority resistance phenotypes

3. **Dilated Causal Convolutions**
   - Better handling of long-range dependencies
   - Useful for full protein sequences

### Validation Tasks

- [ ] Compare MLP encoder vs Transformer encoder
- [ ] Test latent space phylogenetic separation
- [ ] Evaluate reconstruction quality with GAN discriminator

---

## 4. Learning to Rank

### Key Methods from Literature

| Method | Type | Advantages | Disadvantages |
|--------|------|------------|---------------|
| ListMLE | Listwise | Direct MLE optimization | Computationally expensive |
| RankNet | Pairwise | Simple BCE loss | O(n²) pairs |
| LambdaRank | Listwise | NDCG optimization | Complex implementation |
| ApproxNDCG | Listwise | Differentiable NDCG | Approximation errors |

### Our Experimental Findings Match Literature

Our experiments confirmed ListMLE as optimal, which aligns with [Wikipedia - Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank):

> "In practice, listwise approaches often outperform pairwise approaches and pointwise approaches."

### Advanced Ranking Methods to Test

1. **LambdaRank**
   - Directly optimizes NDCG
   - Available in TensorFlow Ranking

2. **Neural Feature Selection for LTR** (Apple Research)
   - Can reduce input size by 60% without affecting performance
   - Useful for identifying key resistance positions

3. **Attention-Based LTR**
   - Learn which positions matter for ranking
   - Provides interpretability

### Validation Tasks

- [ ] Implement LambdaRank and compare
- [ ] Test neural feature selection for position importance
- [ ] Add attention mechanism for interpretability

---

## 5. p-Adic Numbers and Topology

### Current Research (2024-2025)

| Application | Key Finding | Reference |
|-------------|-------------|-----------|
| Complex Networks | Natural representation of hierarchy | [Nature Scientific Reports](https://www.nature.com/articles/s41598-020-79507-4) |
| Geospatial | Topology for data modeling | [ISPRS 2024](https://isprs-annals.copernicus.org/articles/X-4-2024/51/2024/) |
| TDA | Applied algebraic topology | [GEOTOP-A](https://seminargeotop-a.com/merida24) |

### Unique Position of Our Work

> "p-adic mathematics and complex network modeling are an emerging frontier rather than an established research area."

**Our p-adic VAE is pioneering**: We're among the first to apply p-adic topology to drug resistance prediction.

### Validation Tasks

- [ ] Demonstrate p-adic structure improves over Euclidean
- [ ] Visualize hierarchical clustering in p-adic space
- [ ] Compare with standard VAE on phylogenetic tasks

---

## 6. Hyperbolic Embeddings

### State-of-the-Art (2024-2025)

| Method | Application | Key Result | Reference |
|--------|-------------|------------|-----------|
| Hyperbolic Genome Embeddings | ICLR 2025 | Better phylogeny representation | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b63ad8c24354b0e5bcb7aea16490beab-Paper-Conference.pdf) |
| Dodonaphy | Phylogenetics | Differentiable tree decoder | [Bioinformatics Advances 2024](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae082/7696335) |
| H-DEPP | Tree placement | Fewer parameters, better performance | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/) |

### Key Insight

From [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b63ad8c24354b0e5bcb7aea16490beab-Paper-Conference.pdf):

> "Hyperbolic methods have correctly modeled established phylogenies, showcasing their supremacy in representing tree-structured data."

**Relevance**: Our p-adic framework is related to hyperbolic geometry - both capture hierarchical structure.

### Integration Opportunities

1. **Poincaré Embeddings**
   - Project latent space onto Poincaré ball
   - Better distance preservation for hierarchies

2. **Hyperbolic MLR**
   - Replace Euclidean classification with hyperbolic
   - Better separation of resistance levels

3. **GE-PHI Approach**
   - Combine knowledge graphs with hyperbolic embeddings

### Validation Tasks

- [ ] Compare p-adic vs Poincaré embeddings
- [ ] Test hyperbolic MLR for resistance classification
- [ ] Evaluate phylogenetic tree recovery

---

## 7. Graph Neural Networks for Drug Resistance

### State-of-the-Art (2024-2025)

| Method | Application | Performance | Reference |
|--------|-------------|-------------|-----------|
| K-mer GNN | MIC prediction | High precision | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40039779/) |
| STM-GNN | Multi-drug resistance | Dynamic patient networks | [2025] |
| GCGACNN | Microbe-drug assoc. | GCN + GAT + CNN hybrid | [Frontiers 2024](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1370013/full) |

### Integration Opportunities

1. **Mutation Co-occurrence Graph**
   - Build graph of mutation relationships
   - GNN learns epistatic interactions

2. **Drug Similarity Graph**
   - Connect drugs by structural similarity
   - Transfer learning across similar drugs

3. **Patient Network Graph**
   - Model treatment history as graph
   - Predict resistance emergence

### Validation Tasks

- [ ] Build mutation co-occurrence graph for each drug class
- [ ] Test GNN vs MLP on graph-structured data
- [ ] Evaluate epistasis detection

---

## 8. Multi-Task Learning

### State-of-the-Art (2024-2025)

| Method | Application | Key Feature | Reference |
|--------|-------------|-------------|-----------|
| BAITSAO | Drug synergy | LLM embeddings | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12081637/) |
| MTMol-GPT | Multi-target generation | Transformer + GAN | [2024] |
| GradNorm | Task weighting | Adaptive weights | [Standard] |

### Our Current Status

We implemented GradNorm but encountered bugs (size mismatch). From literature:

> "Effective drug combination can reduce the drug resistance of monotherapy with relatively lower doses of individual drugs."

### Integration Opportunities

1. **LLM Embeddings for Drugs**
   - Use BAITSAO-style embeddings
   - Context-enriched representations

2. **Progressive Training**
   - Start with easy drugs, add hard ones
   - Curriculum learning

3. **Task Grouping**
   - Group similar drugs (same class)
   - Different heads for different groups

### Validation Tasks

- [ ] Fix GradNorm implementation
- [ ] Test progressive multi-task training
- [ ] Compare task grouping strategies

---

## 9. Uncertainty Quantification

### State-of-the-Art Methods (2025)

| Method | Computational Cost | Calibration | Reference |
|--------|-------------------|-------------|-----------|
| Deep Ensembles | High (5x) | Good | Standard |
| MC Dropout | Low | Moderate | Standard |
| Bayes by Backprop | Moderate | Best | [Nature 2025](https://www.nature.com/articles/s41598-025-27167-7) |
| Evidential DL | Low | Good | [Nature Comms 2025](https://www.nature.com/articles/s41467-025-62235-6) |
| HBLL | Low | Good | [PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881400/) |

### Key Finding

From [Nature 2025](https://www.nature.com/articles/s41598-025-27167-7):

> "Bayes by Backprop was applied for the first time in this study area, offering a novel and promising approach to uncertainty quantification."

### Integration Opportunities

1. **Evidential Deep Learning**
   - Direct uncertainty estimation
   - No ensemble needed

2. **HBLL (Bayesian Last Layer)**
   - Only Bayesian on last layer
   - Computationally efficient

3. **Conformal Prediction**
   - Distribution-free UQ
   - Guaranteed coverage

### Validation Tasks

- [ ] Implement MC Dropout for UQ
- [ ] Compare Evidential DL vs Deep Ensembles
- [ ] Add confidence intervals to predictions

---

## 10. Summary: Priority Research Directions

### Tier 1: High Impact, Already Explored in Literature

| Direction | Expected Benefit | Effort | Priority |
|-----------|-----------------|--------|----------|
| ESM2 embeddings | +5-10% improvement | Medium | **HIGH** |
| ListMLE (already done) | +1% improvement | Done | Done |
| Uncertainty quantification | Clinical reliability | Medium | **HIGH** |
| Hyperbolic embeddings | Better phylogeny | Medium | **HIGH** |

### Tier 2: High Impact, Novel for HIV Resistance

| Direction | Expected Benefit | Effort | Priority |
|-----------|-----------------|--------|----------|
| GNN for mutation graphs | Epistasis detection | High | **MEDIUM** |
| ProT-VAE architecture | Better representations | High | **MEDIUM** |
| LambdaRank optimization | Better ranking | Low | **MEDIUM** |

### Tier 3: Novel Research Directions (Our Unique Contribution)

| Direction | Expected Benefit | Effort | Priority |
|-----------|-----------------|--------|----------|
| p-Adic hierarchical modeling | Novel framework | Done | Done |
| p-Adic + hyperbolic hybrid | Best of both | Medium | **EXPLORE** |
| p-Adic feature selection | Interpretability | Medium | **EXPLORE** |

---

## 11. Specific Experiments to Validate

### Phase 1: Quick Wins (1-2 weeks)
1. **ESM2-650M embeddings** - Replace OneHot
2. **LambdaRank loss** - Compare with ListMLE
3. **MC Dropout** - Add uncertainty estimates

### Phase 2: Architecture Improvements (2-4 weeks)
4. **Transformer encoder** - ProT-VAE style
5. **Hyperbolic latent space** - Poincaré ball
6. **GNN layer** - For mutation interactions

### Phase 3: Novel Contributions (4-8 weeks)
7. **p-Adic + hyperbolic hybrid**
8. **Evidential deep learning**
9. **Clinical validation** - Treatment outcome prediction

---

## 12. Conclusion

Our p-adic VAE framework is well-positioned in the current research landscape:

1. **VAEs are validated** for protein sequence modeling
2. **Ranking losses** (ListMLE) are confirmed as optimal
3. **Hierarchical embeddings** (hyperbolic/p-adic) are cutting-edge
4. **Uncertainty quantification** is a critical gap we can address

The main opportunities for improvement are:
- **Protein language model embeddings** (ESM2)
- **Graph neural networks** for epistasis
- **Uncertainty quantification** for clinical use

---

## Sources

### HIV Drug Resistance
- [Deep Learning for HIV-1 Drug Resistance](https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/)
- [ML Prediction for INSTI Resistance](https://www.biorxiv.org/content/10.1101/2025.04.25.650610v1.full)
- [GDL for Drug Resistance](https://www.sciencedirect.com/science/article/abs/pii/S0169743922001873)

### Protein Language Models
- [ESM2 Scaling Study](https://www.biorxiv.org/content/10.1101/2025.04.25.650688v1.full)
- [Efficient PLM Inference](https://www.cell.com/iscience/fulltext/S2589-0042(25)01756-0)
- [ProT-VAE](https://www.pnas.org/doi/10.1073/pnas.2408737122)

### VAEs for Proteins
- [Protein Ensemble VAE](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01057)
- [Functional Protein VAEs](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008736)
- [TP-VWGAN](https://www.nature.com/articles/s41598-025-94747-y)

### Hyperbolic Embeddings
- [ICLR 2025 Hyperbolic Genomes](https://proceedings.iclr.cc/paper_files/paper/2025/file/b63ad8c24354b0e5bcb7aea16490beab-Paper-Conference.pdf)
- [Dodonaphy](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae082/7696335)
- [H-DEPP](https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/)

### Graph Neural Networks
- [K-mer GNN for MIC](https://pubmed.ncbi.nlm.nih.gov/40039779/)
- [GNNs in Drug Discovery](https://pubs.acs.org/doi/10.1021/acs.chemrev.5c00461)
- [OGNNMDA](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1370013/full)

### Uncertainty Quantification
- [UQ for Protein-Ligand Binding](https://www.nature.com/articles/s41598-025-27167-7)
- [EviDTI](https://www.nature.com/articles/s41467-025-62235-6)
- [UQ with GNNs](https://www.nature.com/articles/s41467-025-58503-0)

### Learning to Rank
- [Learning to Rank Wikipedia](https://en.wikipedia.org/wiki/Learning_to_rank)
- [TensorFlow Ranking](https://www.tensorflow.org/ranking)
- [Neural Feature Selection (Apple)](https://machinelearning.apple.com/research/neural-feature-selection)
