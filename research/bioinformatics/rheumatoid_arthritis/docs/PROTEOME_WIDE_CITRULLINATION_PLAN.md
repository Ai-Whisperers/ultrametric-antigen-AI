# Proteome-Wide Citrullination Analysis Plan

**Date**: 2024-12-17
**Objective**: Scan the entire human proteome to discover novel autoimmune-risk citrullination sites using hyperbolic geometric signatures

---

## Executive Summary

Apply our validated immunogenicity predictor (AUC=0.873) to all ~150,000 arginine positions in the human proteome to:
1. Rank every human arginine by "immunogenic potential"
2. Discover new autoimmune targets beyond known RA epitopes
3. Identify structural/functional patterns that predispose to autoimmunity
4. Generate testable hypotheses for experimental validation

---

## Phase 1: Data Acquisition

### 1.1 Download Human Proteome

**Source**: UniProt Reference Proteome (UP000005640)
- ~20,400 canonical proteins
- ~70,000 including isoforms
- FASTA format with metadata

**Download URL**:
```
https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(organism_id:9606)+AND+(reviewed:true)
```

**Alternative**: UniProt FTP
```
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/
```

### 1.2 Download Protein Annotations

**Gene Ontology annotations** for enrichment analysis:
- Cellular Component (CC): Where is the protein located?
- Biological Process (BP): What pathways is it involved in?
- Molecular Function (MF): What does it do?

**Source**: UniProt ID mapping + GO annotations
```
https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=(organism_id:9606)+AND+(reviewed:true)&fields=accession,gene_names,protein_name,go_c,go_f,go_p,cc_subcellular_location
```

### 1.3 Expected Data Volume

| Component | Count | Size |
|-----------|-------|------|
| Proteins (reviewed) | ~20,400 | ~50 MB |
| Total arginines | ~150,000 | - |
| Arginines in 9-mer windows | ~140,000 | - |
| Features per arginine | 12 | - |
| Total feature matrix | ~1.7M values | ~15 MB |

---

## Phase 2: Processing Pipeline

### 2.1 Script Architecture

```
research/bioinformatics/rheumatoid_arthritis/scripts/
├── 12_download_human_proteome.py      # Phase 1: Data acquisition
├── 13_extract_arginine_contexts.py    # Phase 2: Window extraction
├── 14_compute_geometric_features.py   # Phase 3: Feature computation
├── 15_predict_immunogenicity.py       # Phase 4: Apply predictor
├── 16_enrichment_analysis.py          # Phase 5: GO/pathway enrichment
└── 17_proteome_visualizations.py      # Phase 6: Results visualization
```

### 2.2 Arginine Context Extraction

For each protein:
1. Identify all arginine (R) positions
2. Extract 9-mer window centered on R (±4 residues)
3. Handle edge cases (N/C terminal R)
4. Record protein ID, position, flanking sequence

**Output format**:
```json
{
  "protein_id": "P04406",
  "gene_name": "GAPDH",
  "r_position": 234,
  "window_sequence": "VKVGVNGFGR",  // R at position 9
  "n_terminal_distance": 234,
  "c_terminal_distance": 101,
  "total_arginines_in_protein": 12
}
```

### 2.3 Batch Processing Strategy

Given ~150,000 arginines:
- Process in batches of 1,000
- Checkpoint every 10,000
- Estimated time: 20-30 minutes on CPU
- Memory: ~2 GB peak

---

## Phase 3: Feature Computation

### 3.1 Features Per Arginine Site

| Feature | Description | Source |
|---------|-------------|--------|
| `embedding_norm` | Mean Poincaré ball radius | Window embeddings |
| `embedding_norm_std` | Embedding variance | Window embeddings |
| `cluster_homogeneity` | Cluster consistency | Cluster assignments |
| `mean_neighbor_distance` | Geodesic smoothness | Pairwise distances |
| `boundary_potential` | Distance to cluster edge | Cluster geometry |
| `centroid_shift` | **Citrullination effect** | Before/after centroid |
| `js_divergence` | Distribution shift | Cluster probabilities |
| `entropy_change` | **Key predictor** | Entropy before/after |
| `sequence_length` | Window size | Always 9 |
| `n_arginines` | R count in window | Sequence |
| `r_density` | R/length ratio | Derived |
| `entropy_per_r` | Normalized entropy | Derived |

### 3.2 Citrullination Simulation

For each arginine:
1. Encode original 9-mer window
2. Remove R from embeddings (simulates citrullination)
3. Compute centroid shift, JS divergence, entropy change
4. This mimics R→Cit without needing actual citrulline codon

---

## Phase 4: Immunogenicity Prediction

### 4.1 Apply Trained Model

Use the validated predictor from Script 11:
- Model: Logistic Regression (AUC=0.873 on LOPO-CV)
- Features: 12-dimensional vector per arginine
- Output: P(immunodominant) ∈ [0, 1]

### 4.2 Risk Stratification

| Risk Category | Probability | Expected Count |
|---------------|-------------|----------------|
| Very High | >0.90 | ~1,500 |
| High | 0.75-0.90 | ~7,500 |
| Moderate | 0.50-0.75 | ~30,000 |
| Low | 0.25-0.50 | ~60,000 |
| Very Low | <0.25 | ~50,000 |

### 4.3 Output Format

```json
{
  "protein_id": "P04406",
  "gene_name": "GAPDH",
  "r_position": 234,
  "window_sequence": "VKVGVNGFGR",
  "immunogenic_probability": 0.847,
  "risk_category": "high",
  "centroid_shift": 0.312,
  "entropy_change": 0.089,
  "known_autoantigen": false
}
```

---

## Phase 5: Enrichment Analysis

### 5.1 Questions to Answer

1. **Cellular location**: Are certain compartments enriched for high-risk sites?
   - Hypothesis: Extracellular/membrane proteins higher risk (immune exposure)

2. **Biological processes**: Which pathways have clustered risk?
   - Hypothesis: Inflammatory pathways, apoptosis, NETosis

3. **Molecular function**: Do certain protein classes dominate?
   - Hypothesis: Structural proteins, enzymes with exposed R

4. **Disease associations**: Overlap with known autoimmune targets?
   - Cross-reference with DisGeNET, OMIM, literature

### 5.2 Statistical Tests

- Fisher's exact test for GO term enrichment
- Benjamini-Hochberg FDR correction
- Odds ratios with 95% CI
- Permutation tests for robustness

### 5.3 Validation Against Known Targets

Compare top predictions against:
- Known RA autoantigens (our training set)
- Lupus autoantigens (external validation)
- Sjögren's syndrome targets
- Systemic sclerosis targets
- Type 1 diabetes autoantigens

---

## Phase 6: Visualization

### 6.1 Proteome-Wide Views

1. **Chromosome ideogram**: High-risk sites by genomic location
2. **Protein length vs risk**: Do longer proteins have more risk sites?
3. **Risk distribution**: Histogram of P(immunodominant) across proteome

### 6.2 Top Candidates

1. **Top 100 novel targets**: Proteins with highest-risk R sites not in training data
2. **Protein risk profiles**: Per-protein heatmaps of R site risk
3. **Network view**: Protein-protein interactions among high-risk proteins

### 6.3 Enrichment Plots

1. **GO enrichment**: Dot plots for significant terms
2. **Pathway maps**: KEGG pathways colored by risk density
3. **Cellular compartment**: Subcellular localization pie chart

---

## Phase 7: Deliverables

### 7.1 Data Files

```
research/bioinformatics/rheumatoid_arthritis/results/proteome_wide/
├── human_proteome_arginine_features.parquet  # All features (~150K rows)
├── immunogenicity_predictions.parquet        # All predictions
├── high_risk_candidates.csv                  # Top 1000 novel targets
├── go_enrichment_results.json                # Enrichment statistics
└── validation_against_known.json             # Comparison with literature
```

### 7.2 Documentation

```
research/bioinformatics/rheumatoid_arthritis/docs/
├── PROTEOME_WIDE_ANALYSIS_REPORT.md          # Full analysis report
├── NOVEL_AUTOIMMUNE_CANDIDATES.md            # Top discoveries
└── METHODOLOGY_VALIDATION.md                  # Statistical validation
```

### 7.3 Visualizations

```
research/bioinformatics/rheumatoid_arthritis/results/proteome_wide/figures/
├── risk_distribution.png
├── go_enrichment_dotplot.png
├── top_candidates_heatmap.png
├── compartment_enrichment.png
├── validation_roc.png
└── protein_network.png
```

---

## Implementation Timeline

| Phase | Script | Description | Est. Time |
|-------|--------|-------------|-----------|
| 1 | 12 | Download proteome | 5 min |
| 2 | 13 | Extract R contexts | 2 min |
| 3 | 14 | Compute features | 20 min |
| 4 | 15 | Predict immunogenicity | 2 min |
| 5 | 16 | Enrichment analysis | 5 min |
| 6 | 17 | Visualizations | 3 min |
| **Total** | | | **~40 min** |

---

## Expected Discoveries

### 7.1 Novel Autoimmune Targets

Proteins with high-risk R sites not currently studied in autoimmunity:
- Metabolic enzymes
- Cytoskeletal proteins beyond vimentin
- Chaperones beyond BiP
- Nuclear proteins beyond histones

### 7.2 Structural Principles

Why some R sites are immunogenic:
- Solvent accessibility
- Proximity to other PTM sites
- Evolutionary conservation
- Structural flexibility

### 7.3 Disease-Specific Patterns

Different autoimmune diseases may target different geometric signatures:
- RA: Citrullination in structural proteins
- Lupus: Nuclear/chromatin proteins
- Sjögren's: Secretory proteins

---

## Technical Requirements

### Dependencies

```python
# Core
numpy, pandas, torch, scipy

# ML
scikit-learn

# Data
requests, biopython

# Visualization
matplotlib, seaborn

# Enrichment
goatools, gseapy (optional)
```

### Hardware

- CPU: Any modern processor
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 500 MB for data + results
- GPU: Not required (CPU inference sufficient)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| UniProt API rate limits | Use FTP download, cache locally |
| Memory overflow | Batch processing, chunked I/O |
| False positives | Multiple testing correction, validation set |
| Overfitting to RA | Test on other autoimmune diseases |

---

## Success Criteria

1. **Technical**: Process 100% of human proteome arginines
2. **Predictive**: >50% of known autoimmune R sites in top 10% of predictions
3. **Discovery**: Identify ≥10 novel high-confidence candidates
4. **Biological**: Significant GO enrichment (FDR < 0.05) in interpretable terms

---

## Next Steps After Analysis

1. **Literature validation**: Search PubMed for top candidates
2. **Experimental collaboration**: Partner with immunology lab for validation
3. **Expand to other PTMs**: Acetylation, methylation, phosphorylation
4. **Cross-species**: Compare human vs mouse to identify conserved risk sites
5. **Publication**: Write up methodology and discoveries

---

**Document version**: 1.0
**Status**: Ready for implementation
**Estimated implementation time**: 2-3 hours
