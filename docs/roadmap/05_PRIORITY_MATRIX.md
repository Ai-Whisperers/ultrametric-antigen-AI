# Priority Matrix: Impact vs Effort Analysis

**Version:** 1.0.0
**Created:** December 29, 2025
**Status:** Planning Phase

---

## Overview

This document provides an impact vs effort analysis for all 100 features, enabling data-driven prioritization decisions. Features are scored on multiple dimensions and placed in priority quadrants.

---

## Scoring Methodology

### Impact Score (1-10)
| Score | Criteria |
|-------|----------|
| 10 | Critical for clinical deployment |
| 9 | Major performance improvement (>20%) |
| 8 | Significant new capability |
| 7 | Important enhancement |
| 6 | Notable improvement |
| 5 | Moderate improvement |
| 4 | Minor enhancement |
| 3 | Nice to have |
| 2 | Edge case improvement |
| 1 | Minimal impact |

### Effort Score (1-10)
| Score | Criteria |
|-------|----------|
| 10 | 3+ months, multiple developers, high risk |
| 9 | 2-3 months, significant complexity |
| 8 | 1-2 months, high complexity |
| 7 | 1 month, moderate-high complexity |
| 6 | 2-3 weeks, moderate complexity |
| 5 | 1-2 weeks, straightforward |
| 4 | 1 week, low complexity |
| 3 | 2-3 days |
| 2 | 1 day |
| 1 | Hours |

### Priority Score Formula
```
Priority = (Impact × Impact_Weight) / (Effort × Effort_Weight) × Dependency_Factor × Risk_Factor

Where:
- Impact_Weight = 1.0 (default)
- Effort_Weight = 1.0 (default)
- Dependency_Factor = 1.2 if enabler, 0.8 if heavily dependent
- Risk_Factor = 0.9 for high risk, 1.0 for low risk
```

---

## Priority Quadrants

```
                        HIGH IMPACT
                             │
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    │   QUICK WINS           │   STRATEGIC PROJECTS   │
    │   (Do First)           │   (Plan Carefully)     │
    │                        │                        │
    │   High Impact          │   High Impact          │
    │   Low Effort           │   High Effort          │
    │                        │                        │
LOW ├────────────────────────┼────────────────────────┤ HIGH
EFF │                        │                        │ EFF
ORT │   FILL-INS             │   AVOID/DEFER          │ ORT
    │   (When Time Allows)   │   (Re-evaluate)        │
    │                        │                        │
    │   Low Impact           │   Low Impact           │
    │   Low Effort           │   High Effort          │
    │                        │                        │
    └────────────────────────┼────────────────────────┘
                             │
                        LOW IMPACT
```

---

## Complete Feature Scoring

### Category 1: Disease Analyzers (Features 1-15)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 1 | Dengue Analyzer | 9 | 7 | 1.54 | Strategic |
| 2 | Zika Analyzer | 8 | 6 | 1.60 | Strategic |
| 3 | Norovirus Analyzer | 7 | 5 | 1.68 | Quick Win |
| 4 | Enterovirus Analyzer | 6 | 5 | 1.44 | Quick Win |
| 5 | West Nile Analyzer | 6 | 6 | 1.20 | Fill-in |
| 6 | Chikungunya Analyzer | 6 | 6 | 1.20 | Fill-in |
| 7 | HPV Analyzer | 8 | 7 | 1.37 | Strategic |
| 8 | EBV Analyzer | 5 | 6 | 1.00 | Fill-in |
| 9 | CMV Analyzer | 6 | 6 | 1.20 | Fill-in |
| 10 | Adenovirus Analyzer | 5 | 5 | 1.20 | Fill-in |
| 11 | C. difficile Analyzer | 8 | 6 | 1.60 | Quick Win |
| 12 | E. coli Analyzer | 9 | 6 | 1.80 | Quick Win |
| 13 | Klebsiella Analyzer | 8 | 6 | 1.60 | Quick Win |
| 14 | Acinetobacter Analyzer | 6 | 6 | 1.20 | Fill-in |
| 15 | Pseudomonas Analyzer | 7 | 6 | 1.40 | Quick Win |

**Top 3 Disease Analyzers:**
1. #12 E. coli (Priority 1.80) - High clinical need, extensive AMR data
2. #3 Norovirus (Priority 1.68) - Outbreak relevance, good data availability
3. #2 Zika (Priority 1.60) - Public health importance, shares with Dengue

---

### Category 2: Encoding Methods (Features 16-25)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 16 | 5-adic Amino Acid | 8 | 5 | 1.92 | Quick Win |
| 17 | 7-adic Secondary Structure | 7 | 6 | 1.40 | Quick Win |
| 18 | Multi-Prime Fusion | 8 | 7 | 1.37 | Strategic |
| 19 | Adaptive Prime Selection | 7 | 8 | 1.05 | Strategic |
| 20 | P-adic Wavelet | 6 | 8 | 0.90 | Defer |
| 21 | Codon Context Window | 6 | 5 | 1.44 | Quick Win |
| 22 | Position-Aware P-adic | 6 | 5 | 1.44 | Quick Win |
| 23 | Mutation-Type Embedding | 5 | 3 | 2.00 | Quick Win |
| 24 | Reading Frame Encoding | 5 | 4 | 1.50 | Quick Win |
| 25 | Synonymous Codon | 5 | 5 | 1.20 | Fill-in |

**Top 3 Encoding Methods:**
1. #23 Mutation-Type Embedding (Priority 2.00) - Very low effort, clear value
2. #16 5-adic Amino Acid (Priority 1.92) - Foundation, enables many others
3. #24 Reading Frame Encoding (Priority 1.50) - Low effort, useful for frameshifts

---

### Category 3: Hyperbolic Geometry (Features 26-35)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 26 | Multi-Sheet Hyperbolic | 7 | 8 | 1.05 | Strategic |
| 27 | Hyperbolic Norm. Flows | 6 | 9 | 0.80 | Defer |
| 28 | Product Manifold | 6 | 8 | 0.90 | Defer |
| 29 | Adaptive Curvature | 7 | 7 | 1.20 | Strategic |
| 30 | Hyperbolic Attention | 7 | 7 | 1.20 | Strategic |
| 31 | Geodesic Interpolation | 6 | 5 | 1.44 | Quick Win |
| 32 | Hyperbolic Clustering | 6 | 5 | 1.44 | Quick Win |
| 33 | Busemann Functions | 5 | 7 | 0.86 | Defer |
| 34 | Hyperbolic Diffusion | 5 | 9 | 0.67 | Defer |
| 35 | Multi-Scale Hyperbolic | 6 | 7 | 1.03 | Fill-in |

**Top 3 Hyperbolic Features:**
1. #31 Geodesic Interpolation (Priority 1.44) - Useful for visualization
2. #32 Hyperbolic Clustering (Priority 1.44) - Direct clinical application
3. #29 Adaptive Curvature (Priority 1.20) - Research innovation

---

### Category 4: Uncertainty & Calibration (Features 36-45)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 36 | Conformal Prediction | 10 | 6 | 2.00 | Quick Win |
| 37 | Temperature Scaling | 9 | 3 | 3.60 | Quick Win |
| 38 | Bayesian Neural Nets | 7 | 8 | 1.05 | Strategic |
| 39 | Ensemble Uncertainty | 7 | 6 | 1.40 | Quick Win |
| 40 | Epistemic/Aleatoric | 7 | 5 | 1.68 | Quick Win |
| 41 | OOD Detection | 8 | 6 | 1.60 | Quick Win |
| 42 | Selective Prediction | 7 | 5 | 1.68 | Quick Win |
| 43 | Uncertainty Propagation | 6 | 6 | 1.20 | Fill-in |
| 44 | Calibration Under Shift | 7 | 7 | 1.20 | Strategic |
| 45 | Human-in-Loop | 6 | 7 | 1.03 | Fill-in |

**Top 3 Uncertainty Features:**
1. #37 Temperature Scaling (Priority 3.60) - Very low effort, high impact
2. #36 Conformal Prediction (Priority 2.00) - Critical for clinical
3. #40 Epistemic/Aleatoric (Priority 1.68) - Important for interpretation

---

### Category 5: Transfer Learning (Features 46-55)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 46 | Domain Adversarial | 8 | 7 | 1.37 | Strategic |
| 47 | Cross-Disease Pre-train | 8 | 6 | 1.60 | Quick Win |
| 48 | Meta-Learning | 7 | 8 | 1.05 | Strategic |
| 49 | Progressive Transfer | 6 | 5 | 1.44 | Quick Win |
| 50 | Contrastive Transfer | 6 | 6 | 1.20 | Fill-in |
| 51 | Self-Supervised Pre-train | 10 | 8 | 1.50 | Strategic |
| 52 | Task-Agnostic Rep | 7 | 6 | 1.40 | Quick Win |
| 53 | Adapter Modules | 7 | 5 | 1.68 | Quick Win |
| 54 | LoRA for VAE | 7 | 5 | 1.68 | Quick Win |
| 55 | Continual Learning | 6 | 7 | 1.03 | Fill-in |

**Top 3 Transfer Features:**
1. #53 Adapter Modules (Priority 1.68) - Low effort, parameter-efficient
2. #54 LoRA for VAE (Priority 1.68) - Low effort, proven technique
3. #47 Cross-Disease Pre-train (Priority 1.60) - Natural extension

---

### Category 6: Architecture (Features 56-70)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 56 | Hierarchical VAE | 8 | 8 | 1.20 | Strategic |
| 57 | VQ-VAE for Codons | 7 | 7 | 1.20 | Strategic |
| 58 | Attention-Based VAE | 7 | 6 | 1.40 | Quick Win |
| 59 | Flow-Based Decoder | 6 | 8 | 0.90 | Defer |
| 60 | Disentangled VAE | 7 | 6 | 1.40 | Quick Win |
| 61 | Conditional VAE | 7 | 5 | 1.68 | Quick Win |
| 62 | Semi-Supervised VAE | 7 | 6 | 1.40 | Quick Win |
| 63 | Info-VAE | 6 | 5 | 1.44 | Quick Win |
| 64 | Two-Stage VAE | 6 | 7 | 1.03 | Fill-in |
| 65 | Sparse VAE | 6 | 5 | 1.44 | Quick Win |
| 66 | Recurrent VAE | 6 | 6 | 1.20 | Fill-in |
| 67 | Wasserstein VAE | 6 | 6 | 1.20 | Fill-in |
| 68 | Multi-Modal VAE | 8 | 8 | 1.20 | Strategic |
| 69 | Memory-Augmented VAE | 6 | 7 | 1.03 | Fill-in |
| 70 | Neural ODE VAE | 5 | 8 | 0.75 | Defer |

**Top 3 Architecture Features:**
1. #61 Conditional VAE (Priority 1.68) - Direct clinical use
2. #63 Info-VAE (Priority 1.44) - Solves posterior collapse
3. #65 Sparse VAE (Priority 1.44) - Interpretability

---

### Category 7: Epistasis (Features 71-80)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 71 | Tensor Epistasis | 9 | 9 | 1.20 | Strategic |
| 72 | Sparse Epistasis | 7 | 6 | 1.40 | Quick Win |
| 73 | Temporal Epistasis | 7 | 7 | 1.20 | Strategic |
| 74 | Cross-Gene Epistasis | 7 | 8 | 1.05 | Strategic |
| 75 | Compensatory Mutations | 8 | 6 | 1.60 | Quick Win |
| 76 | Fitness Landscape | 7 | 7 | 1.20 | Strategic |
| 77 | Evolutionary Trajectory | 8 | 8 | 1.20 | Strategic |
| 78 | Mutation Order | 7 | 6 | 1.40 | Quick Win |
| 79 | Codon-Level Epistasis | 6 | 6 | 1.20 | Fill-in |
| 80 | Structural Epistasis | 7 | 8 | 1.05 | Strategic |

**Top 3 Epistasis Features:**
1. #75 Compensatory Mutations (Priority 1.60) - Clinical relevance
2. #72 Sparse Epistasis (Priority 1.40) - Efficiency improvement
3. #78 Mutation Order (Priority 1.40) - Treatment planning

---

### Category 8: Clinical Integration (Features 81-90)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 81 | FHIR R4 Integration | 10 | 8 | 1.50 | Strategic |
| 82 | Clinical Decision API | 10 | 7 | 1.71 | Strategic |
| 83 | Automated Reports | 8 | 5 | 1.92 | Quick Win |
| 84 | Clinical Alerts | 8 | 5 | 1.92 | Quick Win |
| 85 | Drug Interaction | 7 | 6 | 1.40 | Quick Win |
| 86 | Treatment Recomm. | 8 | 7 | 1.37 | Strategic |
| 87 | Resistance Timeline | 7 | 6 | 1.40 | Quick Win |
| 88 | Surveillance Dashboard | 9 | 8 | 1.35 | Strategic |
| 89 | Quality Control | 7 | 5 | 1.68 | Quick Win |
| 90 | Regulatory Compliance | 10 | 9 | 1.33 | Strategic |

**Top 3 Clinical Features:**
1. #83 Automated Reports (Priority 1.92) - Direct clinical value
2. #84 Clinical Alerts (Priority 1.92) - Direct clinical value
3. #82 Clinical Decision API (Priority 1.71) - Foundation for all clinical

---

### Category 9: Training & Optimization (Features 91-100)

| ID | Feature | Impact | Effort | Priority | Quadrant |
|----|---------|--------|--------|----------|----------|
| 91 | Mixed Precision | 7 | 3 | 2.80 | Quick Win |
| 92 | Gradient Checkpoint | 6 | 3 | 2.40 | Quick Win |
| 93 | Distributed Training | 7 | 7 | 1.20 | Strategic |
| 94 | Dynamic Batching | 6 | 4 | 1.80 | Quick Win |
| 95 | Curriculum Learning | 6 | 5 | 1.44 | Quick Win |
| 96 | Label Smoothing | 5 | 2 | 3.00 | Quick Win |
| 97 | SWA | 6 | 3 | 2.40 | Quick Win |
| 98 | Knowledge Distillation | 7 | 6 | 1.40 | Quick Win |
| 99 | Neural Arch Search | 7 | 9 | 0.93 | Defer |
| 100 | Federated Learning | 9 | 10 | 1.08 | Strategic |

**Top 3 Training Features:**
1. #96 Label Smoothing (Priority 3.00) - Trivial to implement
2. #91 Mixed Precision (Priority 2.80) - Big impact, low effort
3. #92 Gradient Checkpointing (Priority 2.40) - Enables larger models

---

## Top 20 Overall Priority Ranking

| Rank | ID | Feature | Category | Impact | Effort | Priority |
|------|-----|---------|----------|--------|--------|----------|
| 1 | 37 | Temperature Scaling | Uncertainty | 9 | 3 | 3.60 |
| 2 | 96 | Label Smoothing | Training | 5 | 2 | 3.00 |
| 3 | 91 | Mixed Precision | Training | 7 | 3 | 2.80 |
| 4 | 92 | Gradient Checkpointing | Training | 6 | 3 | 2.40 |
| 5 | 97 | SWA | Training | 6 | 3 | 2.40 |
| 6 | 23 | Mutation-Type Embedding | Encoding | 5 | 3 | 2.00 |
| 7 | 36 | Conformal Prediction | Uncertainty | 10 | 6 | 2.00 |
| 8 | 16 | 5-adic Amino Acid | Encoding | 8 | 5 | 1.92 |
| 9 | 83 | Automated Reports | Clinical | 8 | 5 | 1.92 |
| 10 | 84 | Clinical Alerts | Clinical | 8 | 5 | 1.92 |
| 11 | 12 | E. coli Analyzer | Disease | 9 | 6 | 1.80 |
| 12 | 94 | Dynamic Batching | Training | 6 | 4 | 1.80 |
| 13 | 82 | Clinical Decision API | Clinical | 10 | 7 | 1.71 |
| 14 | 3 | Norovirus Analyzer | Disease | 7 | 5 | 1.68 |
| 15 | 40 | Epistemic/Aleatoric | Uncertainty | 7 | 5 | 1.68 |
| 16 | 42 | Selective Prediction | Uncertainty | 7 | 5 | 1.68 |
| 17 | 53 | Adapter Modules | Transfer | 7 | 5 | 1.68 |
| 18 | 54 | LoRA for VAE | Transfer | 7 | 5 | 1.68 |
| 19 | 61 | Conditional VAE | Architecture | 7 | 5 | 1.68 |
| 20 | 89 | Quality Control | Clinical | 7 | 5 | 1.68 |

---

## Quick Wins (Do First)

Features with Impact ≥6 and Effort ≤5:

| ID | Feature | Impact | Effort | Category |
|----|---------|--------|--------|----------|
| 37 | Temperature Scaling | 9 | 3 | Uncertainty |
| 96 | Label Smoothing | 5 | 2 | Training |
| 91 | Mixed Precision | 7 | 3 | Training |
| 92 | Gradient Checkpointing | 6 | 3 | Training |
| 97 | SWA | 6 | 3 | Training |
| 23 | Mutation-Type Embedding | 5 | 3 | Encoding |
| 16 | 5-adic Amino Acid | 8 | 5 | Encoding |
| 83 | Automated Reports | 8 | 5 | Clinical |
| 84 | Clinical Alerts | 8 | 5 | Clinical |
| 40 | Epistemic/Aleatoric | 7 | 5 | Uncertainty |
| 42 | Selective Prediction | 7 | 5 | Uncertainty |
| 53 | Adapter Modules | 7 | 5 | Transfer |
| 54 | LoRA for VAE | 7 | 5 | Transfer |
| 61 | Conditional VAE | 7 | 5 | Architecture |
| 89 | Quality Control | 7 | 5 | Clinical |
| 3 | Norovirus Analyzer | 7 | 5 | Disease |
| 4 | Enterovirus Analyzer | 6 | 5 | Disease |
| 21 | Codon Context Window | 6 | 5 | Encoding |
| 22 | Position-Aware P-adic | 6 | 5 | Encoding |
| 49 | Progressive Transfer | 6 | 5 | Transfer |
| 63 | Info-VAE | 6 | 5 | Architecture |
| 65 | Sparse VAE | 6 | 5 | Architecture |
| 31 | Geodesic Interpolation | 6 | 5 | Hyperbolic |
| 32 | Hyperbolic Clustering | 6 | 5 | Hyperbolic |

---

## Strategic Projects (Plan Carefully)

Features with Impact ≥8 and Effort ≥7:

| ID | Feature | Impact | Effort | Category | Risk |
|----|---------|--------|--------|----------|------|
| 51 | Self-Supervised Pre-train | 10 | 8 | Transfer | Medium |
| 81 | FHIR R4 Integration | 10 | 8 | Clinical | Medium |
| 82 | Clinical Decision API | 10 | 7 | Clinical | Low |
| 90 | Regulatory Compliance | 10 | 9 | Clinical | High |
| 100 | Federated Learning | 9 | 10 | Training | High |
| 1 | Dengue Analyzer | 9 | 7 | Disease | Medium |
| 71 | Tensor Epistasis | 9 | 9 | Epistasis | High |
| 88 | Surveillance Dashboard | 9 | 8 | Clinical | Medium |
| 68 | Multi-Modal VAE | 8 | 8 | Architecture | Medium |
| 56 | Hierarchical VAE | 8 | 8 | Architecture | Medium |
| 46 | Domain Adversarial | 8 | 7 | Transfer | Medium |
| 77 | Evolutionary Trajectory | 8 | 8 | Epistasis | Medium |

---

## Defer/Re-evaluate

Features with low priority scores (<1.0):

| ID | Feature | Impact | Effort | Priority | Reason to Defer |
|----|---------|--------|--------|----------|-----------------|
| 27 | Hyperbolic Norm. Flows | 6 | 9 | 0.80 | High complexity, research-focused |
| 28 | Product Manifold | 6 | 8 | 0.90 | Theoretical, limited practical gain |
| 33 | Busemann Functions | 5 | 7 | 0.86 | Niche application |
| 34 | Hyperbolic Diffusion | 5 | 9 | 0.67 | Very high effort, research |
| 59 | Flow-Based Decoder | 6 | 8 | 0.90 | High complexity |
| 70 | Neural ODE VAE | 5 | 8 | 0.75 | Research-focused |
| 99 | Neural Arch Search | 7 | 9 | 0.93 | High compute cost |
| 20 | P-adic Wavelet | 6 | 8 | 0.90 | Complex math, unclear benefit |

---

## Implementation Recommendation

### Immediate (Next Sprint)
1. #37 Temperature Scaling - 2-3 days
2. #96 Label Smoothing - 1 day
3. #91 Mixed Precision - 2-3 days
4. #92 Gradient Checkpointing - 2-3 days

### Short-term (Next Month)
1. #36 Conformal Prediction - 2 weeks
2. #16 5-adic Amino Acid Encoder - 1 week
3. #83 Automated Reports - 1 week
4. #84 Clinical Alerts - 1 week
5. #53 Adapter Modules - 1 week

### Medium-term (Next Quarter)
1. #51 Self-Supervised Pre-training - 6 weeks
2. #82 Clinical Decision API - 4 weeks
3. #1 Dengue Analyzer - 4 weeks
4. #12 E. coli Analyzer - 3 weeks
5. #71 Tensor Epistasis Network - 6 weeks

### Long-term (Next 6 Months)
1. #81 FHIR R4 Integration - 6 weeks
2. #88 Surveillance Dashboard - 6 weeks
3. #100 Federated Learning - 8 weeks
4. #90 Regulatory Compliance - Ongoing

---

## Resource Allocation Recommendations

### By Priority Score
| Priority Range | Features | Recommended Allocation |
|----------------|----------|------------------------|
| >2.0 | 6 features | 30% of resources |
| 1.5-2.0 | 15 features | 35% of resources |
| 1.0-1.5 | 55 features | 30% of resources |
| <1.0 | 8 features | 5% of resources |

### By Category (Based on Aggregate Priority)
| Category | Avg Priority | Allocation |
|----------|--------------|------------|
| Training | 1.85 | 15% |
| Uncertainty | 1.65 | 15% |
| Clinical | 1.56 | 20% |
| Encoding | 1.44 | 10% |
| Disease | 1.44 | 20% |
| Transfer | 1.38 | 10% |
| Epistasis | 1.25 | 5% |
| Architecture | 1.21 | 3% |
| Hyperbolic | 1.08 | 2% |

---

## Risk-Adjusted Priority

For high-risk features, apply 0.9 multiplier:

| ID | Feature | Base Priority | Risk | Adj. Priority |
|----|---------|---------------|------|---------------|
| 100 | Federated Learning | 1.08 | High | 0.97 |
| 90 | Regulatory Compliance | 1.33 | High | 1.20 |
| 71 | Tensor Epistasis | 1.20 | High | 1.08 |
| 27 | Hyperbolic Flows | 0.80 | High | 0.72 |
| 34 | Hyperbolic Diffusion | 0.67 | High | 0.60 |

---

## Dependency-Adjusted Priority

For enabler features, apply 1.2 multiplier:

| ID | Feature | Base Priority | Enabler? | Adj. Priority |
|----|---------|---------------|----------|---------------|
| 51 | Self-Supervised | 1.50 | Yes (15+) | 1.80 |
| 36 | Conformal | 2.00 | Yes (10+) | 2.40 |
| 16 | 5-adic Encoder | 1.92 | Yes (9) | 2.30 |
| 56 | Hierarchical VAE | 1.20 | Yes (8+) | 1.44 |
| 71 | Tensor Epistasis | 1.20 | Yes (10) | 1.44 |
| 82 | Clinical API | 1.71 | Yes (6+) | 2.05 |

---

## Final Prioritized Backlog (Top 30)

| Rank | ID | Feature | Final Score | Phase |
|------|-----|---------|-------------|-------|
| 1 | 37 | Temperature Scaling | 3.60 | 1 |
| 2 | 96 | Label Smoothing | 3.00 | 1 |
| 3 | 91 | Mixed Precision | 2.80 | 1 |
| 4 | 92 | Gradient Checkpointing | 2.40 | 1 |
| 5 | 36 | Conformal Prediction | 2.40* | 1 |
| 6 | 97 | SWA | 2.40 | 1 |
| 7 | 16 | 5-adic Amino Acid | 2.30* | 1 |
| 8 | 82 | Clinical Decision API | 2.05* | 1 |
| 9 | 23 | Mutation-Type Embedding | 2.00 | 1 |
| 10 | 83 | Automated Reports | 1.92 | 1 |
| 11 | 84 | Clinical Alerts | 1.92 | 1 |
| 12 | 51 | Self-Supervised | 1.80* | 1 |
| 13 | 12 | E. coli Analyzer | 1.80 | 2 |
| 14 | 94 | Dynamic Batching | 1.80 | 1 |
| 15 | 3 | Norovirus Analyzer | 1.68 | 2 |
| 16 | 40 | Epistemic/Aleatoric | 1.68 | 1 |
| 17 | 42 | Selective Prediction | 1.68 | 1 |
| 18 | 53 | Adapter Modules | 1.68 | 2 |
| 19 | 54 | LoRA for VAE | 1.68 | 2 |
| 20 | 61 | Conditional VAE | 1.68 | 2 |
| 21 | 89 | Quality Control | 1.68 | 2 |
| 22 | 11 | C. difficile Analyzer | 1.60 | 2 |
| 23 | 13 | Klebsiella Analyzer | 1.60 | 2 |
| 24 | 2 | Zika Analyzer | 1.60 | 2 |
| 25 | 41 | OOD Detection | 1.60 | 2 |
| 26 | 47 | Cross-Disease Pre-train | 1.60 | 2 |
| 27 | 75 | Compensatory Mutations | 1.60 | 3 |
| 28 | 1 | Dengue Analyzer | 1.54 | 2 |
| 29 | 81 | FHIR R4 Integration | 1.50 | 4 |
| 30 | 24 | Reading Frame Encoding | 1.50 | 2 |

*Adjusted for enabler status

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-29 | 1.0.0 | Initial priority matrix |
