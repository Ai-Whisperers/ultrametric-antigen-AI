# TernaryVAE Checkpoint Index & Selection Guide

**Doc-Type:** Technical Index · Version 1.0 · Updated 2026-01-14 · AI Whisperers

Complete catalog of TernaryVAE checkpoints with performance analysis and application recommendations.

---

## Executive Summary

This index catalogs **80+ checkpoints** across 5 major architectural generations (v5.5→v5.12.4), identifying key breakthroughs, failure modes, and production-ready models for specific applications.

**Key Finding**: Different checkpoints optimize for different aspects of the **coverage-hierarchy-richness tradeoff**. No single "best" model exists - selection depends on downstream application.

---

## 🏆 Tier-1 Production Checkpoints

**Curated collection for production use:**

### v5_12_4_best_Q.pt - **CURRENT FLAGSHIP**
- **Status**: ✅ **PRODUCTION READY**
- **Metrics**: Coverage=100%, Hierarchy_B=-0.82, Q=1.96, Richness=0.003146
- **Architecture**: ImprovedEncoder/Decoder (SiLU + LayerNorm + Dropout + logvar clamping)
- **Innovations**: Modern architecture, v5.5 initialization, early convergence (16 epochs)
- **Use Cases**: New bioinformatics projects, general-purpose embeddings
- **Size**: 1.0MB (113,957 parameters)

### homeostatic_rich_best.pt - **THEORETICAL BREAKTHROUGH**
- **Status**: ✅ **RESEARCH GOLD STANDARD**
- **Metrics**: Coverage=100%, Hierarchy_B=-0.8321 (ceiling), Richness=0.00787 (5.8x higher)
- **Innovation**: **Revolutionary RichHierarchyLoss** - operates on level means, preserves variance
- **Breakthrough**: Proved hierarchy and richness are NOT mutually exclusive
- **Use Cases**: ΔΔG prediction (Spearman 0.61), force constant derivation, thermodynamic analysis
- **Size**: 421KB (compact, efficient)

### v5_11_structural_best.pt - **CONTACT PREDICTION SPECIALIST**
- **Status**: ✅ **CONTACT PREDICTION OPTIMIZED**
- **Metrics**: Coverage=100%, Hierarchy_B=-0.74, Low richness (~0.003)
- **Specialty**: Contact prediction AUC=0.67 (Cohen's d=-0.474)
- **Trade-off**: Collapsed richness enables consistent AA-level distances
- **Use Cases**: Protein structure prediction, contact map generation
- **Size**: 1.4MB (hidden_dim=128, dual projection layers)

### v5_11_homeostasis_best.pt - **BALANCED RESEARCH**
- **Status**: ✅ **RESEARCH READY**
- **Metrics**: Coverage=99.9%, Hierarchy_B=-0.83, Best radial separation
- **Innovation**: HomeostasisController with Q-gated dynamic freeze/unfreeze
- **Strength**: Best v0→v9 radial separation (0.932→0.013)
- **Use Cases**: Research experiments, adaptive training validation
- **Size**: 845KB (use_controller=True)

---

## 📊 Performance Matrix by Application

| Application | Primary | Secondary | Rationale |
|-------------|---------|-----------|-----------|
| **ΔΔG Prediction** | homeostatic_rich_best.pt | v5_12_4_best_Q.pt | High richness preserves thermodynamic signals |
| **Contact Prediction** | v5_11_structural_best.pt | - | Collapsed shells give consistent pairwise distances |
| **Force Constants** | homeostatic_rich_best.pt | - | Richness correlates with vibrational frequencies (r=0.86) |
| **New Projects** | v5_12_4_best_Q.pt | homeostatic_rich_best.pt | Latest architecture with proven foundations |
| **Research/Training** | v5_11_homeostasis_best.pt | v5_12_4_best_Q.pt | HomeostasisController for adaptive experiments |
| **Codon Analysis** | v5_11_structural_best.pt | homeostatic_rich_best.pt | Stable embeddings for sequence analysis |

---

## 🏗️ Architectural Evolution Timeline

### v5.5 Era: StateNet Foundation
**Key Innovation**: First adaptive control system

**v5_5/best.pt** - **HISTORICAL FOUNDATION**
- **Status**: 🔬 **HISTORICAL RESEARCH**
- **Metrics**: Coverage A=97.1%, Coverage B=89.3%, Gradient balance=1.15
- **Innovation**: StateNet with 101 dynamic corrections over 100 epochs
- **Significance**: Proved automated hyperparameter adaptation works
- **Training**: 101 epochs, phase progression (1→2), extensive logging
- **Legacy**: Foundation for all homeostatic control systems

### v5.6-v5.10 Era: Attention Experiments
**Key Innovation**: Hierarchical attention mechanisms

**purposeful_v5.6** - **DUAL ENCODER PROOF-OF-CONCEPT**
- **Status**: 🔬 **EXPERIMENTAL**
- **Metrics**: Coverage ~4.8-4.9%, High entropy (exploration mode)
- **Innovation**: Basic dual encoder with 10-parameter StateNet
- **Training**: 10 epochs, moderate coverage, entropy 2.32→1.94
- **Lesson**: Dual encoders require sophisticated control mechanisms

**purposeful_v5.10** - **ATTENTION BREAKTHROUGH**
- **Status**: 🔬 **EXPERIMENTAL SUCCESS**
- **Metrics**: Similar coverage but more stable, entropy 1.09→0.01 (convergence)
- **Innovation**: 21-parameter StateNet with hierarchical attention
- **Features**: curriculum_attention, hyperbolic_attention, metric_attention
- **Significance**: Validated attention mechanisms for VAE training control
- **Impact**: Led to v5.11+ sophisticated control systems

### v5.11 Era: Dual-Encoder Breakthroughs
**Key Innovation**: Complementary learning systems + homeostatic control

#### ✅ **SUCCESS STORIES**

**v5_11_structural** - **ARCHITECTURAL DEPTH APPROACH**
- **Philosophy**: "Network capacity over dynamic control"
- **Architecture**: hidden_dim=128, n_projection_layers=2, use_controller=False
- **Loss**: PAdicGeodesicLoss (unified hierarchy-correlation)
- **Training**: 44 epochs, static parameters, stability-focused
- **Achievement**: 100% coverage + good hierarchy through architectural complexity

**v5_11_homeostasis** - **DYNAMIC CONTROL APPROACH**
- **Philosophy**: "Adaptive training with automated orchestration"
- **Architecture**: hidden_dim=64, use_controller=True (DifferentiableController)
- **Loss**: RichHierarchyLoss + HomeostasisController
- **Training**: 49 epochs, dynamic freeze/unfreeze, Q-gated annealing
- **Achievement**: Mathematical ceiling hierarchy (-0.83) through dynamic control

#### ❌ **FAILURE MODE**

**v5_11_progressive** - **PROGRESSIVE UNFREEZING FAILURE**
- **Status**: ⚠️ **FAILURE MODE EXAMPLE**
- **Problem**: VAE-B hierarchy inversion (+0.78) due to frequency bias
- **Root Cause**: Progressive unfreezing created competing encoder objectives
- **Lesson**: Sequential specialization >> progressive multi-encoder adaptation
- **Impact**: Led to "differential optimization, not binary switches" insight

#### 🔬 **OTHER VARIANTS**

**v5_11_9_zero** - Zero initialization experiment
**v5_11_9_homeo_zero** - Homeostatic zero initialization
**v5_11_12_validation** - Validation-focused training
**v5_11_11_production** - Production attempt (weak hierarchy +0.06)
**v5_11_epsilon_coupled** - Epsilon coupling experiments

### homeostatic_rich Era: Perfect Balance
**Key Innovation**: Richness-hierarchy coexistence proof

**homeostatic_rich** - **THEORETICAL BREAKTHROUGH**
- **Revolutionary Insight**: Hierarchy and richness are NOT mutually exclusive
- **Method**: RichHierarchyLoss operating on level means (preserves within-level variance)
- **Training**: homeostatic_rich training script with explicit richness defense
- **Proof**: Mathematical ceiling (-0.8321) + highest richness (0.00787)
- **Impact**: Became template for all subsequent high-performance models

### v5.12.4 Era: Production Maturity
**Key Innovation**: Modern architecture + proven training

**v5_12_4** - **PRODUCTION EXCELLENCE**
- **Achievement**: Near-ceiling hierarchy (-0.82) + preserved richness (0.003146)
- **Innovation**: First system to solve richness collapse problem
- **Training**: 16 epochs (fastest convergence), optimal balance
- **Architecture**: Modern components (SiLU, LayerNorm, Dropout) + v5.5 initialization

---

## 🚨 Anti-Patterns & Failure Modes

### Progressive Unfreezing (v5_11_progressive)
- **Problem**: VAE-B hierarchy inversion (+0.78)
- **Cause**: Frequency bias overwhelms p-adic structure during dual adaptation
- **Lesson**: Use sequential specialization, not progressive multi-encoder

### Richness Collapse (v5_11 variants)
- **Problem**: Achieving hierarchy by forcing all samples to exact target radii
- **Cause**: Individual sample supervision instead of level mean supervision
- **Solution**: homeostatic_rich RichHierarchyLoss approach

### Naive Peptide VAE (peptide_vae_attempt_01/02)
- **Problem**: LSTM VAE from scratch ignoring existing infrastructure
- **Status**: DEPRECATED - use carlos_brizuela PeptideVAE instead
- **Lesson**: Build on proven TernaryVAE foundations

---

## 🔧 Technical Specifications

### Model Architectures

| Version | Encoder | Decoder | Hidden Dim | Controller | Loss Function |
|---------|---------|---------|------------|------------|---------------|
| **v5.5** | Dual MLP | Dual (MLP+ResNet) | 256/128/64 | StateNet | Multi-component |
| **v5.6** | Dual MLP | Dual MLP | - | Basic StateNet | Coverage-focused |
| **v5.10** | Dual MLP | Dual MLP | - | Attention StateNet | Multi-attention |
| **v5.11.s** | Dual MLP | Dual MLP | 128 | None | PAdicGeodesicLoss |
| **v5.11.h** | Dual MLP | Dual MLP | 64 | DifferentiableController | RichHierarchyLoss |
| **homeostatic_rich** | Dual MLP | Dual MLP | - | HomeostasisController | RichHierarchyLoss+ |
| **v5.12.4** | ImprovedEncoder | ImprovedDecoder | 64 | DifferentiableController | RichHierarchyLoss |

### Performance Targets

| Metric | Minimum | Good | Excellent | Mathematical Limit |
|--------|---------|------|-----------|-------------------|
| **Coverage** | 95% | 99% | 99.9% | 100% |
| **Hierarchy_B** | -0.5 | -0.7 | -0.8 | -0.8321 (ceiling) |
| **Richness** | 0.001 | 0.003 | 0.005+ | No upper limit |
| **Q-metric** | 1.0 | 1.5 | 2.0+ | No upper limit |

### File Sizes & Parameters

| Checkpoint | Size | Parameters | Notes |
|-----------|------|------------|--------|
| v5.5 | 2.0MB | 168,772 | StateNet overhead |
| v5.11.s | 1.4MB | ~103K | Large hidden_dim=128 |
| v5.11.h | 845KB | ~103K | Efficient hidden_dim=64 |
| homeostatic_rich | 421KB | ~103K | Most compact |
| v5.12.4 | 1.0MB | 113,957 | Modern components overhead |

---

## 🎯 Selection Algorithm

### By Performance Priority
```python
def select_checkpoint(priority):
    if priority == "hierarchy":
        return "homeostatic_rich_best.pt"      # -0.8321 (ceiling)
    elif priority == "coverage":
        return "v5_12_4_best_Q.pt"             # 100% guaranteed
    elif priority == "richness":
        return "homeostatic_rich_best.pt"      # 0.00787 (highest)
    elif priority == "balance":
        return "v5_12_4_best_Q.pt"             # All metrics good
    elif priority == "contacts":
        return "v5_11_structural_best.pt"      # AUC=0.67
```

### By Application Domain
```python
def select_by_application(application):
    bioinformatics = {
        "ddg_prediction": "homeostatic_rich_best.pt",
        "contact_prediction": "v5_11_structural_best.pt",
        "codon_analysis": "v5_11_structural_best.pt",
        "force_constants": "homeostatic_rich_best.pt",
        "general_embedding": "v5_12_4_best_Q.pt"
    }

    research = {
        "architecture_study": "v5_11_homeostasis_best.pt",
        "training_experiments": "v5_11_homeostasis_best.pt",
        "mathematical_analysis": "homeostatic_rich_best.pt",
        "new_loss_functions": "v5_12_4_best_Q.pt"
    }
```

---

## 🧬 Bioinformatics Application Matrix

### Protein Stability (ΔΔG) Prediction
- **Primary**: homeostatic_rich_best.pt (LOO Spearman 0.61)
- **Features**: High richness preserves geometric diversity for thermodynamic signals
- **Validation**: S669 dataset, 52 mutations, +105% over baseline

### Contact Prediction
- **Primary**: v5_11_structural_best.pt (AUC=0.67, Cohen's d=-0.474)
- **Features**: Collapsed shells give consistent AA-level distances
- **Trade-off**: Low richness but excellent pairwise distance discrimination
- **Validation**: Insulin B-chain, 30 residues

### Force Constant Derivation
- **Primary**: homeostatic_rich_best.pt (correlation r=0.86 with mass/radius)
- **Formula**: `k = radius × mass / 100`
- **Applications**: Vibrational frequency prediction, dynamics analysis

### Codon Analysis
- **Primary**: v5_11_structural_best.pt (stable embeddings)
- **Features**: Consistent geometric relationships for sequence analysis
- **Applications**: Genetic code optimization, codon usage bias

---

## 🔬 Research Applications

### Mathematical P-adic Research
- **Primary**: homeostatic_rich_best.pt
- **Validation**: Achieved mathematical ceiling with preserved structure
- **Applications**: p-adic number theory, ultrametric spaces

### Training Methodology Research
- **Primary**: v5_11_homeostasis_best.pt
- **Features**: HomeostasisController for adaptive experiment design
- **Applications**: Multi-objective optimization, automated ML

### Architecture Development
- **Reference**: v5_12_4_best_Q.pt
- **Features**: Modern components as template for new architectures
- **Applications**: New VAE designs, hyperbolic geometry networks

---

## ⚠️ Critical Usage Notes

### Checkpoint Compatibility
- **v5.5-v5.11**: Use `TernaryVAEV5_11_PartialFreeze`
- **v5.12.4**: Use `TernaryVAEV5_12` with `ImprovedEncoder/Decoder`
- **Loading**: Always use `strict=False` for cross-version loading

### Performance vs Application Trade-offs
- **High hierarchy + Low richness**: Good for contacts, bad for ΔΔG
- **High richness + Moderate hierarchy**: Good for ΔΔG, bad for contacts
- **Balanced**: Good for general use, not optimal for specialized tasks

### Validation Requirements
- **Coverage**: Always validate >99% for production use
- **Hierarchy**: Negative values required for p-adic validity
- **Richness**: >0.001 required for non-trivial geometry

---

## 📚 Documentation References

### Training Scripts
- `scripts/epsilon_vae/train_homeostatic_rich.py` - Best balance approach
- `scripts/epsilon_vae/train_hierarchy_focused.py` - Hierarchy-first
- `scripts/epsilon_vae/analyze_all_checkpoints.py` - Comparative analysis

### Research Documentation
- `research/contact-prediction/docs/VALIDATION_RESULTS.md` - Contact prediction validation
- `research/codon-encoder/training/results/` - DDG prediction results
- `docs/audits/CODEBASE_ARCHITECTURE_AUDIT.md` - Architecture analysis

### Configuration Templates
- `configs/v5_12_4.yaml` - Production configuration
- `configs/homeostatic_rich.yaml` - Research configuration
- `configs/v5_11_homeostasis.yaml` - Adaptive training

---

## 🔄 Checkpoint Update Protocol

### Adding New Checkpoints
1. **Performance validation**: Coverage >99%, Hierarchy <-0.5
2. **Tier classification**: Production/Research/Experimental/Deprecated
3. **Documentation update**: Add to this index with metrics and use cases
4. **Size optimization**: Ensure <2MB for practical deployment

### Deprecation Criteria
- **Performance regression**: New checkpoints significantly outperform
- **Architecture obsolescence**: Replaced by improved designs
- **Compatibility issues**: Cannot load with current codebase
- **Failed experiments**: Demonstrated harmful patterns

---

## 🎉 Conclusion

The TernaryVAE checkpoint collection represents **2+ years of systematic architecture development**, from early StateNet experiments to production-ready models achieving theoretical mathematical limits.

**Key Insights**:
1. **No universal "best"** - checkpoint selection depends on application
2. **Hierarchy-richness tradeoff is real** but can be optimized with sophisticated loss design
3. **Homeostatic control** enables superior training outcomes vs static approaches
4. **Modern architecture components** (SiLU, LayerNorm) provide stability without sacrificing performance

**Production Recommendation**: Start with `v5_12_4_best_Q.pt` for general use, switch to specialized checkpoints based on validation performance in your specific application domain.

---

**Index Version**: 1.0
**Last Updated**: 2026-01-14
**Total Checkpoints Cataloged**: 80+
**Production Ready**: 4
**Research Grade**: 15+
**Historical**: 60+
