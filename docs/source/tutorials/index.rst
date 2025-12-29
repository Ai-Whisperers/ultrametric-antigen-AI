Tutorials
=========

Step-by-step tutorials for common tasks with Ternary VAE.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   gpu_quick_start
   basic_training
   full_training

.. toctree::
   :maxdepth: 1
   :caption: Applications

   hiv_resistance
   codon_analysis
   predictors

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   epsilon_vae
   meta_learning
   uncertainty
   transfer_learning
   structure_aware
   epistasis


Quick Start Guide
-----------------

New to Ternary VAE? Start here:

1. **GPU Setup** (:doc:`gpu_quick_start`)
   Verify your GPU works with a 5-minute test

2. **Basic Training** (:doc:`basic_training`)
   Train your first model on ternary operations

3. **Full Training** (:doc:`full_training`)
   Complete guide to V5.11 and V5.11.11 training


Application Tutorials
---------------------

HIV Drug Resistance (:doc:`hiv_resistance`)
    Analyze resistance mutations using hyperbolic embeddings

Codon Analysis (:doc:`codon_analysis`)
    Deep dive into codon structure and synonymous mutations

Predictors (:doc:`predictors`)
    Build resistance, escape, and tropism predictors


Advanced Topics
---------------

Epsilon-VAE (:doc:`epsilon_vae`)
    Meta-learning over checkpoint landscapes

Meta-Learning (:doc:`meta_learning`)
    Few-shot adaptation with MAML and Reptile

Uncertainty Quantification (:doc:`uncertainty`)
    MC Dropout, Evidential, and Ensemble uncertainty methods with calibration

Transfer Learning (:doc:`transfer_learning`)
    Pre-train on multiple diseases, fine-tune on target with MAML, LoRA, and Adapters

Structure-Aware Modeling (:doc:`structure_aware`)
    Integrate AlphaFold2 3D structures with SE(3)-equivariant encoders

Epistasis Analysis (:doc:`epistasis`)
    Model mutation interactions for multi-mutation resistance patterns


Prerequisites
~~~~~~~~~~~~~

1. Python 3.10+
2. PyTorch 2.0+ with CUDA
3. Ternary VAE installed: ``pip install -e .``


Quick Commands
~~~~~~~~~~~~~~

.. code-block:: bash

    # GPU smoke test
    python scripts/quick_train.py

    # Full training
    python src/train.py --mode v5.11 --epochs 100

    # HIV analysis
    python scripts/hiv/run_full_hiv_pipeline.py --stage all

    # Train predictors
    python scripts/train_predictors.py --predictor all


Next Steps
~~~~~~~~~~

After completing these tutorials:

- Explore the :doc:`/api/index` for detailed module documentation
- Read the :doc:`/guide/analysis` for analysis workflows
- Check :doc:`/contributing` to contribute improvements
