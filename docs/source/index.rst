Ternary VAE Documentation
=========================

Ternary VAE is a dual neural variational autoencoder framework for learning 3-adic algebraic
structure in ternary operation space using hyperbolic geometry.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/training
   guide/analysis
   guide/cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Key Features
------------

* **Dual VAE Architecture**: VAE-A explores, VAE-B refines, StateNet Controller allows both VAEs to share hyperparameters, creating "homeostasis"
* **Hyperbolic Geometry**: Poincaré ball latent space
* **P-adic Structure**: 3-adic valuation for codon relationships
* **HIV Analysis**: Drug resistance, CTL escape, neutralization
* **Vaccine Target Identification**: Multi-constraint optimization


Quick Example
-------------

.. code-block:: python

    from src import TernaryVAE, load_config
    from src.data import generate_all_ternary_operations

    # Generate data
    x, indices = generate_all_ternary_operations()

    # Create model
    model = TernaryVAE(latent_dim=16, hidden_dim=64)

    # Train
    from src.training import TernaryVAETrainer
    trainer = TernaryVAETrainer(model, config)
    trainer.train(x)


CLI Usage
---------

.. code-block:: bash

    # Train a model
    ternary-vae train run --epochs 100

    # Analyze data
    ternary-vae analyze stanford --drug-class PI

    # Check data status
    ternary-vae data status


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
