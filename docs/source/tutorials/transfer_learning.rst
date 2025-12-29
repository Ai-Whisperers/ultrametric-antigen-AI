Transfer Learning
=================

This tutorial covers how to use transfer learning to improve drug resistance prediction, especially for diseases with limited data.

Overview
--------

Transfer learning leverages knowledge from data-rich diseases to improve predictions on data-scarce diseases.

The pipeline supports multiple strategies:

1. **Frozen Encoder**: Pre-train encoder, freeze it, train new prediction head
2. **Full Fine-tuning**: Pre-train, then fine-tune all parameters
3. **Adapter Layers**: Add small trainable adapter modules
4. **LoRA**: Low-rank adaptation of large models
5. **MAML**: Model-agnostic meta-learning for few-shot adaptation

Quick Start
-----------

.. code-block:: python

    from src.training.transfer_pipeline import (
        TransferLearningPipeline,
        TransferConfig,
        TransferStrategy,
    )

    # Configure transfer learning
    config = TransferConfig(
        latent_dim=32,
        pretrain_epochs=100,
        finetune_epochs=50,
        strategy=TransferStrategy.FROZEN_ENCODER,
    )

    # Create pipeline
    pipeline = TransferLearningPipeline(config)

    # Pre-train on all diseases
    all_disease_data = {
        "hiv": hiv_dataset,
        "hbv": hbv_dataset,
        "tuberculosis": tb_dataset,
        # ... more diseases
    }
    pretrained = pipeline.pretrain(all_disease_data)

    # Fine-tune on target disease
    finetuned = pipeline.finetune(
        target_disease="candida",
        target_data=candida_dataset,
    )

    # Evaluate transfer
    metrics = pipeline.evaluate_transfer(
        source="hiv",
        target="candida",
    )


Pre-training
------------

Multi-task pre-training learns shared representations across diseases:

.. code-block:: python

    from src.training.transfer_pipeline import TransferLearningPipeline

    # Load datasets for all diseases
    datasets = {
        "hiv": load_hiv_data(),
        "hbv": load_hbv_data(),
        "tuberculosis": load_tb_data(),
        "influenza": load_flu_data(),
        "sars_cov2": load_covid_data(),
    }

    # Pre-train shared encoder
    pipeline = TransferLearningPipeline(config)
    pretrained_model = pipeline.pretrain(
        datasets,
        epochs=100,
        log_every=10,
    )

    # Save checkpoint
    pipeline.save_checkpoint("pretrained_multi_disease.pt")


Transfer Strategies
-------------------

Frozen Encoder
~~~~~~~~~~~~~~

Best for: Small target datasets, preventing overfitting

.. code-block:: python

    config = TransferConfig(
        strategy=TransferStrategy.FROZEN_ENCODER,
        finetune_lr=1e-3,  # Higher LR okay since only training head
    )

    finetuned = pipeline.finetune(
        "target_disease",
        target_data,
    )


Full Fine-tuning
~~~~~~~~~~~~~~~~

Best for: Moderate target datasets, maximum flexibility

.. code-block:: python

    config = TransferConfig(
        strategy=TransferStrategy.FULL_FINETUNE,
        finetune_lr=1e-4,  # Lower LR to preserve pre-trained knowledge
        finetune_epochs=30,
    )


Adapter Layers
~~~~~~~~~~~~~~

Best for: Efficient fine-tuning, keeping base model frozen

.. code-block:: python

    config = TransferConfig(
        strategy=TransferStrategy.ADAPTER,
        adapter_dim=64,  # Bottleneck dimension
    )


LoRA (Low-Rank Adaptation)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Best for: Large models, memory-efficient fine-tuning

.. code-block:: python

    config = TransferConfig(
        strategy=TransferStrategy.LORA,
        lora_rank=8,  # Low-rank dimension
    )


MAML (Few-Shot Learning)
~~~~~~~~~~~~~~~~~~~~~~~~

Best for: Very small target datasets (5-50 samples)

.. code-block:: python

    config = TransferConfig(
        strategy=TransferStrategy.MAML,
        maml_inner_lr=0.01,
        maml_inner_steps=5,
    )

    # Adapt with just a few examples
    adapted = pipeline.finetune(
        "rare_pathogen",
        few_shot_data,  # Only 10-50 samples
        few_shot=True,
    )


Cross-Disease Transfer
----------------------

Evaluate how well knowledge transfers between disease pairs:

.. code-block:: python

    # Evaluate all transfer pairs
    transfer_matrix = {}
    diseases = ["hiv", "hbv", "tuberculosis", "influenza", "sars_cov2"]

    for source in diseases:
        for target in diseases:
            if source != target:
                metrics = pipeline.evaluate_transfer(source, target)
                transfer_matrix[(source, target)] = metrics["spearman"]

    # Print transfer matrix
    print("Transfer Matrix (Spearman correlation):")
    for (src, tgt), score in sorted(transfer_matrix.items()):
        print(f"  {src} -> {tgt}: {score:.3f}")


Expected Transfer Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on biological similarity:

- **High Transfer**: HIV ↔ HBV (both retroviruses), Flu ↔ RSV (respiratory)
- **Moderate Transfer**: Tuberculosis ↔ MRSA (both bacterial)
- **Low Transfer**: Viral ↔ Bacterial ↔ Fungal ↔ Parasitic


Best Practices
--------------

1. **Use diverse pre-training data**: Include all available diseases
2. **Match data modality**: Pre-train on similar sequence types (nucleotide vs protein)
3. **Monitor for negative transfer**: If target performance decreases, try frozen encoder
4. **Validate on held-out data**: Always evaluate on data not seen during training
5. **Consider domain adaptation**: Use unlabeled target data for domain adaptation


Checkpointing
-------------

Save and load transfer learning checkpoints:

.. code-block:: python

    # Save after pre-training
    pipeline.save_checkpoint(
        "checkpoints/pretrained_v1.pt",
        include_optimizer=True,
    )

    # Load for fine-tuning
    pipeline = TransferLearningPipeline.from_checkpoint(
        "checkpoints/pretrained_v1.pt"
    )

    # Fine-tune on new disease
    pipeline.finetune("new_disease", new_data)


See Also
--------

- :doc:`/api/training` for full API reference
- :doc:`meta_learning` for advanced few-shot techniques
- :doc:`hiv_resistance` for HIV-specific examples
