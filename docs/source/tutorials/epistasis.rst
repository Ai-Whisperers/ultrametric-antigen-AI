Epistasis Analysis
==================

This tutorial covers how to model mutation interactions (epistasis) for drug resistance prediction.

Overview
--------

**Epistasis** occurs when the effect of one mutation depends on the presence of other mutations:

- **Additive**: Combined effect = sum of individual effects
- **Synergistic**: Combined effect > sum (mutations enhance each other)
- **Antagonistic**: Combined effect < sum (mutations cancel each other)
- **Sign Epistasis**: Combined effect has opposite sign from expected

Understanding epistasis is critical for predicting multi-mutation resistance patterns.

Quick Start
-----------

.. code-block:: python

    from src.models.epistasis_module import EpistasisModule, EpistasisResult
    import torch

    # Create epistasis module
    epistasis = EpistasisModule(
        n_positions=300,  # Sequence length
        embed_dim=64,
        n_amino_acids=21,
        use_higher_order=True,  # Include 3+ mutation interactions
    )

    # Get interaction score for mutation combination
    positions = torch.tensor([[65, 184, 215]])  # K65R, M184V, K215Y
    result = epistasis(positions)

    print(f"Interaction score: {result.interaction_score.item():.3f}")
    print(f"Synergistic: {result.synergistic.item()}")
    print(f"Antagonistic: {result.antagonistic.item()}")


Epistasis Module
----------------

The EpistasisModule learns mutation interaction patterns:

.. code-block:: python

    from src.models.epistasis_module import (
        EpistasisModule,
        PairwiseInteractionModule,
        HigherOrderInteractionNet,
    )

    # Full epistasis module
    module = EpistasisModule(
        n_positions=300,
        embed_dim=64,
        n_amino_acids=21,
        use_position_embedding=True,
        use_aa_embedding=True,
        use_higher_order=True,
        max_order=4,  # Up to 4-way interactions
    )

    # Analyze mutation combination
    positions = torch.tensor([[65, 184]])  # Two mutations
    aa_indices = torch.tensor([[3, 15]])  # Amino acids at those positions

    result = module(positions, aa_indices)


Pairwise Interactions
---------------------

Learn how pairs of mutations interact:

.. code-block:: python

    from src.models.epistasis_module import PairwiseInteractionModule

    pairwise = PairwiseInteractionModule(
        n_positions=300,
        embed_dim=64,
    )

    # Get pairwise interaction matrix
    positions = torch.tensor([[65, 184, 215]])
    scores = pairwise(positions)

    # Visualize pairwise interactions
    import matplotlib.pyplot as plt

    matrix = module.get_epistasis_matrix()  # (n_positions, n_positions)
    plt.imshow(matrix.detach().numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Interaction Strength")
    plt.title("Learned Epistasis Matrix")
    plt.show()


Higher-Order Interactions
-------------------------

Model 3+ mutation interactions:

.. code-block:: python

    from src.models.epistasis_module import HigherOrderInteractionNet

    higher_order = HigherOrderInteractionNet(
        embed_dim=64,
        hidden_dim=128,
        max_order=4,
    )

    # Three-way interaction
    positions = torch.tensor([[65, 184, 215]])
    embeddings = position_encoder(positions)
    ho_score = higher_order(embeddings)


Epistasis Loss
--------------

Train models to predict epistatic effects:

.. code-block:: python

    from src.losses.epistasis_loss import (
        EpistasisLoss,
        LearnedEpistasisLoss,
        DrugInteractionLoss,
        MarginRankingLoss,
    )

    # Unified epistasis loss
    loss_fn = EpistasisLoss(
        latent_dim=16,
        n_drugs=23,
        weights={
            "epistasis": 0.3,
            "coevolution": 0.3,
            "drug_interaction": 0.2,
            "margin": 0.2,
        },
    )

    # Compute loss
    result = loss_fn(
        model_output={
            "predictions": predictions,
            "z": latent,
            "single_effects": individual_mutation_effects,
        },
        targets={
            "resistance": true_resistance,
        },
    )

    print(f"Total loss: {result.total_loss.item():.4f}")
    print(f"Epistasis: {result.epistasis_loss.item():.4f}")
    print(f"Coevolution: {result.coevolution_loss.item():.4f}")


Drug Cross-Resistance
---------------------

Model resistance patterns across drugs using hyperbolic geometry:

.. code-block:: python

    from src.losses.epistasis_loss import DrugInteractionLoss

    drug_loss = DrugInteractionLoss(
        n_drugs=23,
        embed_dim=32,
        curvature=1.0,  # Hyperbolic curvature
        temperature=0.1,
    )

    # Set drug class relationships
    drug_classes = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0,  # PI (8 drugs)
        1, 1, 1, 1, 1, 1,        # NRTI (6 drugs)
        2, 2, 2, 2, 2,           # NNRTI (5 drugs)
        3, 3, 3, 3,              # INI (4 drugs)
    ])
    drug_loss.set_drug_classes(drug_classes)

    # Loss encourages similar drugs to have similar resistance
    loss = drug_loss(resistance_predictions)


Real-World Example: HIV TAM Pathway
-----------------------------------

The Thymidine Analog Mutations (TAMs) show strong epistasis:

.. code-block:: python

    # TAM positions in HIV RT
    tam_positions = {
        "M41L": 41,
        "D67N": 67,
        "K70R": 70,
        "L210W": 210,
        "T215Y": 215,
        "K219Q": 219,
    }

    # Analyze TAM pathway epistasis
    tam_combos = [
        [41, 215],        # M41L + T215Y (synergistic)
        [67, 70, 219],    # D67N + K70R + K219Q (alternative pathway)
    ]

    for combo in tam_combos:
        positions = torch.tensor([combo])
        result = epistasis(positions)
        print(f"TAMs {combo}: score={result.interaction_score.item():.3f}")


M184V Resensitization
~~~~~~~~~~~~~~~~~~~~~

M184V antagonizes TAMs (resensitization):

.. code-block:: python

    # M184V alone
    m184v = torch.tensor([[184]])
    result_m184v = epistasis(m184v)

    # M184V + TAMs
    m184v_tams = torch.tensor([[184, 41, 215]])
    result_combo = epistasis(m184v_tams)

    # Expected: antagonistic interaction (resensitization)
    print(f"M184V alone: {result_m184v.interaction_score.item():.3f}")
    print(f"M184V + TAMs: {result_combo.interaction_score.item():.3f}")
    print(f"Antagonistic: {result_combo.antagonistic.item()}")


Visualizing Epistasis
---------------------

Create epistasis heatmaps:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get full epistasis matrix
    matrix = epistasis.get_epistasis_matrix()

    # Focus on resistance-associated positions
    positions_of_interest = [41, 65, 67, 70, 74, 184, 210, 215, 219]
    sub_matrix = matrix[positions_of_interest][:, positions_of_interest]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sub_matrix.detach().numpy(),
        xticklabels=positions_of_interest,
        yticklabels=positions_of_interest,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("HIV RT Epistasis Matrix")
    plt.tight_layout()
    plt.savefig("epistasis_matrix.png")


Integration with Disease Analyzers
----------------------------------

Use epistasis in disease prediction:

.. code-block:: python

    from src.diseases.hiv_analyzer import HIVAnalyzer

    analyzer = HIVAnalyzer()

    # Analyze sequence with epistasis
    results = analyzer.analyze(
        sequences=["PISPIET..."],
        include_epistasis=True,
    )

    # Access epistasis scores
    for mutation_combo in results["epistasis"]:
        print(f"Mutations: {mutation_combo['positions']}")
        print(f"Interaction: {mutation_combo['score']:.3f}")
        print(f"Type: {mutation_combo['type']}")  # synergistic/antagonistic


Best Practices
--------------

1. **Use position + AA embeddings**: Both contribute to interaction patterns
2. **Include higher-order**: 3-4 way interactions are biologically relevant
3. **Validate with known epistasis**: Check against published TAM/M184V patterns
4. **Visualize epistasis matrix**: Identify unexpected interaction patterns
5. **Use with cross-resistance**: Drug class relationships inform epistasis


See Also
--------

- :doc:`/api/models` for EpistasisModule API
- :doc:`/api/losses` for EpistasisLoss API
- :doc:`hiv_resistance` for HIV-specific mutation analysis
