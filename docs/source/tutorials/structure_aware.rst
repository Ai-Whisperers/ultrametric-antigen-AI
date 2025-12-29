Structure-Aware Modeling
========================

This tutorial covers how to incorporate 3D protein structure information into drug resistance prediction using AlphaFold2 predictions.

Overview
--------

Protein structure affects drug binding and resistance mechanisms. The Structure-Aware VAE integrates:

1. **AlphaFold2 structures**: Predicted 3D coordinates
2. **SE(3)-equivariant encoding**: Rotation/translation invariant features
3. **pLDDT confidence**: AlphaFold's per-residue confidence scores
4. **Sequence-structure fusion**: Cross-attention between modalities

Quick Start
-----------

.. code-block:: python

    from src.models.structure_aware_vae import (
        StructureAwareVAE,
        StructureConfig,
    )
    from src.encoders.alphafold_encoder import AlphaFoldEncoder

    # Configure structure integration
    structure_config = StructureConfig(
        use_structure=True,
        structure_dim=64,
        n_structure_layers=3,
        cutoff=10.0,  # Angstroms for distance cutoff
        use_plddt=True,  # Weight by AlphaFold confidence
        fusion_type="cross_attention",  # or "gated", "concat"
    )

    # Create structure-aware VAE
    model = StructureAwareVAE(
        input_dim=128,
        latent_dim=32,
        structure_config=structure_config,
    )

    # Forward pass with structure
    outputs = model(
        x=sequence_embeddings,
        structure=coordinates,  # (batch, n_residues, 3)
        plddt=confidence_scores,  # (batch, n_residues)
    )


Loading AlphaFold Structures
----------------------------

The AlphaFold encoder handles structure loading and caching:

.. code-block:: python

    from src.encoders.alphafold_encoder import AlphaFoldStructureLoader

    # Initialize loader with cache
    loader = AlphaFoldStructureLoader(cache_dir=".alphafold_cache")

    # Load structure for a UniProt ID
    structure = loader.get_structure("P04637")  # p53 tumor suppressor

    print(f"Coordinates shape: {structure['coords'].shape}")  # (L, 3)
    print(f"pLDDT scores: {structure['plddt'].mean():.1f}")  # ~80-90 for good predictions
    print(f"Sequence: {structure['sequence'][:50]}...")


SE(3)-Equivariant Encoding
--------------------------

The SE3Encoder processes 3D coordinates while respecting rotational/translational symmetry:

.. code-block:: python

    from src.models.structure_aware_vae import SE3Encoder

    encoder = SE3Encoder(
        node_dim=64,
        edge_dim=32,
        n_layers=3,
        cutoff=10.0,  # Only consider atoms within 10 Angstroms
    )

    # Encode structure
    structure_embedding = encoder(
        coords=coordinates,  # (batch, n_residues, 3)
        aa_indices=amino_acid_indices,  # Optional (batch, n_residues)
    )

**Key Properties**:

- **Rotation invariant**: Same output regardless of protein orientation
- **Translation invariant**: Same output regardless of protein position
- **Distance-based**: Uses pairwise distances, not absolute coordinates


Invariant Point Attention
-------------------------

IPA from AlphaFold2 for structure-aware attention:

.. code-block:: python

    from src.models.structure_aware_vae import InvariantPointAttention

    ipa = InvariantPointAttention(
        embed_dim=64,
        n_heads=4,
        n_query_points=4,
        n_value_points=4,
    )

    # Apply IPA
    output = ipa(
        features=residue_features,  # (batch, n_residues, dim)
        coords=coordinates,  # (batch, n_residues, 3)
        mask=attention_mask,  # Optional
    )


Sequence-Structure Fusion
-------------------------

Three fusion strategies are available:

Cross-Attention (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = StructureConfig(
        fusion_type="cross_attention",
    )

    # Sequence attends to structure
    # Structure provides spatial context


Gated Fusion
~~~~~~~~~~~~

.. code-block:: python

    config = StructureConfig(
        fusion_type="gated",
    )

    # Learned gate controls structure contribution
    # gate = sigmoid(W * [seq, struct])
    # output = gate * transform([seq, struct])


Concatenation
~~~~~~~~~~~~~

.. code-block:: python

    config = StructureConfig(
        fusion_type="concat",
    )

    # Simple concatenation followed by projection
    # output = W * [seq; struct]


pLDDT Confidence Weighting
--------------------------

AlphaFold's pLDDT score indicates prediction confidence:

- **>90**: Very high confidence
- **70-90**: Confident
- **50-70**: Low confidence
- **<50**: Very low confidence (likely disordered)

.. code-block:: python

    config = StructureConfig(
        use_plddt=True,  # Enable pLDDT weighting
    )

    # High pLDDT regions contribute more to the embedding
    # Low pLDDT regions (disordered) are downweighted


Full Pipeline Example
---------------------

Complete example for HIV resistance prediction with structure:

.. code-block:: python

    from src.models.structure_aware_vae import StructureAwareVAE, StructureConfig
    from src.encoders.alphafold_encoder import AlphaFoldStructureLoader
    from src.diseases.hiv_analyzer import HIVAnalyzer
    import torch

    # 1. Load AlphaFold structure for HIV RT
    loader = AlphaFoldStructureLoader()
    rt_structure = loader.get_structure("P03366")  # HIV-1 RT

    # 2. Configure structure-aware model
    config = StructureConfig(
        use_structure=True,
        structure_dim=64,
        use_plddt=True,
        fusion_type="cross_attention",
    )

    model = StructureAwareVAE(
        input_dim=128,
        latent_dim=32,
        structure_config=config,
    )

    # 3. Encode sequence with structure
    sequence_embedding = encode_sequence(hiv_sequence)
    coords = torch.tensor(rt_structure["coords"]).unsqueeze(0)
    plddt = torch.tensor(rt_structure["plddt"]).unsqueeze(0)

    outputs = model(
        x=sequence_embedding,
        structure=coords,
        plddt=plddt,
    )

    # 4. Use latent for downstream prediction
    z = outputs["z"]
    resistance_pred = resistance_head(z)


Without Structure (Fallback)
----------------------------

The model gracefully falls back to sequence-only mode:

.. code-block:: python

    # No structure provided - uses sequence only
    outputs = model(x=sequence_embedding)

    # Structure provided but use_structure=False
    config = StructureConfig(use_structure=False)
    model = StructureAwareVAE(..., structure_config=config)
    outputs = model(x=sequence_embedding, structure=coords)  # Structure ignored


Best Practices
--------------

1. **Use pLDDT weighting**: Always enable for AlphaFold structures
2. **Check structure quality**: Verify pLDDT > 70 for drug binding regions
3. **Align structures**: Ensure consistent orientation for training
4. **Cache structures**: Use AlphaFoldStructureLoader caching
5. **Handle missing structure**: Gracefully fall back to sequence-only


Performance Considerations
--------------------------

- Structure encoding adds ~20-30% compute time
- Cache AlphaFold structures to avoid repeated downloads
- Use smaller cutoff (8-10 Angstroms) for speed
- Batch similar-length proteins together


See Also
--------

- :doc:`/api/models` for StructureAwareVAE API
- :doc:`/api/encoders` for AlphaFoldEncoder API
- :doc:`hiv_resistance` for HIV-specific structure analysis
