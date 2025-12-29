# Image Proposals and Improvements

## Critique of Existing Specifications

The existing 30 images cover the biological results comprehensively. However, the _methodological_ innovation (Ternary VAEs, Hyperbolic/Ultrametric space) is under-represented. To convince the "Virologists and Medical Researchers" audience of the _validity_ of these new "Genetic Distance" metrics, we must visually explain the model.

## Proposed Additional Images

### Image 31: The 3-Adic Latent Space

**Rationale**: The paper claims to use a "Ternary" VAE and "Geometric" modeling. We need to show what this looks like.
**Type**: 3D Scatter or Tree diagram.
**Concept**: Visualizing the hierarchical clustering of sequences.
**Layout**: A tree-like structure where branches represent major mutations.
**Filename**: `31_3adic_latent_space_visualization.png`

### Image 32: Ternary VAE Architecture

**Rationale**: Explains how "Codons" become "Vectors".
**Type**: Schematic Diagram.
**Concept**: Input (Sequence) -> Encoder -> Latent Space (Hyperbolic) -> Decoder -> Output (Reconstructed Sequence).
**Filename**: `32_ternary_vae_architecture.png`

### Image 33: Geometric "Goldilocks Zone" (Data Plot)

**Rationale**: Image 11 is a conceptual diagram. A real scatter plot of the data would be scientifically stronger.
**Type**: Scatter Plot.
**Data**: X-axis = Genetic Distance, Y-axis = Escape Velocity.
**Highlight**: The region where fitness cost is manageable but escape is effective.
**Filename**: `33_goldilocks_zone_scatter.png`

## Improvements to Existing Images

- **Image 22 (Venn Diagram)**: Ensure the "96% combined" region is visually distinct and clearly labeled as the "Functional Cure Target".
- **Image 26 (Conflict Diagram)**: Add specific examples of _common_ conflicting positions (like K103N) directly onto the diagram for better context.
