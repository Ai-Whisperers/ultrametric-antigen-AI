"""VAE ↔ Groupoid Integration."""

from .vae_groupoid_builder import (
    build_amino_acid_groupoid,
    analyze_escape_paths,
    validate_with_substitution_data,
    load_vae_embeddings,
)

__all__ = [
    'build_amino_acid_groupoid',
    'analyze_escape_paths',
    'validate_with_substitution_data',
    'load_vae_embeddings',
]
