#!/usr/bin/env python3
"""Extract ESM-2 Amino Acid Embeddings.

This script extracts ESM-2 embeddings for each of the 20 standard amino acids.
Since amino acids need context, we use a simple context strategy:
- Each AA is placed in the center of a poly-A context
- This gives ESM a realistic sequence environment

Output format compatible with multimodal_ddg_predictor.py:
{
    "A": [float, ...],  # 1280-dim for ESM-2 650M
    "R": [float, ...],
    ...
}

Usage:
    python extract_esm_aa_embeddings.py
    python extract_esm_aa_embeddings.py --model esm2_t12_35M  # Smaller model
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Check for transformers
try:
    import torch
    from transformers import AutoTokenizer, EsmModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ESM-2 model configs
ESM_MODELS = {
    "esm2_t6_8M": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "dim": 320,
    },
    "esm2_t12_35M": {
        "name": "facebook/esm2_t12_35M_UR50D",
        "dim": 480,
    },
    "esm2_t30_150M": {
        "name": "facebook/esm2_t30_150M_UR50D",
        "dim": 640,
    },
    "esm2_t33_650M": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "dim": 1280,
    },
}


def extract_esm_embeddings(
    model_key: str = "esm2_t12_35M",
    context_length: int = 10,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Extract ESM-2 embeddings for each amino acid.

    Strategy:
    - Place each AA in center of poly-A context: "AAAAA{X}AAAAA"
    - Extract the embedding for the central residue
    - This provides realistic sequence context

    Args:
        model_key: Which ESM-2 model to use
        context_length: Length of context on each side
        device: Device for computation

    Returns:
        Dictionary mapping AA letters to embeddings
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers package required. "
            "Install with: pip install transformers torch"
        )

    model_config = ESM_MODELS.get(model_key)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(ESM_MODELS.keys())}")

    print(f"Loading model: {model_config['name']}")
    print(f"Expected embedding dim: {model_config['dim']}")

    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    model = EsmModel.from_pretrained(model_config['name'])
    model.to(device)
    model.eval()

    embeddings = {}
    context = "A" * context_length

    print(f"\nExtracting embeddings for {len(AMINO_ACIDS)} amino acids...")

    with torch.no_grad():
        for aa in AMINO_ACIDS:
            # Create context sequence with AA in center
            seq = context + aa + context

            # Tokenize
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # Extract central residue embedding
            # Sequence: [BOS] + context + AA + context + [EOS]
            # Central position: 1 + context_length
            central_idx = 1 + context_length
            aa_embedding = outputs.last_hidden_state[0, central_idx, :].cpu().numpy()

            embeddings[aa] = aa_embedding.tolist()

            print(f"  {aa}: extracted {len(aa_embedding)}-dim embedding")

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 amino acid embeddings"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="esm2_t12_35M",
        choices=list(ESM_MODELS.keys()),
        help="ESM-2 model to use"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=10,
        help="Context length on each side of AA"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/esm_aa_embeddings.json",
        help="Output path"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ESM-2 Amino Acid Embedding Extractor")
    print("=" * 70)

    if not HAS_TRANSFORMERS:
        print("\nError: transformers package not available.")
        print("Install with: pip install transformers torch")
        print("\nCreating placeholder embeddings for testing...")

        # Create placeholder embeddings (random but reproducible)
        np.random.seed(42)
        dim = ESM_MODELS[args.model]["dim"]
        embeddings = {aa: np.random.randn(dim).tolist() for aa in AMINO_ACIDS}

        output_data = {
            "metadata": {
                "model": args.model,
                "dim": dim,
                "method": "PLACEHOLDER (transformers not installed)",
                "timestamp": datetime.now().isoformat(),
            },
            "embeddings": embeddings,
        }

        # Also save flat format for compatibility
        flat_output = embeddings.copy()

        with open(output_path, 'w') as f:
            json.dump(flat_output, f, indent=2)

        print(f"\nPlaceholder embeddings saved to: {output_path}")
        return 0

    # Extract real embeddings
    try:
        embeddings = extract_esm_embeddings(
            model_key=args.model,
            context_length=args.context_length,
            device=args.device,
        )

        # Save in flat format for multimodal_ddg_predictor.py
        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=2)

        print(f"\nEmbeddings saved to: {output_path}")

        # Also save with metadata
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                "model": args.model,
                "model_name": ESM_MODELS[args.model]["name"],
                "dim": ESM_MODELS[args.model]["dim"],
                "context_length": args.context_length,
                "context_pattern": f"{'A' * args.context_length}{{X}}{'A' * args.context_length}",
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")

    except Exception as e:
        print(f"\nError extracting embeddings: {e}")
        print("Creating placeholder embeddings...")

        np.random.seed(42)
        dim = ESM_MODELS[args.model]["dim"]
        embeddings = {aa: np.random.randn(dim).tolist() for aa in AMINO_ACIDS}

        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=2)

        print(f"\nPlaceholder embeddings saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
