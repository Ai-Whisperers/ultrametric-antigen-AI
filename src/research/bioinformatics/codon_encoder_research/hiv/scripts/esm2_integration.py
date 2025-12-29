"""
ESM-2 Protein Language Model Integration for HIV Analysis.

This module integrates Meta's ESM-2 protein language model for enhanced
sequence analysis, combining learned evolutionary features with our
p-adic hyperbolic geometry framework.

Key features:
1. ESM-2 embeddings for HIV protein sequences
2. Zero-shot mutation effect prediction
3. Attention-based contact map prediction
4. Integration with hyperbolic codon encoding
5. Embedding visualization and comparison

Based on papers:
- Lin et al. 2023: ESM-2 language model
- Meier et al. 2021: Language models for protein engineering
- Rives et al. 2021: Biological structure from scaling

Requirements:
    pip install torch transformers

Author: Research Team
Date: December 2025
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def poincare_distance_np(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance between two Poincare ball embeddings.

    V5.12.2: Proper hyperbolic distance formula instead of Euclidean norm.

    Uses: d(x,y) = arccosh(1 + 2 * ||x-y||² / ((1-||x||²)(1-||y||²)))
    """
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    diff_norm_sq = np.sum((x - y) ** 2)

    # Clamp norms to stay inside the ball
    x_norm_sq = np.clip(x_norm_sq, 0, 0.999)
    y_norm_sq = np.clip(y_norm_sq, 0, 0.999)

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    arg = 1 + 2 * c * diff_norm_sq / (denom + 1e-10)

    return float(np.arccosh(np.clip(arg, 1.0, 1e10)))


# Lazy imports for optional dependencies
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ESM2Config:
    """Configuration for ESM-2 model."""

    model_name: str = "facebook/esm2_t6_8M_UR50D"  # Smallest model for testing
    device: str = "cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    max_length: int = 1024
    batch_size: int = 8
    use_mean_pooling: bool = True
    cache_dir: Optional[Path] = None


# Available ESM-2 model variants
ESM2_MODELS = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",      # 8M params, fastest
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",   # 35M params
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D", # 150M params
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D", # 650M params
    "esm2_t36_3B": "facebook/esm2_t36_3B_UR50D",    # 3B params, most accurate
}


class ESM2Embedder:
    """
    ESM-2 protein embedding extractor.

    Provides sequence embeddings and mutation effect predictions
    using the ESM-2 protein language model.
    """

    def __init__(self, config: Optional[ESM2Config] = None):
        """
        Initialize ESM-2 embedder.

        Args:
            config: ESM2Config object or None for defaults
        """
        if not _TORCH_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "ESM-2 integration requires torch and transformers. "
                "Install with: pip install torch transformers"
            )

        self.config = config or ESM2Config()
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self):
        """Load ESM-2 model and tokenizer."""
        if self._is_loaded:
            return

        print(f"Loading ESM-2 model: {self.config.model_name}")
        print(f"Device: {self.config.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )

        self.model = self.model.to(self.config.device)
        self.model.eval()

        self._is_loaded = True
        print("Model loaded successfully!")

    def get_embedding(
        self,
        sequence: str,
        return_attention: bool = False,
    ) -> dict:
        """
        Get ESM-2 embedding for a protein sequence.

        Args:
            sequence: Amino acid sequence string
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - 'embedding': numpy array of shape (seq_len, hidden_dim) or (hidden_dim,)
                - 'per_residue': numpy array of per-residue embeddings
                - 'attention': attention weights if requested
        """
        if not self._is_loaded:
            self.load_model()

        # Clean sequence
        sequence = sequence.upper().replace(" ", "").replace("\n", "")

        # Truncate if too long
        if len(sequence) > self.config.max_length:
            warnings.warn(
                f"Sequence length {len(sequence)} exceeds max_length "
                f"{self.config.max_length}. Truncating."
            )
            sequence = sequence[:self.config.max_length]

        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )

        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=return_attention,
            )

        # Extract embeddings (last hidden state)
        hidden_states = outputs.last_hidden_state.cpu().numpy()

        # Remove special tokens (CLS at start, EOS at end)
        per_residue = hidden_states[0, 1:-1, :]  # (seq_len, hidden_dim)

        # Mean pooling for sequence-level embedding
        if self.config.use_mean_pooling:
            embedding = np.mean(per_residue, axis=0)
        else:
            embedding = per_residue[0]  # CLS token

        result = {
            "embedding": embedding,
            "per_residue": per_residue,
            "sequence_length": len(sequence),
        }

        if return_attention:
            # Average attention across heads and layers
            attentions = outputs.attentions
            avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))
            result["attention"] = avg_attention.cpu().numpy()

        return result

    def get_batch_embeddings(
        self,
        sequences: list[str],
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Get embeddings for multiple sequences.

        Args:
            sequences: List of amino acid sequences
            show_progress: Whether to show progress bar

        Returns:
            List of embedding dictionaries
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sequences, desc="Embedding sequences")
            except ImportError:
                iterator = sequences
        else:
            iterator = sequences

        for seq in iterator:
            try:
                emb = self.get_embedding(seq)
                results.append(emb)
            except Exception as e:
                warnings.warn(f"Failed to embed sequence: {e}")
                results.append(None)

        return results

    def predict_mutation_effect(
        self,
        wild_type: str,
        mutant: str,
        position: Optional[int] = None,
    ) -> dict:
        """
        Predict the effect of a mutation using embedding distance.

        Args:
            wild_type: Wild-type amino acid sequence
            mutant: Mutant amino acid sequence
            position: Optional position of mutation for detailed analysis

        Returns:
            Dictionary with mutation effect metrics
        """
        wt_emb = self.get_embedding(wild_type)
        mut_emb = self.get_embedding(mutant)

        # Calculate embedding distance
        seq_distance = np.linalg.norm(wt_emb["embedding"] - mut_emb["embedding"])

        # Per-residue changes if same length
        if wt_emb["per_residue"].shape[0] == mut_emb["per_residue"].shape[0]:
            per_res_diff = np.linalg.norm(
                wt_emb["per_residue"] - mut_emb["per_residue"],
                axis=1
            )
            max_change_pos = int(np.argmax(per_res_diff))
            max_change = float(per_res_diff[max_change_pos])
        else:
            per_res_diff = None
            max_change_pos = None
            max_change = None

        # Estimate fitness effect from distance
        # Larger distance = likely more disruptive
        fitness_effect = -0.1 * seq_distance  # Negative = deleterious

        return {
            "embedding_distance": float(seq_distance),
            "per_residue_changes": per_res_diff,
            "max_change_position": max_change_pos,
            "max_change_magnitude": max_change,
            "estimated_fitness_effect": fitness_effect,
            "predicted_effect": (
                "neutral" if seq_distance < 1.0 else
                "moderate" if seq_distance < 2.0 else
                "severe"
            ),
        }

    def get_log_likelihood_ratio(
        self,
        sequence: str,
        position: int,
        original_aa: str,
        mutant_aa: str,
    ) -> float:
        """
        Calculate log-likelihood ratio for a single-site mutation.

        This provides a zero-shot prediction of mutation effect.

        Args:
            sequence: Amino acid sequence
            position: 0-indexed position of mutation
            original_aa: Original amino acid at position
            mutant_aa: Mutant amino acid

        Returns:
            Log-likelihood ratio (positive = favorable mutation)
        """
        if not self._is_loaded:
            self.load_model()

        # Mask the position of interest
        sequence_list = list(sequence)
        sequence_list[position] = self.tokenizer.mask_token

        masked_seq = "".join(sequence_list)

        inputs = self.tokenizer(
            masked_seq,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits at masked position (+1 for CLS token)
        logits = outputs.last_hidden_state[0, position + 1, :]

        # Get token IDs for original and mutant
        orig_token_id = self.tokenizer.convert_tokens_to_ids(original_aa)
        mut_token_id = self.tokenizer.convert_tokens_to_ids(mutant_aa)

        # Calculate log-likelihood ratio
        log_probs = torch.log_softmax(logits, dim=-1)

        llr = float(log_probs[mut_token_id] - log_probs[orig_token_id])

        return llr


def combine_esm2_with_hyperbolic(
    esm2_embedding: np.ndarray,
    hyperbolic_embedding: np.ndarray,
    combination_method: str = "concatenate",
    esm2_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine ESM-2 and hyperbolic embeddings.

    Args:
        esm2_embedding: ESM-2 sequence embedding
        hyperbolic_embedding: P-adic hyperbolic codon embedding
        combination_method: 'concatenate', 'average', or 'weighted'
        esm2_weight: Weight for ESM-2 in weighted combination

    Returns:
        Combined embedding array
    """
    if combination_method == "concatenate":
        return np.concatenate([esm2_embedding, hyperbolic_embedding])

    elif combination_method == "average":
        # Project to same dimension first
        min_dim = min(len(esm2_embedding), len(hyperbolic_embedding))
        return (esm2_embedding[:min_dim] + hyperbolic_embedding[:min_dim]) / 2

    elif combination_method == "weighted":
        min_dim = min(len(esm2_embedding), len(hyperbolic_embedding))
        return (
            esm2_weight * esm2_embedding[:min_dim] +
            (1 - esm2_weight) * hyperbolic_embedding[:min_dim]
        )

    else:
        raise ValueError(f"Unknown combination method: {combination_method}")


def extract_contact_predictions(
    esm2_embedder: ESM2Embedder,
    sequence: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Extract predicted contact map from ESM-2 attention.

    Contact maps indicate which residues are spatially close in 3D structure.

    Args:
        esm2_embedder: ESM2Embedder instance
        sequence: Amino acid sequence
        threshold: Threshold for binarizing contacts

    Returns:
        Contact map array of shape (seq_len, seq_len)
    """
    result = esm2_embedder.get_embedding(sequence, return_attention=True)

    if "attention" not in result:
        raise ValueError("Attention weights not available")

    attention = result["attention"]

    # Symmetrize attention matrix
    contact_map = (attention + attention.T) / 2

    # Apply threshold
    binary_contacts = (contact_map > threshold).astype(np.float32)

    return binary_contacts


def analyze_hiv_protein(
    esm2_embedder: ESM2Embedder,
    sequence: str,
    protein_name: str = "unknown",
) -> dict:
    """
    Comprehensive ESM-2 analysis of an HIV protein sequence.

    Args:
        esm2_embedder: ESM2Embedder instance
        sequence: Amino acid sequence
        protein_name: Name of the protein (e.g., "gp120", "PR")

    Returns:
        Dictionary with analysis results
    """
    # Get embeddings
    emb_result = esm2_embedder.get_embedding(sequence, return_attention=True)

    # Per-residue analysis
    per_residue = emb_result["per_residue"]

    # Calculate per-residue "importance" as embedding magnitude
    residue_importance = np.linalg.norm(per_residue, axis=1)

    # Find most/least important regions
    top_indices = np.argsort(residue_importance)[-10:][::-1]
    bottom_indices = np.argsort(residue_importance)[:10]

    return {
        "protein_name": protein_name,
        "sequence_length": len(sequence),
        "embedding_dim": per_residue.shape[1],
        "sequence_embedding": emb_result["embedding"],
        "per_residue_embeddings": per_residue,
        "residue_importance": residue_importance,
        "most_important_positions": top_indices.tolist(),
        "least_important_positions": bottom_indices.tolist(),
        "mean_importance": float(np.mean(residue_importance)),
        "importance_std": float(np.std(residue_importance)),
    }


def scan_mutation_effects(
    esm2_embedder: ESM2Embedder,
    sequence: str,
    positions: Optional[list[int]] = None,
) -> dict:
    """
    Scan all possible single mutations at specified positions.

    Args:
        esm2_embedder: ESM2Embedder instance
        sequence: Wild-type amino acid sequence
        positions: List of positions to scan (None = all positions)

    Returns:
        Dictionary with mutation effect predictions
    """
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    if positions is None:
        positions = list(range(len(sequence)))

    results = []

    for pos in positions:
        original_aa = sequence[pos]

        for mut_aa in AMINO_ACIDS:
            if mut_aa == original_aa:
                continue

            # Create mutant sequence
            mutant = sequence[:pos] + mut_aa + sequence[pos + 1:]

            # Calculate effect
            effect = esm2_embedder.predict_mutation_effect(sequence, mutant, pos)

            results.append({
                "position": pos,
                "original": original_aa,
                "mutant": mut_aa,
                "mutation": f"{original_aa}{pos + 1}{mut_aa}",
                "embedding_distance": effect["embedding_distance"],
                "predicted_effect": effect["predicted_effect"],
            })

    return {
        "sequence_length": len(sequence),
        "positions_scanned": len(positions),
        "total_mutations": len(results),
        "mutations": results,
    }


class ESM2HyperbolicHybrid:
    """
    Hybrid model combining ESM-2 embeddings with hyperbolic geometry.

    This class provides a unified interface for analyzing HIV sequences
    using both learned (ESM-2) and geometric (hyperbolic) representations.
    """

    def __init__(
        self,
        esm2_config: Optional[ESM2Config] = None,
        hyperbolic_encoder=None,
    ):
        """
        Initialize hybrid model.

        Args:
            esm2_config: Configuration for ESM-2
            hyperbolic_encoder: Function or class for hyperbolic encoding
        """
        self.esm2_embedder = ESM2Embedder(esm2_config) if _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE else None
        self.hyperbolic_encoder = hyperbolic_encoder
        self._embedding_cache = {}

    def get_hybrid_embedding(
        self,
        sequence: str,
        codons: Optional[list[str]] = None,
    ) -> dict:
        """
        Get combined ESM-2 and hyperbolic embedding.

        Args:
            sequence: Amino acid sequence
            codons: List of codons (for hyperbolic encoding)

        Returns:
            Dictionary with both embedding types and combined embedding
        """
        result = {"sequence": sequence}

        # ESM-2 embedding
        if self.esm2_embedder:
            esm2_result = self.esm2_embedder.get_embedding(sequence)
            result["esm2_embedding"] = esm2_result["embedding"]
            result["esm2_per_residue"] = esm2_result["per_residue"]

        # Hyperbolic embedding
        if self.hyperbolic_encoder and codons:
            hyp_embeddings = [self.hyperbolic_encoder(c) for c in codons]
            result["hyperbolic_embeddings"] = np.array(hyp_embeddings)
            result["hyperbolic_mean"] = np.mean(hyp_embeddings, axis=0)

        # Combined embedding
        if "esm2_embedding" in result and "hyperbolic_mean" in result:
            result["combined_embedding"] = combine_esm2_with_hyperbolic(
                result["esm2_embedding"],
                result["hyperbolic_mean"],
                combination_method="concatenate",
            )

        return result

    def compare_sequences(
        self,
        seq1: str,
        seq2: str,
        codons1: Optional[list[str]] = None,
        codons2: Optional[list[str]] = None,
    ) -> dict:
        """
        Compare two sequences using both embedding types.

        Returns:
            Dictionary with distance metrics from both representations
        """
        emb1 = self.get_hybrid_embedding(seq1, codons1)
        emb2 = self.get_hybrid_embedding(seq2, codons2)

        result = {}

        if "esm2_embedding" in emb1 and "esm2_embedding" in emb2:
            result["esm2_distance"] = float(np.linalg.norm(
                emb1["esm2_embedding"] - emb2["esm2_embedding"]
            ))

        if "hyperbolic_mean" in emb1 and "hyperbolic_mean" in emb2:
            # V5.12.2: Use proper hyperbolic distance
            result["hyperbolic_distance"] = poincare_distance_np(
                emb1["hyperbolic_mean"], emb2["hyperbolic_mean"]
            )

        if "combined_embedding" in emb1 and "combined_embedding" in emb2:
            result["combined_distance"] = float(np.linalg.norm(
                emb1["combined_embedding"] - emb2["combined_embedding"]
            ))

        return result


def generate_esm2_report(analysis_results: dict) -> str:
    """
    Generate a formatted report from ESM-2 analysis.

    Args:
        analysis_results: Dictionary from analyze_hiv_protein()

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"ESM-2 PROTEIN ANALYSIS: {analysis_results['protein_name']}",
        "=" * 60,
        "",
        "SEQUENCE STATISTICS:",
        f"  Length: {analysis_results['sequence_length']} residues",
        f"  Embedding dimension: {analysis_results['embedding_dim']}",
        "",
        "IMPORTANCE ANALYSIS:",
        f"  Mean importance: {analysis_results['mean_importance']:.4f}",
        f"  Std deviation: {analysis_results['importance_std']:.4f}",
        "",
        "MOST IMPORTANT POSITIONS (top 10):",
    ]

    for pos in analysis_results["most_important_positions"]:
        imp = analysis_results["residue_importance"][pos]
        lines.append(f"  Position {pos + 1}: importance = {imp:.4f}")

    lines.extend([
        "",
        "LEAST IMPORTANT POSITIONS (bottom 10):",
    ])

    for pos in analysis_results["least_important_positions"]:
        imp = analysis_results["residue_importance"][pos]
        lines.append(f"  Position {pos + 1}: importance = {imp:.4f}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


# Mock embedder for testing without GPU/models
class MockESM2Embedder:
    """
    Mock ESM-2 embedder for testing without actual model.

    Generates random embeddings with consistent dimensions.
    """

    def __init__(self, hidden_dim: int = 320):
        self.hidden_dim = hidden_dim

    def get_embedding(self, sequence: str, return_attention: bool = False) -> dict:
        """Generate mock embedding."""
        np.random.seed(hash(sequence) % (2**32))

        seq_len = len(sequence)
        per_residue = np.random.randn(seq_len, self.hidden_dim).astype(np.float32)
        embedding = np.mean(per_residue, axis=0)

        result = {
            "embedding": embedding,
            "per_residue": per_residue,
            "sequence_length": seq_len,
        }

        if return_attention:
            result["attention"] = np.random.rand(seq_len, seq_len).astype(np.float32)

        return result

    def predict_mutation_effect(
        self,
        wild_type: str,
        mutant: str,
        position: Optional[int] = None,
    ) -> dict:
        """Generate mock mutation effect prediction."""
        wt_emb = self.get_embedding(wild_type)
        mut_emb = self.get_embedding(mutant)

        distance = np.linalg.norm(wt_emb["embedding"] - mut_emb["embedding"])

        return {
            "embedding_distance": float(distance),
            "per_residue_changes": None,
            "max_change_position": None,
            "max_change_magnitude": None,
            "estimated_fitness_effect": -0.1 * distance,
            "predicted_effect": (
                "neutral" if distance < 1.0 else
                "moderate" if distance < 2.0 else
                "severe"
            ),
        }


# Example usage
if __name__ == "__main__":
    print("Testing ESM-2 Integration Module")
    print("=" * 50)

    # Check dependencies
    print(f"\nPyTorch available: {_TORCH_AVAILABLE}")
    print(f"Transformers available: {_TRANSFORMERS_AVAILABLE}")

    # Test with mock embedder
    print("\nTesting with mock embedder...")
    mock_embedder = MockESM2Embedder(hidden_dim=320)

    # Example HIV protease sequence (first 50 residues)
    hiv_pr = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGI"

    # Get embedding
    result = mock_embedder.get_embedding(hiv_pr)
    print(f"\nSequence length: {result['sequence_length']}")
    print(f"Embedding shape: {result['embedding'].shape}")
    print(f"Per-residue shape: {result['per_residue'].shape}")

    # Test mutation effect
    mutant = hiv_pr[:29] + "N" + hiv_pr[30:]  # D30N mutation
    effect = mock_embedder.predict_mutation_effect(hiv_pr, mutant, position=29)
    print("\nMutation D30N effect:")
    print(f"  Embedding distance: {effect['embedding_distance']:.4f}")
    print(f"  Predicted effect: {effect['predicted_effect']}")

    # Test analysis
    analysis = analyze_hiv_protein(mock_embedder, hiv_pr, "HIV-1 Protease")
    print("\n" + generate_esm2_report(analysis))

    # Test hybrid model
    print("\nTesting hybrid model...")

    def mock_hyperbolic_encoder(codon):
        """Mock hyperbolic encoder."""
        np.random.seed(hash(codon) % (2**32))
        return np.random.randn(3).astype(np.float32)

    hybrid = ESM2HyperbolicHybrid(hyperbolic_encoder=mock_hyperbolic_encoder)
    hybrid.esm2_embedder = mock_embedder  # Use mock

    codons = ["CCT", "CAA", "ATC", "ACT"]  # First 4 codons
    hybrid_result = hybrid.get_hybrid_embedding(hiv_pr[:4], codons)

    if "combined_embedding" in hybrid_result:
        print(f"Combined embedding shape: {hybrid_result['combined_embedding'].shape}")

    # If real model is available, test it
    if _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 50)
        print("Testing with real ESM-2 model...")
        try:
            config = ESM2Config(model_name="facebook/esm2_t6_8M_UR50D")
            real_embedder = ESM2Embedder(config)
            real_embedder.load_model()

            result = real_embedder.get_embedding(hiv_pr[:50])
            print(f"Real embedding shape: {result['embedding'].shape}")

            effect = real_embedder.predict_mutation_effect(hiv_pr[:50], mutant[:50])
            print(f"Real mutation effect: {effect['predicted_effect']}")

        except Exception as e:
            print(f"Could not load real model: {e}")

    print("\n" + "=" * 50)
    print("Module testing complete!")
