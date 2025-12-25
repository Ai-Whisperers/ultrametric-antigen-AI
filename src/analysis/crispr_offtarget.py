# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CRISPR Off-Target Landscape Analysis using Hyperbolic Geometry.

This module implements analysis of CRISPR-Cas9 guide RNA (gRNA) off-target
effects using p-adic distances and hyperbolic embeddings. The key insight
is that sequence similarity for CRISPR targeting follows ultrametric
patterns that can be naturally modeled in hyperbolic space.

Key concepts:
- Guide RNA (gRNA): 20nt sequence that directs Cas9 to target DNA
- PAM (Protospacer Adjacent Motif): NGG sequence required for Cas9 binding
- Off-target effects: Unintended cleavage at similar sequences
- Safety radius: Hyperbolic distance within which off-targets are likely

The p-adic framework captures:
- Position-dependent mismatch tolerance (seed region vs non-seed)
- Hierarchical nature of sequence similarity
- Ultrametric clustering of off-target sites

References:
- Clockwork integration ideas (Idea #1: CRISPR off-target landscapes)
- Hsu et al. (2013) DNA targeting specificity of RNA-guided Cas9 nucleases
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import project_to_poincare as geometry_project_to_poincare


class MismatchType(Enum):
    """Types of mismatches in CRISPR targeting."""

    MATCH = "match"
    TRANSITION = "transition"  # Purine-purine or pyrimidine-pyrimidine
    TRANSVERSION = "transversion"  # Purine-pyrimidine
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass
class OffTargetSite:
    """Represents a potential off-target site."""

    sequence: str  # 20nt target sequence
    chromosome: str
    position: int
    strand: str  # + or -
    pam: str  # PAM sequence (typically NGG)
    mismatches: list[tuple[int, str, str]]  # (position, ref, alt)
    mismatch_count: int
    seed_mismatches: int  # Mismatches in seed region (positions 1-12)
    padic_distance: float
    hyperbolic_distance: float
    predicted_activity: float  # 0-1 probability of cleavage


@dataclass
class GuideSafetyProfile:
    """Safety profile for a guide RNA."""

    guide_sequence: str
    total_offtargets: int
    high_risk_offtargets: int  # Activity > 0.1
    seed_region_offtargets: int
    min_padic_distance: float
    safety_radius: float  # Hyperbolic safety radius
    specificity_score: float  # 0-1, higher is safer
    recommended: bool


# Nucleotide encoding
NUCLEOTIDE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
IDX_TO_NUCLEOTIDE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

# PAM sequences for different Cas variants
PAM_SEQUENCES = {
    "SpCas9": "NGG",
    "SaCas9": "NNGRRT",
    "Cas12a": "TTTV",
    "xCas9": "NG",
}

# Position weights for mismatch tolerance (empirical from literature)
# Seed region (1-12 from PAM) is more critical
POSITION_WEIGHTS = torch.tensor([
    0.1, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # Non-seed (1-10)
    0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # Seed (11-20)
])


def encode_sequence(sequence: str) -> torch.Tensor:
    """Encode DNA sequence to tensor.

    Args:
        sequence: DNA sequence string

    Returns:
        Tensor of nucleotide indices
    """
    indices = [NUCLEOTIDE_TO_IDX.get(nt.upper(), 4) for nt in sequence]
    return torch.tensor(indices)


def sequence_to_onehot(sequence: str) -> torch.Tensor:
    """Convert sequence to one-hot encoding.

    Args:
        sequence: DNA sequence string

    Returns:
        One-hot tensor of shape (seq_len, 4)
    """
    indices = encode_sequence(sequence)
    onehot = F.one_hot(indices.clamp(0, 3), num_classes=4).float()
    return onehot


class PAdicSequenceDistance:
    """Computes p-adic distances between DNA sequences.

    Uses position-weighted p-adic norm to capture the
    hierarchical importance of different sequence positions.
    """

    def __init__(self, p: int = 3, seed_start: int = 12):
        """Initialize distance calculator.

        Args:
            p: Prime for p-adic calculations
            seed_start: Position where seed region begins (from PAM)
        """
        self.p = p
        self.seed_start = seed_start

    def mismatch_positions(
        self,
        seq1: str,
        seq2: str,
    ) -> list[int]:
        """Find mismatch positions between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            List of mismatch positions (0-indexed)
        """
        seq1 = seq1.upper()
        seq2 = seq2.upper()
        return [i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b]

    def padic_valuation(self, position: int) -> int:
        """Compute p-adic valuation for a position.

        Higher valuation = position is more divisible by p = less critical.
        Lower valuation = position is less divisible = more critical.

        Args:
            position: Sequence position (1-indexed)

        Returns:
            p-adic valuation
        """
        if position == 0:
            return 100  # Infinity for position 0

        v = 0
        n = position
        while n % self.p == 0:
            v += 1
            n //= self.p
        return v

    def compute_distance(
        self,
        target: str,
        offtarget: str,
        position_weighted: bool = True,
    ) -> float:
        """Compute p-adic distance between target and off-target.

        Args:
            target: Target sequence (guide RNA target)
            offtarget: Potential off-target sequence
            position_weighted: Whether to weight by position importance

        Returns:
            p-adic distance (0 = identical, higher = more different)
        """
        mismatches = self.mismatch_positions(target, offtarget)

        if not mismatches:
            return 0.0

        # Compute p-adic contribution from each mismatch
        total_distance = 0.0
        for pos in mismatches:
            # Base p-adic contribution
            v = self.padic_valuation(pos + 1)  # 1-indexed
            padic_contrib = self.p ** (-v)

            # Apply position weight if requested
            if position_weighted and pos < len(POSITION_WEIGHTS):
                padic_contrib *= POSITION_WEIGHTS[pos].item()

            total_distance += padic_contrib

        return total_distance


class HyperbolicOfftargetEmbedder(nn.Module):
    """Embeds CRISPR target/off-target pairs in hyperbolic space.

    The hyperbolic embedding captures:
    - Distance from guide = off-target risk
    - Radial position = overall specificity
    - Angular position = mismatch pattern
    """

    def __init__(
        self,
        seq_len: int = 20,
        embedding_dim: int = 64,
        curvature: float = 1.0,
        max_norm: float = 0.95,
    ):
        """Initialize embedder.

        Args:
            seq_len: Length of guide sequence (typically 20)
            embedding_dim: Dimension of hyperbolic embedding
            curvature: Hyperbolic curvature parameter
            max_norm: Maximum norm for Poincaré ball
        """
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_norm = max_norm

        # Nucleotide embedding
        self.nt_embedding = nn.Embedding(5, 16)  # 4 nucleotides + N

        # Position encoding
        self.pos_embedding = nn.Embedding(seq_len, 16)

        # Sequence encoder
        self.seq_encoder = nn.Sequential(
            nn.Linear(32 * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        # Mismatch-aware attention
        self.mismatch_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True,
        )

    def project_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball.

        Delegates to geometry module for single source of truth.

        Args:
            x: Input tensor

        Returns:
            Projected tensor with norm < max_norm
        """
        return geometry_project_to_poincare(x, max_norm=self.max_norm, c=self.curvature)

    def encode_sequence(self, sequences: list[str]) -> torch.Tensor:
        """Encode multiple sequences.

        Args:
            sequences: List of DNA sequences

        Returns:
            Encoded tensor (batch, seq_len, 32)
        """
        batch_size = len(sequences)

        # Encode nucleotides
        nt_indices = torch.stack([encode_sequence(s) for s in sequences])
        nt_emb = self.nt_embedding(nt_indices)  # (batch, seq_len, 16)

        # Add position embeddings
        positions = torch.arange(self.seq_len, device=nt_emb.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        # Combine
        combined = torch.cat([nt_emb, pos_emb], dim=-1)  # (batch, seq_len, 32)

        return combined

    def forward(
        self,
        target_sequences: list[str],
        offtarget_sequences: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Embed sequences in hyperbolic space.

        Args:
            target_sequences: List of guide RNA targets
            offtarget_sequences: Optional list of off-target sequences

        Returns:
            Dictionary with embeddings and distances
        """
        # Encode targets
        target_encoded = self.encode_sequence(target_sequences)

        # Apply attention
        target_attended, _ = self.mismatch_attention(
            target_encoded, target_encoded, target_encoded
        )

        # Flatten and project
        target_flat = target_attended.flatten(start_dim=1)
        target_emb = self.seq_encoder(target_flat)
        target_emb = self.project_to_poincare(target_emb)

        result = {"target_embeddings": target_emb}

        # If off-targets provided, compute relative embeddings
        if offtarget_sequences is not None:
            offtarget_encoded = self.encode_sequence(offtarget_sequences)

            # Cross-attention with target
            offtarget_attended, attn_weights = self.mismatch_attention(
                offtarget_encoded, target_encoded, target_encoded
            )

            offtarget_flat = offtarget_attended.flatten(start_dim=1)
            offtarget_emb = self.seq_encoder(offtarget_flat)
            offtarget_emb = self.project_to_poincare(offtarget_emb)

            result["offtarget_embeddings"] = offtarget_emb
            result["attention_weights"] = attn_weights

            # Compute hyperbolic distances
            diff = target_emb - offtarget_emb
            euclidean_dist = torch.norm(diff, dim=-1)

            # Poincaré distance approximation
            target_norm = torch.norm(target_emb, dim=-1)
            offtarget_norm = torch.norm(offtarget_emb, dim=-1)
            hyperbolic_dist = torch.acosh(
                1 + 2 * euclidean_dist ** 2 /
                ((1 - target_norm ** 2) * (1 - offtarget_norm ** 2) + 1e-8)
            )

            result["hyperbolic_distances"] = hyperbolic_dist

        return result


class OfftargetActivityPredictor(nn.Module):
    """Predicts cleavage activity at off-target sites.

    Uses learned embeddings and mismatch patterns to predict
    the probability of Cas9 cleavage at a given off-target.
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_dim: int = 128,
    ):
        """Initialize predictor.

        Args:
            seq_len: Length of guide sequence
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.seq_len = seq_len

        # Mismatch encoding (one-hot for each position)
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(seq_len * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Sequence similarity features
        self.similarity_encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim // 2),
            nn.ReLU(),
        )

        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_mismatch_features(
        self,
        target: str,
        offtarget: str,
    ) -> torch.Tensor:
        """Compute mismatch feature vector.

        Args:
            target: Target sequence
            offtarget: Off-target sequence

        Returns:
            Mismatch features (seq_len * 4,)
        """
        features = torch.zeros(self.seq_len, 4)

        for i, (t, o) in enumerate(zip(target.upper(), offtarget.upper())):
            if t == o:
                features[i, 0] = 1.0  # Match
            elif (t in "AG" and o in "AG") or (t in "CT" and o in "CT"):
                features[i, 1] = 1.0  # Transition
            else:
                features[i, 2] = 1.0  # Transversion

        return features.flatten()

    def forward(
        self,
        targets: list[str],
        offtargets: list[str],
    ) -> torch.Tensor:
        """Predict off-target activity.

        Args:
            targets: List of target sequences
            offtargets: List of corresponding off-target sequences

        Returns:
            Predicted activity probabilities (batch,)
        """
        batch_size = len(targets)

        # Compute mismatch features
        mismatch_features = torch.stack([
            self.compute_mismatch_features(t, o)
            for t, o in zip(targets, offtargets)
        ])

        # Compute similarity (1 - mismatch fraction)
        similarity = torch.zeros(batch_size, self.seq_len)
        for i, (t, o) in enumerate(zip(targets, offtargets)):
            for j, (nt1, nt2) in enumerate(zip(t.upper(), o.upper())):
                similarity[i, j] = 1.0 if nt1 == nt2 else 0.0

        # Encode features
        mismatch_encoded = self.mismatch_encoder(mismatch_features)
        similarity_encoded = self.similarity_encoder(similarity)

        # Combine and predict
        combined = torch.cat([mismatch_encoded, similarity_encoded], dim=-1)
        activity = self.predictor(combined).squeeze(-1)

        return activity


class CRISPROfftargetAnalyzer:
    """Complete CRISPR off-target analysis pipeline.

    Combines p-adic distance computation, hyperbolic embedding,
    and activity prediction for comprehensive off-target analysis.
    """

    def __init__(
        self,
        p: int = 3,
        embedding_dim: int = 64,
        activity_threshold: float = 0.1,
    ):
        """Initialize analyzer.

        Args:
            p: Prime for p-adic calculations
            embedding_dim: Dimension for hyperbolic embeddings
            activity_threshold: Threshold for high-risk off-targets
        """
        self.p = p
        self.activity_threshold = activity_threshold

        self.padic_distance = PAdicSequenceDistance(p=p)
        self.embedder = HyperbolicOfftargetEmbedder(embedding_dim=embedding_dim)
        self.activity_predictor = OfftargetActivityPredictor()

    def find_mismatches(
        self,
        target: str,
        offtarget: str,
    ) -> list[tuple[int, str, str]]:
        """Find all mismatches between sequences.

        Args:
            target: Target sequence
            offtarget: Off-target sequence

        Returns:
            List of (position, target_nt, offtarget_nt)
        """
        mismatches = []
        for i, (t, o) in enumerate(zip(target.upper(), offtarget.upper())):
            if t != o:
                mismatches.append((i, t, o))
        return mismatches

    def analyze_offtarget(
        self,
        guide: str,
        offtarget_seq: str,
        chromosome: str = "unknown",
        position: int = 0,
        strand: str = "+",
        pam: str = "NGG",
    ) -> OffTargetSite:
        """Analyze a single off-target site.

        Args:
            guide: Guide RNA sequence (20nt)
            offtarget_seq: Off-target DNA sequence
            chromosome: Chromosome name
            position: Genomic position
            strand: Strand (+ or -)
            pam: PAM sequence

        Returns:
            OffTargetSite with analysis results
        """
        # Find mismatches
        mismatches = self.find_mismatches(guide, offtarget_seq)
        seed_mismatches = sum(1 for pos, _, _ in mismatches if pos >= 12)

        # Compute p-adic distance
        padic_dist = self.padic_distance.compute_distance(guide, offtarget_seq)

        # Compute hyperbolic distance
        with torch.no_grad():
            result = self.embedder([guide], [offtarget_seq])
            hyper_dist = result["hyperbolic_distances"][0].item()

        # Predict activity
        with torch.no_grad():
            activity = self.activity_predictor([guide], [offtarget_seq])
            predicted_activity = activity[0].item()

        return OffTargetSite(
            sequence=offtarget_seq,
            chromosome=chromosome,
            position=position,
            strand=strand,
            pam=pam,
            mismatches=mismatches,
            mismatch_count=len(mismatches),
            seed_mismatches=seed_mismatches,
            padic_distance=padic_dist,
            hyperbolic_distance=hyper_dist,
            predicted_activity=predicted_activity,
        )

    def analyze_guide(
        self,
        guide: str,
        offtarget_sequences: list[tuple[str, dict[str, Any]]],
    ) -> GuideSafetyProfile:
        """Analyze safety profile for a guide RNA.

        Args:
            guide: Guide RNA sequence
            offtarget_sequences: List of (sequence, metadata) tuples

        Returns:
            GuideSafetyProfile with safety analysis
        """
        offtargets = []

        for seq, metadata in offtarget_sequences:
            site = self.analyze_offtarget(
                guide,
                seq,
                chromosome=metadata.get("chromosome", "unknown"),
                position=metadata.get("position", 0),
                strand=metadata.get("strand", "+"),
                pam=metadata.get("pam", "NGG"),
            )
            offtargets.append(site)

        # Compute summary statistics
        high_risk = [o for o in offtargets if o.predicted_activity > self.activity_threshold]
        seed_hits = [o for o in offtargets if o.seed_mismatches <= 1]
        min_padic = min((o.padic_distance for o in offtargets), default=float("inf"))

        # Compute safety radius (minimum hyperbolic distance to high-risk site)
        if high_risk:
            safety_radius = min(o.hyperbolic_distance for o in high_risk)
        else:
            safety_radius = float("inf")

        # Compute specificity score
        if offtargets:
            avg_distance = sum(o.padic_distance for o in offtargets) / len(offtargets)
            high_risk_ratio = len(high_risk) / len(offtargets)
            specificity = (1 - high_risk_ratio) * min(1.0, avg_distance)
        else:
            specificity = 1.0

        return GuideSafetyProfile(
            guide_sequence=guide,
            total_offtargets=len(offtargets),
            high_risk_offtargets=len(high_risk),
            seed_region_offtargets=len(seed_hits),
            min_padic_distance=min_padic,
            safety_radius=safety_radius,
            specificity_score=specificity,
            recommended=specificity > 0.7 and len(high_risk) < 3,
        )

    def compare_guides(
        self,
        guides: list[str],
        offtarget_db: dict[str, list[tuple[str, dict[str, Any]]]],
    ) -> list[GuideSafetyProfile]:
        """Compare safety profiles of multiple guides.

        Args:
            guides: List of guide sequences
            offtarget_db: Dictionary mapping guide -> off-target list

        Returns:
            List of GuideSafetyProfile sorted by specificity
        """
        profiles = []
        for guide in guides:
            offtargets = offtarget_db.get(guide, [])
            profile = self.analyze_guide(guide, offtargets)
            profiles.append(profile)

        # Sort by specificity (highest first)
        profiles.sort(key=lambda p: p.specificity_score, reverse=True)
        return profiles

    def compute_landscape(
        self,
        guide: str,
        offtarget_sequences: list[str],
    ) -> dict[str, torch.Tensor]:
        """Compute hyperbolic landscape of off-targets.

        Args:
            guide: Guide RNA sequence
            offtarget_sequences: List of off-target sequences

        Returns:
            Dictionary with embeddings and distance matrices
        """
        # Get embeddings
        with torch.no_grad():
            target_result = self.embedder([guide])
            target_emb = target_result["target_embeddings"]

            offtarget_result = self.embedder(
                [guide] * len(offtarget_sequences),
                offtarget_sequences,
            )
            offtarget_embs = offtarget_result["offtarget_embeddings"]

        # Compute pairwise distances
        n = len(offtarget_sequences)
        padic_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                padic_matrix[i, j] = self.padic_distance.compute_distance(
                    offtarget_sequences[i], offtarget_sequences[j]
                )

        return {
            "target_embedding": target_emb,
            "offtarget_embeddings": offtarget_embs,
            "padic_distance_matrix": padic_matrix,
            "distances_to_target": offtarget_result["hyperbolic_distances"],
        }


class GuideDesignOptimizer:
    """Optimizes guide RNA design for minimal off-target effects.

    Uses p-adic and hyperbolic analysis to suggest modifications
    that improve specificity.
    """

    def __init__(self, analyzer: CRISPROfftargetAnalyzer | None = None):
        """Initialize optimizer.

        Args:
            analyzer: CRISPROfftargetAnalyzer instance
        """
        self.analyzer = analyzer or CRISPROfftargetAnalyzer()

    def suggest_modifications(
        self,
        guide: str,
        problematic_offtargets: list[OffTargetSite],
    ) -> list[tuple[int, str, str, float]]:
        """Suggest nucleotide modifications to reduce off-target binding.

        Args:
            guide: Original guide sequence
            problematic_offtargets: High-risk off-target sites

        Returns:
            List of (position, original_nt, suggested_nt, improvement_score)
        """
        suggestions = []

        # Analyze which positions are shared across problematic off-targets
        position_counts = {}
        for site in problematic_offtargets:
            for pos, _, _ in site.mismatches:
                position_counts[pos] = position_counts.get(pos, 0) + 1

        # Find positions that could be modified
        for pos in range(len(guide)):
            original_nt = guide[pos]

            # Skip positions that are already mismatched with most off-targets
            if position_counts.get(pos, 0) > len(problematic_offtargets) // 2:
                continue

            # Try alternative nucleotides
            for alt_nt in "ACGT":
                if alt_nt == original_nt:
                    continue

                # Compute improvement score
                # (simplified - would need full re-analysis in practice)
                seed_bonus = 2.0 if pos >= 12 else 1.0
                mismatch_penalty = position_counts.get(pos, 0) / len(problematic_offtargets)
                improvement = seed_bonus * (1 - mismatch_penalty)

                if improvement > 0.5:
                    suggestions.append((pos, original_nt, alt_nt, improvement))

        # Sort by improvement score
        suggestions.sort(key=lambda x: x[3], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions

    def generate_variants(
        self,
        guide: str,
        n_variants: int = 10,
    ) -> list[str]:
        """Generate variant guides with potential improved specificity.

        Args:
            guide: Original guide sequence
            n_variants: Number of variants to generate

        Returns:
            List of variant guide sequences
        """
        variants = [guide]
        guide_list = list(guide)

        # Generate single-nucleotide variants
        for pos in range(len(guide)):
            for alt_nt in "ACGT":
                if alt_nt != guide[pos]:
                    variant = guide_list.copy()
                    variant[pos] = alt_nt
                    variants.append("".join(variant))

                    if len(variants) >= n_variants:
                        return variants

        return variants[:n_variants]
