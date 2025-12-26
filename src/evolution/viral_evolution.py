# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Viral Evolution Predictor using p-adic framework.

This module predicts viral escape mutations and evolutionary trajectories
using p-adic distance metrics and immune pressure modeling.

Key Concepts:
    - P-adic codon distances predict mutational accessibility
    - Immune pressure modeling (antibody binding sites, T-cell epitopes)
    - Fitness landscape based on codon usage and protein stability
    - Escape mutation prediction combining all factors

Applications:
    - Vaccine design: anticipate escape variants
    - Therapeutic development: identify resistant mutations
    - Surveillance: predict emerging variants

Research References:
    - RESEARCH_PROPOSALS/Autoimmunity_Codon_Adaptation/proposal.md
    - COMPREHENSIVE_RESEARCH_REPORT.md Section 2.8
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class SelectionType(Enum):
    """Types of evolutionary selection pressure."""

    POSITIVE = "positive"  # Favors change (immune escape)
    NEGATIVE = "negative"  # Disfavors change (functional constraint)
    NEUTRAL = "neutral"  # No selection pressure
    BALANCING = "balancing"  # Maintains diversity


# Amino acid physicochemical properties for mutation impact assessment
AMINO_ACID_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "charge": 0, "polarity": 0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "charge": 1, "polarity": 1},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "charge": 0, "polarity": 1},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "charge": -1, "polarity": 1},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "charge": 0, "polarity": 0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "charge": -1, "polarity": 1},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "charge": 0, "polarity": 1},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "charge": 0, "polarity": 0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "charge": 0.5, "polarity": 1},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "charge": 0, "polarity": 0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "charge": 0, "polarity": 0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "charge": 1, "polarity": 1},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "charge": 0, "polarity": 0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "charge": 0, "polarity": 0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "charge": 0, "polarity": 0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "charge": 0, "polarity": 1},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "charge": 0, "polarity": 1},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "charge": 0, "polarity": 0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "charge": 0, "polarity": 1},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "charge": 0, "polarity": 0},
}


@dataclass
class EscapeMutation:
    """Represents a predicted escape mutation."""

    position: int
    original_aa: str
    mutant_aa: str
    escape_score: float  # 0-1, probability of immune escape
    fitness_cost: float  # 0-1, cost to viral fitness
    padic_distance: float  # P-adic distance between codons
    selection_type: SelectionType = SelectionType.POSITIVE


@dataclass
class MutationHotspot:
    """Region with elevated mutation potential."""

    start: int
    end: int
    mutation_rate: float
    dominant_selection: SelectionType
    epitope_overlap: bool = False
    structural_importance: float = 0.0


@dataclass
class EscapePrediction:
    """Complete prediction result for a viral sequence."""

    mutations: List[EscapeMutation]
    hotspots: List[MutationHotspot]
    overall_escape_risk: float
    trajectory_confidence: float
    key_findings: List[str] = field(default_factory=list)


@dataclass
class EvolutionaryPressure:
    """Evolutionary pressure at a specific site."""

    position: int
    dn_ds_ratio: float  # dN/dS ratio (>1 = positive selection)
    immune_pressure: float  # 0-1
    structural_constraint: float  # 0-1
    codon_accessibility: float  # How many mutations are accessible


class ViralEvolutionPredictor(nn.Module):
    """Predict viral evolution and escape mutations using p-adic framework.

    Combines p-adic codon distances with immune pressure modeling to
    predict likely escape mutations and evolutionary trajectories.
    """

    def __init__(
        self,
        p: int = 3,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        n_epitope_types: int = 3,
    ):
        """Initialize evolution predictor.

        Args:
            p: Prime for p-adic metric (3 for ternary/codon)
            latent_dim: Dimension for sequence embeddings
            hidden_dim: Hidden layer dimension
            n_epitope_types: Number of epitope categories (B-cell, CD4, CD8)
        """
        super().__init__()
        self.p = p
        self.latent_dim = latent_dim
        self.n_epitope_types = n_epitope_types

        # Codon embedding for p-adic distance computation
        self.codon_embedding = nn.Embedding(64, latent_dim)

        # Epitope pressure encoder
        self.epitope_encoder = nn.Sequential(
            nn.Linear(n_epitope_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Mutation predictor head
        self.mutation_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),  # 20 amino acids
            nn.Softmax(dim=-1),
        )

        # Escape score predictor
        self.escape_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Fitness predictor
        self.fitness_head = nn.Sequential(
            nn.Linear(latent_dim + 20, hidden_dim),  # codon + AA properties
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def compute_padic_distance(self, codon1: int, codon2: int) -> float:
        """Compute p-adic distance between two codon indices.

        Args:
            codon1: First codon index (0-63)
            codon2: Second codon index (0-63)

        Returns:
            P-adic distance
        """
        if codon1 == codon2:
            return 0.0

        diff = abs(codon1 - codon2)
        if diff == 0:
            return 0.0

        # Find p-adic valuation (highest power of p dividing diff)
        v = 0
        while diff % self.p == 0:
            diff //= self.p
            v += 1

        return float(self.p ** (-v))

    def compute_mutation_accessibility(self, codon_idx: int) -> Dict[int, float]:
        """Compute accessibility of all possible mutations from a codon.

        Args:
            codon_idx: Source codon index

        Returns:
            Dictionary mapping target codon indices to accessibility scores
        """
        accessibility = {}

        for target in range(64):
            if target != codon_idx:
                # P-adic distance gives accessibility (lower = more accessible)
                dist = self.compute_padic_distance(codon_idx, target)
                # Invert: high distance = low accessibility
                accessibility[target] = 1.0 / (1.0 + dist * 10)

        return accessibility

    def encode_epitope_pressure(
        self,
        b_cell_epitopes: torch.Tensor,
        cd4_epitopes: torch.Tensor,
        cd8_epitopes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode immune pressure from epitope annotations.

        Args:
            b_cell_epitopes: B-cell epitope scores (B, L)
            cd4_epitopes: CD4+ T-cell epitope scores (B, L)
            cd8_epitopes: CD8+ T-cell epitope scores (B, L)

        Returns:
            Encoded pressure (B, L, latent_dim)
        """
        # Stack epitope types
        pressure = torch.stack([b_cell_epitopes, cd4_epitopes, cd8_epitopes], dim=-1)

        # Encode
        return self.epitope_encoder(pressure)

    def predict_mutation_probabilities(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Predict probability distribution over mutations at each position.

        Args:
            codon_indices: Current codon sequence (B, L)
            epitope_pressure: Encoded immune pressure (B, L, latent_dim)

        Returns:
            Mutation probabilities (B, L, 20)
        """
        # Embed current codons
        codon_emb = self.codon_embedding(codon_indices)  # (B, L, latent_dim)

        # Combine with pressure
        combined = torch.cat([codon_emb, epitope_pressure], dim=-1)

        # Predict mutations
        return self.mutation_head(combined)

    def compute_escape_scores(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Compute escape probability at each position.

        Args:
            codon_indices: Current codon sequence (B, L)
            epitope_pressure: Encoded immune pressure (B, L, latent_dim)

        Returns:
            Escape scores (B, L)
        """
        # Embed codons
        codon_emb = self.codon_embedding(codon_indices)

        # Add pressure contribution
        combined = codon_emb + epitope_pressure

        return self.escape_head(combined).squeeze(-1)

    def compute_fitness_landscape(
        self,
        codon_indices: torch.Tensor,
        aa_properties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute fitness at each position for all possible mutations.

        Args:
            codon_indices: Current codon sequence (B, L)
            aa_properties: Optional amino acid property tensor

        Returns:
            Fitness scores (B, L)
        """
        # Embed codons
        codon_emb = self.codon_embedding(codon_indices)

        # Default AA properties if not provided
        if aa_properties is None:
            aa_properties = torch.zeros(
                codon_indices.shape[0], codon_indices.shape[1], 20,
                device=codon_indices.device
            )

        combined = torch.cat([codon_emb, aa_properties], dim=-1)

        return self.fitness_head(combined).squeeze(-1)

    def forward(
        self,
        codon_indices: torch.Tensor,
        epitope_annotations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for evolution prediction.

        Args:
            codon_indices: Codon sequence (B, L)
            epitope_annotations: Optional dict with b_cell, cd4, cd8 tensors

        Returns:
            Dictionary with all predictions
        """
        B, L = codon_indices.shape
        device = codon_indices.device

        # Default epitope pressure if not provided
        if epitope_annotations is None:
            b_cell = torch.zeros(B, L, device=device)
            cd4 = torch.zeros(B, L, device=device)
            cd8 = torch.zeros(B, L, device=device)
        else:
            b_cell = epitope_annotations.get("b_cell", torch.zeros(B, L, device=device))
            cd4 = epitope_annotations.get("cd4", torch.zeros(B, L, device=device))
            cd8 = epitope_annotations.get("cd8", torch.zeros(B, L, device=device))

        # Encode epitope pressure
        epitope_pressure = self.encode_epitope_pressure(b_cell, cd4, cd8)

        # Predictions
        mutation_probs = self.predict_mutation_probabilities(codon_indices, epitope_pressure)
        escape_scores = self.compute_escape_scores(codon_indices, epitope_pressure)
        fitness_scores = self.compute_fitness_landscape(codon_indices)

        # Combined escape-fitness score
        combined_scores = escape_scores * (1.0 - fitness_scores * 0.5)  # Escape penalized by fitness cost

        return {
            "mutation_probabilities": mutation_probs,
            "escape_scores": escape_scores,
            "fitness_scores": fitness_scores,
            "combined_scores": combined_scores,
            "epitope_pressure": epitope_pressure,
        }

    def identify_hotspots(
        self,
        escape_scores: torch.Tensor,
        threshold: float = 0.5,
        min_length: int = 3,
    ) -> List[MutationHotspot]:
        """Identify mutation hotspot regions from escape scores.

        Args:
            escape_scores: Escape scores per position (L,)
            threshold: Minimum score to be considered hotspot
            min_length: Minimum length of hotspot region

        Returns:
            List of identified hotspots
        """
        if isinstance(escape_scores, torch.Tensor):
            scores = escape_scores.detach().cpu().numpy()
        else:
            scores = np.array(escape_scores)

        hotspots = []
        in_hotspot = False
        start = 0

        for i, score in enumerate(scores):
            if score >= threshold:
                if not in_hotspot:
                    start = i
                    in_hotspot = True
            else:
                if in_hotspot:
                    if i - start >= min_length:
                        hotspots.append(
                            MutationHotspot(
                                start=start,
                                end=i,
                                mutation_rate=float(np.mean(scores[start:i])),
                                dominant_selection=SelectionType.POSITIVE,
                                epitope_overlap=True,
                            )
                        )
                    in_hotspot = False

        # Check if sequence ends in hotspot
        if in_hotspot and len(scores) - start >= min_length:
            hotspots.append(
                MutationHotspot(
                    start=start,
                    end=len(scores),
                    mutation_rate=float(np.mean(scores[start:])),
                    dominant_selection=SelectionType.POSITIVE,
                    epitope_overlap=True,
                )
            )

        return hotspots

    def predict_escape_mutations(
        self,
        codon_indices: torch.Tensor,
        epitope_annotations: Optional[Dict[str, torch.Tensor]] = None,
        top_k: int = 10,
    ) -> EscapePrediction:
        """Generate complete escape mutation predictions.

        Args:
            codon_indices: Codon sequence (L,) or (1, L)
            epitope_annotations: Optional epitope annotations
            top_k: Number of top mutations to return

        Returns:
            Complete prediction with mutations and hotspots
        """
        if codon_indices.dim() == 1:
            codon_indices = codon_indices.unsqueeze(0)

        # Forward pass
        results = self.forward(codon_indices, epitope_annotations)

        escape_scores = results["escape_scores"][0]
        mutation_probs = results["mutation_probabilities"][0]
        fitness_scores = results["fitness_scores"][0]

        # Find top escape positions
        top_positions = torch.topk(escape_scores, min(top_k, len(escape_scores)))

        mutations = []
        aa_list = list("ARNDCEQGHILKMFPSTWYV")

        for idx, (score, pos) in enumerate(zip(top_positions.values, top_positions.indices)):
            pos = pos.item()
            original_codon = codon_indices[0, pos].item()

            # Find most likely mutation
            mut_probs = mutation_probs[pos]
            best_mut_idx = torch.argmax(mut_probs).item()

            mutations.append(
                EscapeMutation(
                    position=pos,
                    original_aa=aa_list[int(original_codon) % 20],  # Simplified mapping
                    mutant_aa=aa_list[int(best_mut_idx)],
                    escape_score=score.item(),
                    fitness_cost=1.0 - fitness_scores[pos].item(),
                    padic_distance=0.33,  # Placeholder
                    selection_type=SelectionType.POSITIVE,
                )
            )

        # Identify hotspots
        hotspots = self.identify_hotspots(escape_scores)

        # Overall metrics
        overall_escape = escape_scores.mean().item()
        trajectory_confidence = 1.0 - escape_scores.std().item()

        # Key findings
        findings = []
        if overall_escape > 0.5:
            findings.append("High overall escape risk detected")
        if len(hotspots) > 3:
            findings.append(f"Multiple mutation hotspots identified ({len(hotspots)})")
        if mutations and mutations[0].escape_score > 0.8:
            findings.append(f"Critical escape site at position {mutations[0].position}")

        return EscapePrediction(
            mutations=mutations,
            hotspots=hotspots,
            overall_escape_risk=overall_escape,
            trajectory_confidence=trajectory_confidence,
            key_findings=findings,
        )

    def compute_evolutionary_pressure(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> List[EvolutionaryPressure]:
        """Compute evolutionary pressure at each position.

        Args:
            codon_indices: Codon sequence (L,)
            epitope_pressure: Encoded epitope pressure (L, latent_dim)

        Returns:
            List of evolutionary pressure metrics per position
        """
        pressures = []

        for pos in range(len(codon_indices)):
            codon = int(codon_indices[pos].item())

            # Compute codon accessibility
            accessibility = self.compute_mutation_accessibility(codon)
            mean_accessibility = float(np.mean(list(accessibility.values())))

            # Immune pressure from epitope encoding
            immune = torch.norm(epitope_pressure[pos]).item()

            # Estimate dN/dS (simplified)
            # High immune pressure + high accessibility = positive selection
            dn_ds = 1.0 + (immune * mean_accessibility)

            pressures.append(
                EvolutionaryPressure(
                    position=pos,
                    dn_ds_ratio=dn_ds,
                    immune_pressure=min(immune, 1.0),
                    structural_constraint=0.5,  # Default
                    codon_accessibility=mean_accessibility,
                )
            )

        return pressures
