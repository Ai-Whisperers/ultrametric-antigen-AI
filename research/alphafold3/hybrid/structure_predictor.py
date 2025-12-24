# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hybrid Structure Predictor: Ternary VAE + PDB Structures.

Combines our ternary VAE model's geometric analysis with PDB structural
context for HIV integrase reveal mutation validation.

This replaces AlphaFold3's 630GB setup with:
- Our ternary VAE (v5_11_11_production/best.pt)
- BioPython for PDB parsing
- ~100MB of HIV integrase structures
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .pdb_analyzer import PDBAnalyzer, HAS_BIOPYTHON


# Standard genetic code for codon analysis
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


class HybridStructurePredictor:
    """Hybrid predictor combining ternary VAE with PDB structural context.

    This class provides reveal mutation analysis without requiring
    AlphaFold3's massive computational infrastructure.

    Architecture:
        1. Ternary VAE (v5_11_11_production/best.pt) - Codon geometry
        2. BioPython PDB Parser - Structural context
        3. AF3 residue constants - Amino acid properties

    Usage:
        predictor = HybridStructurePredictor()
        result = predictor.predict_reveal_effect(
            wt_sequence="FLDGIDKAQE...",
            mutation="E166K",
            pdb_id="2B4J"
        )
    """

    # Default reference PDB for HIV-1 Integrase-LEDGF complex
    DEFAULT_PDB = "2B4J"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pdb_dir: Optional[Path] = None,
        device: str = "cpu"
    ):
        """Initialize the hybrid predictor.

        Args:
            model_path: Path to ternary VAE checkpoint
            pdb_dir: Directory containing PDB structures
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.device = device

        # PDB analyzer is optional - works without BioPython for basic functions
        if HAS_BIOPYTHON:
            self.pdb_analyzer = PDBAnalyzer(pdb_dir)
        else:
            self.pdb_analyzer = None
            print("Note: BioPython not available - structural context disabled")

        # Model path defaults
        if model_path is None:
            model_path = project_root / "sandbox-training" / "checkpoints" / "v5_11_11_production" / "best.pt"

        self.model_path = model_path
        self.model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazy load the ternary VAE model."""
        if self._model_loaded:
            return

        if not HAS_TORCH:
            print("Warning: PyTorch not available, using fallback distance calculation")
            return

        if not self.model_path.exists():
            print(f"Warning: Model not found at {self.model_path}")
            print("Using fallback codon distance calculation")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            # Extract model if wrapped in checkpoint dict
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            else:
                self.model = checkpoint
            self._model_loaded = True
            print(f"Loaded ternary model from {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Using fallback codon distance calculation")

    def _codon_to_ternary(self, codon: str) -> np.ndarray:
        """Convert codon to ternary representation.

        Each nucleotide maps to ternary digits:
        T/U -> 0, C -> 1, A -> 2, G -> 3 (mod 3)

        Args:
            codon: 3-letter DNA/RNA codon

        Returns:
            9-dimensional ternary vector (3 positions x 3 digits each)
        """
        nuc_to_val = {'T': 0, 'U': 0, 'C': 1, 'A': 2, 'G': 0}  # G maps to 0 (mod 3)
        result = np.zeros(9, dtype=np.float32)

        for i, nuc in enumerate(codon.upper()):
            if nuc in nuc_to_val:
                val = nuc_to_val[nuc]
                # One-hot encode the ternary value
                result[i * 3 + (val % 3)] = 1.0

        return result

    def _fallback_codon_distance(self, codon1: str, codon2: str) -> float:
        """Calculate p-adic-inspired distance between codons.

        This is a fallback when the model is not available.
        Uses Hamming distance weighted by position.

        Args:
            codon1: First codon
            codon2: Second codon

        Returns:
            Distance value (higher = more different)
        """
        # Position weights: 3rd position most variable, 1st least
        weights = [3.0, 2.0, 1.0]  # First position most conserved

        distance = 0.0
        for i, (n1, n2) in enumerate(zip(codon1.upper(), codon2.upper())):
            if n1 != n2:
                distance += weights[i]

        return distance

    def _poincare_distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Calculate Poincare distance between embeddings.

        Args:
            z1: First embedding
            z2: Second embedding

        Returns:
            Hyperbolic distance
        """
        # Euclidean norms
        norm_z1 = np.linalg.norm(z1)
        norm_z2 = np.linalg.norm(z2)
        norm_diff = np.linalg.norm(z1 - z2)

        # Poincare distance formula
        denominator = (1 - norm_z1**2) * (1 - norm_z2**2)
        if denominator <= 0:
            denominator = 1e-10

        arg = 1 + 2 * (norm_diff**2) / denominator
        return np.arccosh(max(1.0, arg))

    def compute_reveal_score(
        self,
        wt_codon: str,
        mut_codon: str
    ) -> float:
        """Compute reveal score for a codon mutation.

        The reveal score measures how much a mutation "exposes"
        the virus to immune recognition, based on hyperbolic
        geometry in codon space.

        Args:
            wt_codon: Wild-type codon (3 letters)
            mut_codon: Mutant codon (3 letters)

        Returns:
            Reveal score (higher = more exposing)
        """
        self._load_model()

        if self.model is not None and HAS_TORCH:
            # Use the ternary model
            wt_input = torch.tensor(self._codon_to_ternary(wt_codon)).unsqueeze(0)
            mut_input = torch.tensor(self._codon_to_ternary(mut_codon)).unsqueeze(0)

            with torch.no_grad():
                # Get latent representations
                wt_z = self.model.encode(wt_input)[0]  # mu
                mut_z = self.model.encode(mut_input)[0]

            # Compute hyperbolic distance
            return self._poincare_distance(
                wt_z.numpy().flatten(),
                mut_z.numpy().flatten()
            )
        else:
            # Fallback calculation
            return self._fallback_codon_distance(wt_codon, mut_codon)

    def parse_mutation(self, mutation: str) -> Tuple[str, int, str]:
        """Parse mutation string into components.

        Args:
            mutation: Mutation string (e.g., "E166K")

        Returns:
            Tuple of (wt_aa, position, mut_aa)
        """
        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        position = int(mutation[1:-1])
        return wt_aa, position, mut_aa

    def predict_reveal_effect(
        self,
        wt_sequence: str,
        mutation: str,
        pdb_id: Optional[str] = None,
        chain_id: str = 'A'
    ) -> Dict:
        """Predict the structural impact of a reveal mutation.

        Combines:
        1. Geometric analysis from ternary VAE
        2. Structural context from PDB
        3. Interface/catalytic site information

        Args:
            wt_sequence: Wild-type protein sequence
            mutation: Mutation string (e.g., "E166K")
            pdb_id: PDB structure for context (default: 2B4J)
            chain_id: Chain identifier in PDB

        Returns:
            Dictionary with comprehensive analysis
        """
        pdb_id = pdb_id or self.DEFAULT_PDB
        wt_aa, position, mut_aa = self.parse_mutation(mutation)

        # Validate mutation against sequence
        if position <= 0 or position > len(wt_sequence):
            raise ValueError(f"Position {position} out of range for sequence length {len(wt_sequence)}")

        seq_aa = wt_sequence[position - 1]
        if seq_aa != wt_aa:
            print(f"Warning: Sequence has {seq_aa} at position {position}, mutation specifies {wt_aa}")

        # Get structural context from PDB (if BioPython available)
        if self.pdb_analyzer is not None:
            structural_analysis = self.pdb_analyzer.analyze_mutation_site(
                pdb_id=pdb_id,
                position=position,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                chain_id=chain_id
            )
        else:
            # Fallback when BioPython not available
            structural_analysis = {
                'position': position,
                'mutation': f"{wt_aa}{position}{mut_aa}",
                'structural_context': {},
                'charge_change': 0,
                'hydrophobicity_change': 0,
                'mechanisms': [],
                'is_ledgf_interface': position in range(166, 176),  # Approximate
                'is_catalytic_adjacent': position in {64, 116, 152},
                'solvent_accessibility': 0.5,
                'n_contacts': 0,
            }

        # Compute geometric reveal score
        # For simplicity, use a representative codon for each amino acid
        # In practice, you'd want the actual codon from the nucleotide sequence
        representative_codons = {
            'A': 'GCT', 'R': 'CGT', 'N': 'AAT', 'D': 'GAT', 'C': 'TGT',
            'Q': 'CAA', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
            'L': 'CTT', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCT',
            'S': 'TCT', 'T': 'ACT', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTT',
        }

        wt_codon = representative_codons.get(wt_aa, 'NNN')
        mut_codon = representative_codons.get(mut_aa, 'NNN')

        reveal_score = self.compute_reveal_score(wt_codon, mut_codon)

        # Determine overall mechanism
        mechanisms = structural_analysis['mechanisms']
        is_ledgf = (
            self.pdb_analyzer.is_ledgf_interface(position)
            if self.pdb_analyzer is not None
            else structural_analysis['is_ledgf_interface']
        )

        if is_ledgf:
            primary_mechanism = "LEDGF interface disruption"
        elif structural_analysis['is_catalytic_adjacent']:
            primary_mechanism = "Catalytic site perturbation"
        elif structural_analysis['charge_change'] != 0:
            primary_mechanism = "Electrostatic disruption"
        elif 'aromatic_loss' in mechanisms or 'aromatic_gain' in mechanisms:
            primary_mechanism = "Aromatic interaction change"
        else:
            primary_mechanism = "Local structural perturbation"

        # Confidence estimation based on available data
        confidence = 0.5  # Base confidence
        if pdb_id == self.DEFAULT_PDB:
            confidence += 0.2  # Reference structure
        if structural_analysis['n_contacts'] > 5:
            confidence += 0.1  # Good structural context
        if is_ledgf:
            confidence += 0.1  # Well-characterized interface
        confidence = min(confidence, 1.0)

        return {
            'mutation': mutation,
            'position': position,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'reveal_score': reveal_score,
            'primary_mechanism': primary_mechanism,
            'all_mechanisms': mechanisms,
            'structural_context': structural_analysis['structural_context'],
            'is_ledgf_interface': structural_analysis['is_ledgf_interface'],
            'is_catalytic_adjacent': structural_analysis['is_catalytic_adjacent'],
            'charge_change': structural_analysis['charge_change'],
            'solvent_accessibility': structural_analysis['solvent_accessibility'],
            'n_contacts': structural_analysis['n_contacts'],
            'confidence': confidence,
            'pdb_reference': pdb_id,
        }

    def batch_predict(
        self,
        wt_sequence: str,
        mutations: List[str],
        pdb_id: Optional[str] = None
    ) -> List[Dict]:
        """Predict effects for multiple mutations.

        Args:
            wt_sequence: Wild-type protein sequence
            mutations: List of mutation strings
            pdb_id: PDB structure for context

        Returns:
            List of prediction dictionaries, sorted by reveal_score
        """
        results = []
        for mutation in mutations:
            try:
                result = self.predict_reveal_effect(wt_sequence, mutation, pdb_id)
                results.append(result)
            except Exception as e:
                print(f"Error processing {mutation}: {e}")

        # Sort by reveal score (highest first)
        return sorted(results, key=lambda x: x['reveal_score'], reverse=True)


def main():
    """Example usage of the hybrid predictor."""
    # HIV-1 Integrase catalytic domain sequence (HXB2 reference)
    integrase_sequence = (
        "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
        "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
        "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
        "QKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
        "RQED"
    )

    # Top reveal mutation candidates from our analysis
    reveal_mutations = [
        "E166K",  # Salt bridge reversal at LEDGF interface
        "K173E",  # Charge reversal at helix
        "W131A",  # Aromatic cap removal
        "D167N",  # Charge neutralization
    ]

    print("=" * 60)
    print("Hybrid Structure Predictor - HIV Integrase Reveal Mutations")
    print("=" * 60)

    try:
        predictor = HybridStructurePredictor()

        for mutation in reveal_mutations:
            print(f"\n--- {mutation} ---")
            try:
                result = predictor.predict_reveal_effect(
                    wt_sequence=integrase_sequence,
                    mutation=mutation
                )
                print(f"Reveal Score: {result['reveal_score']:.2f}")
                print(f"Mechanism: {result['primary_mechanism']}")
                print(f"LEDGF Interface: {result['is_ledgf_interface']}")
                print(f"Confidence: {result['confidence']:.1%}")
            except FileNotFoundError as e:
                print(f"PDB structure not available: {e}")
                print("Run: python scripts/download_integrase_structures.py")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install biopython")


if __name__ == "__main__":
    main()
