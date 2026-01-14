"""
HIV drug resistance analysis integration with p-adic hyperbolic encoding.

Connects Stanford HIVDB data with the codon encoder research framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import re

import pandas as pd

from ..clients import HIVDBClient


@dataclass
class MutationInfo:
    """Parsed mutation information."""
    position: int
    wild_type: str
    mutant: str
    gene: str = "RT"

    @property
    def notation(self) -> str:
        """Standard mutation notation (e.g., 'M184V')."""
        return f"{self.wild_type}{self.position}{self.mutant}"

    @property
    def codon_change(self) -> tuple[str, str]:
        """Get wild-type and mutant codons (placeholder - requires sequence context)."""
        # This would require the actual sequence to determine codons
        # Returns amino acid as placeholder
        return (self.wild_type, self.mutant)


@dataclass
class ResistanceResult:
    """Drug resistance analysis result."""
    drug: str
    drug_class: str
    score: float
    level: str
    mutations: list[str] = field(default_factory=list)

    @property
    def is_resistant(self) -> bool:
        """Check if result indicates resistance."""
        return self.score > 0 or "Resistance" in self.level


@dataclass
class SequenceAnalysis:
    """Complete sequence analysis result."""
    sequence_id: str
    gene: str
    mutations: list[MutationInfo]
    resistance_results: list[ResistanceResult]
    validation_messages: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert resistance results to DataFrame."""
        records = []
        for r in self.resistance_results:
            records.append({
                "sequence_id": self.sequence_id,
                "gene": self.gene,
                "drug": r.drug,
                "drug_class": r.drug_class,
                "score": r.score,
                "level": r.level,
                "is_resistant": r.is_resistant,
                "mutations": ",".join(r.mutations) if r.mutations else "",
            })
        return pd.DataFrame(records)


class HIVAnalysisIntegration:
    """
    Integration layer between HIVDB API and p-adic hyperbolic analysis.

    Provides methods to:
    - Analyze HIV sequences for drug resistance
    - Extract mutation information for codon encoding
    - Map resistance patterns to hyperbolic space
    """

    def __init__(self, hivdb_client: Optional[HIVDBClient] = None):
        """
        Initialize integration layer.

        Args:
            hivdb_client: HIVDB client instance (created if not provided)
        """
        self.hivdb = hivdb_client or HIVDBClient()

    def parse_mutation(self, mutation_str: str, gene: str = "RT") -> MutationInfo:
        """
        Parse mutation string into structured format.

        Args:
            mutation_str: Mutation notation (e.g., "M184V", "K103N")
            gene: Gene name (RT, PR, IN)

        Returns:
            Parsed mutation information
        """
        # Pattern: optional gene prefix, wild-type AA, position, mutant AA(s)
        pattern = r"(?:([A-Za-z]+):)?([A-Z])(\d+)([A-Z]+)"
        match = re.match(pattern, mutation_str.upper())

        if not match:
            raise ValueError(f"Invalid mutation format: {mutation_str}")

        gene_prefix, wild_type, position, mutant = match.groups()

        return MutationInfo(
            position=int(position),
            wild_type=wild_type,
            mutant=mutant[0],  # Take first if multiple
            gene=gene_prefix or gene
        )

    def analyze_sequence(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> SequenceAnalysis:
        """
        Analyze HIV sequence for drug resistance.

        Args:
            sequence: HIV nucleotide or amino acid sequence
            sequence_id: Identifier for the sequence

        Returns:
            Complete sequence analysis with mutations and resistance results
        """
        # Call HIVDB API
        result = self.hivdb.analyze_sequence(sequence)

        # Parse response
        mutations = []
        resistance_results = []
        validation_messages = []
        gene = "Unknown"

        if "data" in result and "viewer" in result["data"]:
            viewer = result["data"]["viewer"]
            if "sequenceAnalysis" in viewer and viewer["sequenceAnalysis"]:
                analysis = viewer["sequenceAnalysis"][0]

                # Extract validation messages
                if "validationResults" in analysis:
                    validation_messages = [
                        v.get("message", "") for v in analysis["validationResults"]
                    ]

                # Extract drug resistance
                if "drugResistance" in analysis:
                    for dr in analysis["drugResistance"]:
                        drug_class = dr.get("drugClass", {}).get("name", "Unknown")
                        gene = dr.get("gene", {}).get("name", gene)

                        if "drugScores" in dr:
                            for ds in dr["drugScores"]:
                                drug = ds.get("drug", {}).get("name", "Unknown")
                                score = ds.get("score", 0.0)
                                level = ds.get("text", "Unknown")

                                # Extract mutations contributing to this drug
                                drug_mutations = []
                                if "partialScores" in ds:
                                    for ps in ds["partialScores"]:
                                        if "mutations" in ps:
                                            for m in ps["mutations"]:
                                                mut_text = m.get("text", "")
                                                if mut_text:
                                                    drug_mutations.append(mut_text)

                                resistance_results.append(ResistanceResult(
                                    drug=drug,
                                    drug_class=drug_class,
                                    score=score,
                                    level=level,
                                    mutations=drug_mutations,
                                ))

                # Extract all mutations
                if "alignedGeneSequences" in analysis:
                    for ags in analysis["alignedGeneSequences"]:
                        gene = ags.get("gene", {}).get("name", gene)
                        if "mutations" in ags:
                            for m in ags["mutations"]:
                                mut_text = m.get("text", "")
                                if mut_text:
                                    try:
                                        mutations.append(
                                            self.parse_mutation(mut_text, gene)
                                        )
                                    except ValueError:
                                        pass

        return SequenceAnalysis(
            sequence_id=sequence_id,
            gene=gene,
            mutations=mutations,
            resistance_results=resistance_results,
            validation_messages=validation_messages,
        )

    def analyze_mutations(
        self,
        mutations: list[str],
        gene: str = "RT",
    ) -> pd.DataFrame:
        """
        Analyze a list of mutations for drug resistance.

        Args:
            mutations: List of mutation strings (e.g., ["M184V", "K103N"])
            gene: Gene name (RT, PR, IN)

        Returns:
            DataFrame with resistance analysis
        """
        result = self.hivdb.get_mutations_analysis(mutations, gene)

        records = []
        if "data" in result and "viewer" in result["data"]:
            viewer = result["data"]["viewer"]
            if "mutationsAnalysis" in viewer:
                for ma in viewer["mutationsAnalysis"]:
                    mutation_type = ma.get("mutationType", "Unknown")
                    comments = [c.get("text", "") for c in ma.get("comments", [])]

                    # Get drug scores if available
                    if "drugScores" in ma:
                        for ds in ma["drugScores"]:
                            records.append({
                                "mutation": mutations[0] if mutations else "Unknown",
                                "type": mutation_type,
                                "drug": ds.get("drug", {}).get("name", "Unknown"),
                                "score": ds.get("score", 0.0),
                                "comments": "; ".join(comments),
                            })

        if not records:
            # Return basic info if no detailed analysis
            for mut in mutations:
                records.append({
                    "mutation": mut,
                    "type": "Unknown",
                    "drug": "N/A",
                    "score": 0.0,
                    "comments": "",
                })

        return pd.DataFrame(records)

    def get_resistance_profile(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> pd.DataFrame:
        """
        Get a complete resistance profile as a DataFrame.

        Args:
            sequence: HIV nucleotide or amino acid sequence
            sequence_id: Identifier for the sequence

        Returns:
            DataFrame with drug-by-drug resistance profile
        """
        analysis = self.analyze_sequence(sequence, sequence_id)
        return analysis.to_dataframe()

    def extract_mutations_for_encoding(
        self,
        sequence: str,
    ) -> list[dict]:
        """
        Extract mutations in format suitable for p-adic encoding.

        Args:
            sequence: HIV sequence

        Returns:
            List of mutation dictionaries with position and amino acid info
        """
        analysis = self.analyze_sequence(sequence)

        mutations_for_encoding = []
        for mut in analysis.mutations:
            mutations_for_encoding.append({
                "gene": mut.gene,
                "position": mut.position,
                "wild_type_aa": mut.wild_type,
                "mutant_aa": mut.mutant,
                "notation": mut.notation,
                # For codon encoding, we would need the actual codon
                # This requires sequence context
                "wild_type_codon": None,
                "mutant_codon": None,
            })

        return mutations_for_encoding

    def batch_analyze(
        self,
        sequences: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Analyze multiple sequences in batch.

        Args:
            sequences: List of (sequence_id, sequence) tuples

        Returns:
            Combined DataFrame with all resistance results
        """
        all_results = []

        for seq_id, sequence in sequences:
            try:
                df = self.get_resistance_profile(sequence, seq_id)
                all_results.append(df)
            except Exception as e:
                # Log error and continue
                all_results.append(pd.DataFrame([{
                    "sequence_id": seq_id,
                    "error": str(e),
                }]))

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
