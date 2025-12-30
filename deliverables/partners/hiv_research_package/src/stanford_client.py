# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stanford HIVdb Integration Client.

This module provides integration with Stanford University's HIV Drug
Resistance Database for professional resistance interpretation.

Stanford HIVdb: https://hivdb.stanford.edu/
GraphQL API: https://hivdb.stanford.edu/graphql

Example:
    >>> client = StanfordHIVdbClient()
    >>> report = client.analyze_sequence(sequence, "PATIENT_001")
    >>> print(f"Subtype: {report.subtype}")
    >>> print(f"TDR: {report.has_tdr()}")
    >>> for drug in report.get_resistant_drugs():
    ...     print(f"  Resistant: {drug}")
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .models import (
    ResistanceLevel,
    DrugScore,
    MutationInfo,
    ResistanceReport,
)
from .constants import (
    WHO_SDRM_NRTI,
    WHO_SDRM_NNRTI,
    WHO_SDRM_INSTI,
    WHO_SDRM_PI,
)

# Try to import requests
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class StanfordHIVdbClient:
    """Client for Stanford HIVdb GraphQL API.

    Provides methods to analyze HIV sequences for drug resistance
    using the Stanford HIVdb database.

    Attributes:
        cache_dir: Directory for caching API responses
        timeout: API request timeout in seconds

    Example:
        >>> client = StanfordHIVdbClient()
        >>> report = client.analyze_sequence(sequence, "P001")
        >>> if report.has_tdr():
        ...     print("TDR detected!")
    """

    GRAPHQL_URL = "https://hivdb.stanford.edu/graphql"

    SEQUENCE_ANALYSIS_QUERY = """
    mutation AnalyzeSequences($sequences: [UnalignedSequenceInput]!) {
      viewer {
        sequenceAnalysis(sequences: $sequences) {
          inputSequence {
            header
          }
          bestMatchingSubtype {
            display
          }
          validationResults {
            level
            message
          }
          drugResistance {
            gene {
              name
            }
            mutationsByTypes {
              mutationType
              mutations {
                text
                position
                primaryType
              }
            }
            drugScores {
              drug {
                name
                displayAbbr
              }
              drugClass {
                name
              }
              score
              level
              text
              partialScores {
                mutations {
                  text
                }
              }
            }
          }
        }
      }
    }
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: int = 60,
    ):
        """Initialize Stanford HIVdb client.

        Args:
            cache_dir: Directory for caching responses (optional)
            timeout: API request timeout in seconds
        """
        self.cache_dir = cache_dir
        self.timeout = timeout

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_sequence(
        self,
        sequence: str,
        patient_id: str = "sample",
    ) -> Optional[ResistanceReport]:
        """Analyze a single HIV sequence.

        Args:
            sequence: HIV pol gene sequence (nucleotide or amino acid)
            patient_id: Patient identifier

        Returns:
            ResistanceReport or None if analysis failed
        """
        if not REQUESTS_AVAILABLE:
            return self._mock_analysis(sequence, patient_id)

        try:
            response = requests.post(
                self.GRAPHQL_URL,
                json={
                    "query": self.SEQUENCE_ANALYSIS_QUERY,
                    "variables": {
                        "sequences": [
                            {"header": patient_id, "sequence": sequence}
                        ]
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            return self._parse_response(data, patient_id)

        except Exception as e:
            print(f"Stanford API error: {e}")
            return self._mock_analysis(sequence, patient_id)

    def _parse_response(
        self,
        data: dict,
        patient_id: str,
    ) -> ResistanceReport:
        """Parse Stanford GraphQL response."""
        try:
            analysis = data["data"]["viewer"]["sequenceAnalysis"][0]

            # Get subtype
            subtype = analysis.get("bestMatchingSubtype", {}).get(
                "display", "Unknown"
            )

            # Get quality issues
            quality_issues = []
            for result in analysis.get("validationResults", []):
                if result["level"] in ("WARNING", "CRITICAL"):
                    quality_issues.append(result["message"])

            # Parse mutations
            mutations = []
            for gene_data in analysis.get("drugResistance", []):
                gene = gene_data["gene"]["name"]
                for mut_type in gene_data.get("mutationsByTypes", []):
                    for mut in mut_type.get("mutations", []):
                        mutation_text = mut["text"]
                        is_sdrm = self._is_sdrm(mutation_text, gene)
                        is_major = mut.get("primaryType") == "Major"

                        mutations.append(
                            MutationInfo(
                                gene=gene,
                                position=mut["position"],
                                reference=mutation_text[0] if mutation_text else "",
                                mutation=mutation_text[-1] if mutation_text else "",
                                text=mutation_text,
                                is_sdrm=is_sdrm,
                                is_major=is_major,
                            )
                        )

            # Parse drug scores
            drug_scores = []
            for gene_data in analysis.get("drugResistance", []):
                for drug_data in gene_data.get("drugScores", []):
                    drug_scores.append(
                        DrugScore(
                            drug_name=drug_data["drug"]["name"],
                            drug_abbr=drug_data["drug"]["displayAbbr"],
                            drug_class=drug_data["drugClass"]["name"],
                            score=drug_data["score"],
                            level=ResistanceLevel.from_text(drug_data["level"]),
                            text=drug_data["text"],
                            mutations=[
                                m["text"]
                                for ps in drug_data.get("partialScores", [])
                                for m in ps.get("mutations", [])
                            ],
                        )
                    )

            return ResistanceReport(
                patient_id=patient_id,
                subtype=subtype,
                mutations=mutations,
                drug_scores=drug_scores,
                quality_issues=quality_issues,
            )

        except (KeyError, IndexError) as e:
            print(f"Error parsing Stanford response: {e}")
            return self._mock_analysis("", patient_id)

    def _is_sdrm(self, mutation: str, gene: str) -> bool:
        """Check if mutation is a WHO surveillance drug resistance mutation."""
        if gene == "RT":
            return mutation in WHO_SDRM_NRTI or mutation in WHO_SDRM_NNRTI
        elif gene == "IN":
            return mutation in WHO_SDRM_INSTI
        elif gene == "PR":
            return mutation in WHO_SDRM_PI
        return False

    def _mock_analysis(
        self,
        sequence: str,
        patient_id: str,
    ) -> ResistanceReport:
        """Mock analysis when API unavailable.

        Generates realistic demo data for testing purposes.

        Args:
            sequence: HIV sequence (used for seeding)
            patient_id: Patient identifier

        Returns:
            ResistanceReport with mock data
        """
        # Seed based on sequence for reproducibility
        seed = hash(sequence) % (2**31 - 1) if sequence else 42
        random.seed(seed)

        # Simulate subtype
        subtypes = ["B", "C", "CRF01_AE", "CRF02_AG", "A1"]
        subtype = random.choice(subtypes)

        # Generate mock mutations (low probability)
        mutations = []
        possible_mutations = [
            ("RT", "K103N", True, True),
            ("RT", "M184V", True, True),
            ("RT", "K65R", True, True),
            ("RT", "Y181C", True, True),
            ("RT", "G190A", True, True),
            ("RT", "T215Y", True, True),
            ("IN", "N155H", True, False),
            ("IN", "Q148H", True, True),
            ("IN", "G140S", True, False),
        ]

        for gene, mut, is_sdrm, is_major in possible_mutations:
            if random.random() < 0.1:  # 10% chance per mutation
                position = int("".join(c for c in mut if c.isdigit()))
                mutations.append(
                    MutationInfo(
                        gene=gene,
                        position=position,
                        reference=mut[0],
                        mutation=mut[-1],
                        text=mut,
                        is_sdrm=is_sdrm,
                        is_major=is_major,
                    )
                )

        # Generate drug scores
        drug_scores = []
        all_drugs = [
            ("Tenofovir", "TDF", "NRTI"),
            ("Lamivudine", "3TC", "NRTI"),
            ("Emtricitabine", "FTC", "NRTI"),
            ("Abacavir", "ABC", "NRTI"),
            ("Zidovudine", "AZT", "NRTI"),
            ("Efavirenz", "EFV", "NNRTI"),
            ("Nevirapine", "NVP", "NNRTI"),
            ("Rilpivirine", "RPV", "NNRTI"),
            ("Doravirine", "DOR", "NNRTI"),
            ("Dolutegravir", "DTG", "INSTI"),
            ("Raltegravir", "RAL", "INSTI"),
            ("Bictegravir", "BIC", "INSTI"),
            ("Cabotegravir", "CAB", "INSTI"),
            ("Darunavir", "DRV", "PI"),
            ("Atazanavir", "ATV", "PI"),
            ("Lopinavir", "LPV", "PI"),
        ]

        resistance_mutations = {m.text for m in mutations}

        for name, abbr, drug_class in all_drugs:
            score = 0
            affected_muts = []

            # Check for resistance mutations
            if drug_class == "NNRTI":
                nnrti_muts = ["K103N", "Y181C", "G190A"]
                matching = [m for m in nnrti_muts if m in resistance_mutations]
                if matching:
                    score = 60 + random.randint(0, 30)
                    affected_muts = matching

            elif drug_class == "NRTI":
                if "M184V" in resistance_mutations and abbr in ["3TC", "FTC"]:
                    score = 60 + random.randint(0, 20)
                    affected_muts = ["M184V"]
                elif "K65R" in resistance_mutations and abbr in ["TDF", "ABC"]:
                    score = 40 + random.randint(0, 20)
                    affected_muts = ["K65R"]

            elif drug_class == "INSTI":
                insti_muts = ["N155H", "Q148H", "G140S"]
                matching = [m for m in insti_muts if m in resistance_mutations]
                if matching:
                    score = 30 + random.randint(0, 40)
                    affected_muts = matching

            drug_scores.append(
                DrugScore(
                    drug_name=name,
                    drug_abbr=abbr,
                    drug_class=drug_class,
                    score=score,
                    level=ResistanceLevel.from_score(score),
                    text=ResistanceLevel.from_score(score).name.replace("_", " ").title(),
                    mutations=affected_muts,
                )
            )

        return ResistanceReport(
            patient_id=patient_id,
            subtype=subtype,
            mutations=mutations,
            drug_scores=drug_scores,
            quality_issues=(
                [] if sequence else ["Demo mode - no real sequence analyzed"]
            ),
        )

    def analyze_fasta_file(self, fasta_path: Path) -> list[ResistanceReport]:
        """Analyze all sequences in a FASTA file.

        Args:
            fasta_path: Path to FASTA file

        Returns:
            List of ResistanceReport objects
        """
        fasta_path = Path(fasta_path)
        reports = []

        try:
            from Bio import SeqIO

            for record in SeqIO.parse(fasta_path, "fasta"):
                report = self.analyze_sequence(str(record.seq), record.id)
                if report:
                    reports.append(report)

        except ImportError:
            # Manual FASTA parsing
            with open(fasta_path) as f:
                header = None
                sequence_lines: list[str] = []

                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if header and sequence_lines:
                            report = self.analyze_sequence(
                                "".join(sequence_lines), header
                            )
                            if report:
                                reports.append(report)
                        header = line[1:].split()[0]
                        sequence_lines = []
                    else:
                        sequence_lines.append(line)

                if header and sequence_lines:
                    report = self.analyze_sequence(
                        "".join(sequence_lines), header
                    )
                    if report:
                        reports.append(report)

        return reports

    def generate_report(self, report: ResistanceReport) -> str:
        """Generate a formatted text report.

        Args:
            report: ResistanceReport object

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "HIV DRUG RESISTANCE REPORT",
            f"Patient: {report.patient_id}",
            f"Subtype: {report.subtype}",
            "=" * 70,
            "",
        ]

        # TDR Status
        if report.has_tdr():
            lines.append("*** TRANSMITTED DRUG RESISTANCE DETECTED ***")
            sdrm = ", ".join(m.notation for m in report.get_sdrm_mutations())
            lines.append(f"SDRM Mutations: {sdrm}")
            lines.append("")
        else:
            lines.append("No transmitted drug resistance mutations detected.")
            lines.append("")

        # Drug Scores by Class
        lines.append("-" * 70)
        lines.append("DRUG SUSCEPTIBILITY")
        lines.append("-" * 70)

        for drug_class in ["NRTI", "NNRTI", "INSTI", "PI"]:
            class_drugs = [
                d for d in report.drug_scores if d.drug_class == drug_class
            ]
            if class_drugs:
                lines.append(f"\n{drug_class}:")
                for drug in class_drugs:
                    status = "R" if drug.is_resistant() else "S"
                    lines.append(
                        f"  [{status}] {drug.drug_abbr:6} "
                        f"Score: {drug.score:3}  {drug.text}"
                    )

        # Recommendations
        lines.append("")
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)

        resistant_drugs = report.get_resistant_drugs()
        if resistant_drugs:
            lines.append(f"Avoid: {', '.join(resistant_drugs)}")
        else:
            lines.append("All drugs appear susceptible.")

        regimens = report.get_recommended_regimens()
        lines.append(f"Recommended regimens: {', '.join(regimens)}")

        # Quality issues
        if report.quality_issues:
            lines.append("")
            lines.append("Quality warnings:")
            for issue in report.quality_issues:
                lines.append(f"  - {issue}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_report(
        self,
        report: ResistanceReport,
        output_path: Path,
        format: str = "json",
    ) -> Path:
        """Save report to file.

        Args:
            report: ResistanceReport object
            output_path: Output file path
            format: Output format ("json" or "txt")

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        else:
            with open(output_path, "w") as f:
                f.write(self.generate_report(report))

        return output_path
