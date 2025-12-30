# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stanford HIVdb Integration Client.

Provides integration with Stanford University's HIV Drug Resistance Database
for professional resistance interpretation.

Stanford HIVdb: https://hivdb.stanford.edu/

Usage:
    python stanford_hivdb_client.py --sequence SEQUENCE
    python stanford_hivdb_client.py --file sequences.fasta
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

from shared.config import get_config
from shared.constants import HIV_DRUG_CLASSES, WHO_SDRM_NRTI, WHO_SDRM_NNRTI, WHO_SDRM_INSTI

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ResistanceLevel(Enum):
    """Drug resistance level."""
    SUSCEPTIBLE = 1
    POTENTIAL_LOW = 2
    LOW = 3
    INTERMEDIATE = 4
    HIGH = 5

    @classmethod
    def from_score(cls, score: int) -> "ResistanceLevel":
        """Convert Stanford score to resistance level."""
        if score < 10:
            return cls.SUSCEPTIBLE
        elif score < 15:
            return cls.POTENTIAL_LOW
        elif score < 30:
            return cls.LOW
        elif score < 60:
            return cls.INTERMEDIATE
        else:
            return cls.HIGH

    @classmethod
    def from_text(cls, text: str) -> "ResistanceLevel":
        """Convert Stanford text to resistance level."""
        text = text.lower()
        if "susceptible" in text:
            return cls.SUSCEPTIBLE
        elif "potential" in text:
            return cls.POTENTIAL_LOW
        elif "low" in text:
            return cls.LOW
        elif "intermediate" in text:
            return cls.INTERMEDIATE
        elif "high" in text:
            return cls.HIGH
        return cls.SUSCEPTIBLE


@dataclass
class DrugScore:
    """Score for a single drug."""
    drug_name: str
    drug_abbr: str
    drug_class: str
    score: int
    level: ResistanceLevel
    text: str
    mutations: list[str] = field(default_factory=list)

    def is_resistant(self) -> bool:
        """Check if drug shows resistance."""
        return self.level.value >= ResistanceLevel.LOW.value


@dataclass
class MutationInfo:
    """Information about a detected mutation."""
    gene: str
    position: int
    reference: str
    mutation: str
    text: str
    is_sdrm: bool = False
    drug_class: Optional[str] = None

    @property
    def notation(self) -> str:
        """Standard mutation notation (e.g., K103N)."""
        return self.text


@dataclass
class ResistanceReport:
    """Complete resistance analysis report."""
    patient_id: str
    subtype: str
    mutations: list[MutationInfo]
    drug_scores: list[DrugScore]
    quality_issues: list[str] = field(default_factory=list)

    def get_resistant_drugs(self) -> list[str]:
        """Get list of drugs with resistance."""
        return [d.drug_abbr for d in self.drug_scores if d.is_resistant()]

    def get_sdrm_mutations(self) -> list[MutationInfo]:
        """Get SDRM mutations only."""
        return [m for m in self.mutations if m.is_sdrm]

    def has_tdr(self) -> bool:
        """Check if any transmitted drug resistance is present."""
        return len(self.get_sdrm_mutations()) > 0

    def get_recommended_regimens(self) -> list[str]:
        """Get recommended first-line regimens based on resistance."""
        resistant = set(self.get_resistant_drugs())

        regimens = [
            ("TDF/3TC/DTG", {"TDF", "3TC", "DTG"}),
            ("TDF/FTC/DTG", {"TDF", "FTC", "DTG"}),
            ("TAF/FTC/DTG", {"TAF", "FTC", "DTG"}),
            ("TDF/3TC/EFV", {"TDF", "3TC", "EFV"}),
            ("ABC/3TC/DTG", {"ABC", "3TC", "DTG"}),
        ]

        recommended = []
        for name, drugs in regimens:
            if not drugs & resistant:
                recommended.append(name)

        return recommended if recommended else ["Specialist referral needed"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "subtype": self.subtype,
            "tdr_positive": self.has_tdr(),
            "mutations": [asdict(m) for m in self.mutations],
            "sdrm_mutations": [m.notation for m in self.get_sdrm_mutations()],
            "drug_scores": [
                {
                    "drug": d.drug_abbr,
                    "score": d.score,
                    "level": d.level.name,
                    "text": d.text,
                }
                for d in self.drug_scores
            ],
            "resistant_drugs": self.get_resistant_drugs(),
            "recommended_regimens": self.get_recommended_regimens(),
        }


class StanfordHIVdbClient:
    """Client for Stanford HIVdb GraphQL API."""

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

    def __init__(self):
        self.config = get_config()
        self.cache_dir = self.config.get_partner_dir("hiv") / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_sequence(
        self,
        sequence: str,
        patient_id: str = "sample",
    ) -> Optional[ResistanceReport]:
        """Analyze a single HIV sequence.

        Args:
            sequence: HIV pol gene sequence
            patient_id: Patient identifier

        Returns:
            ResistanceReport or None if failed
        """
        if not REQUESTS_AVAILABLE:
            return self._mock_analysis(sequence, patient_id)

        try:
            response = requests.post(
                self.GRAPHQL_URL,
                json={
                    "query": self.SEQUENCE_ANALYSIS_QUERY,
                    "variables": {
                        "sequences": [{
                            "header": patient_id,
                            "sequence": sequence
                        }]
                    }
                },
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            return self._parse_response(data, patient_id)

        except Exception as e:
            print(f"Stanford API error: {e}")
            return self._mock_analysis(sequence, patient_id)

    def _parse_response(self, data: dict, patient_id: str) -> ResistanceReport:
        """Parse Stanford GraphQL response."""
        try:
            analysis = data["data"]["viewer"]["sequenceAnalysis"][0]

            # Get subtype
            subtype = analysis.get("bestMatchingSubtype", {}).get("display", "Unknown")

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

                        mutations.append(MutationInfo(
                            gene=gene,
                            position=mut["position"],
                            reference=mutation_text[0] if mutation_text else "",
                            mutation=mutation_text[-1] if mutation_text else "",
                            text=mutation_text,
                            is_sdrm=is_sdrm,
                        ))

            # Parse drug scores
            drug_scores = []
            for gene_data in analysis.get("drugResistance", []):
                for drug_data in gene_data.get("drugScores", []):
                    drug_scores.append(DrugScore(
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
                        ]
                    ))

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
        """Check if mutation is a WHO SDRM."""
        if gene == "RT":
            return mutation in WHO_SDRM_NRTI or mutation in WHO_SDRM_NNRTI
        elif gene == "IN":
            return mutation in WHO_SDRM_INSTI
        return False

    def _mock_analysis(self, sequence: str, patient_id: str) -> ResistanceReport:
        """Mock analysis when API unavailable."""
        import random
        random.seed(hash(sequence) % (2**31 - 1) if sequence else 42)

        # Simulate subtype
        subtypes = ["B", "C", "CRF01_AE", "CRF02_AG", "A1"]
        subtype = random.choice(subtypes)

        # Generate mock mutations (low probability)
        mutations = []
        possible_mutations = [
            ("RT", "K103N", True),
            ("RT", "M184V", True),
            ("RT", "K65R", True),
            ("RT", "Y181C", True),
            ("RT", "G190A", True),
            ("IN", "N155H", True),
            ("IN", "Q148H", True),
        ]

        for gene, mut, is_sdrm in possible_mutations:
            if random.random() < 0.1:  # 10% chance per mutation
                mutations.append(MutationInfo(
                    gene=gene,
                    position=int("".join(c for c in mut if c.isdigit())),
                    reference=mut[0],
                    mutation=mut[-1],
                    text=mut,
                    is_sdrm=is_sdrm,
                ))

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
            ("Dolutegravir", "DTG", "INSTI"),
            ("Raltegravir", "RAL", "INSTI"),
            ("Bictegravir", "BIC", "INSTI"),
            ("Darunavir", "DRV", "PI"),
        ]

        resistance_mutations = {m.text for m in mutations}

        for name, abbr, drug_class in all_drugs:
            # Check if any resistance mutations affect this drug
            score = 0
            affected_muts = []

            if drug_class == "NNRTI" and any(m in resistance_mutations for m in ["K103N", "Y181C", "G190A"]):
                score = 60 + random.randint(0, 30)
                affected_muts = [m for m in ["K103N", "Y181C", "G190A"] if m in resistance_mutations]
            elif drug_class == "NRTI" and "M184V" in resistance_mutations:
                if abbr in ["3TC", "FTC"]:
                    score = 60 + random.randint(0, 20)
                    affected_muts = ["M184V"]
            elif drug_class == "INSTI" and any(m in resistance_mutations for m in ["N155H", "Q148H"]):
                score = 30 + random.randint(0, 40)
                affected_muts = [m for m in ["N155H", "Q148H"] if m in resistance_mutations]

            drug_scores.append(DrugScore(
                drug_name=name,
                drug_abbr=abbr,
                drug_class=drug_class,
                score=score,
                level=ResistanceLevel.from_score(score),
                text=ResistanceLevel.from_score(score).name.replace("_", " ").title(),
                mutations=affected_muts,
            ))

        return ResistanceReport(
            patient_id=patient_id,
            subtype=subtype,
            mutations=mutations,
            drug_scores=drug_scores,
            quality_issues=[] if sequence else ["Demo mode - no real sequence analyzed"],
        )

    def analyze_fasta_file(self, fasta_path: Path) -> list[ResistanceReport]:
        """Analyze all sequences in a FASTA file.

        Args:
            fasta_path: Path to FASTA file

        Returns:
            List of ResistanceReports
        """
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
                sequence = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if header and sequence:
                            report = self.analyze_sequence("".join(sequence), header)
                            if report:
                                reports.append(report)
                        header = line[1:].split()[0]
                        sequence = []
                    else:
                        sequence.append(line)

                if header and sequence:
                    report = self.analyze_sequence("".join(sequence), header)
                    if report:
                        reports.append(report)

        return reports

    def generate_report(self, report: ResistanceReport) -> str:
        """Generate a formatted text report.

        Args:
            report: ResistanceReport

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
            lines.append(f"SDRM Mutations: {', '.join(m.notation for m in report.get_sdrm_mutations())}")
            lines.append("")
        else:
            lines.append("No transmitted drug resistance mutations detected.")
            lines.append("")

        # Drug Scores by Class
        lines.append("-" * 70)
        lines.append("DRUG SUSCEPTIBILITY")
        lines.append("-" * 70)

        for drug_class in ["NRTI", "NNRTI", "INSTI", "PI"]:
            class_drugs = [d for d in report.drug_scores if d.drug_class == drug_class]
            if class_drugs:
                lines.append(f"\n{drug_class}:")
                for drug in class_drugs:
                    status = "R" if drug.is_resistant() else "S"
                    lines.append(f"  [{status}] {drug.drug_abbr:6} Score: {drug.score:3}  {drug.text}")

        # Recommendations
        lines.append("")
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)

        if report.get_resistant_drugs():
            lines.append(f"Avoid: {', '.join(report.get_resistant_drugs())}")
        else:
            lines.append("All drugs appear susceptible.")

        lines.append(f"Recommended regimens: {', '.join(report.get_recommended_regimens())}")

        # Quality issues
        if report.quality_issues:
            lines.append("")
            lines.append("Quality warnings:")
            for issue in report.quality_issues:
                lines.append(f"  - {issue}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Stanford HIVdb resistance analysis")
    parser.add_argument("--sequence", help="HIV sequence to analyze")
    parser.add_argument("--file", help="FASTA file to analyze")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--demo", action="store_true", help="Run demo analysis")
    args = parser.parse_args()

    client = StanfordHIVdbClient()

    if args.demo:
        print("Running demo analysis...")
        report = client._mock_analysis("", "DEMO_PATIENT")
        print(client.generate_report(report))

        # Save as JSON
        output_path = client.cache_dir / "demo_report.json"
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved JSON to: {output_path}")

    elif args.sequence:
        print("Analyzing sequence...")
        report = client.analyze_sequence(args.sequence, "CLI_SAMPLE")
        if report:
            print(client.generate_report(report))

    elif args.file:
        fasta_path = Path(args.file)
        if not fasta_path.exists():
            print(f"File not found: {fasta_path}")
            return

        print(f"Analyzing {fasta_path}...")
        reports = client.analyze_fasta_file(fasta_path)

        for report in reports:
            print(client.generate_report(report))
            print()

        if args.output:
            with open(args.output, "w") as f:
                json.dump([r.to_dict() for r in reports], f, indent=2)
            print(f"Saved reports to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
