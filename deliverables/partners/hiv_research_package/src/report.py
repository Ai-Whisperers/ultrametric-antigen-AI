# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Clinical Report Generation for HIV decision support.

This module provides professional clinical report generation for
TDR screening and LA injectable selection results.

Example:
    >>> generator = ClinicalReportGenerator()
    >>> report_text = generator.generate_tdr_report(tdr_result)
    >>> pdf_path = generator.export_pdf(report_text, "patient_report.pdf")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import TDRResult, LASelectionResult, ResistanceReport, PatientData


class ClinicalReportGenerator:
    """Generate clinical reports for HIV decision support tools.

    Produces formatted text, JSON, and (optionally) PDF reports
    suitable for clinical documentation.

    Attributes:
        institution: Institution name for report header
        provider: Provider name for report header

    Example:
        >>> gen = ClinicalReportGenerator(institution="City Hospital")
        >>> report = gen.generate_tdr_report(tdr_result)
        >>> gen.save_report(report, "report.txt")
    """

    def __init__(
        self,
        institution: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """Initialize report generator.

        Args:
            institution: Institution name for report header
            provider: Provider name for report header
        """
        self.institution = institution or "HIV Clinical Decision Support"
        self.provider = provider

    def generate_tdr_report(
        self,
        result: TDRResult,
        include_details: bool = True,
    ) -> str:
        """Generate TDR screening clinical report.

        Args:
            result: TDRResult from TDR screening
            include_details: Whether to include detailed susceptibility

        Returns:
            Formatted report string
        """
        lines = self._generate_header("TRANSMITTED DRUG RESISTANCE (TDR) SCREENING")

        # Patient info
        lines.extend([
            f"Patient ID: {result.patient_id}",
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ])

        # TDR Status (prominent)
        lines.append("=" * 70)
        if result.tdr_positive:
            lines.append("*** TDR STATUS: POSITIVE ***")
            lines.append("")
            lines.append(f"Mutations Detected: {len(result.detected_mutations)}")
            lines.append(f"Summary: {result.resistance_summary}")
        else:
            lines.append("TDR STATUS: NEGATIVE")
            lines.append("")
            lines.append("No transmitted drug resistance mutations detected.")
        lines.extend(["", "=" * 70, ""])

        # Recommendations (prominent)
        lines.extend([
            "TREATMENT RECOMMENDATION",
            "-" * 40,
            f"Primary Regimen: {result.recommended_regimen}",
        ])

        if result.alternative_regimens:
            lines.append(f"Alternatives: {', '.join(result.alternative_regimens)}")

        lines.append(f"Confidence: {result.confidence * 100:.0f}%")
        lines.append("")

        # Detailed susceptibility (if requested)
        if include_details and result.tdr_positive:
            lines.extend([
                "DRUG SUSCEPTIBILITY PROFILE",
                "-" * 40,
            ])

            # Group by class
            by_class: dict[str, list[tuple[str, str, float]]] = {}
            for drug, info in result.drug_susceptibility.items():
                drug_class = info.get("class", "Other")
                if drug_class not in by_class:
                    by_class[drug_class] = []
                by_class[drug_class].append((drug, info["status"], info["score"]))

            for drug_class, drugs in by_class.items():
                lines.append(f"\n{drug_class}:")
                for drug, status, score in drugs:
                    status_symbol = self._get_status_symbol(status)
                    lines.append(f"  {status_symbol} {drug}: {status.upper()}")

            lines.append("")

        # Mutations detail
        if result.detected_mutations:
            lines.extend([
                "DETECTED MUTATIONS",
                "-" * 40,
            ])

            for mut in result.detected_mutations:
                level_indicator = self._get_level_indicator(mut["resistance_level"])
                lines.append(
                    f"  {level_indicator} {mut['mutation']} ({mut['drug_class']})"
                )
                lines.append(f"      Affects: {', '.join(mut['affected_drugs'])}")

            lines.append("")

        # Footer
        lines.extend(self._generate_footer())

        return "\n".join(lines)

    def generate_la_report(
        self,
        result: LASelectionResult,
        patient: Optional[PatientData] = None,
    ) -> str:
        """Generate LA injectable selection clinical report.

        Args:
            result: LASelectionResult from assessment
            patient: PatientData for additional context

        Returns:
            Formatted report string
        """
        lines = self._generate_header("LONG-ACTING INJECTABLE ELIGIBILITY ASSESSMENT")

        # Patient info
        lines.extend([
            f"Patient ID: {result.patient_id}",
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ])

        if patient:
            lines.extend([
                f"Age/Sex: {patient.age}y / {patient.sex}",
                f"BMI: {patient.bmi:.1f} ({patient.get_bmi_category()})",
                f"Viral Load: {'<50' if patient.viral_load < 50 else patient.viral_load} c/mL",
                f"CD4 Count: {patient.cd4_count} cells/mm3",
            ])

        lines.append("")

        # Eligibility Status (prominent)
        lines.append("=" * 70)
        if result.eligible:
            lines.append("*** ELIGIBILITY: APPROVED ***")
        else:
            lines.append("ELIGIBILITY: NOT APPROVED")

        lines.extend([
            "",
            f"Success Probability: {result.success_probability * 100:.0f}%",
            f"Risk Category: {result.get_risk_category().replace('_', ' ').title()}",
            "",
            f"RECOMMENDATION: {result.recommendation}",
            "",
            "=" * 70,
            "",
        ])

        # Risk Assessment
        lines.extend([
            "RISK ASSESSMENT",
            "-" * 40,
            f"CAB Resistance Risk: {result.cab_resistance_risk * 100:.0f}%",
            f"RPV Resistance Risk: {result.rpv_resistance_risk * 100:.0f}%",
            f"PK Adequacy: {result.pk_adequacy_score * 100:.0f}%",
            f"Adherence Score: {result.adherence_score * 100:.0f}%",
            "",
        ])

        # Risk Factors
        if result.risk_factors:
            lines.extend([
                "IDENTIFIED RISK FACTORS",
                "-" * 40,
            ])
            for rf in result.risk_factors:
                lines.append(f"  ! {rf}")
            lines.append("")

        # Detected Mutations
        if result.detected_mutations:
            lines.extend([
                "LA-RELEVANT MUTATIONS",
                "-" * 40,
            ])
            for mut in result.detected_mutations:
                lines.append(
                    f"  {mut['mutation']} ({mut['drug']}): "
                    f"{mut['fold_change']:.1f}x fold-change"
                )
            lines.append("")

        # Monitoring Plan
        if result.monitoring_plan:
            lines.extend([
                "RECOMMENDED MONITORING",
                "-" * 40,
            ])
            for i, item in enumerate(result.monitoring_plan, 1):
                lines.append(f"  {i}. {item}")
            lines.append("")

        # Footer
        lines.extend(self._generate_footer())

        return "\n".join(lines)

    def generate_resistance_report(
        self,
        report: ResistanceReport,
    ) -> str:
        """Generate Stanford HIVdb resistance report.

        Args:
            report: ResistanceReport from Stanford analysis

        Returns:
            Formatted report string
        """
        lines = self._generate_header("HIV DRUG RESISTANCE ANALYSIS")

        # Patient info
        lines.extend([
            f"Patient ID: {report.patient_id}",
            f"HIV-1 Subtype: {report.subtype}",
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ])

        # TDR Status
        lines.append("=" * 70)
        if report.has_tdr():
            lines.append("*** TRANSMITTED DRUG RESISTANCE DETECTED ***")
            sdrm = [m.notation for m in report.get_sdrm_mutations()]
            lines.append(f"SDRM Mutations: {', '.join(sdrm)}")
        else:
            lines.append("No transmitted drug resistance mutations detected.")
        lines.extend(["", "=" * 70, ""])

        # All Mutations
        if report.mutations:
            lines.extend([
                "ALL DETECTED MUTATIONS",
                "-" * 40,
            ])

            by_gene: dict[str, list[str]] = {}
            for mut in report.mutations:
                if mut.gene not in by_gene:
                    by_gene[mut.gene] = []
                marker = "*" if mut.is_sdrm else ""
                by_gene[mut.gene].append(f"{mut.notation}{marker}")

            for gene, muts in by_gene.items():
                lines.append(f"  {gene}: {', '.join(muts)}")

            lines.append("  (* = WHO SDRM)")
            lines.append("")

        # Drug Susceptibility by Class
        lines.extend([
            "DRUG SUSCEPTIBILITY",
            "-" * 40,
        ])

        for drug_class in ["NRTI", "NNRTI", "INSTI", "PI"]:
            class_drugs = [
                d for d in report.drug_scores if d.drug_class == drug_class
            ]
            if class_drugs:
                lines.append(f"\n{drug_class}:")
                for drug in class_drugs:
                    status = "[R]" if drug.is_resistant() else "[S]"
                    lines.append(
                        f"  {status} {drug.drug_abbr:6} "
                        f"Score: {drug.score:3}  {drug.text}"
                    )

        lines.append("")

        # Recommendations
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 40,
        ])

        resistant = report.get_resistant_drugs()
        if resistant:
            lines.append(f"Avoid: {', '.join(resistant)}")
        else:
            lines.append("All drugs appear susceptible.")

        regimens = report.get_recommended_regimens()
        lines.append(f"Recommended: {', '.join(regimens)}")
        lines.append("")

        # Quality Issues
        if report.quality_issues:
            lines.extend([
                "QUALITY WARNINGS",
                "-" * 40,
            ])
            for issue in report.quality_issues:
                lines.append(f"  - {issue}")
            lines.append("")

        # Footer
        lines.extend(self._generate_footer())

        return "\n".join(lines)

    def generate_batch_summary(
        self,
        tdr_results: Optional[list[TDRResult]] = None,
        la_results: Optional[list[LASelectionResult]] = None,
    ) -> str:
        """Generate summary report for batch analysis.

        Args:
            tdr_results: List of TDR screening results
            la_results: List of LA selection results

        Returns:
            Formatted summary report
        """
        lines = self._generate_header("BATCH ANALYSIS SUMMARY")
        lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        if tdr_results:
            tdr_positive = sum(1 for r in tdr_results if r.tdr_positive)
            lines.extend([
                "TDR SCREENING SUMMARY",
                "-" * 40,
                f"Total Screened: {len(tdr_results)}",
                f"TDR Positive: {tdr_positive}",
                f"TDR Prevalence: {tdr_positive / len(tdr_results) * 100:.1f}%",
                "",
            ])

            if tdr_positive > 0:
                lines.append("TDR-Positive Patients:")
                for r in tdr_results:
                    if r.tdr_positive:
                        lines.append(f"  - {r.patient_id}: {r.resistance_summary}")
                lines.append("")

        if la_results:
            eligible = sum(1 for r in la_results if r.eligible)
            avg_success = sum(r.success_probability for r in la_results) / len(la_results)

            lines.extend([
                "LA INJECTABLE SELECTION SUMMARY",
                "-" * 40,
                f"Total Assessed: {len(la_results)}",
                f"Eligible: {eligible} ({eligible / len(la_results) * 100:.1f}%)",
                f"Mean Success Probability: {avg_success * 100:.1f}%",
                "",
            ])

            lines.append("Individual Results:")
            for r in la_results:
                status = "ELIGIBLE" if r.eligible else "NOT ELIGIBLE"
                lines.append(
                    f"  - {r.patient_id}: {status} "
                    f"({r.success_probability * 100:.0f}% success)"
                )
            lines.append("")

        lines.extend(self._generate_footer())
        return "\n".join(lines)

    def save_report(
        self,
        report_text: str,
        output_path: Path,
        format: str = "txt",
    ) -> Path:
        """Save report to file.

        Args:
            report_text: Generated report text
            output_path: Output file path
            format: Output format ("txt" or "json")

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report_text)

        return output_path

    def export_json(
        self,
        results: list[TDRResult] | list[LASelectionResult],
        output_path: Path,
    ) -> Path:
        """Export results to JSON format.

        Args:
            results: List of result objects
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "report_date": datetime.now().isoformat(),
            "institution": self.institution,
            "results": [r.to_dict() for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def _generate_header(self, title: str) -> list[str]:
        """Generate report header."""
        lines = [
            "=" * 70,
            self.institution.center(70),
            title.center(70),
            "=" * 70,
            "",
        ]

        if self.provider:
            lines.insert(2, f"Provider: {self.provider}".center(70))

        return lines

    def _generate_footer(self) -> list[str]:
        """Generate report footer."""
        return [
            "",
            "-" * 70,
            "DISCLAIMER: This report is for clinical decision support only.",
            "Final treatment decisions should be made by qualified clinicians",
            "in consultation with patients.",
            "-" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "HIV Clinical Decision Support System v1.0",
            "",
        ]

    def _get_status_symbol(self, status: str) -> str:
        """Get symbol for susceptibility status."""
        symbols = {
            "susceptible": "[S]",
            "possible_resistance": "[?]",
            "resistant": "[R]",
        }
        return symbols.get(status, "[?]")

    def _get_level_indicator(self, level: str) -> str:
        """Get indicator for resistance level."""
        indicators = {
            "high": "[!!!]",
            "moderate": "[!!]",
            "low": "[!]",
        }
        return indicators.get(level, "[?]")
