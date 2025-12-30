# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for HIV Research Package.

Tests TDR screening, LA injectable selection, Stanford client,
and clinical report generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent
sys.path.insert(0, str(deliverables_dir))
sys.path.insert(0, str(deliverables_dir / "partners" / "hiv_research_package"))

from partners.hiv_research_package.src.models import (
    TDRResult,
    PatientData,
    LASelectionResult,
    ResistanceLevel,
    DrugScore,
    MutationInfo,
    ResistanceReport,
)
from partners.hiv_research_package.src.constants import (
    TDR_MUTATIONS,
    FIRST_LINE_DRUGS,
    LA_DRUGS,
    WHO_SDRM_NRTI,
)
from partners.hiv_research_package.src.tdr import TDRScreener, generate_demo_sequence
from partners.hiv_research_package.src.la_injectable import LASelector, generate_demo_patients
from partners.hiv_research_package.src.stanford_client import StanfordHIVdbClient
from partners.hiv_research_package.src.report import ClinicalReportGenerator


class TestResistanceLevel:
    """Tests for ResistanceLevel enum."""

    def test_from_score_susceptible(self):
        """Test score < 10 is susceptible."""
        assert ResistanceLevel.from_score(0) == ResistanceLevel.SUSCEPTIBLE
        assert ResistanceLevel.from_score(9) == ResistanceLevel.SUSCEPTIBLE

    def test_from_score_potential_low(self):
        """Test score 10-14 is potential low."""
        assert ResistanceLevel.from_score(10) == ResistanceLevel.POTENTIAL_LOW
        assert ResistanceLevel.from_score(14) == ResistanceLevel.POTENTIAL_LOW

    def test_from_score_low(self):
        """Test score 15-29 is low."""
        assert ResistanceLevel.from_score(15) == ResistanceLevel.LOW
        assert ResistanceLevel.from_score(29) == ResistanceLevel.LOW

    def test_from_score_intermediate(self):
        """Test score 30-59 is intermediate."""
        assert ResistanceLevel.from_score(30) == ResistanceLevel.INTERMEDIATE
        assert ResistanceLevel.from_score(59) == ResistanceLevel.INTERMEDIATE

    def test_from_score_high(self):
        """Test score >= 60 is high."""
        assert ResistanceLevel.from_score(60) == ResistanceLevel.HIGH
        assert ResistanceLevel.from_score(100) == ResistanceLevel.HIGH

    def test_from_text(self):
        """Test conversion from text description."""
        assert ResistanceLevel.from_text("Susceptible") == ResistanceLevel.SUSCEPTIBLE
        assert ResistanceLevel.from_text("High-Level") == ResistanceLevel.HIGH
        assert ResistanceLevel.from_text("INTERMEDIATE") == ResistanceLevel.INTERMEDIATE

    def test_is_resistant(self):
        """Test is_resistant method."""
        assert not ResistanceLevel.SUSCEPTIBLE.is_resistant()
        assert not ResistanceLevel.POTENTIAL_LOW.is_resistant()
        assert ResistanceLevel.LOW.is_resistant()
        assert ResistanceLevel.INTERMEDIATE.is_resistant()
        assert ResistanceLevel.HIGH.is_resistant()


class TestPatientData:
    """Tests for PatientData dataclass."""

    def test_patient_creation(self):
        """Test creating a PatientData instance."""
        patient = PatientData(
            patient_id="P001",
            age=35,
            sex="M",
            bmi=24.5,
            viral_load=0,
            cd4_count=650,
            prior_regimens=["TDF/FTC/DTG"],
            adherence_history="excellent",
        )
        assert patient.patient_id == "P001"
        assert patient.age == 35
        assert patient.bmi == 24.5

    def test_is_virally_suppressed(self):
        """Test viral suppression check."""
        suppressed = PatientData(
            patient_id="P1", age=30, sex="F", bmi=22,
            viral_load=0, cd4_count=500,
            prior_regimens=[], adherence_history="good"
        )
        assert suppressed.is_virally_suppressed()

        unsuppressed = PatientData(
            patient_id="P2", age=30, sex="F", bmi=22,
            viral_load=500, cd4_count=500,
            prior_regimens=[], adherence_history="good"
        )
        assert not unsuppressed.is_virally_suppressed()

    def test_bmi_category(self):
        """Test BMI category classification."""
        def make_patient(bmi: float) -> PatientData:
            return PatientData(
                patient_id="P", age=30, sex="M", bmi=bmi,
                viral_load=0, cd4_count=500,
                prior_regimens=[], adherence_history="good"
            )

        assert make_patient(17.0).get_bmi_category() == "underweight"
        assert make_patient(22.0).get_bmi_category() == "normal"
        assert make_patient(27.0).get_bmi_category() == "overweight"
        assert make_patient(32.0).get_bmi_category() == "obese_1"
        assert make_patient(37.0).get_bmi_category() == "obese_2"
        assert make_patient(42.0).get_bmi_category() == "obese_3"


class TestTDRScreener:
    """Tests for TDRScreener class."""

    def test_screener_creation(self):
        """Test creating a TDRScreener instance."""
        screener = TDRScreener()
        assert screener.use_stanford is False

        screener_stanford = TDRScreener(use_stanford=True)
        assert screener_stanford.use_stanford is True

    def test_parse_sequence_valid(self):
        """Test parsing valid sequences."""
        screener = TDRScreener()

        # Simple sequence
        result = screener.parse_sequence("MKVLIYG")
        assert result == "MKVLIYG"

        # With whitespace
        result = screener.parse_sequence("MKV LIY G")
        assert result == "MKVLIYG"

        # Lowercase
        result = screener.parse_sequence("mkvliyg")
        assert result == "MKVLIYG"

    def test_parse_sequence_invalid(self):
        """Test parsing invalid sequences raises error."""
        screener = TDRScreener()

        with pytest.raises(ValueError) as exc:
            screener.parse_sequence("MKVB123")
        assert "Invalid amino acids" in str(exc.value)

    def test_screen_patient_basic(self):
        """Test basic patient screening."""
        screener = TDRScreener()
        sequence = generate_demo_sequence()

        result = screener.screen_patient(sequence, patient_id="TEST_001")

        assert isinstance(result, TDRResult)
        assert result.patient_id == "TEST_001"
        assert isinstance(result.tdr_positive, bool)
        assert result.recommended_regimen is not None
        assert 0 <= result.confidence <= 1

    def test_screen_patient_reproducible(self):
        """Test that screening is reproducible with same sequence."""
        screener = TDRScreener()
        sequence = generate_demo_sequence()

        result1 = screener.screen_patient(sequence, "P1")
        result2 = screener.screen_patient(sequence, "P2")

        # Same sequence should yield same mutations (seeded by sequence)
        assert result1.tdr_positive == result2.tdr_positive
        assert len(result1.detected_mutations) == len(result2.detected_mutations)

    def test_predict_drug_susceptibility(self):
        """Test drug susceptibility prediction."""
        screener = TDRScreener()

        # With no mutations
        susceptibility = screener.predict_drug_susceptibility([])
        assert all(d["status"] == "susceptible" for d in susceptibility.values())

        # With M184V mutation
        mutations = [{
            "mutation": "M184V",
            "drug_class": "NRTI",
            "affected_drugs": ["3TC", "FTC"],
            "resistance_level": "high",
            "prevalence": 5.2,
        }]
        susceptibility = screener.predict_drug_susceptibility(mutations)
        assert susceptibility["3TC"]["status"] == "resistant"
        assert susceptibility["FTC"]["status"] == "resistant"
        assert susceptibility["DTG"]["status"] == "susceptible"

    def test_recommend_regimen(self):
        """Test regimen recommendation."""
        screener = TDRScreener()

        # Susceptible to all
        susceptibility = {
            drug: {"status": "susceptible", "score": 0}
            for class_drugs in FIRST_LINE_DRUGS.values()
            for drug in class_drugs
        }
        primary, alternatives = screener.recommend_regimen(susceptibility, [])
        assert "DTG" in primary  # DTG-based preferred

        # With 3TC/FTC resistance
        susceptibility["3TC"]["status"] = "resistant"
        susceptibility["FTC"]["status"] = "resistant"
        primary, alternatives = screener.recommend_regimen(susceptibility, [])
        # Should still recommend something


class TestLASelector:
    """Tests for LASelector class."""

    def test_selector_creation(self):
        """Test creating an LASelector instance."""
        selector = LASelector()
        assert selector.use_stanford is False

    def test_compute_pk_adequacy(self):
        """Test PK adequacy computation."""
        selector = LASelector()

        # Normal BMI
        normal_patient = PatientData(
            patient_id="P1", age=35, sex="M", bmi=22,
            viral_load=0, cd4_count=600,
            prior_regimens=[], adherence_history="excellent"
        )
        pk = selector.compute_pk_adequacy(normal_patient)
        assert pk == 1.0

        # High BMI
        obese_patient = PatientData(
            patient_id="P2", age=35, sex="M", bmi=38,
            viral_load=0, cd4_count=600,
            prior_regimens=[], adherence_history="excellent"
        )
        pk = selector.compute_pk_adequacy(obese_patient)
        assert pk < 1.0  # Should be reduced

    def test_compute_adherence_score(self):
        """Test adherence score computation."""
        selector = LASelector()

        for history, expected_min in [
            ("excellent", 0.9),
            ("good", 0.8),
            ("moderate", 0.6),
            ("poor", 0.4),
        ]:
            patient = PatientData(
                patient_id="P", age=35, sex="M", bmi=22,
                viral_load=0, cd4_count=600,
                prior_regimens=[], adherence_history=history
            )
            score = selector.compute_adherence_score(patient)
            assert score >= expected_min

    def test_assess_eligibility_suppressed_patient(self):
        """Test eligibility for virally suppressed patient."""
        selector = LASelector()

        patient = PatientData(
            patient_id="P001", age=35, sex="M", bmi=24,
            viral_load=0, cd4_count=650,
            prior_regimens=["TDF/FTC/DTG"],
            adherence_history="excellent"
        )

        result = selector.assess_eligibility(patient)

        assert isinstance(result, LASelectionResult)
        assert result.patient_id == "P001"
        assert result.eligible is True
        assert result.success_probability > 0.8

    def test_assess_eligibility_unsuppressed_patient(self):
        """Test eligibility for unsuppressed patient."""
        selector = LASelector()

        patient = PatientData(
            patient_id="P002", age=35, sex="M", bmi=24,
            viral_load=500,  # Not suppressed
            cd4_count=350,
            prior_regimens=["TDF/3TC/EFV"],
            adherence_history="poor"
        )

        result = selector.assess_eligibility(patient)

        assert result.eligible is False
        assert "Not virally suppressed" in result.risk_factors


class TestStanfordHIVdbClient:
    """Tests for StanfordHIVdbClient."""

    def test_client_creation(self):
        """Test creating a client instance."""
        client = StanfordHIVdbClient()
        assert client.timeout == 60

    def test_mock_analysis(self):
        """Test mock analysis when API unavailable."""
        client = StanfordHIVdbClient()
        sequence = generate_demo_sequence()

        report = client._mock_analysis(sequence, "TEST_001")

        assert isinstance(report, ResistanceReport)
        assert report.patient_id == "TEST_001"
        assert report.subtype in ["B", "C", "CRF01_AE", "CRF02_AG", "A1"]

    def test_mock_analysis_reproducible(self):
        """Test mock analysis is reproducible."""
        client = StanfordHIVdbClient()
        sequence = generate_demo_sequence()

        report1 = client._mock_analysis(sequence, "P1")
        report2 = client._mock_analysis(sequence, "P2")

        # Same sequence should give same subtype
        assert report1.subtype == report2.subtype

    def test_is_sdrm(self):
        """Test SDRM detection."""
        client = StanfordHIVdbClient()

        # Known SDRMs
        assert client._is_sdrm("K103N", "RT") is True
        assert client._is_sdrm("M184V", "RT") is True
        assert client._is_sdrm("N155H", "IN") is True

        # Non-SDRMs
        assert client._is_sdrm("A62V", "RT") is False

    def test_generate_report(self):
        """Test report generation."""
        client = StanfordHIVdbClient()
        sequence = generate_demo_sequence()
        analysis = client._mock_analysis(sequence, "TEST")

        report_text = client.generate_report(analysis)

        assert "HIV DRUG RESISTANCE REPORT" in report_text
        assert "TEST" in report_text
        assert "DRUG SUSCEPTIBILITY" in report_text


class TestClinicalReportGenerator:
    """Tests for ClinicalReportGenerator."""

    def test_generator_creation(self):
        """Test creating a report generator."""
        gen = ClinicalReportGenerator()
        assert gen.institution == "HIV Clinical Decision Support"

        gen_custom = ClinicalReportGenerator(institution="Test Hospital")
        assert gen_custom.institution == "Test Hospital"

    def test_generate_tdr_report(self):
        """Test TDR report generation."""
        gen = ClinicalReportGenerator()

        result = TDRResult(
            patient_id="P001",
            sequence_id=None,
            detected_mutations=[],
            drug_susceptibility={
                "DTG": {"status": "susceptible", "score": 0, "class": "INSTI"},
            },
            tdr_positive=False,
            recommended_regimen="TDF/3TC/DTG",
            alternative_regimens=["TDF/FTC/DTG"],
            resistance_summary="No TDR detected",
            confidence=0.95,
        )

        report = gen.generate_tdr_report(result)

        assert "TRANSMITTED DRUG RESISTANCE" in report
        assert "P001" in report
        assert "TDF/3TC/DTG" in report

    def test_generate_la_report(self):
        """Test LA selection report generation."""
        gen = ClinicalReportGenerator()

        result = LASelectionResult(
            patient_id="P001",
            eligible=True,
            success_probability=0.92,
            cab_resistance_risk=0.0,
            rpv_resistance_risk=0.0,
            pk_adequacy_score=1.0,
            adherence_score=0.95,
            detected_mutations=[],
            recommendation="ELIGIBLE - Recommend LA switch",
            risk_factors=[],
            monitoring_plan=["HIV RNA at 1 month"],
        )

        patient = PatientData(
            patient_id="P001", age=35, sex="M", bmi=24,
            viral_load=0, cd4_count=650,
            prior_regimens=["TDF/FTC/DTG"],
            adherence_history="excellent"
        )

        report = gen.generate_la_report(result, patient)

        assert "LONG-ACTING INJECTABLE" in report
        assert "APPROVED" in report
        assert "92%" in report


class TestConstants:
    """Tests for constants module."""

    def test_tdr_mutations_structure(self):
        """Test TDR mutations have correct structure."""
        for drug_class, mutations in TDR_MUTATIONS.items():
            assert drug_class in ["NRTI", "NNRTI", "INSTI", "PI"]
            for mut_name, mut_info in mutations.items():
                assert "drugs" in mut_info
                assert "level" in mut_info
                assert "prevalence" in mut_info
                assert mut_info["level"] in ["high", "moderate", "low"]

    def test_first_line_drugs(self):
        """Test first line drugs structure."""
        assert "NRTI" in FIRST_LINE_DRUGS
        assert "INSTI" in FIRST_LINE_DRUGS
        assert "DTG" in FIRST_LINE_DRUGS["INSTI"]

    def test_la_drugs(self):
        """Test LA drugs structure."""
        assert "CAB" in LA_DRUGS
        assert "RPV" in LA_DRUGS
        assert LA_DRUGS["CAB"]["class"] == "INSTI"
        assert LA_DRUGS["RPV"]["class"] == "NNRTI"

    def test_who_sdrm_lists(self):
        """Test WHO SDRM lists contain known mutations."""
        assert "M184V" in WHO_SDRM_NRTI
        assert "K65R" in WHO_SDRM_NRTI


class TestDemoDataGeneration:
    """Tests for demo data generation functions."""

    def test_generate_demo_sequence(self):
        """Test demo sequence generation."""
        seq = generate_demo_sequence()
        assert len(seq) == 500
        assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)

    def test_generate_demo_patients(self):
        """Test demo patient generation."""
        patients = generate_demo_patients(5)
        assert len(patients) == 5

        for p in patients:
            assert isinstance(p, PatientData)
            assert 25 <= p.age <= 65
            assert p.sex in ["M", "F"]
            assert p.adherence_history in ["excellent", "good", "moderate", "poor"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
