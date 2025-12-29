# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Drug-Drug Interaction Checker.

Tests cover:
- Drug name normalization
- Known interaction detection
- Mechanism-based interaction inference
- Regimen checking
- Severity classification
- Risk assessment
- Alternative drug suggestions
"""

from __future__ import annotations

import pytest

from src.clinical.drug_interactions import (
    DrugInteractionChecker,
    InteractionSeverity,
    InteractionMechanism,
    DrugCategory,
    DrugInfo,
    Interaction,
    RegimenReport,
)


class TestDrugInteractionChecker:
    """Tests for DrugInteractionChecker class."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_initialization(self, checker):
        """Test checker initialization."""
        assert len(checker.drug_database) > 50
        assert len(checker.known_interactions) > 10

    def test_drug_database_contains_hiv_drugs(self, checker):
        """Test drug database contains HIV drugs."""
        assert "ritonavir" in checker.drug_database
        assert "dolutegravir" in checker.drug_database
        assert "tenofovir" in checker.drug_database

    def test_drug_database_contains_antibiotics(self, checker):
        """Test drug database contains antibiotics."""
        assert "vancomycin" in checker.drug_database
        assert "gentamicin" in checker.drug_database
        assert "meropenem" in checker.drug_database


class TestDrugNameNormalization:
    """Tests for drug name normalization."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_normalize_generic_name(self, checker):
        """Test normalization of generic names."""
        assert checker._normalize_drug_name("ritonavir") == "ritonavir"
        assert checker._normalize_drug_name("Ritonavir") == "ritonavir"
        assert checker._normalize_drug_name("RITONAVIR") == "ritonavir"

    def test_normalize_by_alias(self, checker):
        """Test normalization using drug aliases."""
        assert checker._normalize_drug_name("RTV") == "ritonavir"
        assert checker._normalize_drug_name("Norvir") == "ritonavir"
        assert checker._normalize_drug_name("3TC") == "lamivudine"

    def test_normalize_with_hyphens(self, checker):
        """Test normalization handles hyphens."""
        assert checker._normalize_drug_name("piperacillin-tazobactam") == "piperacillin_tazobactam"

    def test_normalize_unknown_drug(self, checker):
        """Test normalization returns None for unknown drugs."""
        assert checker._normalize_drug_name("unknown_drug_xyz") is None


class TestGetDrugInfo:
    """Tests for getting drug information."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_get_drug_info_by_generic(self, checker):
        """Test getting drug info by generic name."""
        info = checker.get_drug_info("ritonavir")
        assert info is not None
        assert info.name == "Ritonavir"
        assert info.cyp3a4_inhibitor is True

    def test_get_drug_info_by_alias(self, checker):
        """Test getting drug info by alias."""
        info = checker.get_drug_info("RTV")
        assert info is not None
        assert info.name == "Ritonavir"

    def test_get_drug_info_unknown(self, checker):
        """Test getting drug info for unknown drug."""
        info = checker.get_drug_info("unknown_drug")
        assert info is None


class TestKnownInteractions:
    """Tests for known drug interactions."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_ritonavir_simvastatin_contraindicated(self, checker):
        """Test ritonavir + simvastatin is contraindicated."""
        interaction = checker.check_interaction("ritonavir", "simvastatin")
        assert interaction is not None
        assert interaction.severity == InteractionSeverity.CONTRAINDICATED
        assert interaction.mechanism == InteractionMechanism.CYP3A4_INHIBITION
        assert "rhabdomyolysis" in interaction.clinical_effect.lower()

    def test_ritonavir_lovastatin_contraindicated(self, checker):
        """Test ritonavir + lovastatin is contraindicated."""
        interaction = checker.check_interaction("ritonavir", "lovastatin")
        assert interaction is not None
        assert interaction.severity == InteractionSeverity.CONTRAINDICATED

    def test_rifampin_ritonavir_contraindicated(self, checker):
        """Test rifampin + ritonavir is contraindicated."""
        interaction = checker.check_interaction("rifampin", "ritonavir")
        assert interaction is not None
        assert interaction.severity == InteractionSeverity.CONTRAINDICATED
        assert interaction.mechanism == InteractionMechanism.CYP3A4_INDUCTION

    def test_vancomycin_gentamicin_major(self, checker):
        """Test vancomycin + gentamicin is major interaction."""
        interaction = checker.check_interaction("vancomycin", "gentamicin")
        assert interaction is not None
        assert interaction.severity == InteractionSeverity.MAJOR
        assert interaction.mechanism == InteractionMechanism.NEPHROTOXICITY_SYNERGY

    def test_interaction_order_independent(self, checker):
        """Test interaction detection works in both orders."""
        i1 = checker.check_interaction("ritonavir", "simvastatin")
        i2 = checker.check_interaction("simvastatin", "ritonavir")
        assert i1 is not None
        assert i2 is not None
        assert i1.severity == i2.severity


class TestInferredInteractions:
    """Tests for mechanism-based inferred interactions."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_qt_prolongation_inference(self, checker):
        """Test QT prolongation interaction inference."""
        # moxifloxacin + levofloxacin both prolong QT
        interaction = checker.check_interaction("moxifloxacin", "levofloxacin")
        assert interaction is not None
        assert interaction.mechanism == InteractionMechanism.QT_PROLONGATION
        assert "ECG" in str(interaction.monitoring)

    def test_nephrotoxicity_inference(self, checker):
        """Test nephrotoxicity synergy inference."""
        # colistin + amikacin both nephrotoxic
        interaction = checker.check_interaction("colistin", "amikacin")
        assert interaction is not None
        assert interaction.mechanism == InteractionMechanism.NEPHROTOXICITY_SYNERGY

    def test_cyp3a4_inhibitor_substrate(self, checker):
        """Test CYP3A4 inhibitor + substrate inference."""
        # itraconazole (inhibitor) + doravirine (substrate)
        interaction = checker.check_interaction("itraconazole", "doravirine")
        assert interaction is not None
        assert interaction.mechanism == InteractionMechanism.CYP3A4_INHIBITION

    def test_hepatotoxicity_inference(self, checker):
        """Test hepatotoxicity synergy inference."""
        # nevirapine + isoniazid both hepatotoxic (no CYP interaction to override)
        interaction = checker.check_interaction("nevirapine", "isoniazid")
        assert interaction is not None
        assert interaction.mechanism == InteractionMechanism.HEPATOTOXICITY_SYNERGY

    def test_no_interaction_safe_drugs(self, checker):
        """Test no interaction between safe drug pair."""
        # lamivudine + raltegravir - both relatively safe
        interaction = checker.check_interaction("lamivudine", "raltegravir")
        assert interaction is None

    def test_same_drug_no_interaction(self, checker):
        """Test same drug returns no interaction."""
        interaction = checker.check_interaction("ritonavir", "ritonavir")
        assert interaction is None


class TestRegimenChecking:
    """Tests for regimen interaction checking."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_safe_hiv_regimen(self, checker):
        """Test a safe HIV regimen."""
        regimen = ["tenofovir", "lamivudine", "dolutegravir"]
        report = checker.check_regimen(regimen)

        assert report.n_contraindicated == 0
        assert report.safe_to_use is True

    def test_risky_regimen_with_statins(self, checker):
        """Test risky regimen with ritonavir + statin."""
        regimen = ["ritonavir", "darunavir", "simvastatin"]
        report = checker.check_regimen(regimen)

        assert report.n_contraindicated >= 1
        assert report.safe_to_use is False
        assert "contraindicated" in report.summary.lower()

    def test_nephrotoxic_regimen(self, checker):
        """Test nephrotoxic regimen risk assessment."""
        regimen = ["vancomycin", "gentamicin", "colistin"]
        report = checker.check_regimen(regimen)

        assert report.nephrotoxicity_risk == "high"
        assert report.n_major >= 2

    def test_qt_prolonging_regimen(self, checker):
        """Test QT-prolonging regimen risk assessment."""
        regimen = ["moxifloxacin", "azithromycin", "fluconazole"]
        report = checker.check_regimen(regimen)

        assert report.qt_risk in ["moderate", "high"]

    def test_empty_regimen(self, checker):
        """Test empty regimen."""
        report = checker.check_regimen([])
        assert report.n_contraindicated == 0
        assert report.n_major == 0
        assert report.safe_to_use is True

    def test_single_drug_regimen(self, checker):
        """Test single drug regimen."""
        report = checker.check_regimen(["dolutegravir"])
        assert len(report.interactions) == 0
        assert report.safe_to_use is True

    def test_regimen_report_recommendations(self, checker):
        """Test regimen report generates recommendations."""
        regimen = ["ritonavir", "simvastatin", "moxifloxacin", "azithromycin"]
        report = checker.check_regimen(regimen)

        assert len(report.recommendations) > 0
        assert any("URGENT" in r for r in report.recommendations)


class TestSeverityClassification:
    """Tests for interaction severity classification."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_severity_enum_values(self):
        """Test severity enum has expected values."""
        assert InteractionSeverity.CONTRAINDICATED.value == "contraindicated"
        assert InteractionSeverity.MAJOR.value == "major"
        assert InteractionSeverity.MODERATE.value == "moderate"
        assert InteractionSeverity.MINOR.value == "minor"

    def test_contraindicated_severity(self, checker):
        """Test contraindicated interactions are properly classified."""
        interaction = checker.check_interaction("rifampin", "voriconazole")
        assert interaction.severity == InteractionSeverity.CONTRAINDICATED

    def test_major_severity(self, checker):
        """Test major interactions are properly classified."""
        interaction = checker.check_interaction("vancomycin", "piperacillin_tazobactam")
        assert interaction.severity == InteractionSeverity.MAJOR


class TestAlternativeSuggestions:
    """Tests for alternative drug suggestions."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_statin_alternatives(self, checker):
        """Test statin alternative suggestions."""
        alternatives = checker.get_alternatives("simvastatin", DrugCategory.STATIN)
        assert len(alternatives) > 0
        assert "pravastatin" in alternatives or "rosuvastatin" in alternatives

    def test_azole_alternatives(self, checker):
        """Test azole antifungal alternatives."""
        alternatives = checker.get_alternatives("voriconazole", DrugCategory.AZOLE_ANTIFUNGAL)
        assert len(alternatives) > 0

    def test_alternatives_for_unknown_drug(self, checker):
        """Test alternatives for unknown drug."""
        alternatives = checker.get_alternatives("unknown_drug_xyz")
        assert alternatives == []


class TestDrugCategories:
    """Tests for drug category classification."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_hiv_pi_category(self, checker):
        """Test HIV PI categorization."""
        info = checker.get_drug_info("darunavir")
        assert info.category == DrugCategory.HIV_PI

    def test_aminoglycoside_category(self, checker):
        """Test aminoglycoside categorization."""
        info = checker.get_drug_info("gentamicin")
        assert info.category == DrugCategory.AMINOGLYCOSIDE

    def test_azole_category(self, checker):
        """Test azole antifungal categorization."""
        info = checker.get_drug_info("fluconazole")
        assert info.category == DrugCategory.AZOLE_ANTIFUNGAL

    def test_list_drugs_by_category(self, checker):
        """Test listing drugs by category."""
        pis = checker.list_drugs_by_category(DrugCategory.HIV_PI)
        assert len(pis) >= 3
        assert all(d.category == DrugCategory.HIV_PI for d in pis)


class TestInteractionSummary:
    """Tests for drug interaction summary."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_get_interaction_summary(self, checker):
        """Test getting interaction summary for a drug."""
        summary = checker.get_interaction_summary("ritonavir")

        assert summary["drug"] == "ritonavir"
        assert summary["name"] == "Ritonavir"
        assert summary["properties"]["cyp3a4_inhibitor"] is True
        assert summary["n_interactions"] > 0

    def test_interaction_summary_unknown_drug(self, checker):
        """Test interaction summary for unknown drug."""
        summary = checker.get_interaction_summary("unknown_drug")
        assert "error" in summary

    def test_interaction_summary_sorted_by_severity(self, checker):
        """Test interactions are sorted by severity."""
        summary = checker.get_interaction_summary("ritonavir")
        interactions = summary["interactions"]

        if len(interactions) >= 2:
            severity_order = ["contraindicated", "major", "moderate", "minor"]
            for i in range(len(interactions) - 1):
                current_severity = interactions[i]["severity"]
                next_severity = interactions[i + 1]["severity"]
                assert severity_order.index(current_severity) <= severity_order.index(next_severity)


class TestInteractionMechanisms:
    """Tests for interaction mechanism classification."""

    def test_mechanism_enum_values(self):
        """Test mechanism enum has expected values."""
        assert InteractionMechanism.CYP3A4_INHIBITION.value == "cyp3a4_inhibition"
        assert InteractionMechanism.QT_PROLONGATION.value == "qt_prolongation"
        assert InteractionMechanism.NEPHROTOXICITY_SYNERGY.value == "nephrotoxicity_synergy"


class TestDrugInfoDataclass:
    """Tests for DrugInfo dataclass."""

    def test_drug_info_creation(self):
        """Test DrugInfo creation."""
        info = DrugInfo(
            name="Test Drug",
            generic_name="test_drug",
            category=DrugCategory.BETA_LACTAM,
            nephrotoxic=True,
        )
        assert info.name == "Test Drug"
        assert info.nephrotoxic is True
        assert info.qt_prolongation is False  # default

    def test_drug_info_aliases(self):
        """Test DrugInfo with aliases."""
        info = DrugInfo(
            name="Test Drug",
            generic_name="test_drug",
            category=DrugCategory.BETA_LACTAM,
            aliases=["TD", "TestDrug"],
        )
        assert len(info.aliases) == 2


class TestRegimenReportDataclass:
    """Tests for RegimenReport dataclass."""

    def test_regimen_report_defaults(self):
        """Test RegimenReport default values."""
        report = RegimenReport(drugs=["drug1"], interactions=[])
        assert report.n_contraindicated == 0
        assert report.qt_risk == "low"
        assert report.safe_to_use is True


class TestIntegration:
    """Integration tests for drug interaction checker."""

    @pytest.fixture
    def checker(self):
        """Create checker fixture."""
        return DrugInteractionChecker()

    def test_complex_hiv_hcv_regimen(self, checker):
        """Test complex HIV/HCV coinfection regimen."""
        regimen = [
            "tenofovir",
            "emtricitabine",
            "dolutegravir",
            "sofosbuvir",
            "velpatasvir",
        ]
        report = checker.check_regimen(regimen)

        # Should be relatively safe
        assert report.n_contraindicated == 0

    def test_complex_icu_regimen(self, checker):
        """Test complex ICU antibiotic regimen."""
        regimen = [
            "meropenem",
            "vancomycin",
            "fluconazole",
            "amikacin",
        ]
        report = checker.check_regimen(regimen)

        # Should have nephrotoxicity concerns
        assert report.nephrotoxicity_risk in ["moderate", "high"]

    def test_tb_hiv_cotreatment(self, checker):
        """Test TB/HIV cotreatment interactions."""
        # This should detect rifampin-PI interaction
        interaction = checker.check_interaction("rifampin", "lopinavir")
        assert interaction is not None
        assert interaction.severity == InteractionSeverity.CONTRAINDICATED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
