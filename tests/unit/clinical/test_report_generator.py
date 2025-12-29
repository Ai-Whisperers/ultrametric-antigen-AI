# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Clinical Report Generator.

Tests cover:
- Report format enums
- Drug prediction dataclass
- Report configuration
- HTML report rendering
- JSON report rendering
- PDF report rendering
- Report generation workflow
- Report archive management
- Multi-language support
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.clinical.report_generator import (
    BaseReportRenderer,
    DrugPrediction,
    HTMLReportRenderer,
    JSONReportRenderer,
    PDFReportRenderer,
    ReportArchive,
    ReportConfig,
    ReportFormat,
    ReportGenerator,
    ReportLanguage,
    ResistanceReport,
    TRANSLATIONS,
)


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_html_format(self):
        """Test HTML format value."""
        assert ReportFormat.HTML.value == "html"

    def test_pdf_format(self):
        """Test PDF format value."""
        assert ReportFormat.PDF.value == "pdf"

    def test_json_format(self):
        """Test JSON format value."""
        assert ReportFormat.JSON.value == "json"


class TestReportLanguage:
    """Tests for ReportLanguage enum."""

    def test_english_language(self):
        """Test English language code."""
        assert ReportLanguage.ENGLISH.value == "en"

    def test_spanish_language(self):
        """Test Spanish language code."""
        assert ReportLanguage.SPANISH.value == "es"

    def test_french_language(self):
        """Test French language code."""
        assert ReportLanguage.FRENCH.value == "fr"


class TestDrugPrediction:
    """Tests for DrugPrediction dataclass."""

    def test_basic_creation(self):
        """Test basic drug prediction creation."""
        pred = DrugPrediction(
            drug_name="AZT",
            drug_class="NRTI",
            resistance_score=0.25,
            classification="susceptible",
            confidence=0.95,
        )
        assert pred.drug_name == "AZT"
        assert pred.drug_class == "NRTI"
        assert pred.resistance_score == 0.25
        assert pred.classification == "susceptible"
        assert pred.confidence == 0.95

    def test_with_mutations(self):
        """Test prediction with mutations."""
        pred = DrugPrediction(
            drug_name="EFV",
            drug_class="NNRTI",
            resistance_score=0.85,
            classification="resistant",
            confidence=0.92,
            mutations=["K103N", "Y181C"],
        )
        assert len(pred.mutations) == 2
        assert "K103N" in pred.mutations

    def test_with_interpretation(self):
        """Test prediction with interpretation."""
        pred = DrugPrediction(
            drug_name="LPV",
            drug_class="PI",
            resistance_score=0.45,
            classification="intermediate",
            confidence=0.88,
            interpretation="Reduced susceptibility likely",
        )
        assert "Reduced" in pred.interpretation

    def test_default_values(self):
        """Test default values for optional fields."""
        pred = DrugPrediction(
            drug_name="DRV",
            drug_class="PI",
            resistance_score=0.1,
            classification="susceptible",
            confidence=0.99,
        )
        assert pred.mutations == []
        assert pred.interpretation == ""


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        assert config.format == ReportFormat.HTML
        assert config.language == ReportLanguage.ENGLISH
        assert config.include_logo is True
        assert config.include_mutations is True
        assert config.include_interpretation is True
        assert config.include_recommendations is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            format=ReportFormat.PDF,
            language=ReportLanguage.SPANISH,
            include_logo=False,
            organization_name="Test Lab",
        )
        assert config.format == ReportFormat.PDF
        assert config.language == ReportLanguage.SPANISH
        assert config.organization_name == "Test Lab"

    def test_custom_header_footer(self):
        """Test custom header and footer."""
        config = ReportConfig(
            custom_header="Custom Header",
            custom_footer="Custom Footer",
        )
        assert config.custom_header == "Custom Header"
        assert config.custom_footer == "Custom Footer"


class TestTranslations:
    """Tests for translation dictionaries."""

    def test_english_translations_complete(self):
        """Test English translations are complete."""
        en = TRANSLATIONS[ReportLanguage.ENGLISH]
        required_keys = [
            "title", "patient_id", "sequence_length", "analysis_date",
            "disease", "drug_resistance", "drug", "class", "score",
            "classification", "confidence", "mutations", "interpretation",
            "recommendations", "recommended", "avoid", "warnings",
            "overall", "susceptible", "intermediate", "resistant",
        ]
        for key in required_keys:
            assert key in en, f"Missing key: {key}"

    def test_spanish_translations_complete(self):
        """Test Spanish translations are complete."""
        es = TRANSLATIONS[ReportLanguage.SPANISH]
        en_keys = set(TRANSLATIONS[ReportLanguage.ENGLISH].keys())
        es_keys = set(es.keys())
        assert en_keys == es_keys, "Spanish translations incomplete"

    def test_french_translations_complete(self):
        """Test French translations are complete."""
        fr = TRANSLATIONS[ReportLanguage.FRENCH]
        en_keys = set(TRANSLATIONS[ReportLanguage.ENGLISH].keys())
        fr_keys = set(fr.keys())
        assert en_keys == fr_keys, "French translations incomplete"

    def test_translation_content_differs(self):
        """Test translations actually differ between languages."""
        en_title = TRANSLATIONS[ReportLanguage.ENGLISH]["title"]
        es_title = TRANSLATIONS[ReportLanguage.SPANISH]["title"]
        fr_title = TRANSLATIONS[ReportLanguage.FRENCH]["title"]

        assert en_title != es_title
        assert en_title != fr_title
        assert es_title != fr_title


class TestHTMLReportRenderer:
    """Tests for HTML report rendering."""

    @pytest.fixture
    def config(self):
        """Create default config fixture."""
        return ReportConfig()

    @pytest.fixture
    def sample_report(self, config):
        """Create sample report fixture."""
        predictions = [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.15,
                classification="susceptible",
                confidence=0.95,
                mutations=[],
            ),
            DrugPrediction(
                drug_name="EFV",
                drug_class="NNRTI",
                resistance_score=0.85,
                classification="resistant",
                confidence=0.92,
                mutations=["K103N"],
            ),
        ]
        return ResistanceReport(
            report_id="RPT-TEST-001",
            patient_id="P001",
            sequence_length=450,
            analysis_date=datetime(2024, 1, 15, 10, 30),
            disease="HIV",
            predictions=predictions,
            recommended_drugs=["AZT", "3TC", "DRV"],
            avoid_drugs=["EFV", "NVP"],
            warnings=["High resistance to NNRTIs"],
            overall_recommendation="Consider boosted PI regimen",
            mutations_detected=[{"position": 103, "wt": "K", "mut": "N"}],
            config=config,
        )

    def test_render_returns_bytes(self, config, sample_report):
        """Test renderer returns bytes."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report)
        assert isinstance(result, bytes)

    def test_render_valid_html(self, config, sample_report):
        """Test renderer produces valid HTML."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "<!DOCTYPE html>" in result
        assert "<html" in result
        assert "</html>" in result

    def test_render_includes_patient_id(self, config, sample_report):
        """Test HTML includes patient ID."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "P001" in result

    def test_render_includes_drugs(self, config, sample_report):
        """Test HTML includes drug names."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "AZT" in result
        assert "EFV" in result

    def test_render_includes_mutations(self, config, sample_report):
        """Test HTML includes mutations when configured."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "K103N" in result

    def test_render_without_mutations(self, sample_report):
        """Test HTML without mutations when disabled."""
        config = ReportConfig(include_mutations=False)
        sample_report.config = config
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        # Table header for mutations should not be present
        assert "<th>Mutations</th>" not in result

    def test_render_includes_recommendations(self, config, sample_report):
        """Test HTML includes recommendations."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "DRV" in result  # Recommended drug
        assert "Consider boosted PI regimen" in result

    def test_render_includes_warnings(self, config, sample_report):
        """Test HTML includes warnings."""
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "High resistance to NNRTIs" in result

    def test_render_includes_organization(self, sample_report):
        """Test HTML includes organization name."""
        config = ReportConfig(organization_name="Test Lab")
        sample_report.config = config
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "Test Lab" in result

    def test_render_spanish(self, sample_report):
        """Test HTML rendering in Spanish."""
        config = ReportConfig(language=ReportLanguage.SPANISH)
        sample_report.config = config
        renderer = HTMLReportRenderer(config)
        result = renderer.render(sample_report).decode("utf-8")
        assert "Informe de Analisis de Resistencia" in result

    def test_resistance_colors(self, config):
        """Test resistance color mapping."""
        renderer = HTMLReportRenderer(config)
        assert "#28a745" in renderer._get_resistance_color("susceptible")  # Green
        assert "#dc3545" in renderer._get_resistance_color("high_resistance")  # Red

    def test_resistance_bar(self, config):
        """Test resistance bar generation."""
        renderer = HTMLReportRenderer(config)
        bar = renderer._get_resistance_bar(0.75)
        assert "75%" in bar
        assert "#dc3545" in bar  # Red for high resistance


class TestJSONReportRenderer:
    """Tests for JSON report rendering."""

    @pytest.fixture
    def config(self):
        """Create default config fixture."""
        return ReportConfig(format=ReportFormat.JSON)

    @pytest.fixture
    def sample_report(self, config):
        """Create sample report fixture."""
        predictions = [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.15,
                classification="susceptible",
                confidence=0.95,
            ),
        ]
        return ResistanceReport(
            report_id="RPT-TEST-002",
            patient_id="P002",
            sequence_length=300,
            analysis_date=datetime(2024, 2, 20, 14, 0),
            disease="HIV",
            predictions=predictions,
            recommended_drugs=["AZT"],
            avoid_drugs=[],
            warnings=[],
            overall_recommendation="Treatment effective",
            mutations_detected=[],
            config=config,
        )

    def test_render_returns_bytes(self, config, sample_report):
        """Test renderer returns bytes."""
        renderer = JSONReportRenderer(config)
        result = renderer.render(sample_report)
        assert isinstance(result, bytes)

    def test_render_valid_json(self, config, sample_report):
        """Test renderer produces valid JSON."""
        renderer = JSONReportRenderer(config)
        result = renderer.render(sample_report)
        data = json.loads(result.decode("utf-8"))
        assert isinstance(data, dict)

    def test_json_contains_report_id(self, config, sample_report):
        """Test JSON contains report ID."""
        renderer = JSONReportRenderer(config)
        result = renderer.render(sample_report)
        data = json.loads(result.decode("utf-8"))
        assert data["report_id"] == "RPT-TEST-002"

    def test_json_contains_predictions(self, config, sample_report):
        """Test JSON contains predictions."""
        renderer = JSONReportRenderer(config)
        result = renderer.render(sample_report)
        data = json.loads(result.decode("utf-8"))
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["drug_name"] == "AZT"

    def test_json_contains_config(self, config, sample_report):
        """Test JSON contains config."""
        renderer = JSONReportRenderer(config)
        result = renderer.render(sample_report)
        data = json.loads(result.decode("utf-8"))
        assert "config" in data
        assert data["config"]["format"] == "json"


class TestPDFReportRenderer:
    """Tests for PDF report rendering."""

    @pytest.fixture
    def config(self):
        """Create PDF config fixture."""
        return ReportConfig(format=ReportFormat.PDF)

    @pytest.fixture
    def sample_report(self, config):
        """Create sample report fixture."""
        predictions = [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.15,
                classification="susceptible",
                confidence=0.95,
            ),
        ]
        return ResistanceReport(
            report_id="RPT-TEST-003",
            patient_id="P003",
            sequence_length=400,
            analysis_date=datetime.now(),
            disease="HIV",
            predictions=predictions,
            recommended_drugs=["AZT"],
            avoid_drugs=[],
            warnings=[],
            overall_recommendation="Treatment effective",
            mutations_detected=[],
            config=config,
        )

    def test_render_returns_bytes(self, config, sample_report):
        """Test PDF renderer returns bytes."""
        renderer = PDFReportRenderer(config)
        result = renderer.render(sample_report)
        assert isinstance(result, bytes)

    def test_render_fallback_to_html(self, config, sample_report):
        """Test PDF falls back to HTML when weasyprint unavailable."""
        renderer = PDFReportRenderer(config)
        result = renderer.render(sample_report)
        # Without weasyprint, we get HTML with comment
        content = result.decode("utf-8")
        if "weasyprint" in content:
            assert "<!DOCTYPE html>" in content


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator fixture."""
        return ReportGenerator()

    @pytest.fixture
    def predictions(self):
        """Create sample predictions fixture."""
        return [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.15,
                classification="susceptible",
                confidence=0.95,
            ),
            DrugPrediction(
                drug_name="EFV",
                drug_class="NNRTI",
                resistance_score=0.8,
                classification="resistant",
                confidence=0.88,
            ),
        ]

    def test_initialization_default_config(self, generator):
        """Test generator uses default config."""
        assert generator.config.format == ReportFormat.HTML
        assert generator.config.language == ReportLanguage.ENGLISH

    def test_initialization_custom_config(self):
        """Test generator with custom config."""
        config = ReportConfig(format=ReportFormat.JSON)
        generator = ReportGenerator(config)
        assert generator.config.format == ReportFormat.JSON

    def test_generate_report_id(self, generator):
        """Test report ID generation."""
        report_id = generator._generate_report_id()
        assert report_id.startswith("RPT-")
        assert len(report_id) > 10

    def test_generate_html_report(self, generator, predictions):
        """Test generating HTML report."""
        result = generator.generate(
            sequence="MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            predictions=predictions,
            disease="HIV",
            patient_id="TEST001",
        )
        assert isinstance(result, bytes)
        content = result.decode("utf-8")
        assert "<!DOCTYPE html>" in content

    def test_generate_json_report(self, predictions):
        """Test generating JSON report."""
        generator = ReportGenerator(ReportConfig(format=ReportFormat.JSON))
        result = generator.generate(
            sequence="MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            predictions=predictions,
            disease="HIV",
        )
        data = json.loads(result.decode("utf-8"))
        assert data["disease"] == "HIV"

    def test_generate_with_format_override(self, generator, predictions):
        """Test format override in generate call."""
        result = generator.generate(
            sequence="MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            predictions=predictions,
            output_format=ReportFormat.JSON,
        )
        data = json.loads(result.decode("utf-8"))
        assert "report_id" in data

    def test_generate_with_all_options(self, generator, predictions):
        """Test generation with all options."""
        result = generator.generate(
            sequence="MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            predictions=predictions,
            disease="HIV",
            patient_id="FULL001",
            recommended_drugs=["AZT", "3TC"],
            avoid_drugs=["EFV"],
            warnings=["High NNRTI resistance"],
            overall_recommendation="Avoid NNRTIs",
            mutations_detected=[{"pos": 103, "wt": "K", "mut": "N"}],
        )
        content = result.decode("utf-8")
        assert "FULL001" in content
        assert "3TC" in content

    def test_generate_from_api_response(self, generator):
        """Test generating report from API response."""
        api_response = {
            "disease": "HIV",
            "patient_id": "API001",
            "drug_class_results": {
                "NRTI": [
                    {
                        "drug": "AZT",
                        "resistance_score": 0.2,
                        "interpretation": "Susceptible - No resistance",
                        "confidence": 0.95,
                    }
                ],
            },
            "recommended_drugs": ["AZT"],
            "avoid_drugs": [],
        }
        result = generator.generate_from_api_response(
            api_response=api_response,
            sequence="MKWVTVYIG",
        )
        content = result.decode("utf-8")
        assert "AZT" in content

    def test_save_report(self, generator, predictions):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = generator.generate(
                sequence="MKWVTVYIG",
                predictions=predictions,
            )
            filepath = Path(tmpdir) / "test_report"
            saved_path = generator.save_report(
                content,
                filepath,
                ReportFormat.HTML,
            )
            assert saved_path.exists()
            assert saved_path.suffix == ".html"

    def test_save_report_creates_directory(self, generator, predictions):
        """Test save_report creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = generator.generate(
                sequence="MKWVTVYIG",
                predictions=predictions,
                output_format=ReportFormat.JSON,
            )
            filepath = Path(tmpdir) / "subdir" / "nested" / "report.json"
            saved_path = generator.save_report(content, filepath, ReportFormat.JSON)
            assert saved_path.exists()


class TestReportArchive:
    """Tests for ReportArchive class."""

    @pytest.fixture
    def archive(self, tmp_path):
        """Create archive fixture."""
        return ReportArchive(tmp_path / "archive")

    def test_archive_creates_directory(self, tmp_path):
        """Test archive creates directory."""
        archive_dir = tmp_path / "new_archive"
        archive = ReportArchive(archive_dir)
        assert archive_dir.exists()

    def test_archive_report(self, archive):
        """Test archiving a report."""
        content = b"<html>Test Report</html>"
        path = archive.archive(
            report_id="RPT-001",
            content=content,
            patient_id="P001",
        )
        assert path.exists()

    def test_archive_with_metadata(self, archive):
        """Test archiving with metadata."""
        content = b"<html>Test Report</html>"
        path = archive.archive(
            report_id="RPT-002",
            content=content,
            metadata={"disease": "HIV", "drugs_analyzed": 5},
        )
        assert path.exists()

    def test_retrieve_report(self, archive):
        """Test retrieving archived report."""
        content = b"<html>Test Content</html>"
        archive.archive(
            report_id="RPT-003",
            content=content,
        )
        retrieved = archive.retrieve("RPT-003")
        assert retrieved == content

    def test_retrieve_nonexistent(self, archive):
        """Test retrieving non-existent report."""
        result = archive.retrieve("NONEXISTENT")
        assert result is None

    def test_list_reports_all(self, archive):
        """Test listing all reports."""
        archive.archive("RPT-A", b"A", patient_id="P1")
        archive.archive("RPT-B", b"B", patient_id="P2")
        archive.archive("RPT-C", b"C", patient_id="P1")

        reports = archive.list_reports()
        assert len(reports) == 3

    def test_list_reports_by_patient(self, archive):
        """Test listing reports by patient."""
        archive.archive("RPT-A", b"A", patient_id="P1")
        archive.archive("RPT-B", b"B", patient_id="P2")
        archive.archive("RPT-C", b"C", patient_id="P1")

        reports = archive.list_reports(patient_id="P1")
        assert len(reports) == 2
        assert all(r["patient_id"] == "P1" for r in reports)

    def test_list_reports_by_date(self, archive):
        """Test listing reports by date range."""
        archive.archive("RPT-X", b"X", patient_id="P1")

        # Reports archived now should be found
        from datetime import timedelta
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        reports = archive.list_reports(start_date=start, end_date=end)
        assert len(reports) >= 1

    def test_index_persistence(self, tmp_path):
        """Test index persists across instances."""
        archive_dir = tmp_path / "persistent"
        archive1 = ReportArchive(archive_dir)
        archive1.archive("RPT-PERSIST", b"persist", patient_id="P1")

        # Create new instance
        archive2 = ReportArchive(archive_dir)
        reports = archive2.list_reports()
        assert len(reports) == 1
        assert reports[0]["report_id"] == "RPT-PERSIST"


class TestIntegration:
    """Integration tests for report generation."""

    def test_full_workflow(self, tmp_path):
        """Test complete report generation workflow."""
        # Create generator
        config = ReportConfig(
            format=ReportFormat.HTML,
            language=ReportLanguage.ENGLISH,
            organization_name="Integration Test Lab",
        )
        generator = ReportGenerator(config)

        # Create predictions
        predictions = [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.1,
                classification="susceptible",
                confidence=0.98,
                mutations=[],
            ),
            DrugPrediction(
                drug_name="EFV",
                drug_class="NNRTI",
                resistance_score=0.9,
                classification="resistant",
                confidence=0.95,
                mutations=["K103N", "Y181C"],
                interpretation="High-level resistance",
            ),
        ]

        # Generate report
        content = generator.generate(
            sequence="MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            predictions=predictions,
            disease="HIV",
            patient_id="INT001",
            recommended_drugs=["AZT", "3TC", "DRV/r"],
            avoid_drugs=["EFV", "NVP", "ETR"],
            warnings=["High NNRTI resistance detected"],
            overall_recommendation="Consider boosted PI regimen with NRTI backbone",
        )

        # Save report
        report_path = generator.save_report(
            content,
            tmp_path / "report.html",
            ReportFormat.HTML,
        )
        assert report_path.exists()

        # Archive report
        archive = ReportArchive(tmp_path / "archive")
        archive.archive(
            report_id="RPT-INT-001",
            content=content,
            patient_id="INT001",
            metadata={"disease": "HIV", "drugs_analyzed": 2},
        )

        # Retrieve and verify
        retrieved = archive.retrieve("RPT-INT-001")
        assert retrieved == content

    def test_multilingual_reports(self, tmp_path):
        """Test generating reports in multiple languages."""
        predictions = [
            DrugPrediction(
                drug_name="AZT",
                drug_class="NRTI",
                resistance_score=0.5,
                classification="intermediate",
                confidence=0.85,
            ),
        ]

        for lang in [ReportLanguage.ENGLISH, ReportLanguage.SPANISH, ReportLanguage.FRENCH]:
            config = ReportConfig(language=lang)
            generator = ReportGenerator(config)
            content = generator.generate(
                sequence="MKWVTVYIG",
                predictions=predictions,
            )
            assert len(content) > 0

            # Check language-specific content
            text = content.decode("utf-8")
            if lang == ReportLanguage.ENGLISH:
                assert "Drug Resistance" in text
            elif lang == ReportLanguage.SPANISH:
                assert "Resistencia" in text
            elif lang == ReportLanguage.FRENCH:
                assert "Resistance" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
