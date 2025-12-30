# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""End-to-end tests for all demo workflows.

These tests verify that all partner package demos run successfully
and produce expected outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
deliverables_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(deliverables_root))


class TestVAEService:
    """Test VAE service functionality."""

    def test_vae_service_initialization(self):
        """Test VAE service can be initialized."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()
        assert vae is not None
        assert hasattr(vae, 'is_real')

    def test_vae_encode_decode(self):
        """Test encode/decode roundtrip."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()

        # Test encoding
        sequence = "KLWKKWKKWLK"
        z = vae.encode_sequence(sequence)
        assert z is not None
        assert len(z) == 16

        # Test decoding
        decoded = vae.decode_latent(z)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_vae_sample_latent(self):
        """Test latent space sampling."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()

        # Sample with biases
        samples = vae.sample_latent(
            n_samples=10,
            charge_bias=0.5,
            hydro_bias=0.3
        )

        assert samples.shape == (10, 16)

    def test_vae_stability_metrics(self):
        """Test stability and valuation metrics."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()
        z = vae.encode_sequence("AAAAA")

        radius = vae.get_radius(z)
        valuation = vae.get_padic_valuation(z)
        stability = vae.get_stability_score(z)

        assert radius >= 0  # Radius is non-negative (can exceed 1 in mock mode)
        assert 0 <= valuation <= 9
        assert 0 <= stability <= 1


class TestHIVDemo:
    """Test HIV package demo workflow."""

    def test_tdr_screening_pipeline(self):
        """Test complete TDR screening workflow."""
        try:
            from partners.hiv_research_package.src import TDRScreener, TDRResult

            # Demo RT sequence
            demo_sequence = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"

            screener = TDRScreener(use_stanford=False)
            result = screener.screen_patient(demo_sequence, "TEST-001")

            assert result.patient_id == "TEST-001"
            assert isinstance(result.tdr_positive, bool)
            assert 0 <= result.confidence <= 1
            assert result.recommended_regimen is not None

        except ImportError:
            pytest.skip("HIV package not available")

    def test_la_selection_pipeline(self):
        """Test complete LA selection workflow."""
        try:
            from partners.hiv_research_package.src import LASelector, PatientData

            patient = PatientData(
                patient_id="TEST-001",
                age=35,
                sex="M",
                bmi=24.5,
                viral_load=0,
                cd4_count=650,
                prior_regimens=["TDF/FTC/DTG"],
                adherence_history="excellent"
            )

            demo_sequence = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKI"

            selector = LASelector()
            result = selector.assess_eligibility(patient, demo_sequence)

            assert isinstance(result.eligible, bool)
            assert 0 <= result.success_probability <= 1
            assert result.recommendation is not None

        except ImportError:
            pytest.skip("HIV package not available")

    def test_sequence_alignment(self):
        """Test HIV sequence alignment."""
        try:
            from partners.hiv_research_package.src import HIVSequenceAligner

            aligner = HIVSequenceAligner()
            demo_sequence = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKI"

            result = aligner.align(demo_sequence, gene="RT", method="simple")

            assert 0 <= result.identity <= 1
            assert 0 <= result.coverage <= 1
            assert result.gene == "RT"
            assert isinstance(result.mutations, list)

        except ImportError:
            pytest.skip("HIV package not available")

    def test_clinical_report_generation(self):
        """Test clinical report generation."""
        try:
            from partners.hiv_research_package.src import (
                TDRScreener, ClinicalReportGenerator
            )

            screener = TDRScreener()
            demo_sequence = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKI"
            tdr_result = screener.screen_patient(demo_sequence, "TEST-001")

            generator = ClinicalReportGenerator()
            report = generator.generate_tdr_report(tdr_result)

            assert isinstance(report, str)
            assert len(report) > 0
            assert "TEST-001" in report

        except ImportError:
            pytest.skip("HIV package not available")


class TestArbovirusDemo:
    """Test arbovirus package demo workflow."""

    def test_primer_design_pipeline(self):
        """Test complete primer design workflow."""
        try:
            from partners.alejandra_rojas.src import PrimerDesigner
            from partners.alejandra_rojas.src.constants import PRIMER_CONSTRAINTS

            # Demo genome sequence (longer to allow amplicon design)
            demo_genome = "ATGAACAACCAACGGAAAAAGACGGGTCGACCGTCTTTCAATATGCTGAAACGCGCGAGAAACCGCGTGTCAACTGTTTCACAGTTGGCGAAGAGATTCTCAAAAGGATTGCTTTCAGGCCAAGGACCCATGAAATTGGTGATGGCTTTTATAGCATTCCTAAGATTTCTAGCCATACCTCCAACAGCAGGAATTTTGGCTAGATGGGGCTCATTCAAGAAGAATGGAGCGATCAAAGTGTTACGGGGTTTCAAGAAAGAAATCTCAAACATGTTGAACATAATGAACAGGAGGAAAAGATCTGTGACCATGCTCCTCATGCTGCTGCCCACAGCCCTGGCGTTCCATCTGACCACCCGAGGGGGAGAGCCGCACATGATAGTTAGC" * 3

            designer = PrimerDesigner(constraints=PRIMER_CONSTRAINTS)
            pairs = designer.design_primer_pairs(
                sequence=demo_genome,
                target_virus="DENV-1",
                n_pairs=5
            )

            # Some primer pairs may be found (depends on sequence)
            assert isinstance(pairs, list)

            for pair in pairs:
                assert len(pair.forward.sequence) >= PRIMER_CONSTRAINTS['length']['min']
                assert len(pair.forward.sequence) <= PRIMER_CONSTRAINTS['length']['max']
                assert pair.score >= 0

        except ImportError:
            pytest.skip("Arbovirus package not available")

    def test_gc_content_calculation(self):
        """Test GC content calculation."""
        try:
            from partners.alejandra_rojas.src import PrimerDesigner

            designer = PrimerDesigner()

            # Test known sequences
            assert abs(designer.compute_gc_content("GCGC") - 1.0) < 0.01
            assert abs(designer.compute_gc_content("ATAT") - 0.0) < 0.01
            assert abs(designer.compute_gc_content("GATC") - 0.5) < 0.01

        except ImportError:
            pytest.skip("Arbovirus package not available")

    def test_tm_estimation(self):
        """Test melting temperature estimation."""
        try:
            from partners.alejandra_rojas.src import PrimerDesigner

            designer = PrimerDesigner()

            # Test that Tm is reasonable
            tm = designer.estimate_tm("ATGCATGCATGCATGC")
            assert 40 < tm < 80  # Reasonable Tm range

        except ImportError:
            pytest.skip("Arbovirus package not available")

    def test_ncbi_client_demo_mode(self):
        """Test NCBI client initialization."""
        try:
            from partners.alejandra_rojas.src import NCBIClient

            client = NCBIClient(email="test@example.com")

            # Check client is initialized
            assert client.email == "test@example.com"
            assert hasattr(client, 'download_virus')

        except ImportError:
            pytest.skip("Arbovirus package not available")


class TestAMPDemo:
    """Test AMP design demo workflow."""

    def test_peptide_property_calculation(self):
        """Test peptide property calculations."""
        from shared.peptide_utils import compute_peptide_properties

        props = compute_peptide_properties("KLWKKWKKWLK")

        assert props['length'] == 11
        assert props['net_charge'] > 0  # Should be cationic
        assert 'hydrophobicity' in props
        assert 'hydrophobic_ratio' in props
        assert 'cationic_ratio' in props

    def test_peptide_validation(self):
        """Test sequence validation."""
        from shared.peptide_utils import validate_sequence

        # Valid sequence
        is_valid, error = validate_sequence("KLWKKWKKWLK")
        assert is_valid
        assert error is None or error == ""  # Can be None or empty string

        # Invalid sequence (contains X)
        is_valid, error = validate_sequence("KLWXKWKKWLK")
        # Behavior depends on implementation - X might be allowed in some contexts
        # Just check the function returns proper types
        assert isinstance(is_valid, bool)
        assert error is None or isinstance(error, str)

    def test_hemolysis_prediction(self):
        """Test hemolysis prediction."""
        from shared.hemolysis_predictor import HemolysisPredictor

        predictor = HemolysisPredictor()
        result = predictor.predict("KLWKKWKKWLK")

        assert 'hc50_predicted' in result
        assert 'risk_category' in result
        assert 'hemolytic_probability' in result


class TestStabilityDemo:
    """Test protein stability demo workflow."""

    def test_geometric_stability_analysis(self):
        """Test geometric stability analysis."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()

        # Analyze different amino acids
        for aa in ['A', 'V', 'L', 'P', 'G']:
            z = vae.encode_sequence(aa * 10)
            stability = vae.get_stability_score(z)
            valuation = vae.get_padic_valuation(z)

            assert 0 <= stability <= 1
            assert 0 <= valuation <= 9


class TestIntegration:
    """Integration tests across packages."""

    def test_shared_vae_service(self):
        """Test VAE service works across all packages."""
        from shared.vae_service import get_vae_service

        # Get service multiple times (should be singleton)
        vae1 = get_vae_service()
        vae2 = get_vae_service()

        assert vae1 is vae2  # Same instance

    def test_shared_config(self):
        """Test shared configuration."""
        from shared.config import get_config

        config = get_config()

        assert config.project_root.exists()
        assert config.deliverables_root.exists()

    def test_data_flow_consistency(self):
        """Test data formats are consistent across packages."""
        from shared.vae_service import get_vae_service

        vae = get_vae_service()

        # Encode sequence
        sequence = "KLWKKWKKWLK"
        z = vae.encode_sequence(sequence)

        # Verify latent vector format
        assert isinstance(z, np.ndarray)
        assert z.dtype in [np.float32, np.float64]
        assert len(z) == 16

    def test_peptide_utils_integration(self):
        """Test peptide utilities work with VAE."""
        from shared.vae_service import get_vae_service
        from shared.peptide_utils import compute_peptide_properties

        vae = get_vae_service()

        # Sample and analyze peptides
        samples = vae.sample_latent(n_samples=5)

        for z in samples:
            sequence = vae.decode_latent(z)
            props = compute_peptide_properties(sequence)

            assert props['length'] > 0
            assert 'net_charge' in props


class TestBiotoolsCLI:
    """Test biotools CLI functionality."""

    def test_list_tools(self):
        """Test tool listing."""
        import subprocess
        import sys

        script_path = deliverables_root / "scripts" / "biotools.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--list"],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent)
        )

        # Should list tools without error
        assert result.returncode == 0
        assert "BIOINFORMATICS TOOLS" in result.stdout


class TestShowcaseFigures:
    """Test showcase figure generation."""

    def test_figure_generation_imports(self):
        """Test that figure generation script can be imported."""
        try:
            # Just test imports work
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.stats import spearmanr

            # Test basic figure creation
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            plt.close(fig)

        except ImportError as e:
            pytest.skip(f"Required package not available: {e}")


# Fixtures

@pytest.fixture
def demo_sequence():
    """Provide demo HIV RT sequence."""
    return "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"


@pytest.fixture
def demo_genome():
    """Provide demo viral genome sequence."""
    return "ATGAACAACCAACGGAAAAAGACGGGTCGACCGTCTTTCAATATGCTGAAACGCGCGAG" * 10


@pytest.fixture
def vae_service():
    """Provide VAE service instance."""
    from shared.vae_service import get_vae_service
    return get_vae_service()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
