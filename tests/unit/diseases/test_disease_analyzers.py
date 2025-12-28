# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for all disease analyzers.

Tests the 11-disease framework including:
- SARS-CoV-2, TB, Influenza, HCV, HBV, Malaria, MRSA, Candida, RSV, Cancer
"""

import pytest
import numpy as np


class TestSARSCoV2Analyzer:
    """Tests for SARS-CoV-2 analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import SARSCoV2Analyzer, SARSCoV2Gene

        assert SARSCoV2Analyzer is not None
        assert SARSCoV2Gene is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import SARSCoV2Analyzer

        analyzer = SARSCoV2Analyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "analyze")
        assert hasattr(analyzer, "validate_predictions")

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_sars_cov2_dataset

        X, y, ids = create_sars_cov2_dataset()
        assert X.shape[0] > 0
        assert X.shape[0] == len(y)
        assert X.shape[0] == len(ids)
        assert X.dtype == np.float32

    def test_gene_enum(self):
        """Test gene enumeration."""
        from src.diseases import SARSCoV2Gene

        # Should have key genes
        assert hasattr(SARSCoV2Gene, "SPIKE")
        assert hasattr(SARSCoV2Gene, "NSP5")  # Mpro


class TestTuberculosisAnalyzer:
    """Tests for Tuberculosis analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import TuberculosisAnalyzer, TBGene, TBDrug

        assert TuberculosisAnalyzer is not None
        assert TBGene is not None
        assert TBDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import TuberculosisAnalyzer

        analyzer = TuberculosisAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_tb_synthetic_dataset

        X, y, ids = create_tb_synthetic_dataset()
        assert X.shape[0] > 0
        assert X.shape[0] == len(y)

    def test_mdr_classification(self):
        """Test MDR classification enum."""
        from src.diseases import ResistanceLevel

        assert hasattr(ResistanceLevel, "SUSCEPTIBLE")
        assert hasattr(ResistanceLevel, "LOW")
        assert hasattr(ResistanceLevel, "INTERMEDIATE")
        assert hasattr(ResistanceLevel, "HIGH")


class TestInfluenzaAnalyzer:
    """Tests for Influenza analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import InfluenzaAnalyzer, InfluenzaSubtype, InfluenzaDrug

        assert InfluenzaAnalyzer is not None
        assert InfluenzaSubtype is not None
        assert InfluenzaDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import InfluenzaAnalyzer

        analyzer = InfluenzaAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "recommend_vaccine_strain")

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_influenza_synthetic_dataset

        X, y, ids = create_influenza_synthetic_dataset()
        assert X.shape[0] > 0

    def test_subtypes(self):
        """Test influenza subtypes."""
        from src.diseases import InfluenzaSubtype

        assert hasattr(InfluenzaSubtype, "H1N1")
        assert hasattr(InfluenzaSubtype, "H3N2")


class TestHCVAnalyzer:
    """Tests for HCV analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import HCVAnalyzer, HCVGenotype, HCVDrug

        assert HCVAnalyzer is not None
        assert HCVGenotype is not None
        assert HCVDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import HCVAnalyzer

        analyzer = HCVAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_hcv_synthetic_dataset

        X, y, ids = create_hcv_synthetic_dataset()
        assert X.shape[0] > 0

    def test_genotypes(self):
        """Test HCV genotypes."""
        from src.diseases import HCVGenotype

        assert hasattr(HCVGenotype, "GT1A")
        assert hasattr(HCVGenotype, "GT1B")
        assert hasattr(HCVGenotype, "GT3")


class TestHBVAnalyzer:
    """Tests for HBV analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import HBVAnalyzer, HBVGenotype, HBVDrug

        assert HBVAnalyzer is not None
        assert HBVGenotype is not None
        assert HBVDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import HBVAnalyzer

        analyzer = HBVAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_hbv_synthetic_dataset

        X, y, ids = create_hbv_synthetic_dataset()
        assert X.shape[0] > 0


class TestMalariaAnalyzer:
    """Tests for Malaria analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import MalariaAnalyzer, PlasmodiumSpecies, MalariaDrug

        assert MalariaAnalyzer is not None
        assert PlasmodiumSpecies is not None
        assert MalariaDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import MalariaAnalyzer

        analyzer = MalariaAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_malaria_synthetic_dataset

        X, y, ids = create_malaria_synthetic_dataset()
        assert X.shape[0] > 0

    def test_species(self):
        """Test Plasmodium species."""
        from src.diseases import PlasmodiumSpecies

        assert hasattr(PlasmodiumSpecies, "P_FALCIPARUM")


class TestMRSAAnalyzer:
    """Tests for MRSA analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import MRSAAnalyzer, StaphGene, Antibiotic

        assert MRSAAnalyzer is not None
        assert StaphGene is not None
        assert Antibiotic is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import MRSAAnalyzer

        analyzer = MRSAAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_mrsa_synthetic_dataset

        X, y, ids = create_mrsa_synthetic_dataset()
        assert X.shape[0] > 0


class TestCandidaAnalyzer:
    """Tests for Candida auris analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import CandidaAnalyzer, CandidaClade, Antifungal

        assert CandidaAnalyzer is not None
        assert CandidaClade is not None
        assert Antifungal is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import CandidaAnalyzer

        analyzer = CandidaAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_candida_synthetic_dataset

        X, y, ids = create_candida_synthetic_dataset()
        assert X.shape[0] > 0


class TestRSVAnalyzer:
    """Tests for RSV analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import RSVAnalyzer, RSVSubtype, RSVDrug

        assert RSVAnalyzer is not None
        assert RSVSubtype is not None
        assert RSVDrug is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import RSVAnalyzer

        analyzer = RSVAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_rsv_synthetic_dataset

        X, y, ids = create_rsv_synthetic_dataset()
        assert X.shape[0] > 0


class TestCancerAnalyzer:
    """Tests for Cancer analyzer."""

    def test_import(self):
        """Test analyzer can be imported."""
        from src.diseases import CancerAnalyzer, CancerType, CancerGene, TargetedTherapy

        assert CancerAnalyzer is not None
        assert CancerType is not None
        assert CancerGene is not None
        assert TargetedTherapy is not None

    def test_initialization(self):
        """Test analyzer initialization."""
        from src.diseases import CancerAnalyzer

        analyzer = CancerAnalyzer()
        assert analyzer is not None

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.diseases import create_cancer_synthetic_dataset

        X, y, ids = create_cancer_synthetic_dataset()
        assert X.shape[0] > 0

    def test_cancer_types(self):
        """Test cancer types."""
        from src.diseases import CancerType

        assert hasattr(CancerType, "NSCLC")
        assert hasattr(CancerType, "MELANOMA")


class TestDiseaseRegistry:
    """Tests for disease registry."""

    def test_registry_import(self):
        """Test registry can be imported."""
        from src.diseases import DiseaseRegistry, get_disease_config

        assert DiseaseRegistry is not None
        assert get_disease_config is not None

    def test_multi_disease_loss(self):
        """Test multi-disease loss import."""
        from src.diseases import MultiDiseaseLoss

        assert MultiDiseaseLoss is not None


class TestVariantEscape:
    """Tests for variant escape prediction."""

    def test_escape_predictors(self):
        """Test escape predictor imports."""
        from src.diseases import (
            VariantEscapeHead,
            DrugResistancePredictor,
            ImmuneEscapePredictor,
            FitnessPredictor,
        )

        assert VariantEscapeHead is not None
        assert DrugResistancePredictor is not None
        assert ImmuneEscapePredictor is not None
        assert FitnessPredictor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
