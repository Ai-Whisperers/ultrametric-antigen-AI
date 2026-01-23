# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Pytest configuration and shared fixtures for deliverables tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent
sys.path.insert(0, str(deliverables_dir))

# Add project root to path (for src.core imports)
project_root = deliverables_dir.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def demo_sequence():
    """Generate a demo amino acid sequence."""
    np.random.seed(42)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(np.random.choice(list(aa), size=500))


@pytest.fixture
def demo_nucleotide_sequence():
    """Generate a demo nucleotide sequence."""
    np.random.seed(42)
    bases = "ATGC"
    return "".join(np.random.choice(list(bases), size=1000))


@pytest.fixture
def sample_patient_data():
    """Create sample patient data for HIV tests."""
    from partners.hiv_research_package.src.models import PatientData

    return PatientData(
        patient_id="TEST_001",
        age=35,
        sex="M",
        bmi=24.5,
        viral_load=0,
        cd4_count=650,
        prior_regimens=["TDF/FTC/DTG"],
        adherence_history="excellent",
        injection_site_concerns=False,
        psychiatric_history=False,
    )


@pytest.fixture
def sample_tdr_result():
    """Create sample TDR result for testing."""
    from partners.hiv_research_package.src.models import TDRResult

    return TDRResult(
        patient_id="TEST_001",
        sequence_id=None,
        detected_mutations=[],
        drug_susceptibility={
            "DTG": {"status": "susceptible", "score": 0, "class": "INSTI"},
            "TDF": {"status": "susceptible", "score": 0, "class": "NRTI"},
            "3TC": {"status": "susceptible", "score": 0, "class": "NRTI"},
        },
        tdr_positive=False,
        recommended_regimen="TDF/3TC/DTG",
        alternative_regimens=["TDF/FTC/DTG", "TAF/FTC/DTG"],
        resistance_summary="No TDR detected",
        confidence=0.95,
    )


@pytest.fixture
def sample_la_result():
    """Create sample LA selection result for testing."""
    from partners.hiv_research_package.src.models import LASelectionResult

    return LASelectionResult(
        patient_id="TEST_001",
        eligible=True,
        success_probability=0.92,
        cab_resistance_risk=0.0,
        rpv_resistance_risk=0.0,
        pk_adequacy_score=1.0,
        adherence_score=0.95,
        detected_mutations=[],
        recommendation="ELIGIBLE - Recommend LA injectable switch",
        risk_factors=[],
        monitoring_plan=["HIV RNA at 1, 3, and 6 months post-switch"],
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for test exports."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_rotamer_data():
    """Generate mock rotamer data for Colbes tests."""
    np.random.seed(42)

    rotamers = []
    amino_acids = ["VAL", "LEU", "ILE", "PHE", "TYR", "TRP"]

    for i in range(20):
        rotamers.append({
            "pdb_id": f"TEST{i:03d}",
            "chain_id": "A",
            "residue_id": i + 1,
            "residue_name": np.random.choice(amino_acids),
            "chi_angles": [
                np.random.uniform(-np.pi, np.pi)
                for _ in range(np.random.randint(1, 4))
            ],
        })

    return rotamers


@pytest.fixture
def mock_arbovirus_sequences():
    """Generate mock arbovirus sequences for Rojas tests."""
    np.random.seed(42)

    def generate_seq(length: int) -> str:
        bases = list("ATGC")
        return "".join(np.random.choice(bases, size=length))

    return {
        "DENV-1": generate_seq(1000),
        "DENV-2": generate_seq(1000),
        "ZIKV": generate_seq(1000),
        "CHIKV": generate_seq(1000),
    }


# Skip markers for tests requiring external dependencies
pytest_plugins = []


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network access"
    )
    config.addinivalue_line(
        "markers", "optional: marks tests for optional features"
    )
