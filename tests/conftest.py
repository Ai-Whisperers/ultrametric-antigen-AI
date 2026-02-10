"""Pytest configuration and shared fixtures for hiv-antigen-ai tests."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_fasta_content():
    """Fixture providing sample FASTA content for testing."""
    return """>seq1
ATCGATCGATCG
>seq2
GCTAGCTAGCTA
>seq3
TTAATTAATTAA
"""


@pytest.fixture
def sample_sequence_data():
    """Fixture providing sample sequence data for testing."""
    return {
        "seq1": "ATCGATCGATCG",
        "seq2": "GCTAGCTAGCTA", 
        "seq3": "TTAATTAATTAA"
    }


# Configure pytest for better output
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )