"""
Pytest configuration and shared fixtures for data access tests.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch
from typing import Generator

# Mark all tests in this directory as data_access tests
pytestmark = pytest.mark.data_access


@pytest.fixture(scope="session")
def mock_env() -> Generator[dict, None, None]:
    """Provide mock environment variables for testing."""
    env_vars = {
        "NCBI_EMAIL": "test@example.com",
        "NCBI_API_KEY": "test_api_key",
        "HIVDB_ENDPOINT": "https://hivdb.stanford.edu/graphql",
        "CBIOPORTAL_URL": "https://www.cbioportal.org/api",
        "CARD_API_URL": "https://card.mcmaster.ca/api",
        "BVBRC_API_URL": "https://www.bv-brc.org/api",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    def _create(json_data=None, status_code=200, text=""):
        response = MagicMock()
        response.json.return_value = json_data or {}
        response.status_code = status_code
        response.text = text
        response.raise_for_status = MagicMock()
        if status_code >= 400:
            from requests.exceptions import HTTPError
            response.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
        return response
    return _create


@pytest.fixture
def sample_hiv_sequence() -> str:
    """Provide a sample HIV-1 sequence for testing."""
    return (
        "CCTCAGATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGGCAACTAAAGGAAGCTCTAT"
        "TAGATACAGGAGCAGATGATACAGTATTAGAAGAAATGAGTTTGCCAGGAAGATGGAAACCAAAAATGAT"
        "AGGGGGAATTGGAGGTTTTATCAAAGTAAGACAGTATGATCAGATACTCATAGAAATCTGTGGACATAAA"
    )


@pytest.fixture
def sample_mutations() -> list[str]:
    """Provide sample HIV mutations for testing."""
    return ["M41L", "K65R", "K103N", "M184V", "T215Y"]


@pytest.fixture
def sample_codon_sequence() -> str:
    """Provide a sample codon sequence for testing."""
    return "ATGTTTAGAACAGGAGGT"  # MFR TGG


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "live_api: marks tests that require live API access")
    config.addinivalue_line("markers", "data_access: marks tests for data access module")
