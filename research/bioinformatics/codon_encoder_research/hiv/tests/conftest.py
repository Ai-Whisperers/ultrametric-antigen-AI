"""
Shared test fixtures.
"""
import pytest
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_codons():
    """Sample codons for testing."""
    return ["ATG", "TTT", "GGG", "CCC", "AAA", "TAA"]


@pytest.fixture
def sample_sequence():
    """Sample DNA sequence."""
    return "ATGTTTATGCCCGGGTAA"


@pytest.fixture
def sample_mutations():
    """Sample mutations for testing."""
    return ["D30N", "M46I", "V82A", "L90M"]


@pytest.fixture
def sample_epitope_data():
    """Sample epitope data."""
    return {
        "sequence": "SLYNTVATL",
        "protein": "Gag",
        "start": 77,
        "end": 85,
        "hla": ["A*02:01"],
    }
