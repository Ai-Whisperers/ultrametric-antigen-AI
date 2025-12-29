"""
Data loader utilities for RA visualization results.
Centralizes all data loading and provides structured access.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"


@dataclass
class HLAData:
    """HLA analysis results."""

    n_alleles: int = 0
    n_positions: int = 0
    top_positions: List[Dict] = field(default_factory=list)
    alleles: Dict[str, Dict] = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)


@dataclass
class CitrullinationData:
    """Citrullination boundary analysis."""

    n_epitopes: int = 0
    n_with_cit_site: int = 0
    n_boundary_crossed: int = 0
    epitopes: Dict[str, Dict] = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)


@dataclass
class GoldilocksData:
    """Goldilocks zone shift analysis."""

    all_shifts: List[Dict] = field(default_factory=list)
    comparisons: Dict[str, Dict] = field(default_factory=dict)
    acpa_correlation: Dict = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)


@dataclass
class RegenerativeAxisData:
    """Regenerative axis pathway analysis."""

    n_proteins: int = 0
    pathways: List[str] = field(default_factory=list)
    hypothesis: Dict = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)


@dataclass
class CodonOptimizationData:
    """Codon optimization safety results."""

    proteins: Dict[str, Dict] = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)


class RADataLoader:
    """Central data loader for all RA visualization data."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self._cache = {}

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file with caching."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self._cache[filename] = data
        return data

    def load_hla_data(self) -> HLAData:
        """Load HLA expanded analysis results."""
        raw = self._load_json("hla_expanded_results.json")
        return HLAData(
            n_alleles=raw.get("n_alleles", 0),
            n_positions=raw.get("n_positions", 0),
            top_positions=raw.get("top_discriminative_positions", []),
            alleles=raw.get("alleles", {}),
            raw=raw,
        )

    def load_citrullination_data(self) -> CitrullinationData:
        """Load citrullination boundary analysis."""
        raw = self._load_json("citrullination_results.json")
        return CitrullinationData(
            n_epitopes=raw.get("n_epitopes", 0),
            n_with_cit_site=raw.get("n_with_cit_site", 0),
            n_boundary_crossed=raw.get("n_boundary_crossed", 0),
            epitopes=raw.get("epitopes", {}),
            raw=raw,
        )

    def load_goldilocks_data(self) -> GoldilocksData:
        """Load Goldilocks zone shift analysis."""
        raw = self._load_json("citrullination_shift_analysis.json")
        return GoldilocksData(
            all_shifts=raw.get("all_shifts", []),
            comparisons=raw.get("comparisons", {}),
            acpa_correlation=raw.get("acpa_correlation", {}),
            raw=raw,
        )

    def load_regenerative_axis_data(self) -> RegenerativeAxisData:
        """Load regenerative axis pathway analysis."""
        raw = self._load_json("regenerative_axis_results.json")
        return RegenerativeAxisData(
            n_proteins=raw.get("n_proteins", 0),
            pathways=raw.get("pathways", []),
            hypothesis=raw.get("regeneration_hypothesis", {}),
            raw=raw,
        )

    def load_codon_optimization_data(self) -> CodonOptimizationData:
        """Load codon optimization results."""
        raw = self._load_json("codon_optimization_results.json")
        return CodonOptimizationData(proteins=raw.get("proteins", {}), raw=raw)

    def load_all(self) -> Dict[str, Any]:
        """Load all available datasets."""
        return {
            "hla": self.load_hla_data(),
            "citrullination": self.load_citrullination_data(),
            "goldilocks": self.load_goldilocks_data(),
            "regenerative": self.load_regenerative_axis_data(),
            "codon_optimization": self.load_codon_optimization_data(),
        }


# Convenience function
def get_loader(results_dir: Optional[Path] = None) -> RADataLoader:
    """Get a data loader instance."""
    if results_dir:
        return RADataLoader(results_dir)
    return RADataLoader()


# HLA allele risk classification
HLA_RISK_CATEGORIES = {
    "DRB1*04:01": ("high", 4.0),
    "DRB1*04:04": ("high", 3.5),
    "DRB1*04:05": ("high", 3.2),
    "DRB1*04:08": ("high", 2.8),
    "DRB1*01:01": ("moderate", 1.8),
    "DRB1*01:02": ("moderate", 1.6),
    "DRB1*10:01": ("moderate", 1.5),
    "DRB1*15:01": ("neutral", 1.0),
    "DRB1*15:02": ("neutral", 1.0),
    "DRB1*03:01": ("neutral", 1.0),
    "DRB1*08:01": ("neutral", 1.0),
    "DRB1*11:01": ("neutral", 1.0),
    "DRB1*12:01": ("neutral", 1.0),
    "DRB1*07:01": ("protective", 0.6),
    "DRB1*13:01": ("protective", 0.4),
    "DRB1*13:02": ("protective", 0.5),
    "DRB1*14:01": ("protective", 0.5),
}

# Pathophysiology stages
PATHOPHYSIOLOGY_STAGES = [
    {
        "stage": 1,
        "name": "Genetic Susceptibility",
        "key": "genetic",
        "description": "HLA-DRB1 shared epitope determines baseline risk",
        "discovery": "HLA-RA Prediction",
    },
    {
        "stage": 2,
        "name": "Environmental Trigger",
        "key": "environmental",
        "description": "Chronic stress + gut dysbiosis shift autonomic balance",
        "discovery": "Regenerative Axis",
    },
    {
        "stage": 3,
        "name": "Molecular Modification",
        "key": "molecular",
        "description": "PAD enzymes citrullinate sentinel epitopes",
        "discovery": "Citrullination Boundaries",
    },
    {
        "stage": 4,
        "name": "Immune Recognition",
        "key": "immune",
        "description": "Modified peptides in Goldilocks zone break tolerance",
        "discovery": "Goldilocks Autoimmunity",
    },
    {
        "stage": 5,
        "name": "Autoimmune Cascade",
        "key": "autoimmune",
        "description": "ACPA production, epitope spreading, synovitis",
        "discovery": "Clinical Progression",
    },
    {
        "stage": 6,
        "name": "Regenerative Failure",
        "key": "failure",
        "description": "Sympathetic lock-out prevents tissue repair",
        "discovery": "Regenerative Axis",
    },
]

# Three-tier intervention protocol
INTERVENTION_TIERS = [
    {
        "tier": 1,
        "name": "Autonomic Rebalancing",
        "targets": ["VNS", "Breath work", "Mind-body"],
        "pathway": "parasympathetic",
        "rationale": "Restore parasympathetic centrality",
    },
    {
        "tier": 2,
        "name": "Gut Barrier Repair",
        "targets": ["Probiotics", "Glutamine", "Zinc carnosine"],
        "pathway": "gut_barrier",
        "rationale": "Reduce inflammatory signaling",
    },
    {
        "tier": 3,
        "name": "Regeneration Activation",
        "targets": ["Wnt agonists", "LGR5 activation", "Cell therapy"],
        "pathway": "regeneration",
        "rationale": "Enable tissue repair pathways",
    },
]
