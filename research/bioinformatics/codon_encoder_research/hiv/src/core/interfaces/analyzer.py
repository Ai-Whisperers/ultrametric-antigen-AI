"""
Analyzer interfaces - protocols for analysis modules.

Defines contracts for resistance analysis, escape analysis,
tropism analysis, and other domain-specific analyses.
"""
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable
from datetime import datetime


@dataclass(slots=True)
class AnalysisResult:
    """
    Base analysis result container.

    Attributes:
        data: Primary analysis data (interpretation depends on analyzer)
        statistics: Summary statistics
        metadata: Additional metadata
        timestamp: When analysis was performed
    """

    data: dict
    statistics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __getitem__(self, key: str):
        return self.data[key]

    def get(self, key: str, default=None):
        return self.data.get(key, default)


T_Input = TypeVar("T_Input")
T_Result = TypeVar("T_Result", bound=AnalysisResult)


@runtime_checkable
class IAnalyzer(Protocol[T_Input, T_Result]):
    """Generic analyzer protocol."""

    def analyze(self, input_data: T_Input) -> T_Result:
        """
        Perform analysis.

        Args:
            input_data: Data to analyze

        Returns:
            Analysis result
        """
        ...

    def analyze_batch(self, inputs: list[T_Input]) -> list[T_Result]:
        """
        Perform batch analysis.

        Args:
            inputs: List of inputs to analyze

        Returns:
            List of analysis results
        """
        ...

    @property
    def name(self) -> str:
        """Get analyzer name."""
        ...


@runtime_checkable
class IStructuralAnalyzer(Protocol):
    """Protocol for structural analysis."""

    def calculate_net_charge(self, sequence: str) -> float:
        """Calculate net charge of sequence."""
        ...

    def calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity."""
        ...

    def find_glycosylation_sites(self, sequence: str) -> list[int]:
        """Find potential N-glycosylation sites (NXS/NXT)."""
        ...


@runtime_checkable
class IFitnessAnalyzer(Protocol):
    """Protocol for fitness cost analysis."""

    def estimate_fitness_cost(self, mutation: str) -> float:
        """Estimate fitness cost of a mutation."""
        ...

    def estimate_reversion_rate(self, mutation: str) -> float:
        """Estimate reversion rate for a mutation."""
        ...


@runtime_checkable
class IPathwayAnalyzer(Protocol):
    """Protocol for evolutionary pathway analysis."""

    def find_shortest_path(self, start: str, end: str) -> list[str]:
        """Find shortest mutational path between genotypes."""
        ...

    def calculate_path_cost(self, path: list[str]) -> float:
        """Calculate total fitness cost along path."""
        ...

    def identify_bottlenecks(self) -> list[str]:
        """Identify evolutionary bottleneck positions."""
        ...
