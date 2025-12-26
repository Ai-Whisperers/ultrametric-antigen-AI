# Software Architecture Specification

## Clean, Modular, Production-Ready Design

**Version:** 1.0
**Date:** December 2025
**Principles:** SOLID, Clean Architecture, Domain-Driven Design

---

## Table of Contents

1. [Design Principles](#part-i-design-principles)
2. [Package Structure](#part-ii-package-structure)
3. [Core Abstractions](#part-iii-core-abstractions)
4. [Domain Layer](#part-iv-domain-layer)
5. [Infrastructure Layer](#part-v-infrastructure-layer)
6. [Application Layer](#part-vi-application-layer)
7. [Presentation Layer](#part-vii-presentation-layer)
8. [Configuration Management](#part-viii-configuration-management)
9. [Testing Strategy](#part-ix-testing-strategy)
10. [Module Specifications](#part-x-module-specifications)

---

## Part I: Design Principles

### Anti-Patterns to Avoid

```
❌ GOD FILES
   - Files with 1000+ lines
   - Classes doing multiple unrelated things
   - Mixing business logic with I/O

❌ TIGHT COUPLING
   - Direct instantiation of dependencies
   - Hardcoded paths/URLs
   - Concrete class references everywhere

❌ LEAKY ABSTRACTIONS
   - Domain objects knowing about databases
   - Business logic in API endpoints
   - ML models knowing about file formats
```

### Patterns to Follow

```
✅ SINGLE RESPONSIBILITY
   - Each file: one class/concept
   - Each class: one reason to change
   - Each function: one thing well

✅ DEPENDENCY INVERSION
   - Depend on abstractions, not concretions
   - Inject dependencies
   - Use protocols/interfaces

✅ CLEAN ARCHITECTURE
   - Domain at center (no dependencies)
   - Use cases orchestrate domain
   - Infrastructure at edges
```

### File Size Guidelines

| File Type | Max Lines | If Exceeded |
|-----------|-----------|-------------|
| Interface/Protocol | 50 | Split into multiple protocols |
| Data class | 100 | Extract nested classes |
| Implementation | 200 | Extract helper classes |
| Service | 300 | Split by subdomain |
| Module `__init__` | 50 | Re-export only |

---

## Part II: Package Structure

### Complete Directory Tree

```
src/
├── __init__.py
├── py.typed                          # PEP 561 marker
│
├── core/                             # Core domain (no external deps)
│   ├── __init__.py
│   ├── entities/                     # Domain entities
│   │   ├── __init__.py
│   │   ├── codon.py                  # Codon entity (< 50 lines)
│   │   ├── sequence.py               # Sequence entity
│   │   ├── mutation.py               # Mutation entity
│   │   ├── epitope.py                # Epitope entity
│   │   └── embedding.py              # Embedding value object
│   │
│   ├── value_objects/                # Immutable value objects
│   │   ├── __init__.py
│   │   ├── position.py               # HXB2 position
│   │   ├── hla_allele.py             # HLA allele
│   │   ├── drug.py                   # Drug identifier
│   │   └── protein.py                # Protein identifier
│   │
│   ├── interfaces/                   # Abstract interfaces (protocols)
│   │   ├── __init__.py
│   │   ├── encoder.py                # IEncoder protocol
│   │   ├── predictor.py              # IPredictor protocol
│   │   ├── repository.py             # IRepository protocol
│   │   └── analyzer.py               # IAnalyzer protocol
│   │
│   └── exceptions.py                 # Domain exceptions (< 100 lines)
│
├── encoding/                         # Encoding implementations
│   ├── __init__.py
│   ├── padic/                        # P-adic math (isolated)
│   │   ├── __init__.py
│   │   ├── number.py                 # PadicNumber class
│   │   ├── arithmetic.py             # Add, multiply, etc.
│   │   ├── distance.py               # P-adic distance
│   │   └── series.py                 # P-adic series expansions
│   │
│   ├── hyperbolic/                   # Hyperbolic geometry (isolated)
│   │   ├── __init__.py
│   │   ├── poincare/                 # Poincaré disk model
│   │   │   ├── __init__.py
│   │   │   ├── point.py              # Point in disk
│   │   │   ├── operations.py         # Möbius add, exp, log
│   │   │   ├── distance.py           # Hyperbolic distance
│   │   │   └── geodesic.py           # Geodesic computation
│   │   │
│   │   ├── lorentz/                  # Lorentz model (alternative)
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   │
│   │   └── common/                   # Shared hyperbolic utils
│   │       ├── __init__.py
│   │       ├── curvature.py
│   │       └── projection.py
│   │
│   ├── mapping/                      # Codon mapping
│   │   ├── __init__.py
│   │   ├── genetic_code.py           # Genetic code data
│   │   ├── amino_acid_properties.py  # AA property vectors
│   │   ├── codon_to_padic.py         # Codon -> p-adic
│   │   └── hierarchy.py              # Hierarchical encoding
│   │
│   └── encoder.py                    # Main encoder (thin orchestrator)
│
├── data/                             # Data access layer
│   ├── __init__.py
│   ├── repositories/                 # Repository implementations
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseRepository
│   │   ├── stanford.py               # StanfordRepository
│   │   ├── lanl.py                   # LANLRepository
│   │   ├── catnap.py                 # CATNAPRepository
│   │   └── v3.py                     # V3Repository
│   │
│   ├── loaders/                      # File loaders (low-level)
│   │   ├── __init__.py
│   │   ├── csv_loader.py
│   │   ├── fasta_loader.py
│   │   ├── json_loader.py
│   │   └── parquet_loader.py
│   │
│   ├── parsers/                      # Data parsers
│   │   ├── __init__.py
│   │   ├── mutation_parser.py        # Parse mutation strings
│   │   ├── hla_parser.py             # Parse HLA notations
│   │   └── sequence_parser.py        # Parse sequences
│   │
│   ├── alignment/                    # Sequence alignment
│   │   ├── __init__.py
│   │   ├── hxb2_reference.py         # HXB2 reference
│   │   ├── aligner.py                # Alignment algorithms
│   │   └── position_mapper.py        # Position mapping
│   │
│   ├── cache/                        # Caching
│   │   ├── __init__.py
│   │   ├── base.py                   # ICache protocol
│   │   ├── memory.py                 # In-memory cache
│   │   ├── disk.py                   # Disk cache
│   │   └── redis.py                  # Redis cache
│   │
│   └── pipeline.py                   # Unified pipeline (thin)
│
├── analysis/                         # Analysis domain
│   ├── __init__.py
│   ├── resistance/                   # Drug resistance
│   │   ├── __init__.py
│   │   ├── domain/                   # Resistance domain
│   │   │   ├── __init__.py
│   │   │   ├── entities.py           # ResistanceProfile, etc.
│   │   │   └── services.py           # Domain services
│   │   │
│   │   ├── predictors/               # Prediction implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # BaseResistancePredictor
│   │   │   ├── geometric.py          # GeometricPredictor
│   │   │   ├── ml_based.py           # MLPredictor
│   │   │   └── ensemble.py           # EnsemblePredictor
│   │   │
│   │   ├── pathway/                  # Pathway analysis
│   │   │   ├── __init__.py
│   │   │   ├── graph.py              # ResistanceGraph
│   │   │   ├── pathfinding.py        # Shortest path, etc.
│   │   │   └── bottleneck.py         # Bottleneck detection
│   │   │
│   │   └── cross_resistance.py       # Cross-resistance logic
│   │
│   ├── escape/                       # Immune escape
│   │   ├── __init__.py
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   ├── entities.py
│   │   │   └── services.py
│   │   │
│   │   ├── ctl/                      # CTL escape
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py
│   │   │   ├── landscape.py
│   │   │   └── hla_specific.py
│   │   │
│   │   ├── antibody/                 # Antibody escape
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py
│   │   │   ├── epitope_mapper.py
│   │   │   └── neutralization.py
│   │   │
│   │   └── dynamics/                 # Escape dynamics
│   │       ├── __init__.py
│   │       ├── ode_model.py
│   │       └── trajectory.py
│   │
│   ├── tropism/                      # Coreceptor tropism
│   │   ├── __init__.py
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   └── entities.py
│   │   │
│   │   ├── predictor.py
│   │   ├── v3_analyzer.py
│   │   └── switch_detector.py
│   │
│   ├── fitness/                      # Fitness estimation
│   │   ├── __init__.py
│   │   ├── cost_estimator.py
│   │   ├── known_costs.py            # Literature values
│   │   └── tradeoff_analyzer.py
│   │
│   ├── structural/                   # Structural features
│   │   ├── __init__.py
│   │   ├── charge.py                 # Net charge
│   │   ├── hydrophobicity.py         # Hydrophobicity
│   │   ├── glycosylation.py          # Glycan sites
│   │   └── secondary_structure.py
│   │
│   └── coverage/                     # Population coverage
│       ├── __init__.py
│       ├── hla_frequencies.py        # HLA freq data
│       ├── calculator.py             # Coverage calc
│       └── optimizer.py              # Epitope selection
│
├── ml/                               # Machine learning
│   ├── __init__.py
│   ├── base/                         # Base classes
│   │   ├── __init__.py
│   │   ├── model.py                  # BaseModel protocol
│   │   ├── trainer.py                # BaseTrainer
│   │   └── evaluator.py              # BaseEvaluator
│   │
│   ├── datasets/                     # Dataset classes
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseDataset
│   │   ├── resistance.py
│   │   ├── escape.py
│   │   ├── tropism.py
│   │   └── transforms.py             # Data transforms
│   │
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── hyperbolic/               # Hyperbolic models
│   │   │   ├── __init__.py
│   │   │   ├── layers.py             # HyperbolicLinear, etc.
│   │   │   ├── mlp.py                # HyperbolicMLP
│   │   │   ├── attention.py          # HyperbolicAttention
│   │   │   └── encoder.py            # HyperbolicEncoder
│   │   │
│   │   ├── gnn/                      # Graph models
│   │   │   ├── __init__.py
│   │   │   ├── layers.py
│   │   │   └── resistance_gnn.py
│   │   │
│   │   ├── vae/                      # Variational models
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   └── hyperbolic_vae.py
│   │   │
│   │   └── ensemble.py               # Ensemble wrapper
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                # Trainer class
│   │   ├── callbacks.py              # Training callbacks
│   │   ├── schedulers.py             # LR schedulers
│   │   └── early_stopping.py
│   │
│   ├── evaluation/                   # Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                # Metric functions
│   │   ├── cross_validation.py
│   │   └── calibration.py
│   │
│   └── integration/                  # External model integration
│       ├── __init__.py
│       ├── esm2/                     # ESM-2 integration
│       │   ├── __init__.py
│       │   ├── embedder.py
│       │   ├── mutation_effect.py
│       │   └── config.py
│       │
│       └── alphafold/                # AlphaFold integration
│           ├── __init__.py
│           └── structure_fetcher.py
│
├── vaccine/                          # Vaccine design
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities.py               # Epitope, Mosaic, etc.
│   │   └── scoring.py                # Scoring functions
│   │
│   ├── optimization/                 # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseOptimizer
│   │   ├── greedy.py                 # Greedy selection
│   │   ├── genetic.py                # Genetic algorithm
│   │   └── ilp.py                    # Integer linear programming
│   │
│   ├── mosaic/                       # Mosaic design
│   │   ├── __init__.py
│   │   ├── designer.py
│   │   └── polyvalent.py
│   │
│   └── antibody/                     # Antibody combinations
│       ├── __init__.py
│       ├── combination.py
│       ├── synergy.py
│       └── breadth.py
│
├── research/                         # Novel research modules
│   ├── __init__.py
│   ├── quasispecies/                 # Quasispecies analysis
│   │   ├── __init__.py
│   │   ├── cloud.py                  # QuasispeciesCloud
│   │   ├── analyzer.py
│   │   └── dynamics.py
│   │
│   ├── temporal/                     # Temporal evolution
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   ├── trajectory.py
│   │   └── prediction.py
│   │
│   ├── transmission/                 # Transmission analysis
│   │   ├── __init__.py
│   │   ├── bottleneck.py
│   │   └── founder.py
│   │
│   └── reservoir/                    # Latent reservoir
│       ├── __init__.py
│       └── analyzer.py
│
├── application/                      # Application services (use cases)
│   ├── __init__.py
│   ├── services/                     # Application services
│   │   ├── __init__.py
│   │   ├── sequence_analysis.py      # SequenceAnalysisService
│   │   ├── resistance_prediction.py  # ResistancePredictionService
│   │   ├── escape_prediction.py      # EscapePredictionService
│   │   ├── vaccine_design.py         # VaccineDesignService
│   │   └── surveillance.py           # SurveillanceService
│   │
│   ├── dto/                          # Data transfer objects
│   │   ├── __init__.py
│   │   ├── requests.py               # Request DTOs
│   │   └── responses.py              # Response DTOs
│   │
│   └── factories.py                  # Service factories
│
├── presentation/                     # Presentation layer
│   ├── __init__.py
│   ├── api/                          # REST API
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI app (thin)
│   │   ├── routers/                  # Route handlers
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── sequences.py
│   │   │   ├── resistance.py
│   │   │   ├── escape.py
│   │   │   └── vaccine.py
│   │   │
│   │   ├── middleware/               # Middleware
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── logging.py
│   │   │   └── error_handling.py
│   │   │
│   │   └── schemas/                  # Pydantic schemas
│   │       ├── __init__.py
│   │       ├── sequence.py
│   │       ├── resistance.py
│   │       └── common.py
│   │
│   ├── cli/                          # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py                   # CLI entry point
│   │   ├── commands/                 # CLI commands
│   │   │   ├── __init__.py
│   │   │   ├── analyze.py
│   │   │   ├── predict.py
│   │   │   └── train.py
│   │   │
│   │   └── formatters/               # Output formatters
│   │       ├── __init__.py
│   │       ├── json.py
│   │       └── table.py
│   │
│   └── dashboard/                    # Web dashboard
│       ├── __init__.py
│       ├── app.py                    # Streamlit app
│       └── pages/                    # Dashboard pages
│           ├── __init__.py
│           ├── home.py
│           ├── sequence.py
│           └── resistance.py
│
├── visualization/                    # Visualization
│   ├── __init__.py
│   ├── hyperbolic/                   # Hyperbolic plots
│   │   ├── __init__.py
│   │   ├── poincare_disk.py
│   │   ├── trajectory.py
│   │   └── regions.py
│   │
│   ├── networks/                     # Network plots
│   │   ├── __init__.py
│   │   ├── resistance_graph.py
│   │   └── pathway.py
│   │
│   ├── statistical/                  # Statistical plots
│   │   ├── __init__.py
│   │   ├── distributions.py
│   │   └── correlations.py
│   │
│   └── export/                       # Export utilities
│       ├── __init__.py
│       ├── figure_export.py
│       └── publication.py
│
├── infrastructure/                   # Infrastructure
│   ├── __init__.py
│   ├── config/                       # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py               # Settings class
│   │   ├── loader.py                 # Config loader
│   │   └── validation.py             # Config validation
│   │
│   ├── logging/                      # Logging
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── handlers.py
│   │
│   ├── monitoring/                   # Monitoring
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   │
│   └── external/                     # External services
│       ├── __init__.py
│       ├── genbank.py
│       ├── gisaid.py
│       └── alphafold_api.py
│
└── utils/                            # Utilities (minimal)
    ├── __init__.py
    ├── typing.py                     # Type aliases
    ├── decorators.py                 # Utility decorators
    └── validators.py                 # Validation helpers
```

---

## Part III: Core Abstractions

### 3.1 Encoder Interface

**File:** `src/core/interfaces/encoder.py`

```python
"""
Encoder interface - the core abstraction for sequence encoding.
"""
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic

from ..entities.sequence import Sequence
from ..entities.embedding import Embedding


class IEncoder(Protocol):
    """Protocol for sequence encoders."""

    def encode(self, sequence: Sequence) -> Embedding:
        """Encode a sequence to embedding."""
        ...

    def encode_batch(self, sequences: list[Sequence]) -> list[Embedding]:
        """Encode multiple sequences."""
        ...

    def distance(self, emb1: Embedding, emb2: Embedding) -> float:
        """Calculate distance between embeddings."""
        ...


T = TypeVar('T', bound=Embedding)


class BaseEncoder(ABC, Generic[T]):
    """Abstract base class for encoders."""

    @abstractmethod
    def encode(self, sequence: Sequence) -> T:
        """Encode sequence."""
        pass

    def encode_batch(self, sequences: list[Sequence]) -> list[T]:
        """Default batch implementation."""
        return [self.encode(seq) for seq in sequences]

    @abstractmethod
    def distance(self, emb1: T, emb2: T) -> float:
        """Calculate distance."""
        pass
```

### 3.2 Predictor Interface

**File:** `src/core/interfaces/predictor.py`

```python
"""
Predictor interface - abstraction for all prediction models.
"""
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass


@dataclass
class Prediction:
    """Base prediction result."""
    value: float
    confidence: float
    metadata: dict


class IPredictor(Protocol):
    """Protocol for predictors."""

    def predict(self, input_data) -> Prediction:
        """Make prediction."""
        ...

    def predict_batch(self, inputs: list) -> list[Prediction]:
        """Batch prediction."""
        ...


class ITrainablePredictor(IPredictor, Protocol):
    """Protocol for trainable predictors."""

    def train(self, train_data, val_data) -> dict:
        """Train the model."""
        ...

    def save(self, path: str) -> None:
        """Save model."""
        ...

    def load(self, path: str) -> None:
        """Load model."""
        ...


T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output', bound=Prediction)


class BasePredictor(ABC, Generic[T_Input, T_Output]):
    """Abstract base predictor."""

    @abstractmethod
    def predict(self, input_data: T_Input) -> T_Output:
        """Make prediction."""
        pass

    def predict_batch(self, inputs: list[T_Input]) -> list[T_Output]:
        """Default batch implementation."""
        return [self.predict(inp) for inp in inputs]
```

### 3.3 Repository Interface

**File:** `src/core/interfaces/repository.py`

```python
"""
Repository interface - abstraction for data access.
"""
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, Optional


T = TypeVar('T')


class IRepository(Protocol[T]):
    """Protocol for repositories."""

    def get(self, id: str) -> Optional[T]:
        """Get by ID."""
        ...

    def get_all(self) -> list[T]:
        """Get all."""
        ...

    def find(self, **criteria) -> list[T]:
        """Find by criteria."""
        ...


class ISequenceRepository(Protocol):
    """Protocol for sequence repositories."""

    def get_sequence(self, id: str) -> Optional['Sequence']:
        ...

    def get_by_protein(self, protein: str) -> list['Sequence']:
        ...

    def get_by_mutation(self, mutation: str) -> list['Sequence']:
        ...


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository."""

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        pass

    @abstractmethod
    def get_all(self) -> list[T]:
        pass

    def find(self, **criteria) -> list[T]:
        """Default filter implementation."""
        all_items = self.get_all()
        return [
            item for item in all_items
            if self._matches_criteria(item, criteria)
        ]

    def _matches_criteria(self, item: T, criteria: dict) -> bool:
        """Check if item matches criteria."""
        for key, value in criteria.items():
            if getattr(item, key, None) != value:
                return False
        return True
```

### 3.4 Analyzer Interface

**File:** `src/core/interfaces/analyzer.py`

```python
"""
Analyzer interface - abstraction for analysis modules.
"""
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Base analysis result."""
    data: dict
    metadata: dict


class IAnalyzer(Protocol):
    """Protocol for analyzers."""

    def analyze(self, input_data) -> AnalysisResult:
        """Perform analysis."""
        ...


T_Input = TypeVar('T_Input')
T_Result = TypeVar('T_Result', bound=AnalysisResult)


class BaseAnalyzer(ABC, Generic[T_Input, T_Result]):
    """Abstract base analyzer."""

    @abstractmethod
    def analyze(self, input_data: T_Input) -> T_Result:
        """Perform analysis."""
        pass

    def analyze_batch(self, inputs: list[T_Input]) -> list[T_Result]:
        """Batch analysis."""
        return [self.analyze(inp) for inp in inputs]
```

---

## Part IV: Domain Layer

### 4.1 Entity Definitions

**File:** `src/core/entities/codon.py`

```python
"""
Codon entity - fundamental unit of genetic encoding.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)  # Immutable
class Codon:
    """Represents a single codon."""

    nucleotides: str  # 3-letter string (e.g., "ATG")

    def __post_init__(self):
        if len(self.nucleotides) != 3:
            raise ValueError(f"Codon must be 3 nucleotides: {self.nucleotides}")
        if not all(n in "ACGTU" for n in self.nucleotides.upper()):
            raise ValueError(f"Invalid nucleotides: {self.nucleotides}")

    @property
    def amino_acid(self) -> str:
        """Get translated amino acid."""
        from ..data import GENETIC_CODE
        return GENETIC_CODE.get(self.nucleotides.upper(), 'X')

    @property
    def is_stop(self) -> bool:
        """Check if stop codon."""
        return self.nucleotides.upper() in ("TAA", "TAG", "TGA")

    def __str__(self) -> str:
        return self.nucleotides
```

**File:** `src/core/entities/sequence.py`

```python
"""
Sequence entity - represents a biological sequence.
"""
from dataclasses import dataclass, field
from typing import Optional, Iterator
from enum import Enum

from .codon import Codon


class SequenceType(Enum):
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


@dataclass
class Sequence:
    """Represents a biological sequence."""

    raw: str
    sequence_type: SequenceType
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.raw = self.raw.upper().replace(" ", "")

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index) -> str:
        return self.raw[index]

    def codons(self) -> Iterator[Codon]:
        """Iterate over codons (for DNA/RNA)."""
        if self.sequence_type == SequenceType.PROTEIN:
            raise ValueError("Cannot get codons from protein sequence")

        for i in range(0, len(self.raw) - 2, 3):
            yield Codon(self.raw[i:i+3])

    def translate(self) -> 'Sequence':
        """Translate to protein sequence."""
        if self.sequence_type == SequenceType.PROTEIN:
            return self

        aa_seq = ''.join(codon.amino_acid for codon in self.codons())
        return Sequence(aa_seq, SequenceType.PROTEIN, metadata=self.metadata)
```

**File:** `src/core/entities/mutation.py`

```python
"""
Mutation entity - represents a sequence change.
"""
from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class Mutation:
    """Represents a mutation."""

    wild_type: str      # Original residue
    position: int       # 1-indexed position
    mutant: str         # New residue
    protein: Optional[str] = None

    @classmethod
    def from_string(cls, mutation_str: str, protein: Optional[str] = None) -> 'Mutation':
        """Parse mutation string like 'D30N' or 'PR:D30N'."""
        # Handle protein prefix
        if ':' in mutation_str:
            protein, mutation_str = mutation_str.split(':', 1)

        # Parse mutation
        match = re.match(r'([A-Z])(\d+)([A-Z*])', mutation_str)
        if not match:
            raise ValueError(f"Invalid mutation format: {mutation_str}")

        return cls(
            wild_type=match.group(1),
            position=int(match.group(2)),
            mutant=match.group(3),
            protein=protein
        )

    def __str__(self) -> str:
        base = f"{self.wild_type}{self.position}{self.mutant}"
        if self.protein:
            return f"{self.protein}:{base}"
        return base

    @property
    def is_synonymous(self) -> bool:
        """Check if synonymous (same AA)."""
        return self.wild_type == self.mutant
```

**File:** `src/core/entities/embedding.py`

```python
"""
Embedding value object - represents a vector embedding.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Embedding:
    """Immutable embedding vector."""

    vector: tuple[float, ...]  # Immutable tuple
    space: str = "euclidean"   # or "hyperbolic", "padic"
    metadata: Optional[dict] = None

    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs) -> 'Embedding':
        """Create from numpy array."""
        return cls(vector=tuple(arr.tolist()), **kwargs)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.vector)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def __len__(self) -> int:
        return self.dimension
```

---

## Part V: Infrastructure Layer

### 5.1 Configuration Management

**File:** `src/infrastructure/config/settings.py`

```python
"""
Application settings - centralized configuration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class EncoderSettings:
    """Encoder configuration."""
    prime: int = 3
    embedding_dim: int = 3
    curvature: float = 1.0


@dataclass
class DataSettings:
    """Data configuration."""
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    cache_enabled: bool = True
    cache_ttl: int = 3600


@dataclass
class MLSettings:
    """ML configuration."""
    device: str = "cuda"
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100


@dataclass
class APISettings:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class Settings:
    """Main settings container."""
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    data: DataSettings = field(default_factory=DataSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    api: APISettings = field(default_factory=APISettings)

    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment."""
        return cls(
            encoder=EncoderSettings(
                prime=int(os.getenv("ENCODER_PRIME", 3)),
                embedding_dim=int(os.getenv("ENCODER_DIM", 3)),
            ),
            data=DataSettings(
                data_dir=Path(os.getenv("DATA_DIR", "data")),
                cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ),
            # ... etc
        )
```

**File:** `src/infrastructure/config/loader.py`

```python
"""
Configuration loader - load from various sources.
"""
from pathlib import Path
from typing import Optional
import yaml
import json

from .settings import Settings


class ConfigLoader:
    """Load configuration from files."""

    @staticmethod
    def from_yaml(path: Path) -> Settings:
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return ConfigLoader._dict_to_settings(data)

    @staticmethod
    def from_json(path: Path) -> Settings:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return ConfigLoader._dict_to_settings(data)

    @staticmethod
    def _dict_to_settings(data: dict) -> Settings:
        """Convert dict to Settings."""
        # Implementation
        pass
```

### 5.2 Dependency Injection

**File:** `src/infrastructure/container.py`

```python
"""
Dependency injection container.
"""
from dataclasses import dataclass
from typing import TypeVar, Type, Callable, Optional

from .config.settings import Settings


T = TypeVar('T')


class Container:
    """Simple dependency injection container."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._registry: dict[type, Callable] = {}
        self._singletons: dict[type, object] = {}

    def register(self, interface: Type[T], factory: Callable[[], T],
                 singleton: bool = True) -> None:
        """Register a dependency."""
        self._registry[interface] = (factory, singleton)

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a dependency."""
        if interface in self._singletons:
            return self._singletons[interface]

        if interface not in self._registry:
            raise KeyError(f"No registration for {interface}")

        factory, singleton = self._registry[interface]
        instance = factory()

        if singleton:
            self._singletons[interface] = instance

        return instance

    def register_defaults(self) -> None:
        """Register default implementations."""
        from ..encoding.encoder import PadicHyperbolicEncoder
        from ..data.pipeline import UnifiedDataPipeline

        self.register(
            IEncoder,
            lambda: PadicHyperbolicEncoder(self.settings.encoder)
        )
        self.register(
            IDataPipeline,
            lambda: UnifiedDataPipeline(self.settings.data)
        )


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container."""
    global _container
    if _container is None:
        settings = Settings.from_env()
        _container = Container(settings)
        _container.register_defaults()
    return _container


def inject(interface: Type[T]) -> T:
    """Convenience function to inject dependency."""
    return get_container().resolve(interface)
```

---

## Part VI: Application Layer

### 6.1 Application Services

**File:** `src/application/services/sequence_analysis.py`

```python
"""
Sequence analysis service - orchestrates sequence analysis.
"""
from dataclasses import dataclass
from typing import Optional

from ...core.interfaces.encoder import IEncoder
from ...core.interfaces.analyzer import IAnalyzer
from ...core.entities.sequence import Sequence
from ..dto.requests import SequenceAnalysisRequest
from ..dto.responses import SequenceAnalysisResponse


@dataclass
class SequenceAnalysisService:
    """
    Application service for sequence analysis.

    Orchestrates domain objects without containing business logic.
    """

    encoder: IEncoder
    structural_analyzer: IAnalyzer
    fitness_analyzer: IAnalyzer

    def analyze(self, request: SequenceAnalysisRequest) -> SequenceAnalysisResponse:
        """
        Analyze a sequence.

        This is a thin orchestration layer - business logic is in domain.
        """
        # Parse sequence
        sequence = Sequence(
            raw=request.sequence,
            sequence_type=request.sequence_type
        )

        # Encode
        embedding = self.encoder.encode(sequence)

        # Analyze structure
        structural_result = self.structural_analyzer.analyze(sequence)

        # Analyze fitness
        fitness_result = self.fitness_analyzer.analyze(sequence)

        # Compose response
        return SequenceAnalysisResponse(
            sequence_length=len(sequence),
            embedding=embedding.to_array().tolist(),
            structural_features=structural_result.data,
            fitness_features=fitness_result.data,
        )
```

**File:** `src/application/services/resistance_prediction.py`

```python
"""
Resistance prediction service.
"""
from dataclasses import dataclass
from typing import List

from ...core.interfaces.predictor import IPredictor
from ...core.entities.mutation import Mutation
from ..dto.requests import ResistancePredictionRequest
from ..dto.responses import ResistancePredictionResponse


@dataclass
class ResistancePredictionService:
    """Service for resistance prediction."""

    predictor: IPredictor
    cross_resistance_analyzer: 'ICrossResistanceAnalyzer'

    def predict(self, request: ResistancePredictionRequest) -> ResistancePredictionResponse:
        """Predict resistance for mutations."""
        # Parse mutations
        mutations = [
            Mutation.from_string(m)
            for m in request.mutations
        ]

        # Get predictions
        predictions = self.predictor.predict_batch(mutations)

        # Analyze cross-resistance
        cross_resistance = self.cross_resistance_analyzer.analyze(mutations)

        return ResistancePredictionResponse(
            predictions=[
                {"mutation": str(m), "fold_change": p.value, "confidence": p.confidence}
                for m, p in zip(mutations, predictions)
            ],
            cross_resistance=cross_resistance.data,
        )
```

### 6.2 Service Factory

**File:** `src/application/factories.py`

```python
"""
Service factories - create fully configured services.
"""
from typing import Optional

from ..infrastructure.container import inject, Container
from ..core.interfaces.encoder import IEncoder
from .services.sequence_analysis import SequenceAnalysisService
from .services.resistance_prediction import ResistancePredictionService


class ServiceFactory:
    """Factory for creating application services."""

    def __init__(self, container: Optional[Container] = None):
        self.container = container or inject(Container)

    def create_sequence_analysis_service(self) -> SequenceAnalysisService:
        """Create sequence analysis service with dependencies."""
        return SequenceAnalysisService(
            encoder=self.container.resolve(IEncoder),
            structural_analyzer=self.container.resolve(IStructuralAnalyzer),
            fitness_analyzer=self.container.resolve(IFitnessAnalyzer),
        )

    def create_resistance_prediction_service(self) -> ResistancePredictionService:
        """Create resistance prediction service."""
        return ResistancePredictionService(
            predictor=self.container.resolve(IResistancePredictor),
            cross_resistance_analyzer=self.container.resolve(ICrossResistanceAnalyzer),
        )
```

---

## Part VII: Presentation Layer

### 7.1 API Routers (Thin)

**File:** `src/presentation/api/routers/sequences.py`

```python
"""
Sequence analysis API routes.

This is a THIN layer - just HTTP handling, no business logic.
"""
from fastapi import APIRouter, Depends, HTTPException

from ...application.services.sequence_analysis import SequenceAnalysisService
from ...application.factories import ServiceFactory
from ..schemas.sequence import SequenceAnalysisRequestSchema, SequenceAnalysisResponseSchema


router = APIRouter(prefix="/sequences", tags=["Sequences"])


def get_service() -> SequenceAnalysisService:
    """Dependency injection for service."""
    return ServiceFactory().create_sequence_analysis_service()


@router.post("/analyze", response_model=SequenceAnalysisResponseSchema)
async def analyze_sequence(
    request: SequenceAnalysisRequestSchema,
    service: SequenceAnalysisService = Depends(get_service)
):
    """
    Analyze a sequence.

    This endpoint is thin - just converts HTTP to service call.
    """
    try:
        result = service.analyze(request.to_dto())
        return SequenceAnalysisResponseSchema.from_dto(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**File:** `src/presentation/api/routers/resistance.py`

```python
"""
Resistance prediction API routes.
"""
from fastapi import APIRouter, Depends, HTTPException

from ...application.services.resistance_prediction import ResistancePredictionService
from ...application.factories import ServiceFactory
from ..schemas.resistance import (
    ResistancePredictionRequestSchema,
    ResistancePredictionResponseSchema
)


router = APIRouter(prefix="/resistance", tags=["Resistance"])


def get_service() -> ResistancePredictionService:
    return ServiceFactory().create_resistance_prediction_service()


@router.post("/predict", response_model=ResistancePredictionResponseSchema)
async def predict_resistance(
    request: ResistancePredictionRequestSchema,
    service: ResistancePredictionService = Depends(get_service)
):
    """Predict drug resistance."""
    try:
        result = service.predict(request.to_dto())
        return ResistancePredictionResponseSchema.from_dto(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 7.2 Main App (Minimal)

**File:** `src/presentation/api/app.py`

```python
"""
FastAPI application - minimal configuration.

This file should be < 50 lines. All logic is in routers/services.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, sequences, resistance, escape, vaccine
from ...infrastructure.config.settings import Settings


def create_app(settings: Settings = None) -> FastAPI:
    """Create FastAPI application."""
    settings = settings or Settings.from_env()

    app = FastAPI(
        title="HIV P-adic Hyperbolic Analysis API",
        version="1.0.0",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(sequences.router)
    app.include_router(resistance.router)
    app.include_router(escape.router)
    app.include_router(vaccine.router)

    return app


# Default app instance
app = create_app()
```

---

## Part VIII: Testing Strategy

### 8.1 Test Structure

```
tests/
├── __init__.py
├── conftest.py                       # Shared fixtures
│
├── unit/                             # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── core/
│   │   ├── test_entities.py
│   │   └── test_value_objects.py
│   │
│   ├── encoding/
│   │   ├── test_padic.py
│   │   ├── test_hyperbolic.py
│   │   └── test_encoder.py
│   │
│   ├── analysis/
│   │   ├── test_resistance.py
│   │   └── test_escape.py
│   │
│   └── ml/
│       └── test_models.py
│
├── integration/                      # Integration tests
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_prediction_service.py
│   └── test_api_endpoints.py
│
├── e2e/                              # End-to-end tests
│   ├── __init__.py
│   └── test_full_workflow.py
│
└── fixtures/                         # Test data
    ├── sequences.json
    ├── mutations.json
    └── expected_results.json
```

### 8.2 Test Examples

**File:** `tests/unit/encoding/test_padic.py`

```python
"""
Unit tests for p-adic arithmetic.
"""
import pytest
from src.encoding.padic.number import PadicNumber
from src.encoding.padic.distance import padic_distance


class TestPadicNumber:
    """Tests for PadicNumber class."""

    def test_creation(self):
        """Test p-adic number creation."""
        p = PadicNumber(digits=[1, 2, 0], prime=3)
        assert p.prime == 3
        assert p.digits == [1, 2, 0]

    def test_valuation_nonzero(self):
        """Test valuation of nonzero number."""
        p = PadicNumber(digits=[0, 0, 1], prime=3)
        assert p.valuation() == 2

    def test_norm(self):
        """Test p-adic norm."""
        p = PadicNumber(digits=[0, 1], prime=3)
        assert p.norm() == pytest.approx(1/3)


class TestPadicDistance:
    """Tests for p-adic distance."""

    def test_same_number(self):
        """Distance to self is 0."""
        p = PadicNumber(digits=[1, 2], prime=3)
        assert padic_distance(p, p) == 0

    def test_ultrametric(self):
        """Test ultrametric inequality."""
        a = PadicNumber(digits=[1, 0], prime=3)
        b = PadicNumber(digits=[1, 1], prime=3)
        c = PadicNumber(digits=[2, 1], prime=3)

        d_ac = padic_distance(a, c)
        d_ab = padic_distance(a, b)
        d_bc = padic_distance(b, c)

        assert d_ac <= max(d_ab, d_bc)
```

**File:** `tests/conftest.py`

```python
"""
Shared test fixtures.
"""
import pytest
from pathlib import Path

from src.infrastructure.container import Container
from src.infrastructure.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Test settings."""
    return Settings(
        data=DataSettings(
            data_dir=Path("tests/fixtures"),
            cache_enabled=False,
        )
    )


@pytest.fixture
def container(settings) -> Container:
    """Test container with mock dependencies."""
    container = Container(settings)
    container.register_defaults()
    return container


@pytest.fixture
def sample_sequences() -> list[str]:
    """Sample sequences for testing."""
    return [
        "ATGCGATCGATCG",
        "ATGCGATCGATCA",
        "ATGCGATCGAAAA",
    ]


@pytest.fixture
def sample_mutations() -> list[str]:
    """Sample mutations for testing."""
    return ["D30N", "M46I", "V82A", "L90M"]
```

---

## Part IX: Module Size Guidelines

### Maximum File Sizes

| Component | Max Lines | Typical Lines |
|-----------|-----------|---------------|
| Entity | 50 | 20-30 |
| Value Object | 30 | 15-20 |
| Protocol/Interface | 50 | 20-30 |
| Implementation | 200 | 100-150 |
| Service | 150 | 80-120 |
| Router | 100 | 50-80 |
| Test file | 300 | 150-200 |
| Config | 100 | 50-70 |

### When to Split

**Split a class when:**
- It has more than 5-7 public methods
- It has more than 3 responsibilities
- Methods are grouped into distinct clusters
- You find yourself saying "and" to describe it

**Split a file when:**
- It exceeds the max lines
- It contains multiple unrelated classes
- Imports span too many domains
- You scroll a lot to find things

### Example: Splitting a God File

**Before (bad):**
```python
# resistance.py - 800 lines!
class ResistanceAnalyzer:
    def predict_fold_change(self): ...
    def predict_cross_resistance(self): ...
    def build_pathway_graph(self): ...
    def find_shortest_path(self): ...
    def identify_bottlenecks(self): ...
    def visualize_graph(self): ...
    def export_to_networkx(self): ...
    def load_stanford_data(self): ...
    def parse_mutations(self): ...
    # ... 50 more methods
```

**After (good):**
```
analysis/resistance/
├── __init__.py              # Re-exports
├── domain/
│   ├── entities.py          # ResistanceProfile (30 lines)
│   └── services.py          # Domain logic (80 lines)
├── predictors/
│   ├── base.py              # BasePredictor (40 lines)
│   ├── geometric.py         # GeometricPredictor (100 lines)
│   └── cross_resistance.py  # CrossResistancePredictor (80 lines)
├── pathway/
│   ├── graph.py             # ResistanceGraph (120 lines)
│   ├── pathfinding.py       # Pathfinding algos (100 lines)
│   └── bottleneck.py        # Bottleneck detection (60 lines)
└── visualization.py         # Visualization (80 lines)
```

---

## Part X: Dependency Management

### 10.1 Layer Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION                          │
│        (api, cli, dashboard - can depend on all)        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    APPLICATION                           │
│    (services, dto - depends on domain & infra)          │
└─────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌─────────────────────┐       ┌─────────────────────────┐
│       DOMAIN        │       │     INFRASTRUCTURE      │
│ (core, encoding,    │       │ (config, logging, ext)  │
│  analysis, vaccine) │       │                         │
│                     │       │  Implements interfaces  │
│  NO DEPENDENCIES    │◄──────│  defined in domain      │
│  (except core)      │       │                         │
└─────────────────────┘       └─────────────────────────┘
```

### 10.2 Import Rules

```python
# ✅ GOOD - Domain importing from core
from src.core.entities.sequence import Sequence
from src.core.interfaces.encoder import IEncoder

# ✅ GOOD - Application importing from domain
from src.analysis.resistance.predictors.geometric import GeometricPredictor
from src.core.interfaces.predictor import IPredictor

# ✅ GOOD - Presentation importing from application
from src.application.services.resistance_prediction import ResistancePredictionService

# ❌ BAD - Domain importing from infrastructure
from src.infrastructure.config.settings import Settings  # NO!

# ❌ BAD - Core importing from analysis
from src.analysis.resistance import ResistancePredictor  # NO!

# ❌ BAD - Circular imports
from src.encoding.encoder import PadicEncoder  # in core/entities? NO!
```

### 10.3 Allowed Import Matrix

| From \ To | core | encoding | analysis | ml | application | presentation | infrastructure |
|-----------|------|----------|----------|-----|-------------|--------------|----------------|
| core | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| encoding | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| analysis | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| ml | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| application | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| presentation | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| infrastructure | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Summary

### Key Architecture Decisions

1. **Clean Architecture**: Domain at center, infrastructure at edges
2. **Dependency Inversion**: Depend on protocols, not implementations
3. **Small Files**: Max 200 lines per implementation file
4. **Single Responsibility**: One class, one reason to change
5. **Layer Isolation**: Strict import rules between layers
6. **Dependency Injection**: Container-based DI for testability
7. **Thin Presentation**: API routers just do HTTP, no logic

### File Count Estimates

| Layer | Files | Avg Lines | Total Lines |
|-------|-------|-----------|-------------|
| core | 15 | 40 | 600 |
| encoding | 20 | 80 | 1,600 |
| data | 25 | 100 | 2,500 |
| analysis | 40 | 120 | 4,800 |
| ml | 30 | 100 | 3,000 |
| vaccine | 15 | 100 | 1,500 |
| application | 15 | 80 | 1,200 |
| presentation | 20 | 60 | 1,200 |
| infrastructure | 15 | 60 | 900 |
| **Total** | **195** | | **~17,300** |

This is ~17,000 lines across 195 files = **~90 lines per file average**.

No god files. Clean, modular, testable.

---

**Document Version:** 1.0
**Last Updated:** December 2025
