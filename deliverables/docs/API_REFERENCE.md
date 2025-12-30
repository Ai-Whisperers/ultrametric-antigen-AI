# API Reference

**Ternary VAE Bioinformatics Platform**

This document provides comprehensive API documentation for all public interfaces in the deliverables package.

---

## Table of Contents

1. [Shared Services](#shared-services)
   - [VAEService](#vaeservice)
   - [Configuration](#configuration)
   - [Peptide Utilities](#peptide-utilities)
2. [HIV Research Package](#hiv-research-package)
   - [TDRScreener](#tdrscreener)
   - [LASelector](#laselector)
   - [StanfordHIVdbClient](#stanfordhivdbclient)
   - [HIVSequenceAligner](#hivsequencealigner)
   - [ClinicalReportGenerator](#clinicalreportgenerator)
3. [Arbovirus Package (Alejandra Rojas)](#arbovirus-package)
   - [NCBIClient](#ncbiclient)
   - [PrimerDesigner](#primerdesigner)
4. [AMP Design Package (Carlos Brizuela)](#amp-design-package)
5. [Protein Stability Package (Jose Colbes)](#protein-stability-package)

---

## Shared Services

### VAEService

Singleton VAE service providing encoding/decoding for all partner packages.

```python
from shared.vae_service import get_vae_service, VAEService

# Get singleton instance
vae = get_vae_service()

# Check if real model is loaded
if vae.is_real:
    print("Using trained VAE model")
else:
    print("Using mock mode (no checkpoint)")
```

#### Methods

**`encode_sequence(sequence: str) -> np.ndarray`**

Encode an amino acid sequence to latent space.

```python
z = vae.encode_sequence("KLWKKWKKWLK")
print(f"Latent vector shape: {z.shape}")  # (16,)
```

**`decode_latent(z: np.ndarray) -> str`**

Decode a latent vector to amino acid sequence.

```python
sequence = vae.decode_latent(z)
print(f"Decoded: {sequence}")
```

**`sample_latent(n_samples, target_radius, charge_bias, hydro_bias) -> np.ndarray`**

Sample from latent space with optional biasing.

```python
# Sample 10 cationic peptides
samples = vae.sample_latent(
    n_samples=10,
    charge_bias=0.5,    # Bias toward positive charge
    hydro_bias=0.3      # Moderate hydrophobicity
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 1 | Number of samples |
| `target_radius` | float | None | Target radial position |
| `charge_bias` | float | 0 | Positive = cationic bias |
| `hydro_bias` | float | 0 | Positive = hydrophobic bias |

**`get_radius(z: np.ndarray) -> float`**

Get hyperbolic radius of latent vector.

**`get_padic_valuation(z: np.ndarray) -> int`**

Estimate p-adic valuation from radius (0-9).

**`get_stability_score(z: np.ndarray) -> float`**

Get stability score from latent position (0-1).

**`interpolate(z1, z2, steps) -> list[str]`**

Interpolate between two latent vectors and decode.

```python
sequences = vae.interpolate(z1, z2, steps=10)
```

---

### Configuration

Centralized configuration management.

```python
from shared.config import get_config, Config

config = get_config()

# Access paths
print(config.project_root)
print(config.deliverables_root)
print(config.vae_checkpoint)

# Get partner directory
hiv_dir = config.get_partner_dir("hiv")
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `project_root` | Path | Root of the project |
| `deliverables_root` | Path | Deliverables directory |
| `vae_checkpoint` | str | Path to VAE checkpoint |
| `has_vae` | bool | Whether valid checkpoint exists |
| `demo_mode` | bool | Enable demo/mock data |
| `use_gpu` | bool | Use GPU if available |

---

### Peptide Utilities

Common utilities for peptide analysis.

```python
from shared.peptide_utils import (
    compute_peptide_properties,
    validate_sequence,
    compute_charge,
    compute_hydrophobicity
)

# Compute all properties
props = compute_peptide_properties("KLWKKWKKWLK")
print(f"Charge: {props['net_charge']}")
print(f"Hydrophobicity: {props['hydrophobicity']}")
print(f"Length: {props['length']}")

# Validate sequence
is_valid, error = validate_sequence("KLWKKXWLK")
if not is_valid:
    print(f"Invalid: {error}")
```

#### Properties Returned

| Key | Type | Description |
|-----|------|-------------|
| `length` | int | Sequence length |
| `net_charge` | float | Net charge at pH 7 |
| `hydrophobicity` | float | Mean hydrophobicity |
| `hydrophobic_ratio` | float | Fraction hydrophobic |
| `cationic_ratio` | float | Fraction cationic (K, R, H) |
| `aromatic_ratio` | float | Fraction aromatic (F, W, Y) |

---

## HIV Research Package

### TDRScreener

Screen for transmitted drug resistance in HIV sequences.

```python
from partners.hiv_research_package.src import TDRScreener, TDRResult

screener = TDRScreener(use_stanford=False)

# Screen a patient sample
result = screener.screen_patient(
    sequence="PISPIETVPVKLKPGMDGPKVKQWPLTEEKI...",
    patient_id="P001"
)

print(f"TDR Positive: {result.tdr_positive}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Mutations: {len(result.detected_mutations)}")
print(f"Recommendation: {result.recommended_regimen}")
```

#### TDRResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `patient_id` | str | Patient identifier |
| `tdr_positive` | bool | Whether TDR detected |
| `detected_mutations` | list[dict] | Detected resistance mutations |
| `recommended_regimen` | str | Recommended first-line regimen |
| `confidence` | float | Analysis confidence (0-1) |
| `timestamp` | datetime | Analysis timestamp |

---

### LASelector

Assess eligibility for long-acting injectable HIV therapy (CAB-LA/RPV-LA).

```python
from partners.hiv_research_package.src import LASelector, PatientData, LASelectionResult

# Create patient data
patient = PatientData(
    patient_id="P001",
    age=35,
    sex="M",
    bmi=24.5,
    viral_load=0,           # Undetectable
    cd4_count=650,
    prior_regimens=["TDF/FTC/DTG"],
    adherence_history="excellent"
)

# Assess eligibility
selector = LASelector()
result = selector.assess_eligibility(patient, sequence)

print(f"Eligible: {result.eligible}")
print(f"Success Probability: {result.success_probability:.1%}")
print(f"Recommendation: {result.recommendation}")
if result.risk_factors:
    print(f"Risk Factors: {', '.join(result.risk_factors)}")
```

#### PatientData Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patient_id` | str | Yes | Unique identifier |
| `age` | int | Yes | Age in years |
| `sex` | str | Yes | "M" or "F" |
| `bmi` | float | Yes | Body mass index |
| `viral_load` | int | Yes | Copies/mL (0 = undetectable) |
| `cd4_count` | int | Yes | Cells/uL |
| `prior_regimens` | list[str] | No | Previous ART regimens |
| `adherence_history` | str | No | "excellent", "good", "poor" |

#### LASelectionResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `eligible` | bool | Eligible for LA therapy |
| `success_probability` | float | Predicted success rate |
| `recommendation` | str | Clinical recommendation |
| `risk_factors` | list[str] | Identified risk factors |
| `pk_adequacy` | float | Pharmacokinetic adequacy score |

---

### StanfordHIVdbClient

Client for Stanford HIVdb GraphQL API.

```python
from partners.hiv_research_package.src import StanfordHIVdbClient

client = StanfordHIVdbClient()

# Analyze sequence
result = client.analyze_sequence(
    sequence="PISPIETVPVKLKPGMDGPKVKQWPLTEEKI...",
    gene="RT"
)

# Generate report
report = client.generate_report(result)
print(report)
```

#### Methods

**`analyze_sequence(sequence, gene="RT") -> dict`**

Send sequence to Stanford HIVdb for analysis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | str | Amino acid sequence |
| `gene` | str | Gene type: "RT", "PR", or "IN" |

---

### HIVSequenceAligner

Align HIV sequences to HXB2 reference for mutation detection.

```python
from partners.hiv_research_package.src import HIVSequenceAligner, AlignmentResult

aligner = HIVSequenceAligner()

# Align to reference
result = aligner.align(sequence, gene="RT", method="simple")

print(f"Identity: {result.identity:.1%}")
print(f"Coverage: {result.coverage:.1%}")
print(f"Gaps: {result.gaps}")

# Detect mutations
mutations = aligner.detect_mutations(result)
for mut in mutations:
    print(f"  {mut.notation}: {'Resistance' if mut.is_resistance else 'Polymorphism'}")
```

#### AlignmentResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `query_sequence` | str | Original query |
| `aligned_query` | str | Aligned query |
| `aligned_reference` | str | Aligned reference |
| `score` | float | Alignment score |
| `identity` | float | Sequence identity (0-1) |
| `coverage` | float | Alignment coverage (0-1) |
| `gaps` | int | Number of gaps |
| `mutations` | list[dict] | Detected mutations |
| `gene` | str | Gene aligned to |

---

### ClinicalReportGenerator

Generate formatted clinical reports.

```python
from partners.hiv_research_package.src import ClinicalReportGenerator

generator = ClinicalReportGenerator()

# Generate TDR report
tdr_report = generator.generate_tdr_report(tdr_result)

# Generate LA eligibility report
la_report = generator.generate_la_report(la_result)

# Print report
print(tdr_report)
```

---

## Arbovirus Package

### NCBIClient

Client for downloading arbovirus sequences from NCBI.

```python
from partners.alejandra_rojas.src import NCBIClient, ArbovirusDatabase

client = NCBIClient(email="user@example.com")

# Download sequences for a virus
sequences = client.download_virus("DENV-1", max_sequences=50)

# Load or download all viruses
db = client.load_or_download(max_per_virus=50)

# Get sequences from database
denv_seqs = db.get_sequences("DENV-1")
consensus = db.get_consensus("DENV-1")

# Export to FASTA
db.export_fasta("ZIKV", Path("zikv_sequences.fasta"))
```

#### NCBIClient Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `email` | str | Required | NCBI requires email |
| `api_key` | str | None | NCBI API key (faster) |
| `cache_dir` | Path | None | Cache directory |

#### ArbovirusDatabase Methods

| Method | Description |
|--------|-------------|
| `get_sequences(virus)` | Get list of sequences for virus |
| `get_consensus(virus)` | Get consensus sequence |
| `get_viruses()` | List all viruses in database |
| `export_fasta(virus, path)` | Export sequences to FASTA |

---

### PrimerDesigner

Design RT-PCR primers for arboviruses.

```python
from partners.alejandra_rojas.src import PrimerDesigner, PrimerPair
from partners.alejandra_rojas.src.constants import PRIMER_CONSTRAINTS

# Initialize with constraints
designer = PrimerDesigner(
    database=db,  # Optional ArbovirusDatabase
    constraints=PRIMER_CONSTRAINTS
)

# Design primers from database
primers = designer.design_primers("DENV-1", n_pairs=10)

# Design from custom sequence
pairs = designer.design_primer_pairs(
    sequence=genome_sequence,
    target_virus="DENV-1",
    n_pairs=10,
    conserved_regions=[(100, 500), (800, 1200)]
)

for p in pairs:
    print(f"Forward: {p.forward.sequence}")
    print(f"Reverse: {p.reverse.sequence}")
    print(f"Amplicon: {p.amplicon_size} bp")
    print(f"Tm diff: {p.tm_diff:.1f}°C")
    print(f"Score: {p.score:.1f}")
```

#### PrimerPair Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forward` | PrimerCandidate | Forward primer |
| `reverse` | PrimerCandidate | Reverse primer |
| `amplicon_size` | int | Expected amplicon size |
| `amplicon_start` | int | Start position |
| `tm_diff` | float | Tm difference between primers |
| `target_virus` | str | Target virus |
| `score` | float | Quality score |
| `cross_reactive_with` | list[str] | Cross-reactive viruses |
| `conservation_score` | float | Conservation score |

#### PrimerCandidate Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sequence` | str | Primer sequence (5'→3') |
| `position` | int | Start position in genome |
| `length` | int | Primer length |
| `gc_content` | float | GC content (0-1) |
| `tm` | float | Melting temperature (°C) |
| `direction` | str | "forward" or "reverse" |
| `score` | float | Quality score |

#### Cross-Reactivity Analysis

```python
# Check cross-reactivity
result = designer.check_cross_reactivity(primer, threshold=0.8)

if result.is_specific:
    print("Primer is specific")
else:
    print(f"Cross-reactive with: {result.get_cross_reactive_viruses()}")
```

---

## AMP Design Package

### PeptideCandidate

Represents an antimicrobial peptide candidate.

```python
from partners.carlos_brizuela.scripts.amp_navigator import PeptideCandidate

candidate = PeptideCandidate(
    sequence="KLWKKWKKWLK",
    activity_score=0.85,
    toxicity_score=0.15,
    stability_score=0.72
)
```

### NSGA-II Optimizer

Multi-objective optimization for AMP design.

```python
from partners.carlos_brizuela.scripts.amp_navigator import AMPOptimizer

optimizer = AMPOptimizer(
    target_pathogen="S_aureus",
    population_size=100,
    generations=50
)

# Run optimization
pareto_front = optimizer.optimize()

for peptide in pareto_front[:10]:
    print(f"{peptide.sequence}: Activity={peptide.activity_score:.2f}")
```

---

## Protein Stability Package

### GeometricPredictor

Predict mutation effects using p-adic geometric scoring.

```python
from partners.jose_colbes.scripts.geometric_predictor import MutationAnalyzer

analyzer = MutationAnalyzer()

# Analyze mutation
result = analyzer.analyze_mutation(
    wild_type="V",
    mutant="A",
    position=156,
    context="core"  # "core", "surface", or "interface"
)

print(f"Predicted ΔΔG: {result['ddg_predicted']:.2f} kcal/mol")
print(f"Classification: {result['classification']}")  # stabilizing/neutral/destabilizing
```

---

## Data Models Summary

### Enums

**ResistanceLevel** (HIV Package)
```python
from partners.hiv_research_package.src import ResistanceLevel

level = ResistanceLevel.SUSCEPTIBLE
level = ResistanceLevel.POTENTIAL_LOW
level = ResistanceLevel.LOW
level = ResistanceLevel.INTERMEDIATE
level = ResistanceLevel.HIGH

# From score
level = ResistanceLevel.from_score(45)  # Returns INTERMEDIATE

# From text
level = ResistanceLevel.from_text("High-Level Resistance")  # Returns HIGH
```

---

## Constants

### HIV Constants

```python
from partners.hiv_research_package.src import (
    TDR_MUTATIONS,      # Dict of TDR mutations by drug class
    FIRST_LINE_DRUGS,   # List of first-line drugs
    FIRST_LINE_REGIMENS,# Recommended regimens
    LA_DRUGS,           # Long-acting injectable drugs
    WHO_SDRM_NRTI,      # WHO surveillance mutations (NRTI)
    WHO_SDRM_NNRTI,     # WHO surveillance mutations (NNRTI)
    WHO_SDRM_INSTI,     # WHO surveillance mutations (INSTI)
)
```

### Arbovirus Constants

```python
from partners.alejandra_rojas.src.constants import (
    ARBOVIRUS_TARGETS,  # Target viruses with taxids
    PRIMER_CONSTRAINTS, # Primer design constraints
    CONSERVED_REGIONS,  # Known conserved regions
)
```

### Peptide Constants

```python
from shared.constants import (
    AMINO_ACIDS,        # All 20 amino acids
    HYDROPHOBICITY,     # Kyte-Doolittle scale
    CHARGES,            # Amino acid charges at pH 7
    CODON_TABLE,        # Standard genetic code
)
```

---

## Error Handling

All packages follow consistent error handling patterns:

```python
try:
    result = screener.screen_patient(sequence, patient_id)
except ValueError as e:
    print(f"Invalid input: {e}")
except ConnectionError as e:
    print(f"API connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## CLI Usage

All tools can be accessed via the unified CLI:

```bash
# List available tools
python biotools.py --list

# Run specific demos
python biotools.py demo-hiv
python biotools.py demo-amp
python biotools.py demo-primers
python biotools.py demo-stability

# Generate showcase outputs
python biotools.py showcase

# Analyze a peptide
python biotools.py analyze KLWKKWKKWLK
```

---

*API Reference v1.0 - December 2025*
*Ternary VAE Bioinformatics Platform*
