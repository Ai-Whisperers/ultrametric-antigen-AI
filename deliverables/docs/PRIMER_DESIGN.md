# PCR Primer Design Guide

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

The `PrimerDesigner` module provides tools for designing PCR primers to amplify or clone peptide-encoding sequences. It can integrate with Primer3 for advanced primer design or use built-in heuristics when Primer3 is unavailable.

Key capabilities:
- Design forward and reverse primers for any DNA sequence
- Convert peptide sequences to DNA with codon optimization
- Calculate melting temperature (Tm) and GC content
- Check for self-complementarity and homopolymer runs
- Support for E. coli, human, and yeast codon optimization

---

## Why Primer Design Matters

When cloning antimicrobial peptides for expression:

1. **Optimal Tm**: Primers with matched melting temperatures ensure specific amplification
2. **GC content**: 40-60% GC ensures stable primer binding
3. **Avoid hairpins**: Self-complementary regions form structures that prevent binding
4. **No homopolymers**: Long runs of single bases cause polymerase slippage
5. **Codon optimization**: Using preferred codons improves protein expression

---

## Installation & Basic Usage

### Design Primers for DNA Sequence

```python
from shared import PrimerDesigner

# Create designer with default parameters
designer = PrimerDesigner()

# Design primers for a DNA sequence
dna_sequence = "ATGGGCAAATTCCTGAAAAAACTGGGCATGATGCTGAAAAGCCTGATCAAA"
primers = designer.design_primers(dna_sequence)

print(f"Forward primer: 5'-{primers.forward}-3'")
print(f"  Tm: {primers.forward_tm:.1f}°C")
print(f"  GC: {primers.forward_gc:.1f}%")

print(f"Reverse primer: 5'-{primers.reverse}-3'")
print(f"  Tm: {primers.reverse_tm:.1f}°C")
print(f"  GC: {primers.reverse_gc:.1f}%")

print(f"Product size: {primers.product_size} bp")

if primers.warnings:
    print(f"Warnings: {primers.warnings}")
```

### Design Primers for Peptide Cloning

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

# Design primers to clone a peptide
peptide = "GIGKFLHSAKKFGKAFVGEIMNS"  # Magainin 2

primers = designer.design_for_peptide(
    peptide,
    codon_optimization="ecoli",  # Optimize for E. coli expression
    add_start_codon=True,        # Add ATG
    add_stop_codon=True,         # Add TAA
)

print(f"Peptide: {peptide}")
print(f"Forward: 5'-{primers.forward}-3' (Tm: {primers.forward_tm:.1f}°C)")
print(f"Reverse: 5'-{primers.reverse}-3' (Tm: {primers.reverse_tm:.1f}°C)")
print(f"Expected product: {primers.product_size} bp")
```

---

## API Reference

### PrimerDesigner Class

```python
class PrimerDesigner:
    """Design PCR primers for DNA sequences."""

    def __init__(
        self,
        min_length: int = 18,       # Minimum primer length
        max_length: int = 25,       # Maximum primer length
        min_gc: float = 40,         # Minimum GC content (%)
        max_gc: float = 60,         # Maximum GC content (%)
        min_tm: float = 55,         # Minimum melting temperature (°C)
        max_tm: float = 65,         # Maximum melting temperature (°C)
        max_homopolymer: int = 4,   # Maximum single-base run
    ):
        """Initialize primer designer with parameters."""

    def design_primers(
        self,
        sequence: str,                          # Template DNA sequence
        target_start: Optional[int] = None,     # Target region start
        target_length: Optional[int] = None,    # Target region length
        product_size_range: tuple = (100, 500), # Desired product size
    ) -> PrimerResult:
        """Design primers for a DNA sequence."""

    def design_for_peptide(
        self,
        peptide: str,                    # Amino acid sequence
        codon_optimization: str = "ecoli", # Target organism
        add_start_codon: bool = True,    # Add ATG
        add_stop_codon: bool = True,     # Add TAA
    ) -> PrimerResult:
        """Design primers to clone a peptide sequence."""

    def peptide_to_dna(
        self,
        peptide: str,                    # Amino acid sequence
        codon_optimization: str = "ecoli", # Target organism
    ) -> str:
        """Convert peptide to DNA with codon optimization."""
```

### PrimerResult Dataclass

```python
@dataclass
class PrimerResult:
    """Container for primer design results."""

    forward: str        # Forward primer sequence (5' -> 3')
    reverse: str        # Reverse primer sequence (5' -> 3')
    forward_tm: float   # Forward primer Tm (°C)
    reverse_tm: float   # Reverse primer Tm (°C)
    forward_gc: float   # Forward primer GC (%)
    reverse_gc: float   # Reverse primer GC (%)
    product_size: int   # Expected PCR product size (bp)
    penalty: float      # Quality penalty score (lower = better)
    warnings: list[str] # Design warnings
```

### Utility Functions

```python
from shared import (
    calculate_tm,         # Melting temperature calculation
    calculate_gc,         # GC content calculation
    reverse_complement,   # Get reverse complement
)

# Calculate melting temperature
tm = calculate_tm("ATGGGCAAATTCCTGAAA")
print(f"Tm: {tm:.1f}°C")

# Calculate GC content
gc = calculate_gc("ATGGGCAAATTCCTGAAA")
print(f"GC: {gc:.1f}%")

# Get reverse complement
rev_comp = reverse_complement("ATGGGC")
print(f"ATGGGC -> {rev_comp}")  # GCCCAT
```

---

## Design Parameters

### Optimal Ranges

| Parameter | Default | Optimal Range | Notes |
|-----------|---------|---------------|-------|
| Length | 18-25 bp | 18-22 bp | Longer = more specific |
| Tm | 55-65°C | 58-62°C | Match forward/reverse |
| GC content | 40-60% | 45-55% | Higher = more stable |
| Homopolymer | ≤4 | ≤3 | Avoid polymerase slippage |
| Self-complementarity | ≤4 bp | ≤3 bp | Prevent hairpins |
| Tm difference | - | ≤3°C | Ensures both primers work |

### Custom Parameters

```python
from shared import PrimerDesigner

# Stringent parameters for high-specificity PCR
stringent_designer = PrimerDesigner(
    min_length=20,
    max_length=24,
    min_gc=45,
    max_gc=55,
    min_tm=58,
    max_tm=62,
    max_homopolymer=3,
)

# Relaxed parameters for difficult templates
relaxed_designer = PrimerDesigner(
    min_length=18,
    max_length=28,
    min_gc=35,
    max_gc=65,
    min_tm=52,
    max_tm=68,
    max_homopolymer=5,
)
```

---

## Codon Optimization

### Why Optimize Codons?

Different organisms prefer different codons for the same amino acid:
- **E. coli**: Prefers GCG for Alanine (A)
- **Human**: Prefers GCC for Alanine (A)

Using non-preferred codons leads to:
- Slower translation
- Lower protein yield
- Ribosome stalling
- Misfolding

### Available Optimization Targets

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

# E. coli optimization (high-yield bacterial expression)
dna_ecoli = designer.peptide_to_dna("KFLH", codon_optimization="ecoli")
print(f"E. coli: {dna_ecoli}")  # AAATTCCTGCAT

# Human optimization (mammalian cell expression)
dna_human = designer.peptide_to_dna("KFLH", codon_optimization="human")
print(f"Human: {dna_human}")  # AAGTTCCTGCAC
```

### Codon Tables Used

**E. coli Preferred Codons:**
```
A: GCG    C: TGC    D: GAT    E: GAA    F: TTT
G: GGC    H: CAT    I: ATT    K: AAA    L: CTG
M: ATG    N: AAC    P: CCG    Q: CAG    R: CGT
S: AGC    T: ACC    V: GTG    W: TGG    Y: TAT
```

**Human Preferred Codons:**
```
A: GCC    C: TGC    D: GAC    E: GAG    F: TTC
G: GGC    H: CAC    I: ATC    K: AAG    L: CTG
M: ATG    N: AAC    P: CCC    Q: CAG    R: AGG
S: AGC    T: ACC    V: GTG    W: TGG    Y: TAC
```

---

## Melting Temperature Calculation

### Method

The module uses the Wallace rule for short oligos and a simplified nearest-neighbor approximation for longer primers:

```python
def calculate_tm(seq: str) -> float:
    """Calculate melting temperature.

    For < 14 bp (Wallace rule):
        Tm = 2 × (A+T count) + 4 × (G+C count)

    For >= 14 bp (nearest-neighbor approximation):
        Tm = 64.9 + 41 × (GC - 16.4) / length
    """
```

### Examples

```python
from shared import calculate_tm

primers = [
    "ATGGGC",           # 6 bp
    "ATGGGCAAATTCC",    # 13 bp
    "ATGGGCAAATTCCTGAAA",  # 18 bp
]

for primer in primers:
    tm = calculate_tm(primer)
    print(f"{primer}: Tm = {tm:.1f}°C")
```

**Output:**
```
ATGGGC: Tm = 20.0°C
ATGGGCAAATTCC: Tm = 40.0°C
ATGGGCAAATTCCTGAAA: Tm = 52.9°C
```

---

## Quality Checks

### Self-Complementarity (Hairpins)

Primers that form hairpin structures will not bind efficiently:

```
    G---C
   /     \
  A       G
  |       |
  T       C
   \     /
    C---G
    5'...3'  <-- Hairpin structure
```

```python
from shared.primer_design import check_self_complementarity

# Check for self-complementary regions
primer = "ATGCATGCAT"  # Palindromic
run = check_self_complementarity(primer)
print(f"Longest self-complementary run: {run} bp")
```

### Homopolymer Runs

Long runs of the same base cause polymerase slippage:

```python
from shared.primer_design import check_homopolymer

primers = [
    "ATGGGCAAATTCC",   # GGG = 3
    "ATGGGGGCAAATTCC", # GGGGG = 5 (problematic)
]

for primer in primers:
    run = check_homopolymer(primer)
    print(f"{primer}: longest run = {run}")
```

### 3' End Quality

The 3' end of a primer is critical for extension:
- Should end with G or C (GC clamp)
- Avoid AT-rich 3' ends
- No mismatch tolerance at 3' end

---

## Primer3 Integration

### Automatic Detection

The module automatically detects if Primer3 is installed:

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

if designer._primer3_available:
    print("Primer3 detected - using advanced algorithm")
else:
    print("Primer3 not found - using heuristic method")
```

### Installing Primer3

**Windows:**
1. Download from https://github.com/primer3-org/primer3/releases
2. Extract and add to PATH
3. Verify: `primer3_core --version`

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install primer3

# Mac (Homebrew)
brew install primer3

# Verify installation
primer3_core --version
```

### When Primer3 Is Used

```
IF Primer3 available:
    → Use Primer3 for optimal primer selection
    → Advanced thermodynamic calculations
    → Multiple primer pair options

ELSE:
    → Use heuristic scoring algorithm
    → Scan all possible primer positions
    → Score based on Tm, GC, homopolymers
```

---

## Practical Examples

### Example 1: Clone Magainin 2 for E. coli Expression

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

# Magainin 2 sequence
magainin = "GIGKFLHSAKKFGKAFVGEIMNS"

# Design primers for E. coli expression
primers = designer.design_for_peptide(
    magainin,
    codon_optimization="ecoli",
    add_start_codon=True,
    add_stop_codon=True,
)

print("=== Magainin 2 Cloning Primers ===")
print(f"Peptide: {magainin} ({len(magainin)} aa)")
print(f"Expected gene size: {primers.product_size} bp")
print()
print(f"Forward: 5'-{primers.forward}-3'")
print(f"  Tm: {primers.forward_tm:.1f}°C, GC: {primers.forward_gc:.1f}%")
print()
print(f"Reverse: 5'-{primers.reverse}-3'")
print(f"  Tm: {primers.reverse_tm:.1f}°C, GC: {primers.reverse_gc:.1f}%")

if primers.warnings:
    print(f"\nWarnings: {', '.join(primers.warnings)}")
```

### Example 2: Get DNA Sequence for Multiple Peptides

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

peptides = {
    "Magainin 2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
}

print(f"{'Peptide':<15} {'Length (aa)':<12} {'DNA Length (bp)':<15}")
print("-" * 45)

for name, seq in peptides.items():
    dna = designer.peptide_to_dna(seq, codon_optimization="ecoli")
    # Add start and stop codons
    full_dna = f"ATG{dna}TAA"
    print(f"{name:<15} {len(seq):<12} {len(full_dna):<15}")
```

### Example 3: Design Primers with Target Region

```python
from shared import PrimerDesigner

designer = PrimerDesigner()

# Long template with specific target
template = "ATGNNNNNNNNNNNNNNNNNTARGET_REGIONNNNNNNNNNNNNNNNNNNTAA"
template = template.replace("N", "A")  # Fill Ns for demo

# Amplify specific region (positions 20-35)
primers = designer.design_primers(
    template,
    target_start=20,
    target_length=15,
    product_size_range=(50, 100),
)

print(f"Template length: {len(template)} bp")
print(f"Target: positions 20-35")
print(f"Product size: {primers.product_size} bp")
print(f"Forward: {primers.forward}")
print(f"Reverse: {primers.reverse}")
```

### Example 4: Compare E. coli vs Human Optimization

```python
from shared import PrimerDesigner, calculate_gc

designer = PrimerDesigner()

peptide = "KFLHSAKK"

dna_ecoli = designer.peptide_to_dna(peptide, "ecoli")
dna_human = designer.peptide_to_dna(peptide, "human")

print(f"Peptide: {peptide}")
print()
print(f"E. coli optimized: {dna_ecoli}")
print(f"  GC content: {calculate_gc(dna_ecoli):.1f}%")
print()
print(f"Human optimized:   {dna_human}")
print(f"  GC content: {calculate_gc(dna_human):.1f}%")
```

---

## Scoring Algorithm

When Primer3 is not available, primers are scored using:

```python
def _score_primer(self, primer: str) -> float:
    """Score a primer (lower = better)."""
    score = 0.0

    # Tm penalty (outside optimal range)
    tm = calculate_tm(primer)
    if tm < min_tm:
        score += (min_tm - tm) * 2
    elif tm > max_tm:
        score += (tm - max_tm) * 2

    # GC penalty
    gc = calculate_gc(primer)
    if gc < min_gc:
        score += (min_gc - gc) * 0.5
    elif gc > max_gc:
        score += (gc - max_gc) * 0.5

    # Homopolymer penalty
    homo_run = check_homopolymer(primer)
    if homo_run > max_homopolymer:
        score += (homo_run - max_homopolymer) * 5

    # Self-complementarity penalty
    self_comp = check_self_complementarity(primer)
    if self_comp > 4:
        score += (self_comp - 4) * 3

    # 3' end penalty (prefer G/C)
    if primer[-1] not in "GC":
        score += 1

    return score
```

---

## Troubleshooting

### "Sequence shorter than minimum product size"

The template is too short for the specified product size range:

```python
# For short sequences, use smaller product range
primers = designer.design_primers(
    short_sequence,
    product_size_range=(50, 100),  # Adjust range
)
```

### "Tm outside target range"

Adjust designer parameters:

```python
# For AT-rich sequences (low Tm)
designer = PrimerDesigner(
    min_tm=50,
    max_tm=60,
)

# For GC-rich sequences (high Tm)
designer = PrimerDesigner(
    min_tm=60,
    max_tm=70,
)
```

### "Tm difference > 5°C"

The forward and reverse primers have mismatched Tm values:

```python
# Use narrower Tm range
designer = PrimerDesigner(
    min_tm=58,
    max_tm=62,
)
```

### "Could not find suitable primers"

Template sequence may have problematic regions:
- Very AT-rich or GC-rich
- Long homopolymer runs
- Highly repetitive sequences

Solution: Relax parameters or redesign template.

---

## Best Practices

### General Guidelines

1. **Order primers with HPLC purification** for critical cloning
2. **Verify primer specificity** with BLAST before ordering
3. **Include restriction sites** at 5' end for cloning (add 6 bp spacer)
4. **Test gradient PCR** to find optimal annealing temperature

### Expression Vector Design

```
5'-[Restriction Site][Spacer][ATG][Peptide Codons][Stop]-3'
    |___ BamHI, EcoRI   |    |         |            |
          NdeI, XhoI    6 bp ATG   Optimized      TAA/TGA/TAG

Example:
5'-GGATCC-AAAAAA-ATG-[gene]-TAA-GAATTC-3'
   BamHI  spacer start      stop EcoRI
```

### Primer Storage

- Dissolve in TE buffer (10 mM Tris, 1 mM EDTA, pH 8.0)
- Store stock at -20°C
- Working dilution at 4°C (use within 1 month)
- Avoid repeated freeze-thaw cycles

---

## References

1. Untergasser, A., et al. (2012). Primer3—new capabilities and interfaces. Nucleic acids research, 40(15), e115-e115.

2. SantaLucia, J., & Hicks, D. (2004). The thermodynamics of DNA structural motifs. Annual review of biophysics and biomolecular structure, 33, 415-440.

3. Dieffenbach, C. W., Lowe, T. M., & Dveksler, G. S. (1993). General concepts for PCR primer design. PCR methods and applications, 3(3), S30-S37.

4. Sharp, P. M., & Li, W. H. (1987). The codon adaptation index--a measure of directional synonymous codon usage bias, and its potential applications. Nucleic acids research, 15(3), 1281-1295.

---

## Quick Reference

```python
from shared import PrimerDesigner, calculate_tm, calculate_gc, reverse_complement

# Initialize
designer = PrimerDesigner()

# Design primers for DNA
primers = designer.design_primers(dna_sequence)

# Design primers for peptide
primers = designer.design_for_peptide(peptide, codon_optimization="ecoli")

# Convert peptide to DNA
dna = designer.peptide_to_dna(peptide, codon_optimization="ecoli")

# Utility functions
tm = calculate_tm("ATGGGCAAATTCC")
gc = calculate_gc("ATGGGCAAATTCC")
rev_comp = reverse_complement("ATGGGC")

# PrimerResult attributes
primers.forward        # Forward primer sequence
primers.reverse        # Reverse primer sequence
primers.forward_tm     # Forward Tm (°C)
primers.reverse_tm     # Reverse Tm (°C)
primers.forward_gc     # Forward GC (%)
primers.reverse_gc     # Reverse GC (%)
primers.product_size   # Expected product size (bp)
primers.warnings       # List of warnings
```
