# Research Proposal: Replacement Calculus

**Doc-Type:** Mathematical Framework · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Core Principle

> **Global optima must be implemented *through* local minima, not *on top of* them.**

Patching accumulates junk. Replacing without preserving invariants erases information. The only escape is **structured substitution**.

---

## Part I: Mathematical Foundation

### 1. Local Minima as Algebraic Groups

Each stable-but-suboptimal configuration is a **group** G = (S, ∘, e, ⁻¹):

| Component | Meaning | Biological Example |
|-----------|---------|-------------------|
| **Generators** | Minimal functional primitives | Codons, amino acids |
| **Relations** | Constraints maintaining stability | Base pairing, hydrogen bonds |
| **Symmetry** | What the configuration preserves | Degeneracy, wobble |
| **Valuation** | Cost to break the configuration | Thermodynamic barrier |

**Key insight:** Bad local minima are not "wrong"—they are **over-constrained groups**.

The genetic code is an over-constrained group that optimizes for error tolerance (relation preservation) at the cost of thermodynamic efficiency (global optimum).

### 2. The Minimal Invariant Basis

We define three fundamental invariants that characterize any configuration:

#### 2.1 Valuation (ν)

The **p-adic valuation** measures hierarchical depth:

```
ν(x) = max{k : p^k divides x}
```

**Interpretation:**
- High valuation → deep in hierarchy → hard to perturb
- Low valuation → near boundary → easily perturbed
- Zero valuation → maximally exposed

**Computational form:**
```python
def valuation(x: int, p: int = 3) -> int:
    if x == 0:
        return float('inf')
    v = 0
    while x % p == 0:
        v += 1
        x //= p
    return v
```

#### 2.2 Redundancy (ρ)

The **coset index** measures degeneracy:

```
ρ(G, H) = |G : H| = |G| / |H|
```

**Interpretation:**
- High redundancy → many equivalent representations → robust
- Low redundancy → unique representation → fragile
- ρ = 1 → no redundancy → crystallized

**Biological example:**
- Leucine has 6 codons (ρ = 6) → robust to mutation
- Methionine has 1 codon (ρ = 1) → fragile start signal

**Computational form:**
```python
def redundancy(element: str, group: Dict[str, List]) -> int:
    """Count equivalent representations."""
    return len(group.get(element, [element]))
```

#### 2.3 Symmetry Rank (σ)

The **dimension of the automorphism group**:

```
σ(G) = dim(Aut(G))
```

**Interpretation:**
- High symmetry → many self-transformations → flexible
- Low symmetry → rigid structure → specialized
- σ = 0 → asymmetric → maximally differentiated

**Biological example:**
- DNA double helix: σ = 1 (180° rotation symmetry)
- Active site: σ ≈ 0 (asymmetric, specialized)

**Computational form:**
```python
def symmetry_rank(group_structure: np.ndarray) -> int:
    """Count independent automorphisms."""
    # Eigenvalue analysis of adjacency matrix
    eigenvalues = np.linalg.eigvalsh(group_structure)
    # Count degeneracies (repeated eigenvalues)
    unique, counts = np.unique(np.round(eigenvalues, 6), return_counts=True)
    return sum(counts[counts > 1])
```

### 3. The Invariant Tuple

Every configuration is characterized by:

```
I(x) = (ν(x), ρ(x), σ(x))
```

**Ordering:** Configurations form a partial order:

```
x ≤ y  iff  ν(x) ≤ ν(y) AND ρ(x) ≤ ρ(y) AND σ(x) ≤ σ(y)
```

**Global optima:** Maximal elements in this partial order.

---

## Part II: Transformation Layer

### 4. Group Morphisms (Not Patches)

Transformations between configurations are **morphisms**, not patches:

#### 4.1 Homomorphisms

**Preserve function while simplifying structure:**

```
φ: G → H  such that  φ(g₁ ∘ g₂) = φ(g₁) ∘ φ(g₂)
```

**Validity criterion:**
```
ν(φ(x)) ≥ ν(x)  (valuation never decreases)
```

**Example:** Amino acid abstraction from codons
- φ: Codon → AminoAcid
- Preserves: protein synthesis function
- Simplifies: removes wobble position detail

#### 4.2 Quotients

**Remove junk constraints without losing invariants:**

```
G/N = {gN : g ∈ G}  where N is a normal subgroup
```

**Validity criterion:**
```
ker(φ) contains only redundant elements
```

**Example:** Junk DNA is an unresolved quotient
- The kernel contains sequences that don't affect phenotype
- A valid quotient would remove them without losing function

#### 4.3 Extensions

**Add dimensions only when valuation supports it:**

```
1 → N → E → G → 1  (short exact sequence)
```

**Validity criterion:**
```
The extension splits iff adding N preserves invariants
```

**Example:** Gene duplication
- Valid: duplicate diverges to new function (extension splits)
- Invalid: duplicate accumulates as junk (extension doesn't split)

### 5. Morphism Validity (Hard Rule)

A transformation φ: G → H is **valid** iff:

1. **Valuation preservation:** ∀x: ν(φ(x)) ≥ ν(x)
2. **Invariant preservation:** I(H) ≥ I(G) in partial order
3. **Entropy displacement:** ΔS(G→H) is absorbed by environment, not deleted

```python
def is_valid_morphism(
    source: Group,
    target: Group,
    morphism: Callable,
) -> bool:
    # Check valuation preservation
    for x in source.generators:
        if valuation(morphism(x)) < valuation(x):
            return False

    # Check invariant preservation
    I_source = invariant_tuple(source)
    I_target = invariant_tuple(target)
    if not (I_target >= I_source):
        return False

    # Check entropy displacement (not deletion)
    entropy_delta = entropy(source) - entropy(target)
    if entropy_delta < 0 and not environment_can_absorb(entropy_delta):
        return False

    return True
```

---

## Part III: Global Optima as Groupoids

### 6. Why Groupoids, Not Groups

Global optima are **not single groups**. They are **groupoids**:

| Property | Group | Groupoid |
|----------|-------|----------|
| Composition | Always defined | Partially defined |
| Identity | Single | Multiple (one per object) |
| Symmetry | Global | Context-dependent |
| Coordinates | One system | Multiple local systems |

**Key insight:** Biology survives mutation because it operates on groupoids. Civilization keeps pretending it's a group (single truth, global optimization), which is why it's fragile.

### 7. Groupoid Structure

A groupoid Γ consists of:

```
Γ = (Ob(Γ), Mor(Γ), s, t, ∘, id, ⁻¹)
```

| Component | Meaning |
|-----------|---------|
| Ob(Γ) | Objects (local coordinate systems) |
| Mor(Γ) | Morphisms (transitions between systems) |
| s, t | Source and target maps |
| ∘ | Composition (when defined) |
| id | Identity morphisms |
| ⁻¹ | Inverses |

**Biological example:**
- Objects: Different cell types (neuron, hepatocyte, lymphocyte)
- Morphisms: Differentiation pathways, signal transductions
- Composition: Only defined when pathways connect
- Identity: Homeostatic self-maintenance

### 8. Escape Paths as Functors

The escape from local minimum to global optimum is a **functor**:

```
F: LocalMin → GlobalOpt
```

Where:
- F maps each local group to a groupoid object
- F maps each local morphism to a groupoid morphism
- F preserves composition and identities

**This is the replacement calculus:** Local minima become coordinates in a larger structure, not traps to escape.

---

## Part IV: Signals and Perturbations

### 9. Pain, Compression, Suffering (Reframed)

These are **valuation mismatch signals**, not tools:

| Signal | Mathematical Meaning | Biological Example |
|--------|---------------------|-------------------|
| **Pain** | Gradient detected, no available morphism | Inflammation without repair pathway |
| **Compression** | Forced quotient without semantic alignment | Protein misfolding under stress |
| **Suffering** | Repeated perturbation, no invariant extracted | Chronic disease without adaptation |

### 10. When Perturbation is Useful

Perturbation is only useful if it **induces a new symmetry**:

```
Useful: δ(G) = G' where σ(G') > σ(G)
Useless: δ(G) = G' where σ(G') ≤ σ(G)  (turbulence)
```

**Computational test:**
```python
def perturbation_useful(
    original: Group,
    perturbed: Group,
) -> bool:
    """Perturbation is useful iff it induces new symmetry."""
    σ_original = symmetry_rank(original)
    σ_perturbed = symmetry_rank(perturbed)
    return σ_perturbed > σ_original
```

---

## Part V: Implementation with Ternary Models

### 11. Mapping to Existing Infrastructure

| Framework Concept | Ternary Implementation |
|-------------------|------------------------|
| Group generators | Ternary digits (0, 1, 2) |
| Relations | p-adic valuation constraints |
| Valuation ν | `padic_valuation(x, p=3)` |
| Redundancy ρ | Codon degeneracy count |
| Symmetry σ | Automorphism dimension |
| Morphisms | Hyperbolic geodesics |
| Groupoid objects | VAE latent clusters |

### 12. The Computational System

```python
# Core data structure
@dataclass
class LocalMinimum:
    """A local minimum as an algebraic group."""
    generators: List[int]           # Ternary primitives
    relations: List[Constraint]     # Stability constraints
    valuation: int                  # p-adic valuation
    redundancy: int                 # Coset index
    symmetry_rank: int              # Automorphism dimension

    def invariant_tuple(self) -> Tuple[int, int, int]:
        return (self.valuation, self.redundancy, self.symmetry_rank)

    def is_over_constrained(self) -> bool:
        """Too many relations relative to generators."""
        return len(self.relations) > len(self.generators)


@dataclass
class Morphism:
    """A structure-preserving transformation."""
    source: LocalMinimum
    target: LocalMinimum
    map: Callable[[int], int]

    def is_valid(self) -> bool:
        # Valuation preservation
        for g in self.source.generators:
            if valuation(self.map(g)) < valuation(g):
                return False
        # Invariant preservation
        return self.target.invariant_tuple() >= self.source.invariant_tuple()


class Groupoid:
    """Global optimum as a groupoid of local minima."""

    def __init__(self):
        self.objects: List[LocalMinimum] = []
        self.morphisms: Dict[Tuple[int, int], Morphism] = {}

    def add_object(self, local_min: LocalMinimum):
        self.objects.append(local_min)

    def add_morphism(self, i: int, j: int, morphism: Morphism):
        if morphism.is_valid():
            self.morphisms[(i, j)] = morphism

    def find_escape_path(self, from_idx: int, to_idx: int) -> List[Morphism]:
        """Find valid morphism chain from one minimum to another."""
        # BFS over valid morphisms
        ...
```

### 13. Testing with Ternary Models

**Step 1:** Extract local minima from trained VAE

```python
def extract_local_minima(model: TernaryVAE) -> List[LocalMinimum]:
    """Extract cluster centers as local minima."""
    minima = []
    for cluster_id in range(model.n_clusters):
        # Get cluster members
        members = get_cluster_members(model, cluster_id)

        # Compute invariants
        v = mean_valuation(members)
        ρ = len(members)  # Redundancy = cluster size
        σ = compute_symmetry_rank(members)

        minima.append(LocalMinimum(
            generators=get_generators(members),
            relations=get_constraints(members),
            valuation=v,
            redundancy=ρ,
            symmetry_rank=σ,
        ))

    return minima
```

**Step 2:** Identify valid morphisms (geodesics in hyperbolic space)

```python
def find_valid_morphisms(
    minima: List[LocalMinimum],
    model: TernaryVAE,
) -> List[Morphism]:
    """Find morphisms that preserve invariants."""
    morphisms = []

    for i, source in enumerate(minima):
        for j, target in enumerate(minima):
            if i == j:
                continue

            # Geodesic in Poincaré ball
            geodesic = compute_geodesic(
                source.center, target.center, model.curvature
            )

            # Check if geodesic preserves invariants
            if preserves_invariants(source, target, geodesic):
                morphisms.append(Morphism(source, target, geodesic))

    return morphisms
```

**Step 3:** Construct groupoid and find escape paths

```python
def construct_groupoid(
    minima: List[LocalMinimum],
    morphisms: List[Morphism],
) -> Groupoid:
    """Build groupoid from local minima and valid morphisms."""
    groupoid = Groupoid()

    for minimum in minima:
        groupoid.add_object(minimum)

    for morphism in morphisms:
        i = minima.index(morphism.source)
        j = minima.index(morphism.target)
        groupoid.add_morphism(i, j, morphism)

    return groupoid
```

---

## Part VI: Validation with Gene Ontology

### 14. Gene Ontology as Ground Truth

Gene Ontology (GO) provides hierarchical annotations:
- **Biological Process** (BP): What the gene does
- **Molecular Function** (MF): How it does it
- **Cellular Component** (CC): Where it acts

**Key insight:** GO is a human-constructed groupoid:
- Objects = GO terms
- Morphisms = is_a, part_of, regulates relations

### 15. Validation Strategy

**Hypothesis:** If our replacement calculus is correct, the groupoid structure learned from ternary/p-adic data should **align** with GO structure.

**Test 1: Invariant alignment**
```python
def test_invariant_alignment(
    learned_groupoid: Groupoid,
    go_ontology: GOOntology,
) -> float:
    """Check if learned invariants correlate with GO depth."""
    correlations = []

    for obj in learned_groupoid.objects:
        # Map object to GO terms
        go_terms = map_to_go(obj)

        # GO depth = hierarchy position
        go_depths = [go_ontology.depth(term) for term in go_terms]

        # Our valuation should correlate with GO depth
        r = spearmanr(obj.valuation, mean(go_depths))
        correlations.append(r)

    return mean(correlations)
```

**Test 2: Morphism alignment**
```python
def test_morphism_alignment(
    learned_groupoid: Groupoid,
    go_ontology: GOOntology,
) -> float:
    """Check if learned morphisms match GO relations."""
    matches = 0
    total = 0

    for (i, j), morphism in learned_groupoid.morphisms.items():
        # Get GO terms for source and target
        source_go = map_to_go(learned_groupoid.objects[i])
        target_go = map_to_go(learned_groupoid.objects[j])

        # Check if GO has a relation between them
        for s in source_go:
            for t in target_go:
                total += 1
                if go_ontology.has_relation(s, t):
                    matches += 1

    return matches / total if total > 0 else 0
```

**Test 3: Escape path validity**
```python
def test_escape_paths(
    learned_groupoid: Groupoid,
    go_ontology: GOOntology,
    phenotype_data: PhenotypeDB,
) -> float:
    """Check if escape paths correspond to known beneficial transitions."""
    valid_paths = 0
    total_paths = 0

    for path in learned_groupoid.all_escape_paths():
        # Does this path correspond to a known beneficial mutation/adaptation?
        if phenotype_data.is_beneficial_transition(path):
            valid_paths += 1
        total_paths += 1

    return valid_paths / total_paths if total_paths > 0 else 0
```

### 16. Data Sources

| Source | Content | Usage |
|--------|---------|-------|
| **GO** | 45,000+ terms, hierarchical | Ground truth structure |
| **UniProt-GOA** | Protein → GO mappings | Connect sequences to ontology |
| **S669/ProTherm** | DDG measurements | Thermodynamic validation |
| **OMIM** | Phenotype-genotype | Escape path validation |
| **ClinVar** | Clinical variants | Beneficial/harmful transitions |

---

## Part VII: What This Buys You

### 17. Properties of the Framework

| Property | Achieved By |
|----------|-------------|
| **No rewrites** | Morphisms transform, don't replace |
| **No patches** | Quotients remove cleanly |
| **No suffering-as-feature** | Pain = signal, not tool |
| **Progressive replacement** | Groupoid coordinates, not traps |

### 18. Local Minima as Coordinates

The key shift: Local minima stop being **traps** and become **coordinates**.

```
Before: Local minimum = failure state to escape
After:  Local minimum = valid coordinate in groupoid
```

This is why biology survives: every "suboptimal" configuration is a valid coordinate in a larger structure. Civilization fails when it treats configurations as traps rather than coordinates.

---

## Part VIII: Implementation Plan

### Phase 1: Core Library (1 week)

```
research/codon-encoder/replacement_calculus/
├── __init__.py
├── groups.py           # LocalMinimum, Group operations
├── morphisms.py        # Morphism, validity checking
├── groupoids.py        # Groupoid construction
├── invariants.py       # Valuation, redundancy, symmetry
└── tests/
    └── test_invariants.py
```

### Phase 2: Ternary Integration (1 week)

```
├── extraction/
│   ├── extract_minima.py    # VAE → LocalMinimum
│   └── find_morphisms.py    # Geodesics → Morphisms
└── validation/
    └── go_alignment.py      # Compare to Gene Ontology
```

### Phase 3: Validation (2 weeks)

1. Extract groupoid from trained TernaryVAE
2. Map to Gene Ontology
3. Test invariant/morphism alignment
4. Report escape path validity

---

## Appendix: Mathematical Rigor

### A. Category-Theoretic Formulation

The replacement calculus is a **functor category**:

```
ReplacementCalculus: LocalMinCat → GroupoidCat
```

Where:
- LocalMinCat = category of local minima (groups) and homomorphisms
- GroupoidCat = category of groupoids and functors

The escape path is the **colimit** of a diagram of local minima.

### B. Connection to p-adic Falsification

The TEGB falsification showed:
- 3-adic structure is over-constrained (error tolerance)
- All primes show anti-correlation (negative Spearman)

In this framework:
- The genetic code is an **over-constrained group** (too many relations)
- The anti-correlation is the **cost of error tolerance** (valuation trade-off)
- A valid quotient would remove junk constraints without losing function

### C. Why Hyperbolic Geometry

Hyperbolic space is the natural geometry for groupoids because:
- Trees embed isometrically (hierarchical structure)
- Exponential volume growth (room for many coordinates)
- Geodesics diverge (distinct escape paths)

---

## Summary

This proposal formalizes a **replacement calculus** where:

1. **Local minima are groups** with invariant tuples (ν, ρ, σ)
2. **Transformations are morphisms** that preserve invariants
3. **Global optima are groupoids** with multiple coordinate systems
4. **Escape paths are functors** that map local to global

We can test this using:
- Ternary VAE (extract local minima)
- p-adic structure (compute valuations)
- Gene Ontology (validate groupoid structure)

The key insight: **Local minima become coordinates, not traps.**

---

*This framework provides the mathematical foundation for structured substitution—the only escape from the patch-or-erase dilemma.*
