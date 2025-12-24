# SAFE LANGUAGE GUIDE

**Classification**: CONFIDENTIAL
**Purpose**: Approved terminology for external communications

---

## PRINCIPLE

When discussing our technology externally, use **vague but accurate** language that:
- Does not reveal the specific mathematical basis
- Does not hint at the connection between concepts
- Sounds like standard ML/bioinformatics terminology
- Is technically defensible (not misleading)

---

## TERMINOLOGY TRANSLATION TABLE

### Mathematical Foundations

| INTERNAL (Never Say) | EXTERNAL (Safe) |
|:---------------------|:----------------|
| p-adic | "number-theoretic" or "algebraic" |
| 3-adic | "specialized encoding" |
| p-adic distance | "sequence distance metric" |
| p-adic norm | "hierarchical distance" |
| ultrametric | "tree-like distance" |
| non-Archimedean | omit entirely |

### Ternary Concepts

| INTERNAL (Never Say) | EXTERNAL (Safe) |
|:---------------------|:----------------|
| ternary encoding | "discrete encoding" |
| balanced ternary | "symmetric representation" |
| base-3 | "codon-native representation" |
| ternary algebra | "specialized arithmetic" |
| ternary logic | "multi-valued logic" |
| mod 3 | omit entirely |

### Geometric Concepts

| INTERNAL (Never Say) | EXTERNAL (Safe) |
|:---------------------|:----------------|
| hyperbolic space | "geometric embedding space" |
| Poincaré ball | "bounded embedding space" |
| Poincaré disk | "2D embedding space" |
| hyperbolic distance | "geometric distance" |
| geodesic | "optimal path" |
| curvature | "space geometry" (if must mention) |
| exponential map | omit entirely |
| logarithmic map | omit entirely |
| Möbius addition | omit entirely |

### Architectural Concepts

| INTERNAL (Never Say) | EXTERNAL (Safe) |
|:---------------------|:----------------|
| hyperbolic VAE | "geometric generative model" |
| p-adic prior | "structured prior" |
| ternary layers | "specialized layers" |
| mixed architecture | "hybrid architecture" |
| wrapped distribution | "manifold distribution" |

### Combined Concepts

| INTERNAL (Never Say) | EXTERNAL (Safe) |
|:---------------------|:----------------|
| p-adic + hyperbolic | "geometric sequence analysis" |
| ternary + p-adic | "algebraic encoding" |
| the core synergy | "our proprietary method" |

---

## SAFE PHRASES BY CONTEXT

### For Conferences/Pitches

**Describing the technology**:
- "We use a proprietary geometric approach to sequence analysis"
- "Our method captures hierarchical relationships in biological data"
- "We've developed novel distance metrics that predict biological function"
- "The encoding preserves evolutionary relationships"

**Describing results**:
- "Our computational predictions correlate with structural data"
- "The geometric features predict escape difficulty"
- "Distance metrics identify vulnerability zones"

**Avoiding specifics**:
- "The methodology is proprietary"
- "Detailed methods available under NDA"
- "We're preparing patent applications"

### For Scientific Discussions

**If pressed on methodology**:
- "We use a non-Euclidean embedding approach" (OK - many exist)
- "The encoding respects the triplet structure of codons" (OK - obvious)
- "We measure distances in a way that captures evolutionary cost" (OK - vague)

**Deflection phrases**:
- "That's covered by our IP protection"
- "We're happy to discuss under NDA"
- "The specifics are in our provisional filing"
- "Let's focus on the results, which are reproducible"

### For Partner Discussions (Tier 1)

- Can mention "geometric" and "non-Euclidean"
- Can mention "specialized encoding"
- CANNOT mention "p-adic", "ternary", or "hyperbolic"
- CANNOT explain the connection between components

---

## EXAMPLE TRANSFORMATIONS

### BAD (Internal Language)
> "We encode codons using 3-adic numbers and embed them in a Poincaré ball using a hyperbolic VAE with ternary-valued intermediate layers."

### GOOD (External Language)
> "We use a proprietary geometric encoding that captures hierarchical relationships, combined with a generative model architecture designed for biological sequences."

---

### BAD (Internal Language)
> "The p-adic distance between wild-type and mutant correlates with fitness cost because the ultrametric structure respects evolutionary trees."

### GOOD (External Language)
> "Our distance metric correlates with fitness cost. The metric is designed to capture evolutionary relationships."

---

### BAD (Internal Language)
> "Hyperbolic geometry is perfect for this because protein families form tree-like hierarchies, and hyperbolic space embeds trees with low distortion."

### GOOD (External Language)
> "We use a geometric approach that's well-suited to hierarchical biological data."

---

## QUESTION HANDLING

### "What kind of encoding do you use?"
**Safe**: "A proprietary encoding designed for biological sequences that preserves important structural relationships."

### "Is this like word2vec for proteins?"
**Safe**: "It has some similarities in that we learn representations, but the underlying geometry and encoding are quite different. The details are proprietary."

### "What makes your approach novel?"
**Safe**: "The combination of specialized encoding, geometric embedding, and generative modeling that we've developed. The synergy between these components is key."

### "Is this hyperbolic embeddings?"
**Safe**: "We use non-Euclidean geometry, yes. The specific formulation is proprietary."

### "Can you share the code?"
**Safe**: "We share validation protocols and predictions. The core methodology is proprietary but we're open to collaboration under appropriate IP arrangements."

---

## VALIDATION LANGUAGE (CRITICAL)

### The Honest Core Claim

**What we can truthfully say**:
> "A mathematical encoding that, without biological training, produces distance metrics correlating with independent structural predictions, sequence conservation data, and clinical resistance patterns."

### Validation-Specific Safe Phrases

| Reality | Safe External Phrasing |
|:--------|:-----------------------|
| Correlates with AlphaFold3 | "corroborated by independent structural predictions" |
| Matches database patterns | "aligns with observed sequence conservation" |
| Matches clinical hierarchy | "mirrors patterns in clinical resistance data" |
| Internally consistent | "computationally reproducible" |

### Required Qualifiers

**Always include one of these**:
- "in silico" or "computational"
- "correlates with" (not "validates" or "proves")
- "suggests" (not "demonstrates" or "confirms")
- "pending experimental confirmation"

### Overclaims That Will Damage Credibility

| Never Say | Why It's Wrong | Say Instead |
|:----------|:---------------|:------------|
| "Validated" (alone) | Implies wet-lab | "Computationally corroborated" |
| "Experimentally confirmed" | No wet-lab exists | "Correlates with structural predictions" |
| "Clinically proven" | No clinical data | "Aligns with clinical patterns" |
| "We discovered" | Overclaims causation | "Our analysis identifies" |

### The Honest Pitch Template

> "Our geometric encoding produces predictions that correlate with three independent external sources: structural predictions from AlphaFold3, sequence conservation in global databases, and clinical resistance patterns. This correlation—without any biological training—suggests the geometry captures real evolutionary constraints. We're seeking experimental partners to test whether these computational insights translate to therapeutic applications."

---

## RED FLAG PHRASES TO AVOID

If you find yourself about to say any of these, STOP:

- "The key insight is..."
- "What makes it work is..."
- "The connection between X and Y..."
- "We realized that X implies Y..."
- "The mathematical basis is..."
- "Specifically, we use..."
- "We've validated..." (without qualifier)
- "This proves..." (computational ≠ proof)

These phrases often precede IP disclosure or overclaiming.

---

## DOCUMENT REVIEW CHECKLIST

Before any external document (paper, slide, email), check:

- [ ] No forbidden terms (see FORBIDDEN_TERMS.md)
- [ ] No methodology details beyond "geometric" and "proprietary"
- [ ] No architecture diagrams showing internal structure
- [ ] No loss function formulas
- [ ] No hyperparameter values
- [ ] No training procedure details
- [ ] Results only, not methods

---

*Update this guide as new safe phrasings are developed.*
