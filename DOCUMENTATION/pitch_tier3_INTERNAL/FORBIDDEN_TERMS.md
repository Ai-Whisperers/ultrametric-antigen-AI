# FORBIDDEN TERMS

**Classification**: CONFIDENTIAL
**Purpose**: Terms that must NEVER appear in external communications

---

## ABSOLUTE PROHIBITIONS

These terms reveal core IP. **Never use externally under any circumstances.**

### P-adic Related
```
p-adic
3-adic
2-adic
p-adic norm
p-adic valuation
p-adic distance
p-adic metric
p-adic number
non-Archimedean
ultrametric
Ostrowski
Hensel
```

### Ternary Related
```
ternary encoding
balanced ternary
base-3 (in encoding context)
ternary neural network
ternary quantization
ternary algebra
ternary arithmetic
trit (ternary digit)
tryte
mod 3 (in encoding context)
```

### Specific Geometric Terms
```
Poincaré ball
Poincaré disk
Poincaré half-plane
hyperbolic VAE
hyperbolic embedding (when describing our method)
Möbius addition
Möbius transformation
exponential map (in hyperbolic context)
logarithmic map (in hyperbolic context)
Lorentz model
hyperboloid model
Klein model
gyrovector
Riemannian gradient
```

### Architecture Specifics
```
wrapped normal distribution
hyperbolic prior
p-adic prior
curvature parameter (learnable)
geodesic decoder
tangent space encoder
```

### Our Specific Innovations
```
codon encoder (as product name)
3-adic codon
ternary codon encoding
p-adic biological encoding
Goldilocks zone (as our term)
boundary crossing metric
cascade reveal
```

---

## CONTEXT-DEPENDENT PROHIBITIONS

These are OK in general but NOT when describing our method:

### OK in General, NOT for Our Method
```
hyperbolic (OK: "hyperbolic growth", NOT OK: "our hyperbolic model")
geometric (OK alone, but don't combine with specifics)
non-Euclidean (OK alone, risky with details)
manifold (OK alone)
curvature (OK in physics, careful in ML)
```

---

## COMBINATION PROHIBITIONS

These combinations are especially dangerous:

```
p-adic + codon
p-adic + protein
p-adic + sequence
p-adic + embedding
ternary + encoding + biological
ternary + codon
hyperbolic + VAE + biological
hyperbolic + protein
Poincaré + sequence
geometric + p-adic
algebraic + encoding + codon
```

---

## DOCUMENT GREP CHECKLIST

Before releasing any document, grep for:

```bash
# Critical (must have zero matches)
grep -i "p-adic\|padic\|3-adic" document.md
grep -i "ternary\|base-3\|balanced.*ternary" document.md
grep -i "poincare\|poincaré\|hyperbolic.*vae" document.md
grep -i "ultrametric\|non-archimedean" document.md
grep -i "mobius\|möbius\|gyrovector" document.md

# Warning (review context)
grep -i "hyperbolic\|geodesic\|curvature" document.md
grep -i "non-euclidean\|manifold" document.md
```

---

## SAFE REPLACEMENTS

| Forbidden | Safe Replacement |
|:----------|:-----------------|
| p-adic distance | sequence distance metric |
| ternary encoding | discrete encoding |
| hyperbolic space | geometric embedding space |
| Poincaré ball | bounded embedding space |
| ultrametric | tree-respecting distance |
| non-Archimedean | (omit) |
| curvature parameter | geometry parameter |
| geodesic | optimal path |
| Möbius addition | (omit) |
| wrapped distribution | manifold distribution |

---

## VERBAL COMMUNICATION

In verbal discussions (calls, meetings, conferences):

1. **Pause before technical details** - If about to explain "how it works", stop
2. **Use prepared phrases** - Memorize safe phrasings from SAFE_LANGUAGE.md
3. **Redirect to results** - "Let me show you what it predicts..."
4. **Invoke IP protection** - "That's proprietary, but I can share..."
5. **Offer NDA path** - "Happy to discuss under appropriate agreement"

---

## EMERGENCY RESPONSE

If you accidentally disclose a forbidden term:

1. **Don't emphasize it** - Don't say "forget I said that"
2. **Move on quickly** - Continue with safe language
3. **Document internally** - Note what was said, to whom, when
4. **Inform team** - Report to IP lead same day
5. **Assess damage** - Did they understand the significance?

---

## REVIEW AUTHORITY

All external documents must be reviewed for forbidden terms by:
- Primary: [Designated IP reviewer]
- Backup: [Secondary reviewer]

No exceptions for "quick emails" or "informal slides."

---

*This list is not exhaustive. When in doubt, consult SAFE_LANGUAGE.md or ask IP counsel.*
