# Hyperbolic Geometry: Where Trees Live Naturally

**Why curved space solves the tree embedding problem**

---

## 1. The Tree Embedding Problem

### Why Trees Don't Fit in Flat Space

Consider embedding a binary tree of depth 4:
- Level 0: 1 node (root)
- Level 1: 2 nodes
- Level 2: 4 nodes
- Level 3: 8 nodes
- Level 4: 16 nodes (leaves)

In 2D Euclidean space, we can't place 16 equidistant points (the leaves)!
We need exponentially growing room at each level - exactly what Euclidean space doesn't have.

### The Hyperbolic Solution

In **hyperbolic space**, circumference grows **exponentially** with radius:
```
Euclidean: C = 2πr        (linear growth)
Hyperbolic: C = 2π sinh(r) ≈ πe^r  (exponential growth!)
```

This means:
- Infinite room at the boundary
- Perfect for trees with exponentially many leaves
- No distortion needed!

---

## 2. The Poincare Ball Model

### Definition

The **Poincare ball** is hyperbolic space represented as a unit disk:
```
B^n = {x ∈ R^n : ||x|| < 1}
```

Points live **inside** the ball. The boundary (||x|| = 1) represents "infinity".

### Distance Formula

The Poincare distance between points u and v:
```
d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
```

**Key Property**: As points approach the boundary, distances **explode**!
- Near origin: behaves like Euclidean distance
- Near boundary: tiny Euclidean movements = huge hyperbolic distances

### Visualization

```
            Boundary (infinity)
           /                  \
          /                    \
         |      Far outer       |
         |        region        |
         |   /------------\     |
         |  |   Middle    |     |
         |  |   region    |     |
         |  |   /----\    |     |
         |  |  |Origin|   |     |
         |  |  | zone |   |     |
         |  |   \----/    |     |
         |   \-----------/      |
          \                    /
           \                  /
             ----------------
```

---

## 3. Why This Matches Evolution

### Radial Position = Evolutionary Depth

We interpret the Poincare ball as:
- **Origin (center)**: Most ancestral, stable sequences
- **Boundary (edge)**: Most derived, recent sequences

This mapping is justified because:
1. Ancestral sequences are "central" to the tree
2. Derived sequences are "leaves" at the periphery
3. The exponential room at the boundary matches exponential branching

### Geodesic Distance = Evolutionary Distance

The hyperbolic distance between two points reflects:
- How far up the tree you must go to find common ancestor
- Then how far back down to the other sequence

This is exactly what evolutionary distance measures!

---

## 4. Implementation (src/geometry/poincare.py)

### Using Geoopt Library

We use `geoopt` for numerically stable hyperbolic operations:

```python
import geoopt

def get_manifold(c: float = 1.0):
    """Get Poincare ball with curvature c."""
    return geoopt.PoincareBall(c=c)

def poincare_distance(x, y, c=1.0):
    """Compute Poincare distance using geoopt."""
    manifold = get_manifold(c)
    return manifold.dist(x, y)
```

### Key Operations

**Projection to Ball** (keep points inside):
```python
def project_to_poincare(z, max_norm=0.95):
    """Project points onto Poincare ball."""
    manifold = get_manifold()
    z_proj = manifold.projx(z)

    # Additional max_norm constraint
    norm = torch.norm(z_proj, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / norm, 1.0)
    return z_proj * scale
```

**Exponential Map** (tangent space → ball):
```python
def exp_map_zero(v, c=1.0):
    """Map from tangent space at origin to Poincare ball."""
    # exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
    manifold = get_manifold(c)
    return manifold.expmap(origin, v)
```

**Mobius Addition** (hyperbolic "addition"):
```python
def mobius_add(x, y, c=1.0):
    """Non-commutative addition on Poincare ball."""
    manifold = get_manifold(c)
    return manifold.mobius_add(x, y)
```

---

## 5. The Curvature Parameter

### What is Curvature?

The **curvature** `c` controls how "curved" the space is:
- c = 0: Euclidean (flat)
- c > 0: Hyperbolic (saddle-like, negative curvature in classical terms)
- c < 0: Spherical (positive curvature)

Higher curvature = more "room" at the boundary = better for deeper trees.

### In Our Model

We typically use `c = 1.0` or `c = 2.0`:
```python
# From config
curvature: 1.0  # or 2.0 for "more hyperbolic"
max_radius: 0.95  # Stay away from numerical instability at boundary
```

The `max_radius` constraint is critical:
- At ||x|| = 1, the conformal factor explodes
- We clip at 0.95 to maintain numerical stability

---

## 6. Connection to P-adic Structure

### The Deep Link

P-adic space and hyperbolic space both:
1. Have ultrametric-like properties
2. Naturally embed trees
3. Have "boundary at infinity"

**Correspondence**:
```
P-adic valuation v_3(n)  <-->  Radial position ||z||
High valuation (near 0)  <-->  Near origin (central)
Low valuation            <-->  Near boundary (peripheral)
```

### How We Enforce This

Our loss functions create this correspondence:
1. **Geodesic Loss**: Points with similar p-adic valuation should be close in Poincare distance
2. **Radial Loss**: High valuation operations should have small ||z||

---

## 7. Mathematical Properties

### Geodesics (Shortest Paths)

In the Poincare ball, geodesics are:
- **Diameters** (straight lines through origin)
- **Circular arcs** perpendicular to the boundary

```
    Boundary
   /        \
  |    .--.  |
  |   /    \ |
  |  | Path |---  A circular arc geodesic
  |   \    / |
  |    '--'  |
   \        /
```

### Parallel Transport

Moving vectors along geodesics is non-trivial:
```python
def parallel_transport(x, y, v, c=1.0):
    """Transport tangent vector v from x to y."""
    manifold = get_manifold(c)
    return manifold.transp(x, y, v)
```

This is used by Riemannian optimizers to update gradients correctly.

### Riemannian Optimization

Standard gradient descent doesn't work on curved manifolds! We use:
```python
from geoopt.optim import RiemannianAdam

optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
```

Riemannian Adam:
1. Computes Euclidean gradient
2. Converts to Riemannian gradient
3. Updates along geodesics
4. Projects back to manifold

---

## 8. Why This Works for Our VAE

### The Architecture

```
Input (ternary ops) → Encoder → Euclidean latent → PROJECTION → Poincare latent
                                                        ↓
                                              Hyperbolic losses enforce structure
                                                        ↓
                                              Decoder ← Poincare latent
```

### Benefits

1. **Hierarchical Structure**: The radial dimension encodes "fundamentalness"
2. **Efficient Capacity**: Exponential room means no crowding
3. **Geometric Losses**: Distance-based losses have clear interpretation
4. **Natural Regularization**: Boundary avoidance prevents collapse

---

## 9. Key Equations Summary

### Poincare Distance
```
d(u, v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
```

### Conformal Factor (metric scaling)
```
λ_x = 2 / (1 - c||x||^2)
```

### Exponential Map from Origin
```
exp_0(v) = tanh(sqrt(c)||v||) * v / (sqrt(c)||v||)
```

### Logarithmic Map to Origin
```
log_0(y) = arctanh(sqrt(c)||y||) * y / (sqrt(c)||y||)
```

---

## 10. Practical Considerations

### Numerical Stability

Near the boundary, operations become unstable:
```python
# Always project with max_norm < 1
max_norm = 0.95  # or 0.99 if brave
z = project_to_poincare(z, max_norm)
```

### Gradient Clipping

Hyperbolic gradients can explode near boundary:
```python
# In training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Initialization

Start points near origin (safe region):
```python
# Initialize with small norm
z = torch.randn(batch, dim) * 0.1
```

---

## Further Reading

- Nickel & Kiela (2017): "Poincare Embeddings for Learning Hierarchical Representations"
- Mathieu et al. (2019): "Continuous Hierarchical Representations with Poincare VAEs"
- Ganea et al. (2018): "Hyperbolic Neural Networks"

---

*The hyperbolic geometry provides the GEOMETRIC SUBSTRATE. Next, we'll see how biology naturally fits this framework.*
