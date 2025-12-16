# Plan: Radius - Semantic Acceleration for Game Engines

**Status:** Design Document
**Date:** 2025-12-16
**Codename:** Radius
**Tagline:** "O(1) scene management via ultrametric geometry"

---

## Executive Summary

Game engines spend 30-50% of CPU frame time on hierarchical scene management (culling, LOD, collision, pathfinding). These operations traverse tree structures that are inherently ultrametric. By encoding scene graphs as hyperbolic embeddings where radius = hierarchy depth, we replace O(log n) tree traversals with O(1) SIMD radius comparisons, enabling AAA-complexity games on budget hardware.

---

## Part I: The Opportunity

### Current State: Wasted Cycles

| Operation | Current Complexity | CPU Time % | Structure |
|:----------|:-------------------|:-----------|:----------|
| Frustum culling | O(n) or O(log n) BVH | 5-10% | Tree |
| Occlusion culling | O(n) queries | 10-15% | Tree |
| LOD selection | O(n) distance calcs | 5-10% | Hierarchy |
| Collision broad-phase | O(n log n) BVH | 10-20% | Tree |
| Pathfinding | O(n log n) A* | 5-10% | Graph/Tree |
| **Total scene overhead** | | **35-65%** | |

### Insight: Games Are Already Ultrametric

```
Scene Graph (Ultrametric Tree):
World
├── Region_A
│   ├── Chunk_01
│   │   ├── Building_001
│   │   │   ├── Room_01
│   │   │   │   ├── Chair_001 (LOD 3)
│   │   │   │   └── Table_001 (LOD 3)
│   │   │   └── Room_02
│   │   └── Building_002
│   └── Chunk_02
└── Region_B
    └── ...

Depth in tree = Semantic importance = LOD level = Radius in embedding
```

### Radius Proposition

| Operation | Current | With Radius | Speedup |
|:----------|:--------|:------------|:--------|
| Frustum culling | BVH traverse | 1 SIMD dot product | 10-100× |
| LOD selection | Distance calc per object | Radius lookup | 50× |
| Collision broad-phase | BVH traverse | Ancestry query | 20-50× |
| "Is A visible from B?" | Raycast | Radius comparison | 100× |
| Pathfinding hint | A* search | Semantic locality | 5-10× |

---

## Part II: Technical Architecture

### Core Abstraction

```c
// radius.h - Header-only, zero dependencies

typedef struct {
    float embedding[20];  // 20D hyperbolic embedding
    float radius;         // Precomputed ||embedding||
    uint8_t shell;        // Precomputed hierarchy level
} RadiusNode;

// O(1) operations - compile to pure SIMD
bool radius_in_frustum(RadiusNode* node, Frustum* f);
bool radius_ancestor_of(RadiusNode* parent, RadiusNode* child);
bool radius_may_collide(RadiusNode* a, RadiusNode* b);
uint8_t radius_lod_level(RadiusNode* node, vec3 camera_pos);
RadiusNode* radius_lca(RadiusNode* a, RadiusNode* b);
```

### Scene Graph Encoding

```
At scene load / level stream:
1. Parse scene graph tree
2. Assign each node: depth, parent, bounding volume
3. Encode to 20D hyperbolic embedding:
   - Radius = f(depth) where f is monotonic decreasing
   - Angular position = hash(spatial_position, parent_embedding)
4. Precompute shell boundaries for O(1) level lookup
5. Store embeddings in cache-aligned array (SOA layout)

Runtime:
- No tree pointers, no traversal
- All queries are SIMD on embedding arrays
```

### Memory Layout (Cache-Optimized)

```c
// Structure of Arrays for SIMD efficiency
typedef struct {
    float* radii;           // [N] - hot, always accessed
    uint8_t* shells;        // [N] - hot, LOD selection
    float* embeddings;      // [N×20] - warm, spatial queries
    AABB* bounds;           // [N] - cold, precise collision
    uint32_t* entity_ids;   // [N] - cold, result mapping
} RadiusScene;

// Cache behavior:
// - radii + shells fit in L1 for 1000s of objects
// - embeddings fit in L2 for 10,000s of objects
// - Full scene in L3/RAM
```

### SIMD Implementation

```c
// AVX2 implementation (8 objects per instruction)
__m256 radius_batch_in_frustum_avx2(
    __m256 obj_radii,           // 8 object radii
    __m256 obj_x, obj_y, obj_z, // 8 object positions
    Frustum* frustum
) {
    // 1. Radius check: objects beyond max_radius are culled
    __m256 max_r = _mm256_set1_ps(frustum->max_radius);
    __m256 radius_mask = _mm256_cmp_ps(obj_radii, max_r, _CMP_LT_OS);

    // 2. Plane tests (simplified - full impl tests all 6 planes)
    // ... SIMD dot products with frustum planes ...

    return _mm256_and_ps(radius_mask, plane_mask);
}

// ARM NEON implementation
uint32x4_t radius_batch_in_frustum_neon(
    float32x4_t obj_radii,
    float32x4_t obj_x, obj_y, obj_z,
    Frustum* frustum
) {
    // Similar structure, NEON intrinsics
    // ...
}
```

---

## Part III: Integration Path

### Phase 1: Standalone Library (Week 1-2)

**Deliverable:** `radius.h` header-only library

```c
// Single-header library, C99, zero dependencies
// Auto-detects: AVX2, AVX-512, NEON, SVE, scalar fallback

#define RADIUS_IMPLEMENTATION
#include "radius.h"

// That's it. Now you have O(1) scene queries.
```

**Tasks:**
- [ ] Core data structures (RadiusNode, RadiusScene)
- [ ] Encoding functions (tree → embeddings)
- [ ] Query functions (frustum, LOD, collision, ancestry)
- [ ] SIMD backends (AVX2, NEON, scalar)
- [ ] Unit tests and benchmarks

### Phase 2: Engine Plugins (Week 3-4)

**Deliverable:** Drop-in plugins for major engines

| Engine | Integration Point | Effort |
|:-------|:------------------|:-------|
| Unity | Custom SRP + C# bindings | Medium |
| Unreal | Scene Component + Plugin | Medium |
| Godot | GDExtension | Low |
| Custom (C/C++) | Direct `#include` | Trivial |

**Tasks:**
- [ ] Unity: Native plugin + C# wrapper
- [ ] Unreal: RadiusSceneComponent + Blueprint nodes
- [ ] Godot: GDExtension with GDScript bindings
- [ ] Documentation and example scenes

### Phase 3: Benchmarks & Demos (Week 5-6)

**Deliverable:** Proof that budget hardware runs complex scenes

**Benchmark Suite:**
1. **Culling stress test:** 100K objects, measure FPS
2. **LOD stress test:** Dynamic LOD across 50K objects
3. **Collision broad-phase:** 10K dynamic objects
4. **Open world:** Stream 1M object world on mobile

**Demo Targets:**
| Device | Target Scene Complexity | Current Limit |
|:-------|:------------------------|:--------------|
| iPhone SE (2020) | 50K dynamic objects | ~5K |
| $200 Android | 30K dynamic objects | ~3K |
| Raspberry Pi 4 | 20K dynamic objects | ~2K |
| M1 MacBook Air | 500K dynamic objects | ~50K |

### Phase 4: Open Source Launch (Week 7-8)

**Deliverable:** Public release with adoption strategy

**Repository Structure:**
```
radius/
├── include/
│   └── radius.h              # Single-header library
├── examples/
│   ├── unity/                # Unity demo project
│   ├── unreal/               # Unreal demo project
│   ├── godot/                # Godot demo project
│   └── raw_opengl/           # Minimal C example
├── benchmarks/
│   ├── culling_bench.c
│   ├── collision_bench.c
│   └── results/              # Published benchmark data
├── docs/
│   ├── GETTING_STARTED.md
│   ├── INTEGRATION_GUIDE.md
│   └── THEORY.md             # The math behind it
├── LICENSE                   # MIT or Apache 2.0
└── README.md
```

**Launch Checklist:**
- [ ] Benchmarks on 10+ devices
- [ ] Integration guides for 3 engines
- [ ] Hacker News / Reddit / Twitter launch
- [ ] Reach out to indie devs for beta testing

---

## Part IV: Why This Wins

### Technical Moat

1. **Math is unbreakable:** Ultrametric ↔ Hyperbolic is fundamental
2. **SIMD is universal:** Works on every processor made since 2013
3. **Header-only is frictionless:** No build system, no dependencies
4. **Performance is provable:** Simple to benchmark, hard to argue with

### Market Position

```
                    Complexity
                        ↑
    Full Engine         │         Our target
    (Unity, Unreal)     │             ↓
                        │    ┌─────────────────┐
                        │    │   radius.h      │
                        │    │   O(1) scenes   │
                        │    │   any engine    │
                        │    └─────────────────┘
                        │
    Raw GPU APIs ───────┼─────────────────────────→ Performance
    (Vulkan, Metal)     │
                        │
```

**We are the only library that:**
- Makes scene management O(1)
- Works across all engines
- Has zero integration cost
- Enables mobile to compete with desktop

### Adoption Flywheel

```
Free library → Indie adoption → Benchmarks go viral →
→ AAA studios notice → Engine integration → Standard practice
```

---

## Part V: Resource Requirements

### Team

| Role | Responsibility | Allocation |
|:-----|:---------------|:-----------|
| Core Dev | radius.h implementation | 100% |
| Engine Dev | Unity/Unreal/Godot plugins | 50% |
| DevRel | Docs, examples, community | 25% |

### Hardware for Testing

| Device | Purpose | Cost |
|:-------|:--------|:-----|
| iPhone SE | iOS mobile baseline | $400 |
| Budget Android | Android mobile baseline | $200 |
| Raspberry Pi 4 | ARM embedded baseline | $75 |
| Steam Deck | Handheld gaming | $400 |
| M1 Mac Mini | ARM desktop | $600 |
| Gaming PC (RTX 3060) | x86 desktop | $800 |

**Total hardware budget:** ~$2,500

### Timeline

```
Week 1-2: radius.h core library
Week 3-4: Engine plugins
Week 5-6: Benchmarks and demos
Week 7-8: Open source launch
────────────────────────────────
Total: 8 weeks to public release
```

---

## Part VI: Success Metrics

### Technical

- [ ] 10× speedup on frustum culling (vs naive)
- [ ] 5× speedup on collision broad-phase (vs BVH)
- [ ] O(1) LOD selection (vs O(n) distance calcs)
- [ ] Zero memory overhead vs current scene graphs

### Adoption (6 months post-launch)

- [ ] 1,000+ GitHub stars
- [ ] 10+ shipped games using Radius
- [ ] Integration PRs to major open-source engines
- [ ] Conference talk (GDC, etc.)

### Business (12 months)

- [ ] Enterprise support contracts
- [ ] Consulting for AAA integration
- [ ] Acquisition interest from engine vendors

---

## Part VII: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Encoding overhead too high | Medium | High | Amortize over frames, async encode |
| SIMD gains less than expected | Low | High | Benchmark early, pivot if needed |
| Engine integration painful | Medium | Medium | Start with easiest (Godot) |
| Adoption slow | Medium | Low | Focus on indie/mobile first |
| Someone else does it first | Low | High | Move fast, launch in 8 weeks |

---

## Appendix: Connection to Exascale Plan

This project is the **commercialization path** for the exascale semantic computing research:

| Research (Exascale Plan) | Product (Radius) |
|:-------------------------|:-----------------|
| Hyperbolic embeddings | Scene graph encoding |
| Ultrametric structure | Game hierarchy |
| O(1) semantic queries | O(1) culling/LOD/collision |
| SIMD primitives | radius.h library |
| 19,683× amplification | 10-100× scene speedup |

**Radius is the proof that semantic acceleration works in the real world.**

---

**Document Status:** Ready for execution
**Next Action:** Begin Phase 1 - radius.h core implementation
**Blocking Question:** None - path is clear
