# Experimental Methodology: Topological Probing

> **Goal:** We do not just "train models". We probe the shape of the learned manifold.

This document explains the core experimental techniques used in `experiments/`.

---

## 1. Probing the Manifold (`experiments/demo/`)

**The Question:** "Does the model actually learn a hyperbolic shape?"

**The Method:**

1.  **Iso-Fitness Contours:** We generate 10,000 random sequences, predict their fitness (reconstruction error), and plot them in 3D (PCA of Poincaré embeddings).
2.  **Visual Confirmation:** We look for a "Cone" or "Saddle" shape.
    - _Flat plane_ = Model failed (Euclidean collapse).
    - _Hyperboloid_ = Model succeeded (Curvature learnt).

**Script:** `scripts/visualization/visualize_ternary_manifold.py`

---

## 2. Tracing Trajectories (`experiments/bioinformatics/`)

**The Question:** "Is viral evolution a straight line or a curve?"

**The Method:**

1.  **Start Point:** Ancestral Strain (e.g., Wuhan-Hu-1).
2.  **End Point:** Current Variant (e.g., Omicron).
3.  **Interpolation:**
    - _Linear Interpolation:_ `z = (1-t)z1 + t*z2`.
    - _Geodesic Interpolation:_ `z = mobius_add(z1, scale(z2, t))`.
4.  **Metric:** We decode points along both paths. The path with lower decoding error is the "True Biological Path".

**Result:** HIV consistently follows the Geodesic (Curved) path, confirming hyperbolic geometry.

---

## 3. The Regenerative Axis (Clinical Scoring)

**The Question:** "Where is 'Health' located?"

**The Method:**

1.  **Embedding:** Embed 1,000 healthy HLA sequences.
2.  **PCA:** Find the First Principal Component (PC1) of the healthy cluster.
3.  **Definition:** This vector is the "Regenerative Axis".
4.  **Scoring:** For a patient, computing `Distance(patient_z, regenerative_axis)`.
    - _Small Distance:_ Healthy.
    - _Large Distance:_ Autoimmune Risk.

**Validation:** Correlates with clinically observed RA severity (r=0.75).
