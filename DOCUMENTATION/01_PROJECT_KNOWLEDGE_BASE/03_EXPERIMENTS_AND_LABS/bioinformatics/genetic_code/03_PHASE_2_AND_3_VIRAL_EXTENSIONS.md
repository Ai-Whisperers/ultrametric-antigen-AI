# PTM Extension: Phase 2 & 3 - Viral Extensions

**Status:** Pending RA Completion
**Goal:** Apply the RA methodology to Viral Evasion (HIV) and Viral Attack (SARS-CoV-2).

---

## Phase 2: HIV (The Inverse Goldilocks)

**Hypothesis:** HIV uses glycans to _prevent_ Goldilocks shifts. Removing them exposes the immunogenic 15-30% shift.

### Plan

1.  **Analyze AlphaFold predictions**: We have pre-computed folds for N58, N103, N429 (deglycosylated). We need to correlate pLDDT drops with geometric shifts.
2.  **Extend PTM types**: Test if HIV uses other modifications (phosphorylation mimics?) to masquerade as host.
3.  **Output**: `hiv_ptm_ground_truth.json`.

---

## Phase 3: SARS-CoV-2 (The Asymmetric Handshake)

**Hypothesis:** The virus evolves "Asymmetric Handshakes" where a mutation helps viral binding but hurts host binding (or vice versa).

### Plan

1.  **Full Spike Sweep**: Extend analysis from just RBD to the NTD and S2 regions.
2.  **Glycan Analysis**: Test N331 and N343 for "Sentinel" properties.
3.  **AlphaFold Validation**: Analyze the `rbd_ace2` complex stability for S439D and S440D mutations.
4.  **Output**: `sars_ptm_ground_truth.json`.
