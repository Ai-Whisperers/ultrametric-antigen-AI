# DISCLOSURE MATRIX

**Classification**: CONFIDENTIAL
**Purpose**: Precise control of what can be disclosed at each tier

---

## TIER SUMMARY

| Tier | Audience | Trust Level | Documentation |
|:----:|:---------|:-----------:|:--------------|
| **T3** | Internal only | Full | This folder |
| **T2** | Public/Conferences | None | pitch_tier2/ |
| **T1** | Partners/NDA | Medium | pitch/ |
| **T0** | Deep partners/Acquirers | High | Verbal + NDA |

---

## DETAILED DISCLOSURE RULES

### TIER 2: Public Disclosure

**ALLOWED**:
| Category | Specific Items |
|:---------|:---------------|
| Results | Prediction accuracy, correlation values, validation |
| Claims | "We identified X", "Our method predicts Y" |
| Comparisons | "Outperforms baseline by Z%" |
| Terminology | "Geometric", "proprietary", "novel encoding" |
| Data | Sanitized JSONs, FASTA sequences for validation |
| Validation | Protocols using public tools (AF3, LANL, HIVdb) |

**FORBIDDEN**:
| Category | Specific Items |
|:---------|:---------------|
| Mathematics | Any mention of p-adic, ternary, hyperbolic |
| Architecture | Model diagrams, layer specifications |
| Training | Hyperparameters, schedules, data curation |
| Code | Any source code, pseudocode |
| Connections | Relationships between components |

---

### TIER 1: Partner Disclosure (NDA Required)

**ADDITIONALLY ALLOWED** (beyond T2):
| Category | Specific Items |
|:---------|:---------------|
| Methodology | "Non-Euclidean", "geometry-based" |
| Architecture | "VAE-based", "encoder-decoder" |
| Validation | Step-by-step protocols |
| Data | Full sanitized datasets |
| Terminology | "Manifold", "embedding space" |

**STILL FORBIDDEN**:
| Category | Specific Items |
|:---------|:---------------|
| Core math | P-adic, ternary specifics |
| Architecture details | Layer types, activation functions |
| The synergy | How components connect |
| Training details | Learning rates, batch sizes |
| Loss functions | Specific formulations |

---

### TIER 0: Deep Partner Disclosure (Enhanced NDA + Term Sheet)

**ADDITIONALLY ALLOWED** (beyond T1):
| Category | Specific Items |
|:---------|:---------------|
| Concepts | "Hyperbolic geometry", "non-Euclidean VAE" |
| High-level arch | "Encoder → Latent → Decoder" flow |
| General approach | "We encode codons geometrically" |
| Prior art | What exists, how we differ |

**STILL FORBIDDEN** (Until Acquisition/Deep Partnership):
| Category | Specific Items |
|:---------|:---------------|
| P-adic specifics | The prime, the encoding scheme |
| Ternary specifics | The algebra, the representation |
| The synergy | Why p-adic + ternary + hyperbolic |
| Implementation | Code, exact formulas |
| Training secrets | All specifics |

---

### TIER -1: Post-Acquisition / Deep Integration

**Full disclosure** including:
- Complete codebase
- Training procedures
- All mathematical details
- The core synergy explanation
- Hyperparameters and configurations

**Only after**:
- Acquisition complete, OR
- Exclusive licensing with strong protections, OR
- Deep R&D partnership with joint IP

---

## DISCLOSURE DECISION FLOWCHART

```
                    ┌─────────────────┐
                    │ Want to share X │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Is X on         │
               ┌────│ FORBIDDEN_TERMS?│────┐
               │YES └─────────────────┘ NO │
               │                           │
        ┌──────▼──────┐              ┌─────▼─────┐
        │   STOP!     │              │ Is X in   │
        │ Never share │              │ T2 ALLOWED?│
        └─────────────┘              └─────┬─────┘
                                      YES  │  NO
                                     ┌─────┴─────┐
                              ┌──────▼───┐  ┌────▼────┐
                              │Share in  │  │Is there │
                              │T2 context│  │an NDA?  │
                              └──────────┘  └────┬────┘
                                            YES  │  NO
                                           ┌─────┴─────┐
                                    ┌──────▼───┐  ┌────▼────┐
                                    │Is X in   │  │ Request │
                                    │T1 ALLOWED?│  │   NDA   │
                                    └─────┬────┘  └─────────┘
                                     YES  │  NO
                                    ┌─────┴─────┐
                             ┌──────▼───┐  ┌────▼────┐
                             │Share in  │  │Is this  │
                             │T1 context│  │T0 partner?
                             └──────────┘  └────┬────┘
                                           YES  │  NO
                                          ┌─────┴─────┐
                                   ┌──────▼───┐  ┌────▼────┐
                                   │Share with│  │  STOP!  │
                                   │T0 limits │  │Escalate │
                                   └──────────┘  └─────────┘
```

---

## AUDIENCE-SPECIFIC GUIDELINES

### Academic Conferences
- Tier: T2 only
- Focus: Results, validation, "black box" demos
- Deflect: "Methodology in preparation / under review"
- Never: Methodology talks or posters

### Industry Conferences
- Tier: T2, maybe T1 for known contacts
- Focus: Use cases, ROI, partnerships
- NDA: Require before T1 discussions
- Demo: Results only, not method

### Potential Partners
- Start: T2 (public pitch)
- Progress: T1 after NDA
- Deepen: T0 after term sheet / serious discussions
- Full: Only post-deal

### Investors
- Tier: T1 with standard NDA
- Focus: Market, team, results, defensibility
- Technical: "Proprietary geometric approach"
- Due diligence: T0 for serious investors

### Acquirers
- Initial: T1
- Technical DD: T0
- Full disclosure: Only in clean room after LOI

### Academics / Collaborators
- Default: T2
- Collaboration: T1 with academic NDA
- Joint IP: Careful T0 with strong IP agreement

### Press / Media
- Tier: T2 only, strictly
- Prepared statements only
- No live technical Q&A

---

## DISCLOSURE LOG

Maintain a log of all disclosures:

| Date | Recipient | Tier | What Disclosed | By Whom | Notes |
|:-----|:----------|:----:|:---------------|:--------|:------|
| YYYY-MM-DD | Company/Person | T# | Brief description | Name | Context |

**Review**: Monthly review of disclosure log
**Escalate**: Any unplanned T0+ disclosure

---

## EMERGENCY PROCEDURES

### Accidental Disclosure
1. Stop immediately
2. Do not try to "unsay" it
3. Document: what, to whom, context
4. Report to IP lead within 1 hour
5. Assess: Did they understand?
6. Mitigate: NDA if possible, monitor

### Suspected Leak
1. Document suspicion
2. Report to IP lead immediately
3. Investigate source
4. Assess competitive impact
5. Consider legal options
6. Accelerate filing if needed

### Competitor Publication
1. Obtain and analyze publication
2. Assess overlap with our IP
3. Consult IP counsel
4. Adjust strategy if needed
5. Consider defensive actions

---

*This matrix is the authoritative guide for disclosure decisions.*
