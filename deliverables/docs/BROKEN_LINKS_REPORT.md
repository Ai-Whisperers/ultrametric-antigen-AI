# Broken Links Report

**Generated:** 2026-02-03
**Total:** 67 broken links

---

## Summary by Category

| Category | Count | Priority | Action |
|----------|------:|----------|--------|
| DELIVERABLES | 12 | HIGH | Fix paths |
| SRC | 3 | HIGH | Fix or remove |
| DOCS_CONTENT | 27 | MEDIUM | Tutorial stubs - remove or create |
| SPINOFF_TEMPLATE | 2 | LOW | Template placeholders (intentional) |
| RESEARCH_HIV | 18 | LOW | Planned docs never created |
| ARCHIVE | 5 | LOW | Skip or delete refs |

---

## Recommended Batches

### Batch 1: HIGH PRIORITY (15 links) - Path Fixes
Easy fixes - just update paths to correct locations.

**DELIVERABLES (12):**
- CLAUDE files have wrong paths when accessed directly (not via symlinks)
- DELIVERABLES_IMPROVEMENT_PLAN.md has old folder names

**SRC (3):**
- src/README.md references non-existent DOCUMENTATION/ folder

### Batch 2: MEDIUM PRIORITY (27 links) - Remove Stubs
Tutorial/guide references that were never created. Options:
- Remove the broken links
- Create placeholder files
- Mark as "Coming Soon"

**DOCS_CONTENT (27):**
- master_guide.md references 17 non-existent chapter files
- README files reference non-existent tutorials
- Stakeholder docs reference non-existent legal files

### Batch 3: LOW PRIORITY (25 links) - Defer or Archive
Planned documentation that was never written. These are in research/archive areas.

**SPINOFF_TEMPLATE (2):** Intentional placeholders in template
**RESEARCH_HIV (18):** Planned HIV documentation structure
**ARCHIVE (5):** Old archived docs with stale references

---

## Detailed Listings

### DELIVERABLES (12 links)

| File | Line | Broken Link |
|------|------|-------------|
| deliverables/docs/CLAUDE_LITE.md | 100 | `docs/BIOINFORMATICS_GUIDE.md` |
| deliverables/docs/CLAUDE_LITE.md | 101 | `docs/mathematical-foundations/` |
| deliverables/docs/DELIVERABLES_IMPROVEMENT_PLAN.md | 343 | `link` |
| deliverables/docs/DELIVERABLES_IMPROVEMENT_PLAN.md | 383 | `partners/hiv_research_package/notebooks/` |
| deliverables/docs/DELIVERABLES_IMPROVEMENT_PLAN.md | 384 | `partners/alejandra_rojas/notebooks/` |
| deliverables/docs/DELIVERABLES_IMPROVEMENT_PLAN.md | 385 | `partners/carlos_brizuela/notebooks/` |
| deliverables/docs/DELIVERABLES_IMPROVEMENT_PLAN.md | 386 | `partners/jose_colbes/notebooks/` |
| deliverables/docs/CLAUDE_DEV.md | 333 | `docs/mathematical-foundations/` |
| deliverables/docs/CLAUDE_DEV.md | 334 | `docs/mathematical-foundations/V5_12_2_audit/` |
| deliverables/docs/CLAUDE_DEV.md | 338 | `docs/mathematical-foundations/` |
| deliverables/docs/CLAUDE_DEV.md | 339 | `docs/mathematical-foundations/archive/CLAUDE_ORIGINAL.md` |
| deliverables/docs/CLAUDE_BIO.md | 251 | `docs/mathematical-foundations/` |

### SRC (3 links)

| File | Line | Broken Link |
|------|------|-------------|
| src/README.md | 3 | `../DOCUMENTATION/` |
| src/README.md | 3 | `../DOCUMENTATION/.../ARCHITECTURE.md` |
| src/README.md | 3 | `../DOCUMENTATION/QUICK_START.md` |

### DOCS_CONTENT (27 links)

| File | Line | Broken Link |
|------|------|-------------|
| docs/content/research/README.md | 40 | `../../../results/research_discoveries/` |
| docs/content/research/README.md | 44 | `../../../results/clinical_applications/` |
| docs/content/getting-started/README.md | 92-95 | `tutorials/*.md` (4 files) |
| docs/content/getting-started/master_guide.md | 64-100 | Various chapter files (17 files) |
| docs/content/stakeholders/investors.md | 121 | `../../../../LEGAL_AND_IP/AUTHORS.md` |
| docs/content/research/hiv/README.md | 101-103 | Research paths (3 files) |

### SPINOFF_TEMPLATE (2 links)
Template placeholders - intentionally broken for users to fill in.

### RESEARCH_HIV (18 links)
Planned documentation structure that was never created.

### ARCHIVE (5 links)
Old archived documentation with stale references.

---

## Action Plan

1. **Batch 1:** Fix DELIVERABLES + SRC paths (~15 min)
2. **Batch 2:** Clean up DOCS_CONTENT stubs (~30 min)
3. **Batch 3:** Defer RESEARCH_HIV/ARCHIVE to future session
