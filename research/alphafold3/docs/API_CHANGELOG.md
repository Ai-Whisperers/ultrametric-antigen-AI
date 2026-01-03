# AlphaFold API Changelog

Track breaking changes, deprecations, and version updates.

---

## 2026-06-25 (Upcoming)

**Type:** Breaking Change - Field Renaming

| Old Field | New Field |
|-----------|-----------|
| `entryId` | `modelEntityId` |
| `uniprotStart` | `sequenceStart` |
| `uniprotEnd` | `sequenceEnd` |
| `uniprotSequence` | `sequence` |
| `isReviewed` | `isUniProtReviewed` |
| `isReferenceProteome` | `isUniProtReferenceProteome` |

**Removed:** `paeImageUrl` (PAE images no longer provided)

**Action Required:** Update code to use new field names before sunset.

---

## 2025-09-25

**Type:** Dual Support Period Begins

Both old and new field names supported for 9 months.

---

## 2024-10

**Type:** Database Update - v4 Models

AlphaFold DB updated to 214M+ structures aligned with UniProt 2024_05.

---

## Monitoring

Check for updates: https://www.ebi.ac.uk/pdbe/news/

Last verified: 2026-01-03
