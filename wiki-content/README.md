# Wiki Content

This folder contains comprehensive documentation for the GitHub Wiki.

**Total Pages**: 19

## Setup Instructions

To publish this content to the GitHub Wiki:

### Option 1: Manual Initialization (One-Time)

1. Go to https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/wiki
2. Click "Create the first page"
3. Save a blank page or copy content from `Home.md`
4. Then clone and push:

```bash
# Clone the wiki repo
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.wiki.git
cd ternary-vaes-bioinformatics.wiki

# Copy content
cp ../ternary-vaes-bioinformatics/wiki-content/*.md .

# Commit and push
git add .
git commit -m "Add comprehensive wiki documentation"
git push
```

### Option 2: Use Existing Clone

If you already have the wiki cloned at `ternary-vaes-wiki`:

```bash
cd ternary-vaes-wiki
cp ../ternary-vaes-bioinformatics/wiki-content/*.md .
git add .
git commit -m "Update wiki documentation"
git push
```

## Wiki Pages

### Getting Started (3 pages)
| File | Description |
|------|-------------|
| `Home.md` | Main landing page with overview and quick start |
| `Installation.md` | Platform-specific installation instructions |
| `Quick-Start.md` | 5-minute guide to running first experiment |

### Core Concepts (5 pages)
| File | Description |
|------|-------------|
| `Architecture.md` | System design and component overview |
| `Models.md` | TernaryVAE, SwarmVAE documentation |
| `Geometry.md` | Hyperbolic geometry and p-adic numbers |
| `Loss-Functions.md` | Loss registry system |
| `Biological-Context.md` | Why hyperbolic geometry for biology |

### Usage (3 pages)
| File | Description |
|------|-------------|
| `Configuration.md` | Config system and parameters |
| `Training.md` | Training workflows and callbacks |
| `Evaluation.md` | Metrics and benchmarks |

### Reference (3 pages)
| File | Description |
|------|-------------|
| `API-Reference.md` | Module and class reference |
| `Constants.md` | All configuration constants |
| `Glossary.md` | Term definitions A-Z |

### Help (2 pages)
| File | Description |
|------|-------------|
| `FAQ.md` | Frequently asked questions |
| `Troubleshooting.md` | Common issues and solutions |

### Contributing (2 pages)
| File | Description |
|------|-------------|
| `Contributing-Guide.md` | How to contribute |
| `Testing.md` | Testing guide |

### Learning (1 page)
| File | Description |
|------|-------------|
| `Tutorials.md` | Step-by-step tutorials |

### Navigation
| File | Description |
|------|-------------|
| `_Sidebar.md` | Wiki navigation sidebar |

## Page Statistics

- **Total pages**: 19
- **Total lines**: ~5,500
- **Topics covered**: Installation, architecture, geometry, losses, training, evaluation, troubleshooting, tutorials

## Updates

Last updated: 2025-12-25

Changes in this update:
- Added 10 new pages (Installation, Quick-Start, FAQ, Troubleshooting, Glossary, Biological-Context, Tutorials, Evaluation, Testing, Contributing-Guide, Constants)
- Enhanced Home page with badges, diagrams, quick start
- Fixed sidebar navigation
- Added comprehensive conceptual explanations
- Added troubleshooting for common issues
