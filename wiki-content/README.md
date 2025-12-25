# Wiki Content

This folder contains the content for the GitHub Wiki.

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
git commit -m "Add wiki documentation"
git push
```

### Option 2: Use Existing Clone

If you already have the wiki cloned at `ternary-vaes-wiki`:

```bash
cd ternary-vaes-wiki
git push -u origin master
```

## Wiki Pages

| File | Title |
|------|-------|
| `Home.md` | Main landing page |
| `Getting-Started.md` | Installation guide |
| `Architecture.md` | System design |
| `Models.md` | Model documentation |
| `Geometry.md` | Hyperbolic operations |
| `Loss-Functions.md` | Loss system |
| `Configuration.md` | Config system |
| `Training.md` | Training guide |
| `API-Reference.md` | Module reference |
| `_Sidebar.md` | Navigation sidebar |
