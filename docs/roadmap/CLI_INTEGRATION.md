# CLI Integration Roadmap

## Status: Pending
The current CLI (`hiv_analysis/cli.py`) acts as a placeholder. The core functionality exists in standalone scripts but is not yet piped through the main entry point.

## Tasks
- [ ] **Align Command**: Integrate `mafft_wrapper.py` to handle sequence alignment via `hiv-analysis align`.
- [ ] **Score Command**: Integrate `conservation_scorer.py` to calculate metrics via `hiv-analysis score`.
- [ ] **View Command**: Integrate `alignment_viewer.py` for HTML/Text visualization via `hiv-analysis view`.
- [ ] **Export Command**: Integrate `format_exporter.py` for multi-format output via `hiv-analysis export`.

## Implementation Notes
- Use `argparse` in `cli.py` to pass parameters (`--algorithm`, `--threads`, etc.) to the respective script classes.
- Ensure the `hiv-analysis setup` command properly initializes the working directories before other commands are run.
