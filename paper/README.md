# Floatsom Paper Workspace

This directory contains the markdown-first source for the Floatsom paper.

## Structure

- `manuscript.md`: Current non-colours manuscript (primary draft).
- `manuscript_archival.md`: Deprecated (archival only; superseded by the non-colours manuscript).
- `outline.md`: Working structure and section plan.
- `assets/figures/`: Figure files and related notes.
- `assets/tables/`: Table sources and exports.
- `pandoc-paper.css`: HTML export styling, including narrower side margins and wider figures.
- `pandoc-paper.html.defaults.yaml`: Pandoc defaults for HTML export using the paper stylesheet.
- `pandoc-paper.latex.defaults.yaml`: Pandoc defaults for LaTeX/PDF export with matching base geometry.
- `build_pandoc_paper.sh`: Small wrapper for consistent Pandoc export commands.

## Writing approach

- Keep the paper in markdown.
- Add figure/table references directly in `manuscript.md`.
- Keep section-level planning in `outline.md`.

## Pandoc export

- HTML export uses `pandoc-paper.css`, which narrows the text-side margins and lets figures extend into the left and right margins by half a margin on each side.
- LaTeX/PDF export uses separate Pandoc defaults and a LaTeX header file. CSS is not reused by LaTeX, but the export structure is set up so both backends can follow the same layout intent.
- Example commands:
- `./build_pandoc_paper.sh html manuscript.md manuscript.html`
- `./build_pandoc_paper.sh html manuscript_self_contained.md manuscript_self_contained.html`
- `./build_pandoc_paper.sh latex manuscript.md manuscript.tex`
- `./build_pandoc_paper.sh pdf manuscript.md manuscript.pdf`
