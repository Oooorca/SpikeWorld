# SpikeWorld full preprint

`main.tex` is the root source. `sections/` contains the main text and appendix.
`figures/` contains vector plots, numerical tables and editable draw.io/SVG diagrams.

Run these commands in this directory with a TeX distribution installed:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Supplied figures are sufficient: no Python or experiment inputs are needed to
compile. Standard TeX packages are required. The optional Windows helper
`scripts/build_paper.ps1` builds in `build/` and copies the PDF to `main.pdf`.
Figures 1 and 2 are native editable draw.io documents, exported with draw.io
Desktop 32.6.10. Text uses Arial; formula cells use draw.io mathematical
typesetting (enabled in the documents). Their principal connectors are attached
to the corresponding shapes. For example:

```bash
draw.io --export --format pdf --crop --border 12 --theme light --output figures/fig1_workflow.pdf figures/fig1_workflow.drawio
draw.io --export --format pdf --crop --border 12 --theme light --output figures/fig2_targets.pdf figures/fig2_targets.drawio
```

Figures 3 and 4 and all numerical tables can be regenerated from the recorded
results with `python tools/build_paper_assets.py` from the repository root
(Matplotlib required). This script leaves the draw.io diagrams and their
exports unchanged. The figures and tables supplied here are sufficient for
compiling this standalone paper source.

The source ZIP contains publication material only. Internal reviews and local
release records are kept outside the upload source.
