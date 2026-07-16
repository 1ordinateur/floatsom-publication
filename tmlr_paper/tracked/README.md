# Tracked-changes TMLR manuscript

Build this version from the parent `tmlr_paper` directory so it can reuse the
official TMLR files, bibliography, figures, and supplementary-table source:

```sh
cd tmlr_paper
latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tracked/build tracked/tracked_changes.tex
```

The manuscript remains in anonymous TMLR review mode. Build artifacts belong in
`tracked/build`; the reviewed PDF is stored as `tracked/tracked_changes.pdf`.
