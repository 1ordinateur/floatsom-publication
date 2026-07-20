# Tracked-changes TMLR manuscript

Build both versions from the parent `tmlr_paper` directory. This working
directory is required because the tracked source reuses the official TMLR
files, bibliography, figures, and supplementary-table source in that parent
folder. Do not run `latexmk` from inside `tmlr_paper/tracked`.

Clean manuscript:

```sh
cd tmlr_paper
latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build main.tex
cp build/main.pdf main.pdf
```

Tracked manuscript:

```sh
cd tmlr_paper
latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tracked/build tracked/tracked_changes.tex
cp tracked/build/tracked_changes.pdf tracked/tracked_changes.pdf
```

The manuscript remains in anonymous TMLR review mode. Build artifacts belong in
`tracked/build`; the reviewed PDF is stored as `tracked/tracked_changes.pdf`.
