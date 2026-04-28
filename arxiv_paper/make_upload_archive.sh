#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive="${script_dir}/arxiv_paper_source.zip"

cd "${script_dir}"
rm -f "${archive}"

zip -r "${archive}" \
  main.tex \
  main.bbl \
  main.bib \
  assets \
  -x "main.aux" "main.log" "main.out" "main.fls" "main.fdb_latexmk" "main.synctex.gz" "main.pdf" "arxiv_paper_source.zip"

echo "Wrote ${archive}"
