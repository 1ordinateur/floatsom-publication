#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
caller_pwd="$(pwd)"

resolve_input_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  elif [[ -e "${caller_pwd}/${path}" ]]; then
    printf '%s/%s\n' "${caller_pwd}" "${path}"
  elif [[ -e "${script_dir}/${path}" ]]; then
    printf '%s/%s\n' "${script_dir}" "${path}"
  else
    printf '%s/%s\n' "${caller_pwd}" "${path}"
  fi
}

resolve_output_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${caller_pwd}" "${path}"
  fi
}

convert_path_for_pandoc() {
  local path="$1"
  if [[ "${pandoc_bin}" == *.exe ]]; then
    wslpath -w "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

pandoc_bin="${PANDOC_BIN:-}"
if [[ -z "${pandoc_bin}" ]]; then
  if command -v pandoc >/dev/null 2>&1; then
    pandoc_bin="pandoc"
  elif command -v pandoc.exe >/dev/null 2>&1; then
    pandoc_bin="pandoc.exe"
  else
    echo "pandoc is required but was not found on PATH" >&2
    exit 1
  fi
fi

format="${1:-html}"
if [[ $# -ge 2 ]]; then
  source_file="$(resolve_input_path "$2")"
else
  source_file="${script_dir}/manuscript.md"
fi
output_file="${3:-}"
if [[ -n "${output_file}" ]]; then
  output_file="$(resolve_output_path "${output_file}")"
fi

if [[ "${source_file}" = /* ]]; then
  source_file="$(convert_path_for_pandoc "${source_file}")"
fi
if [[ -n "${output_file}" && "${output_file}" = /* ]]; then
  output_file="$(convert_path_for_pandoc "${output_file}")"
fi

cd "${script_dir}"

case "${format}" in
  html)
    output_file="${output_file:-manuscript.html}"
    "${pandoc_bin}" --defaults pandoc-paper.html.defaults.yaml "${source_file}" --output "${output_file}"
    ;;
  latex)
    output_file="${output_file:-manuscript.tex}"
    "${pandoc_bin}" --defaults pandoc-paper.latex.defaults.yaml "${source_file}" --output "${output_file}"
    ;;
  pdf)
    output_file="${output_file:-manuscript.pdf}"
    "${pandoc_bin}" --defaults pandoc-paper.latex.defaults.yaml "${source_file}" --output "${output_file}"
    ;;
  *)
    echo "Usage: $0 [html|latex|pdf] [source.md] [output]" >&2
    exit 1
    ;;
esac
