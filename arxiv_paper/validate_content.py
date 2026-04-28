#!/usr/bin/env python3
"""Validate that arxiv_paper/main.tex matches paper/manuscript_arxiv.md.

This intentionally compares against the canonical structural conversion produced
by build_from_markdown.py. It catches prose, number, citation-key, caption, table,
algorithm, and ordering drift while allowing only the markup transformations
encoded in the generator.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import build_from_markdown


ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(__file__).resolve().parent / "main.tex"


def main() -> int:
    expected = build_from_markdown.build_tex()
    actual = MAIN.read_text(encoding="utf-8")
    if actual == expected:
        print("Content validation passed: arxiv_paper/main.tex matches paper/manuscript_arxiv.md.")
        return 0

    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected-from-paper/manuscript_arxiv.md",
        tofile="arxiv_paper/main.tex",
        lineterm="",
    )
    print("Content validation failed: arxiv_paper/main.tex has drifted from paper/manuscript_arxiv.md.")
    print("\n".join(list(diff)[:240]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
