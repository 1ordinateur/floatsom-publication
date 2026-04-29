#!/usr/bin/env python3
"""Build arXiv-ready LaTeX from paper/manuscript_arxiv.md.

The markdown manuscript is the content source of truth. This script performs only
structural conversion: front matter, headings, citations, equations, figures,
algorithms, and markdown tables become LaTeX markup.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "manuscript_arxiv.md"
OUTPUT = Path(__file__).resolve().parent / "main.tex"


HEADING_COMMANDS = {
    2: "section",
    3: "subsection",
    4: "subsubsection",
}


def strip_number(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()


def escape_latex_text(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    output: list[str] = []
    for index, char in enumerate(text):
        if char in replacements and (index == 0 or text[index - 1] != "\\"):
            output.append(replacements[char])
        else:
            output.append(char)
    return "".join(output)


def split_math(text: str) -> list[tuple[bool, str]]:
    parts: list[tuple[bool, str]] = []
    current: list[str] = []
    in_math = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "$" and (i == 0 or text[i - 1] != "\\"):
            if current:
                parts.append((in_math, "".join(current)))
                current = []
            current.append(char)
            i += 1
            while i < len(text):
                current.append(text[i])
                if text[i] == "$" and text[i - 1] != "\\":
                    i += 1
                    break
                i += 1
            parts.append((True, "".join(current)))
            current = []
            continue
        current.append(char)
        i += 1
    if current:
        parts.append((in_math, "".join(current)))
    return parts


def protect_commands(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def stash(value: str) -> str:
        key = f"@@PROTECTED{len(protected)}@@"
        protected[key] = value
        return key

    text = re.sub(
        r"\[@([A-Za-z0-9:_-]+)((?:;\s*@?[A-Za-z0-9:_-]+)*)\]",
        lambda match: stash(
            r"\citep{"
            + ",".join(
                [match.group(1)]
                + [item.strip().lstrip("@") for item in match.group(2).split(";") if item.strip()]
            )
            + "}"
        ),
        text,
    )
    def convert_code(match: re.Match[str]) -> str:
        value = match.group(1)
        if "/" in value:
            return stash(r"\path{" + value + "}")
        return stash(r"\texttt{" + escape_latex_text(value) + "}")

    text = re.sub(r"`([^`]+)`", convert_code, text)
    return text, protected


def restore_protected(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def convert_inline(text: str) -> str:
    outer_protected: dict[str, str] = {}

    def stash_outer(value: str) -> str:
        key = f"@@OUTER{len(outer_protected)}@@"
        outer_protected[key] = value
        return key

    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda match: stash_outer(r"\textbf{" + convert_inline(match.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"\*([^*\n]+)\*",
        lambda match: stash_outer(r"\emph{" + convert_inline(match.group(1)) + "}"),
        text,
    )
    pieces: list[str] = []
    for is_math, part in split_math(text):
        if is_math:
            pieces.append(part)
            continue
        protected_text, protected = protect_commands(part)
        escaped = escape_latex_text(protected_text)
        pieces.append(restore_protected(escaped, protected))
    return restore_protected("".join(pieces), outer_protected)


def extract_section(lines: list[str], heading: str) -> list[str]:
    start = lines.index(heading) + 1
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    end = start
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return lines[start:end]


def convert_authors(lines: list[str]) -> str:
    return r"""Tony Xu$^{1}$, Sarah Klamt$^{2}$, Katherine Turner$^{3}$\\
Anne Brüstle$^{1}$, Givanna Putri$^{4}$, Felix Marsh-Wakefield$^{5,6}$\\[0.5em]
\small $^{1}$ John Curtin School of Medical Research, Australian National University\\
\small $^{2}$ Department of Information Technology and Electrical Engineering, ETH Zurich\\
\small $^{3}$ Mathematical Data Science Centre, Australian National University\\
\small $^{4}$ Walter and Eliza Hall Institute\\
\small $^{5}$ Centenary Institute of Cancer Medicine and Cell Biology\\
\small $^{6}$ University of Sydney\\[0.5em]
\small Corresponding author: \texttt{tony.xu@anu.edu.au}"""


def figure_width(image_line: str) -> str:
    if "width=50%" in image_line or ".half-width" in image_line:
        return "0.5\\linewidth"
    return "\\linewidth"


def image_pdf_path(image_line: str) -> str:
    match = re.search(r"\(([^)]+)\)", image_line)
    if not match:
        raise ValueError(f"Cannot parse image line: {image_line}")
    source = Path(match.group(1))
    return f"assets/figures/{source.stem}.pdf"


def convert_caption(caption_line: str, supplementary: bool = False) -> str:
    caption = caption_line.strip()
    if caption.startswith("*") and caption.endswith("*"):
        caption = caption[1:-1]
    if supplementary:
        return convert_inline(caption)
    caption = re.sub(r"^Figure\s+\d+\.\s*", "", caption)
    return convert_inline(caption)


def convert_figure(image_line: str, caption_line: str) -> list[str]:
    supplementary = "Supplementary Figure" in image_line or "Supplementary Figure" in caption_line
    caption_command = "caption*" if supplementary else "caption"
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width={figure_width(image_line)}]{{{image_pdf_path(image_line)}}}",
        rf"\{caption_command}{{{convert_caption(caption_line, supplementary=supplementary)}}}",
        r"\end{figure}",
        "",
    ]


def convert_algorithm(code_lines: list[str], caption_line: str) -> list[str]:
    caption = caption_line.strip()
    if caption.startswith("*") and caption.endswith("*"):
        caption = caption[1:-1]
    caption = re.sub(r"^Algorithm\s+\d+\.\s*", "", caption)
    output = [
        r"\begin{algorithm}[H]",
        rf"\caption{{{convert_inline(caption)}}}",
        r"\begin{algorithmic}[1]",
    ]
    for line in code_lines:
        line_text = re.sub(r"^\s*\d+:\s*", "", line)
        if line_text.startswith("if ") and line_text.endswith(" then"):
            condition = line_text.removeprefix("if ").removesuffix(" then").strip()
            output.append(rf"\If{{{convert_inline(condition)}}}")
        elif line_text == "end if":
            output.append(r"\EndIf")
        else:
            output.append(rf"\State {convert_inline(line_text)}")
    output.extend([r"\end{algorithmic}", r"\end{algorithm}", ""])
    return output


def convert_table(caption_line: str, table_lines: list[str]) -> list[str]:
    caption = caption_line.strip()
    caption = re.sub(r"^\*\*", "", caption)
    caption = re.sub(r"\*\*$", "", caption)
    rows = []
    for raw in table_lines:
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    column_count = len(rows[0])
    colspec = " ".join([r">{\raggedright\arraybackslash}X"] * column_count)
    output = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption*{{{convert_inline(caption)}}}",
        r"\small",
        rf"\begin{{tabularx}}{{\linewidth}}{{{colspec}}}",
        r"\toprule",
        " & ".join(convert_inline(cell) for cell in rows[0]) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        output.append(" & ".join(convert_inline(cell) for cell in row) + r" \\")
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return output


def convert_body(lines: list[str]) -> str:
    output: list[str] = []
    i = 0
    pending_table_caption: str | None = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "" or stripped in {"::: {#refs}", ":::"}:
            i += 1
            continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
                pending_table_caption = None
            output.append("% " + stripped[4:-3].strip())
            i += 1
            continue

        if stripped.startswith("```"):
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
                pending_table_caption = None
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines) and lines[i].strip().startswith("*Algorithm"):
                output.extend(convert_algorithm(code_lines, lines[i].strip()))
                i += 1
            else:
                raise ValueError("Unexpected code block without algorithm caption")
            continue

        if stripped.startswith("!["):
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
                pending_table_caption = None
            image_line = stripped
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("*"):
                raise ValueError(f"Image missing caption: {image_line}")
            output.extend(convert_figure(image_line, lines[i].strip()))
            i += 1
            continue

        if stripped.startswith("$$"):
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
                pending_table_caption = None
            equation_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                equation_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            output.append(r"\[")
            output.extend(equation_lines)
            output.append(r"\]")
            output.append("")
            continue

        heading_match = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading_match:
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
                pending_table_caption = None
            level = len(heading_match.group(1))
            title = strip_number(heading_match.group(2))
            if title == "References":
                output.append(r"\section{References}")
                output.append(r"\renewcommand{\bibsection}{}")
                output.append(r"\bibliographystyle{plainnat}")
                output.append(r"\bibliography{main}")
                output.append("")
            else:
                output.append(rf"\{HEADING_COMMANDS[level]}{{{convert_inline(title)}}}")
                output.append("")
            i += 1
            continue

        if stripped.startswith("**Supplementary Table"):
            if pending_table_caption:
                output.append(convert_inline(pending_table_caption))
                output.append("")
            pending_table_caption = stripped
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if pending_table_caption:
                output.extend(convert_table(pending_table_caption, table_lines))
                pending_table_caption = None
            else:
                raise ValueError("Markdown table without supplementary table caption")
            continue

        paragraph_lines = [stripped]
        if pending_table_caption:
            paragraph_lines.insert(0, pending_table_caption)
            pending_table_caption = None
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if (
                candidate == ""
                or candidate.startswith("#")
                or candidate.startswith("![")
                or candidate.startswith("$$")
                or candidate.startswith("```")
                or candidate.startswith("|")
                or candidate.startswith("<!--")
                or candidate in {"::: {#refs}", ":::"}
                or candidate.startswith("**Supplementary Table")
            ):
                break
            paragraph_lines.append(candidate)
            i += 1
        output.append(convert_inline(" ".join(paragraph_lines)))
        output.append("")

    return "\n".join(output).rstrip() + "\n"


def build_tex() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title = extract_section(lines, "## Title")[0].strip()
    authors = extract_section(lines, "## Authors")
    abstract = " ".join(line.strip() for line in extract_section(lines, "## Abstract") if line.strip())
    body_start = next(i for i, line in enumerate(lines) if line.startswith("## 1. "))
    body = convert_body(lines[body_start:])
    return rf"""\documentclass[11pt]{{article}}

\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage[round,authoryear]{{natbib}}
\usepackage[labelsep=period,font=small,labelfont=bf]{{caption}}
\usepackage{{url}}
\usepackage{{hyperref}}

\title{{{convert_inline(title)}}}
\author{{{convert_authors(authors)}}}
\date{{}}

\begin{{document}}

\maketitle

\begin{{abstract}}
{convert_inline(abstract)}
\end{{abstract}}

{body}
\end{{document}}
"""


def main() -> None:
    OUTPUT.write_text(build_tex(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
