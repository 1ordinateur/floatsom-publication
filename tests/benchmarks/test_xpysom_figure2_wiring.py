"""The accepted manuscript must reference existing publication assets."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "tmlr_paper"


def test_manuscript_figure_files_exist():
    text = (PAPER / "main.tex").read_text()
    paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    assert len(paths) == 19
    for name in paths:
        path = PAPER / name
        assert path.is_file(), name
        svg = path.with_suffix(".svg")
        assert svg.is_file(), svg
        ET.parse(svg)


def test_manuscript_supplement_is_self_contained():
    text = (PAPER / "main.tex").read_text()
    assert r"\input{assets/tables/supplementary_tables.tex}" in text
    assert (PAPER / "assets/tables/supplementary_tables.tex").is_file()
    assert "assets_manual/" not in text
