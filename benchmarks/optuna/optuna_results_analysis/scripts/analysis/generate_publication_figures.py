#!/usr/bin/env python3
"""
Generate publication-oriented paired comparison figures using seaborn.

This script delegates to the refactored publication_figures package while
preserving the historical script path and exported helper surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from scripts.analysis.publication_figures import cli as _cli
from scripts.analysis.publication_figures import composition as _composition
from scripts.analysis.publication_figures import constants as _constants
from scripts.analysis.publication_figures import helpers as _helpers
from scripts.analysis.publication_figures import plots as _plots
from scripts.analysis.publication_figures import workflows as _workflows


def _export_public_api() -> None:
    exported_names = []
    for module in (_constants, _helpers, _plots, _composition, _workflows, _cli):
        module_names = getattr(module, "__all__", ())
        for name in module_names:
            if name.startswith("__"):
                continue
            globals()[name] = getattr(module, name)
            exported_names.append(name)
    globals()["__all__"] = sorted(set(exported_names))


_export_public_api()


if __name__ == "__main__":
    raise SystemExit(main())
