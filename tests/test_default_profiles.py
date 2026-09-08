"""Named profiles must resolve without relying on a checkout or working directory."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from floatsom.defaults import profile_path
from floatsom_benchmarks.optuna.config.default_profiles import LEGACY_NAMES, resolve_defaults_path


@pytest.mark.parametrize("old,name", LEGACY_NAMES.items())
def test_named_and_legacy_profiles_resolve_from_another_directory(tmp_path, monkeypatch, old, name):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_defaults_path(name)
    assert resolved.is_file()
    assert resolve_defaults_path(old) == resolved
    if name != "xpysom-untuned":
        assert resolved == profile_path(name)
    assert set(json.loads(resolved.read_text())) == {"full", "random"}


def test_custom_file_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "publication"
    custom.write_text('{"custom": true}')
    assert resolve_defaults_path("publication") == custom


def test_ray_worker_path_keeps_library_and_benchmarks_separate(tmp_path):
    # Load the helper alone so this CPU test does not import CUDA training modules.
    import ast
    source = Path(__file__).resolve().parents[1] / "benchmarks/optuna/run_matched_radius_sweep.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ensure_ray_worker_import_path")
    code = "import os, sys, tempfile\nfrom pathlib import Path\n" + ast.unparse(function)
    code += "\nroot = _ensure_ray_worker_import_path(Path(sys.argv[1]))\n"
    code += "subprocess.run([sys.executable, '-c', 'import floatsom.defaults, floatsom_benchmarks; print(floatsom.defaults.profile_path())'], check=True, cwd=root)\n"
    result = subprocess.run([sys.executable, "-c", "import subprocess\n" + code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "library.json" in result.stdout
