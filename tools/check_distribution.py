"""Check built wheel contents against Python sources and bundled profiles."""
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks"
NAMESPACE = "floatsom_benchmarks"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/check_distribution.py path/to/package.whl")
    wheel = Path(sys.argv[1])
    files = [p for p in SOURCE.rglob("*") if p.is_file() and p.suffix in {".py", ".json"}]
    expected = {NAMESPACE + "/" + p.relative_to(SOURCE).as_posix() for p in files}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = expected - names
        if missing:
            raise SystemExit("Missing distribution files:\n" + "\n".join(sorted(missing)))
        leaked = [n for n in names if "/tests/" in n or n.endswith((".pdf", ".aux", ".log"))
                  or (n.endswith(".py") and not n.startswith(NAMESPACE + "/"))]
        if leaked:
            raise SystemExit("Unexpected non-runtime files in wheel:\n" + "\n".join(leaked))
    print(f"Verified {len(expected)} source and runtime-data files in {wheel.name}")


if __name__ == "__main__":
    main()
