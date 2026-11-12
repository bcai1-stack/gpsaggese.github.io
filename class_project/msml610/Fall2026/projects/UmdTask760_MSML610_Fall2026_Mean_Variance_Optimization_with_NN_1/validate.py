"""Run unit/integration tests and optionally execute both notebooks from scratch."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root, text=True, capture_output=True,
    )
    output = result.stdout + result.stderr
    print(output)
    (root / "docs/test_results.txt").write_text(output)
    if result.returncode:
        raise SystemExit(result.returncode)
    executed = []
    if args.notebooks:
        import nbformat
        from nbclient import NotebookClient
        for name in ("mvo.API.ipynb", "mvo.example.ipynb"):
            print(f"Executing {name} in a fresh kernel", flush=True)
            path = root / name
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(notebook, timeout=1800, resources={
                "metadata": {"path": str(root)}
            }).execute()
            nbformat.write(notebook, path)
            executed.append(name)
    packages = ["numpy", "pandas", "scipy", "torch", "cvxpy", "cvxpylayers",
                "scikit-learn", "diffcp", "clarabel", "scs", "matplotlib"]
    sources = sorted(root.glob("mvo/*.py")) + sorted(root.glob("tests/*.py"))
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "unit_and_integration_tests": "passed", "executed_notebooks": executed,
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
    (root / "docs/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Project validation passed.")


if __name__ == "__main__":
    main()
