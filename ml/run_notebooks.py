"""Execute every notebook in `notebooks/` in place, baking in fresh outputs.

Used locally (`python -m ml.run_notebooks`) and in CI to guarantee the
notebooks actually run against the current codebase rather than going
stale. Exits non-zero if any notebook raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install as install_kernelspec
from nbclient import NotebookClient

from ml.config import PROJECT_ROOT

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
KERNEL_NAME = "causality-pipeline"


def _ensure_kernel_matches_current_interpreter() -> None:
    """Register a kernelspec pointing at *this* interpreter (the one with
    pandas/sklearn/dowhy/etc installed), regardless of machine or CI runner -
    a bare "python3" kernelspec, if one happens to already be registered,
    may point at an unrelated Python install."""
    install_kernelspec(user=True, kernel_name=KERNEL_NAME)


def run_notebook(path: Path) -> None:
    print(f"Executing {path.name} ...")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name=KERNEL_NAME, resources={"metadata": {"path": str(path.parent)}}
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"  done -> {path}")


def main() -> int:
    _ensure_kernel_matches_current_interpreter()
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return 0

    failures = []
    for nb_path in notebooks:
        try:
            run_notebook(nb_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            failures.append(nb_path.name)

    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {failures}")
        return 1

    print(f"\nAll {len(notebooks)} notebook(s) executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
