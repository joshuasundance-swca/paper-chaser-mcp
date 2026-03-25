"""Run a command with the repository's local virtualenv Python."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_venv_python() -> Path:
    root = _repo_root()
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to the current interpreter so the script works in CI
    # environments that install dependencies into the system Python
    # rather than a repo-local virtualenv.
    import warnings

    warnings.warn(
        "No repo-local virtualenv found; falling back to the current Python interpreter.",
        stacklevel=1,
    )
    return Path(sys.executable)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_in_repo_venv.py <command> [args...]")

    venv_python = _repo_venv_python()
    completed = subprocess.run(  # nosec B603
        [str(venv_python), *sys.argv[1:]],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
