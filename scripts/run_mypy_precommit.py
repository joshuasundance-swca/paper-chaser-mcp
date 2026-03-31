"""Run mypy for pre-commit, scoped to changed Python files when possible."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Sequence

MYPY_BASE_COMMAND = ["-m", "mypy", "--config-file", "pyproject.toml", "--no-incremental"]


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
    return Path(sys.executable)


def _requires_full_run(filenames: Sequence[str]) -> bool:
    return any(Path(filename).name == "pyproject.toml" for filename in filenames)


def _python_targets(filenames: Sequence[str]) -> list[str]:
    return [filename for filename in filenames if filename.endswith(".py")]


def build_mypy_command(filenames: Sequence[str]) -> list[str]:
    command = [str(_repo_venv_python()), *MYPY_BASE_COMMAND]
    if _requires_full_run(filenames):
        return command

    python_targets = _python_targets(filenames)
    if not python_targets:
        return command

    return [*command, *python_targets]


def main(argv: Sequence[str] | None = None) -> int:
    filenames = list(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run(  # nosec B603
        build_mypy_command(filenames),
        check=False,
        cwd=_repo_root(),
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
