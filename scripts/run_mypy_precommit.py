"""Run mypy for pre-commit, scoped to changed Python files when possible."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Sequence

MYPY_BASE_COMMAND = ["-m", "mypy", "--config-file", "pyproject.toml"]
MYPY_INTERNAL_ERROR_MARKER = "error: INTERNAL ERROR"
NO_INCREMENTAL_FLAG = "--no-incremental"


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


def _split_precommit_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    passthrough_flags: list[str] = []
    filenames: list[str] = []
    for arg in argv:
        if arg == NO_INCREMENTAL_FLAG:
            passthrough_flags.append(arg)
            continue
        filenames.append(arg)
    return passthrough_flags, filenames


def _base_command(passthrough_flags: Sequence[str]) -> list[str]:
    return [str(_repo_venv_python()), *MYPY_BASE_COMMAND, *passthrough_flags]


def _ensure_no_incremental(command: Sequence[str]) -> list[str]:
    if NO_INCREMENTAL_FLAG in command:
        return list(command)
    return [*command, NO_INCREMENTAL_FLAG]


def build_mypy_command(argv: Sequence[str]) -> list[str]:
    passthrough_flags, filenames = _split_precommit_args(argv)
    command = _base_command(passthrough_flags)
    if _requires_full_run(filenames):
        return command

    python_targets = _python_targets(filenames)
    if not python_targets:
        return command

    return [*command, *python_targets]


def _run_mypy(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        list(command),
        check=False,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )


def _emit_completed_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)


def _is_internal_error_retryable(
    initial_command: Sequence[str],
    full_command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
) -> bool:
    if list(initial_command) == list(full_command):
        return False
    if completed.returncode == 0:
        return False

    return _is_internal_error(completed)


def _is_internal_error(completed: subprocess.CompletedProcess[str]) -> bool:
    combined_output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    return MYPY_INTERNAL_ERROR_MARKER in combined_output


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    passthrough_flags, _filenames = _split_precommit_args(args)
    scoped_command = build_mypy_command(args)
    full_command = _base_command(passthrough_flags)

    completed = _run_mypy(scoped_command)
    if _is_internal_error_retryable(scoped_command, full_command, completed):
        print(
            "Scoped mypy run hit an internal error; retrying full project check without incremental state.",
            file=sys.stderr,
        )
        completed = _run_mypy(_ensure_no_incremental(full_command))
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true" and completed.returncode != 0:
            print(
                f"Full-project mypy retry exited with code {completed.returncode} in GitHub Actions; "
                "deferring to the dedicated workflow mypy step.",
                file=sys.stderr,
            )
            return 0

    _emit_completed_output(completed)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
