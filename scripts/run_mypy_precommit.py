"""Run mypy for pre-commit with a single retry on INTERNAL ERROR.

The script always invokes ``python -m mypy --config-file pyproject.toml`` using
the interpreter that is executing the hook (``sys.executable``). Project scope
is determined entirely by ``[tool.mypy] files`` in ``pyproject.toml``, so the
hook and a direct ``python -m mypy`` invocation cover the same code and the
behaviour is identical locally and in CI.

Incremental mode is used by default (mypy's ``.mypy_cache/``) so local
pre-commit runs stay fast. If mypy hits an ``INTERNAL ERROR`` (a known failure
mode when the incremental cache is stale or corrupted), the script retries
once with ``--no-incremental`` and surfaces the final exit code unchanged.
There is no CI-specific branch; failures always propagate.
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Sequence

MYPY_BASE_COMMAND: list[str] = [
    sys.executable,
    "-m",
    "mypy",
    "--config-file",
    "pyproject.toml",
]
MYPY_INTERNAL_ERROR_MARKER = "error: INTERNAL ERROR"
NO_INCREMENTAL_FLAG = "--no-incremental"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_mypy(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        list(command),
        check=False,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )


def _is_internal_error(completed: subprocess.CompletedProcess[str]) -> bool:
    combined = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    return MYPY_INTERNAL_ERROR_MARKER in combined


def _emit(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    extra_args = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = [*MYPY_BASE_COMMAND, *extra_args]
    completed = _run_mypy(command)

    if completed.returncode != 0 and _is_internal_error(completed) and NO_INCREMENTAL_FLAG not in command:
        print(
            "mypy hit an INTERNAL ERROR; retrying once with --no-incremental.",
            file=sys.stderr,
        )
        completed = _run_mypy([*command, NO_INCREMENTAL_FLAG])

    _emit(completed)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
