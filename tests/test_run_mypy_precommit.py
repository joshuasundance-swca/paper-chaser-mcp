from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess


def _load_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "run_mypy_precommit.py"
    spec = importlib.util.spec_from_file_location("run_mypy_precommit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_base_command() -> list[str]:
    return [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml"]


def test_base_command_uses_running_interpreter() -> None:
    module = _load_module()

    assert module.MYPY_BASE_COMMAND == _expected_base_command()


def test_main_runs_project_scoped_mypy_by_default(monkeypatch, capsys) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        return CompletedProcess(list(command), 0, stdout="Success: no issues found\n", stderr="")

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main([])

    assert exit_code == 0
    assert calls == [_expected_base_command()]
    assert "Success: no issues found" in capsys.readouterr().out


def test_main_retries_with_no_incremental_on_internal_error(monkeypatch, capsys) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        if len(calls) == 1:
            return CompletedProcess(
                list(command),
                2,
                stdout="",
                stderr=f"{module.MYPY_INTERNAL_ERROR_MARKER}\n",
            )
        return CompletedProcess(list(command), 0, stdout="Success: no issues found\n", stderr="")

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main([])

    assert exit_code == 0
    assert calls == [
        _expected_base_command(),
        [*_expected_base_command(), module.NO_INCREMENTAL_FLAG],
    ]
    captured = capsys.readouterr()
    assert "retrying once with --no-incremental" in captured.err
    assert "Success: no issues found" in captured.out


def test_main_does_not_retry_on_regular_type_errors(monkeypatch, capsys) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        return CompletedProcess(
            list(command),
            1,
            stdout="",
            stderr="paper_chaser_mcp/server.py:1: error: boom\n",
        )

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main([])

    assert exit_code == 1
    assert calls == [_expected_base_command()]
    captured = capsys.readouterr()
    assert "retrying" not in captured.err
    assert "error: boom" in captured.err


def test_main_surfaces_persistent_internal_errors(monkeypatch) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        return CompletedProcess(
            list(command),
            2,
            stdout="",
            stderr=f"{module.MYPY_INTERNAL_ERROR_MARKER}\n",
        )

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main([])

    assert exit_code == 2
    assert calls == [
        _expected_base_command(),
        [*_expected_base_command(), module.NO_INCREMENTAL_FLAG],
    ]


def test_main_does_not_recurse_when_no_incremental_already_supplied(monkeypatch) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        return CompletedProcess(
            list(command),
            2,
            stdout="",
            stderr=f"{module.MYPY_INTERNAL_ERROR_MARKER}\n",
        )

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main([module.NO_INCREMENTAL_FLAG])

    assert exit_code == 2
    assert calls == [[*_expected_base_command(), module.NO_INCREMENTAL_FLAG]]


def test_main_passes_through_extra_arguments(monkeypatch) -> None:
    module = _load_module()

    calls: list[list[str]] = []

    def _fake_run(command):
        calls.append(list(command))
        return CompletedProcess(list(command), 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main(["--show-error-codes"])

    assert exit_code == 0
    assert calls == [[*_expected_base_command(), "--show-error-codes"]]
