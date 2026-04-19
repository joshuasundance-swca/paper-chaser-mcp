from __future__ import annotations

import importlib.util
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
    return [
        str(Path(".venv/Scripts/python.exe")),
        "-m",
        "mypy",
        "--config-file",
        "pyproject.toml",
    ]


def test_build_mypy_command_limits_pre_commit_run_to_changed_python_files(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    command = module.build_mypy_command(
        [
            "paper_chaser_mcp/server.py",
            "tests/test_dispatch.py",
        ]
    )

    assert command == [
        *_expected_base_command(),
        "paper_chaser_mcp/server.py",
        "tests/test_dispatch.py",
    ]


def test_build_mypy_command_runs_full_check_when_pyproject_changes(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    command = module.build_mypy_command(["pyproject.toml"])

    assert command == _expected_base_command()


def test_build_mypy_command_ignores_non_python_targets(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    command = module.build_mypy_command(["README.md"])

    assert command == _expected_base_command()


def test_build_mypy_command_is_incremental_by_default(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    command = module.build_mypy_command(["paper_chaser_mcp/server.py"])

    assert module.NO_INCREMENTAL_FLAG not in command


def test_build_mypy_command_passes_through_no_incremental(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    command = module.build_mypy_command([module.NO_INCREMENTAL_FLAG, "paper_chaser_mcp/server.py"])

    assert command == [
        *_expected_base_command(),
        module.NO_INCREMENTAL_FLAG,
        "paper_chaser_mcp/server.py",
    ]


def test_main_retries_full_run_after_retryable_internal_error(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    commands: list[list[str]] = []

    def _fake_run(command: list[str]):
        commands.append(command)
        if len(commands) == 1:
            return CompletedProcess(
                command,
                2,
                stdout="",
                stderr=f"{module.MYPY_INTERNAL_ERROR_MARKER}\n",
            )
        return CompletedProcess(command, 0, stdout="Success: no issues found\n", stderr="")

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main(["paper_chaser_mcp/server.py"])

    assert exit_code == 0
    assert commands == [
        [*_expected_base_command(), "paper_chaser_mcp/server.py"],
        [*_expected_base_command(), module.NO_INCREMENTAL_FLAG],
    ]
    captured = capsys.readouterr()
    assert "retrying full project check without incremental state" in captured.err
    assert "Success: no issues found" in captured.out


def test_main_does_not_retry_full_run_for_non_retryable_failure(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))

    commands: list[list[str]] = []

    def _fake_run(command: list[str]):
        commands.append(command)
        return CompletedProcess(command, 1, stdout="", stderr="paper_chaser_mcp/server.py:1: error: boom\n")

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main(["paper_chaser_mcp/server.py"])

    assert exit_code == 1
    assert commands == [[*_expected_base_command(), "paper_chaser_mcp/server.py"]]
    captured = capsys.readouterr()
    assert "retrying full project check" not in captured.err
    assert "error: boom" in captured.err


def test_main_defers_persistent_internal_errors_to_ci_mypy_step(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_repo_venv_python", lambda: Path(".venv/Scripts/python.exe"))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    commands: list[list[str]] = []

    def _fake_run(command: list[str]):
        commands.append(command)
        return CompletedProcess(
            command,
            245,
            stdout="",
            stderr=f"{module.MYPY_INTERNAL_ERROR_MARKER}\n",
        )

    monkeypatch.setattr(module, "_run_mypy", _fake_run)

    exit_code = module.main(["paper_chaser_mcp/server.py"])

    assert exit_code == 0
    assert commands == [
        [*_expected_base_command(), "paper_chaser_mcp/server.py"],
        [*_expected_base_command(), module.NO_INCREMENTAL_FLAG],
    ]
    captured = capsys.readouterr()
    assert "retrying full project check without incremental state" in captured.err
    assert "Full-project mypy retry exited with code 245 in GitHub Actions" in captured.err
    assert "deferring to the dedicated workflow mypy step" in captured.err
