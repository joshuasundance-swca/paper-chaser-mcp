from __future__ import annotations

import importlib.util
from pathlib import Path


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
