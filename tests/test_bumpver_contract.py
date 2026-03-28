from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read_pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _read_project_version() -> str:
    text = _read_pyproject_text()
    match = re.search(
        r"(?ms)^\[project\]\s+.*?^version\s*=\s*\"(?P<version>[^\"]+)\"",
        text,
    )
    assert match is not None, "pyproject.toml must define [project].version."
    return match.group("version")


def _read_bumpver_current_version() -> str:
    text = _read_pyproject_text()
    match = re.search(
        r"(?ms)^\[tool\.bumpver\]\s+.*?^current_version\s*=\s*\"(?P<version>[^\"]+)\"",
        text,
    )
    assert match is not None, "pyproject.toml must define [tool.bumpver].current_version."
    return match.group("version")


def test_bumpver_current_version_wraps_project_version_with_v_prefix() -> None:
    assert _read_bumpver_current_version() == f"v{_read_project_version()}"


def test_bumpver_uses_v_prefixed_semver_tags() -> None:
    text = _read_pyproject_text()

    pattern_match = re.search(
        r"(?ms)^\[tool\.bumpver\]\s+.*?^version_pattern\s*=\s*\"(?P<pattern>[^\"]+)\"",
        text,
    )
    assert pattern_match is not None, "pyproject.toml must define [tool.bumpver].version_pattern."
    assert pattern_match.group("pattern") == "vMAJOR.MINOR.PATCH"


def test_bumpver_file_patterns_cover_the_checked_in_version_contract() -> None:
    text = _read_pyproject_text()

    assert "[tool.bumpver.file_patterns]" in text
    assert 'current_version = "{version}"' in text
    assert 'version = "{pep440_version}"' in text
    assert '"version": "{pep440_version}",' in text
    assert '"identifier": "ghcr.io/joshuasundance-swca/paper-chaser-mcp:{pep440_version}",' in text
