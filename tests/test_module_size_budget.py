"""Module-size budget guards for the paper_chaser_mcp package.

Phase 1 surface-pinning test (see the Phase 1 plan in the refactor session
notes). Every `.py` file under `paper_chaser_mcp/` must eventually live under
an 800-line soft cap and a 2,500-line hard cap.

Currently-oversized modules are listed in :data:`OVERSIZE_ALLOWLIST` and are
marked ``xfail(strict=False)`` so the baseline stays green. Phase 12 will flip
the xfails to strict and, once each module is split, drop it from the
allowlist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "paper_chaser_mcp"

SOFT_CAP_LINES = 800
HARD_CAP_LINES = 2_500

# The 21 modules > 20 KB enumerated in the plan's Problem Statement table.
# Phase 12 will delete this list.
PLAN_OVERSIZE_MODULES: frozenset[str] = frozenset(
    {
        "paper_chaser_mcp/dispatch.py",
        "paper_chaser_mcp/agentic/graphs.py",
        "paper_chaser_mcp/agentic/provider_openai.py",
        "paper_chaser_mcp/agentic/provider_langchain.py",
        "paper_chaser_mcp/citation_repair.py",
        "paper_chaser_mcp/server.py",
        "paper_chaser_mcp/agentic/planner.py",
        "paper_chaser_mcp/agentic/provider_helpers.py",
        "paper_chaser_mcp/eval_curation.py",
        "paper_chaser_mcp/agentic/workspace.py",
        "paper_chaser_mcp/search.py",
        "paper_chaser_mcp/provider_runtime.py",
        "paper_chaser_mcp/agentic/models.py",
        "paper_chaser_mcp/enrichment.py",
        "paper_chaser_mcp/eval_canary.py",
        "paper_chaser_mcp/search_executor.py",
        "paper_chaser_mcp/agentic/provider_base.py",
        "paper_chaser_mcp/agentic/ranking.py",
        "paper_chaser_mcp/compat.py",
        "paper_chaser_mcp/agentic/answer_modes.py",
        "paper_chaser_mcp/settings.py",
    }
)

# Additional modules currently above the 800-line soft cap that were not
# enumerated in the plan's 20 KB table. Kept separate so Phase 12 can reason
# about plan-scope vs. discovered overhang independently. These are also
# xfail'd so the baseline stays green.
BASELINE_OVERSIZE_EXTRAS: frozenset[str] = frozenset(
    {
        "paper_chaser_mcp/models/common.py",
        "paper_chaser_mcp/models/tools.py",
        "paper_chaser_mcp/clients/semantic_scholar/client.py",
        "paper_chaser_mcp/clients/ecos/client.py",
    }
)

OVERSIZE_ALLOWLIST: frozenset[str] = PLAN_OVERSIZE_MODULES | BASELINE_OVERSIZE_EXTRAS


def _discover_package_files() -> list[str]:
    """Return every .py file under paper_chaser_mcp/ as forward-slash paths."""
    files: list[str] = []
    for root, dirs, names in os.walk(PACKAGE_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if name.endswith(".py"):
                abs_path = Path(root) / name
                rel = abs_path.relative_to(PACKAGE_ROOT.parent).as_posix()
                files.append(rel)
    files.sort()
    return files


def _count_lines(rel_path: str) -> int:
    abs_path = PACKAGE_ROOT.parent / rel_path
    with abs_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


_ALL_PACKAGE_FILES = _discover_package_files()


def _build_params() -> list[Any]:
    params: list[Any] = []
    for rel in _ALL_PACKAGE_FILES:
        marks: tuple[pytest.MarkDecorator, ...] = ()
        if rel in OVERSIZE_ALLOWLIST:
            marks = (
                pytest.mark.xfail(
                    strict=False,
                    reason="Phase 12 flips strict once module is split",
                ),
            )
        params.append(pytest.param(rel, id=rel, marks=marks))
    return params


@pytest.mark.parametrize("rel_path", _build_params())
def test_module_under_soft_cap(rel_path: str) -> None:
    """Every module must stay under the 800-line soft warning cap."""
    lines = _count_lines(rel_path)
    assert lines <= SOFT_CAP_LINES, (
        f"{rel_path} has {lines} lines (> {SOFT_CAP_LINES}). "
        "Split the module or add it to OVERSIZE_ALLOWLIST with a Phase-12 note."
    )


def test_no_new_oversized_modules() -> None:
    """Modules outside the allowlist must never exceed the soft cap.

    This is the strict gate that Phase 2+ work must respect. Any new module
    that pushes past 800 lines has to either be split or explicitly added to
    the allowlist with justification.
    """
    offenders: list[tuple[str, int]] = []
    for rel in _ALL_PACKAGE_FILES:
        if rel in OVERSIZE_ALLOWLIST:
            continue
        lines = _count_lines(rel)
        if lines > SOFT_CAP_LINES:
            offenders.append((rel, lines))
    assert not offenders, (
        "New modules exceed the 800-line soft cap. Split them or, with strong "
        f"justification, add them to OVERSIZE_ALLOWLIST: {offenders}"
    )


def test_hard_cap_not_breached_outside_allowlist() -> None:
    """Non-allowlisted modules must never exceed the 2,500-line hard cap."""
    offenders: list[tuple[str, int]] = []
    for rel in _ALL_PACKAGE_FILES:
        if rel in OVERSIZE_ALLOWLIST:
            continue
        lines = _count_lines(rel)
        if lines > HARD_CAP_LINES:
            offenders.append((rel, lines))
    assert not offenders, (
        f"Modules breach the {HARD_CAP_LINES}-line hard cap: {offenders}"
    )


def test_allowlist_entries_exist() -> None:
    """Keep the allowlist honest - every entry must point at a real file."""
    missing = sorted(
        rel for rel in OVERSIZE_ALLOWLIST if not (PACKAGE_ROOT.parent / rel).is_file()
    )
    assert not missing, f"OVERSIZE_ALLOWLIST references missing files: {missing}"
