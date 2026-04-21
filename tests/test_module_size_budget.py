"""Module-size budget guards for the paper_chaser_mcp package.

Phase 1 surface-pinning test (see the Phase 1 plan in the refactor session
notes). Every `.py` file under `paper_chaser_mcp/` must eventually live under
an 800-line soft cap and a 2,500-line hard cap.

Currently-oversized modules are listed in :data:`OVERSIZE_ALLOWLIST` and are
marked ``xfail(strict=False)`` so the baseline stays green. Phase 12 will flip
the xfails to strict and, once each module is split, drop it from the
allowlist.

In addition, :data:`BASELINE_LINE_COUNTS` pins the current line count of every
allowlisted module so they cannot silently *regrow* while they wait to be
split. Growth beyond ``BASELINE_GROWTH_TOLERANCE_LINES`` fails the guard; any
intentional growth requires updating ``BASELINE_LINE_COUNTS`` in the same
commit with justification.
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
        "paper_chaser_mcp/dispatch/_core.py",
        "paper_chaser_mcp/agentic/graphs/_core.py",
        "paper_chaser_mcp/server.py",
        "paper_chaser_mcp/agentic/planner/_core.py",
        "paper_chaser_mcp/eval_curation.py",
        "paper_chaser_mcp/agentic/workspace.py",
        "paper_chaser_mcp/search.py",
        "paper_chaser_mcp/provider_runtime.py",
        "paper_chaser_mcp/agentic/models.py",
        "paper_chaser_mcp/enrichment.py",
        "paper_chaser_mcp/eval_canary.py",
        "paper_chaser_mcp/search_executor.py",
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
        # Phase 3 leftovers: guided submodules extracted from dispatch/_core.py
        # that exceed the soft cap. Phase 3b will split these further.
        "paper_chaser_mcp/dispatch/guided/trust.py",
        # Phase 7c-4: ``graphs/_core.search_papers_smart`` orchestration body
        # was moved to ``graphs/smart_graph.run_search_papers_smart`` as a
        # Pattern B extraction. The destination file exceeds the 800-line soft
        # cap until the orchestration is split into per-stage helpers in a
        # follow-up phase.
        "paper_chaser_mcp/agentic/graphs/smart_graph.py",
        # Phase 8c: ``provider_openai.py`` was split into the
        # ``providers/openai/`` subpackage. The OpenAI-compatible bundle class
        # is deeply self-referential (self._foo state throughout every
        # method), so the whole class was relocated verbatim into
        # ``bundle.py`` to preserve behavior. A follow-up phase can split it
        # further into mix-ins (chat / embeddings / ranking / adequacy) once
        # the test seam surface is stable.
        "paper_chaser_mcp/agentic/providers/openai/bundle.py",
        # Phase 8d: ``provider_langchain.py`` was split into the
        # ``providers/langchain/`` subpackage. The shared chat bundle class
        # (``LangChainChatProviderBundle``) carries the same deep
        # self-reference pattern as the OpenAI bundle, so it was relocated
        # verbatim into ``bundle.py``. Concrete provider adapters live in
        # ``adapters.py``. A follow-up phase can split the bundle further.
        "paper_chaser_mcp/agentic/providers/langchain/bundle.py",
        # Phase 9a: ``citation_repair/_core.py`` was split into ``normalization``,
        # ``candidates``, and ``api``. ``api.py`` owns the async
        # ``resolve_citation`` orchestrator plus every provider-layered
        # ``_resolve_*`` helper, response serialization, abstention filters,
        # and the famous-paper candidate bridge. Those pieces are tightly
        # coupled through the resolution state machine and are cheaper to
        # keep together than to interleave with ``candidates``. A later phase
        # can peel the serialization helpers into their own module once the
        # Phase 9b ranking rebalance lands and the contract stabilizes.
        "paper_chaser_mcp/citation_repair/api.py",
    }
)

OVERSIZE_ALLOWLIST: frozenset[str] = PLAN_OVERSIZE_MODULES | BASELINE_OVERSIZE_EXTRAS

# Frozen baseline line counts for every allowlisted module, captured on the
# ``refactor/phase-1-harness`` branch. The point of this mapping is to keep
# allowlisted monoliths from growing silently: Phase 12 work must split them,
# not inflate them.
#
# Any intentional growth beyond ``BASELINE_GROWTH_TOLERANCE_LINES`` requires
# updating the corresponding entry in this dict in the SAME commit as the
# growth, with a justification in the commit message (or in a Phase-12
# tracking note). Any shrinkage is fine and does not require an update.
#
# If an allowlisted module has already been split below the 800-line soft cap
# the baseline reflects the CURRENT on-disk size, not a historical high-water
# mark. That way this guard only protects against regressions, not celebrated
# wins.
BASELINE_LINE_COUNTS: dict[str, int] = {
    "paper_chaser_mcp/agentic/answer_modes.py": 658,
    "paper_chaser_mcp/agentic/graphs/_core.py": 3_256,
    "paper_chaser_mcp/agentic/graphs/smart_graph.py": 1_074,
    "paper_chaser_mcp/agentic/models.py": 870,
    "paper_chaser_mcp/agentic/planner/_core.py": 64,
    "paper_chaser_mcp/agentic/providers/langchain/bundle.py": 1_421,
    "paper_chaser_mcp/agentic/providers/openai/bundle.py": 1_826,
    "paper_chaser_mcp/agentic/ranking.py": 690,
    "paper_chaser_mcp/agentic/workspace.py": 979,
    "paper_chaser_mcp/citation_repair/api.py": 880,
    "paper_chaser_mcp/clients/ecos/client.py": 963,
    "paper_chaser_mcp/clients/semantic_scholar/client.py": 1_179,
    "paper_chaser_mcp/compat.py": 690,
    "paper_chaser_mcp/dispatch/_core.py": 3_665,
    "paper_chaser_mcp/dispatch/guided/trust.py": 984,
    "paper_chaser_mcp/enrichment.py": 816,
    "paper_chaser_mcp/eval_canary.py": 728,
    "paper_chaser_mcp/eval_curation.py": 965,
    "paper_chaser_mcp/models/common.py": 1_107,
    "paper_chaser_mcp/models/tools.py": 1_489,
    "paper_chaser_mcp/provider_runtime.py": 920,
    "paper_chaser_mcp/search.py": 992,
    "paper_chaser_mcp/search_executor.py": 855,
    "paper_chaser_mcp/server.py": 1_214,
    "paper_chaser_mcp/settings.py": 569,
}

BASELINE_GROWTH_TOLERANCE_LINES = 50


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
        if rel == "paper_chaser_mcp/dispatch/_core.py":
            marks = (
                pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "Phase 5 finalize: dispatch/_core.py oversize is pinned strict. "
                        "If this XPASSes (module now <= soft cap), remove from OVERSIZE_ALLOWLIST."
                    ),
                ),
            )
        elif rel in OVERSIZE_ALLOWLIST:
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
    assert not offenders, f"Modules breach the {HARD_CAP_LINES}-line hard cap: {offenders}"


def test_allowlist_entries_exist() -> None:
    """Keep the allowlist honest - every entry must point at a real file."""
    missing = sorted(rel for rel in OVERSIZE_ALLOWLIST if not (PACKAGE_ROOT.parent / rel).is_file())
    assert not missing, f"OVERSIZE_ALLOWLIST references missing files: {missing}"


def test_baseline_line_counts_cover_every_allowlisted_module() -> None:
    """``BASELINE_LINE_COUNTS`` must pin a size for every allowlisted module.

    Without this, a newly-allowlisted module could silently dodge the
    no-regrowth check below.
    """
    missing_from_baseline = sorted(OVERSIZE_ALLOWLIST - BASELINE_LINE_COUNTS.keys())
    extra_in_baseline = sorted(BASELINE_LINE_COUNTS.keys() - OVERSIZE_ALLOWLIST)
    assert not missing_from_baseline, (
        "Every OVERSIZE_ALLOWLIST entry needs a BASELINE_LINE_COUNTS row so "
        f"regrowth is blocked: missing={missing_from_baseline}"
    )
    assert not extra_in_baseline, (
        "BASELINE_LINE_COUNTS has stale entries not in OVERSIZE_ALLOWLIST: "
        f"{extra_in_baseline}. Drop them along with the allowlist removal."
    )


def test_allowlisted_modules_do_not_grow_past_baseline() -> None:
    """Allowlisted monoliths must not grow meaningfully past their baseline.

    The whole point of the Phase-12 plan is to *shrink* these modules, not
    let them balloon further. ``BASELINE_GROWTH_TOLERANCE_LINES`` allows
    minor churn (comments, imports, small bug fixes) but blocks the kind of
    silent growth that would make future splits harder.

    Any intentional growth beyond the tolerance requires updating
    ``BASELINE_LINE_COUNTS`` in the same commit with a justification.
    """
    regressions: list[tuple[str, int, int, int]] = []
    for rel, baseline in sorted(BASELINE_LINE_COUNTS.items()):
        actual = _count_lines(rel)
        ceiling = baseline + BASELINE_GROWTH_TOLERANCE_LINES
        if actual > ceiling:
            regressions.append((rel, baseline, actual, actual - baseline))
    assert not regressions, (
        "Allowlisted modules grew past their baseline + tolerance "
        f"({BASELINE_GROWTH_TOLERANCE_LINES} lines). Split them or update "
        "BASELINE_LINE_COUNTS in the same commit with justification: "
        f"{regressions}"
    )
