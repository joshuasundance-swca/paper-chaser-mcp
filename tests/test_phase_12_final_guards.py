"""Phase 12 final guard tests.

These tests encode the invariants Phase 12 locks in:

1. The five modules that were draining out of the allowlist in Phase 12
   (because they fell below the 800-line soft cap) must never be re-added
   to ``PLAN_OVERSIZE_MODULES``.
2. Every baseline line-count entry must match the module's current on-disk
   size within ``BASELINE_GROWTH_TOLERANCE_LINES`` (redundant with the
   existing growth guard, but with sharper error messages that include the
   exact delta per module).
3. Every module outside ``OVERSIZE_ALLOWLIST`` must stay under the
   ``SOFT_CAP_LINES`` ceiling. This is an explicit safety net parallel to
   ``test_no_new_oversized_modules`` in :mod:`tests.test_module_size_budget`.

All three tests intentionally use ``getattr`` on the module-size budget
module to avoid introducing new AST seams (Phase 9a lesson).
"""

from __future__ import annotations

from pathlib import Path

from tests import test_module_size_budget as budget

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "paper_chaser_mcp"

# Modules that Phase 12 drained from the allowlist because they now live
# below the 800-line soft cap. Re-adding any of these without justification
# would regress the Phase 12 guard-tightening work.
PHASE_12_DRAINED_MODULES: frozenset[str] = frozenset(
    {
        "paper_chaser_mcp/agentic/answer_modes.py",
        "paper_chaser_mcp/agentic/planner/_core.py",
        "paper_chaser_mcp/agentic/ranking.py",
        "paper_chaser_mcp/compat.py",
        "paper_chaser_mcp/settings.py",
    }
)


def test_plan_oversize_modules_frozen_minimum() -> None:
    """Phase 12 must not silently re-add drained modules to the allowlist."""
    plan_oversize = getattr(budget, "PLAN_OVERSIZE_MODULES")
    oversize_allowlist = getattr(budget, "OVERSIZE_ALLOWLIST")

    regressed_plan = sorted(PHASE_12_DRAINED_MODULES & plan_oversize)
    regressed_allowlist = sorted(PHASE_12_DRAINED_MODULES & oversize_allowlist)

    assert not regressed_plan, (
        "Phase 12 drained these modules from PLAN_OVERSIZE_MODULES because "
        "they now sit under the 800-line soft cap. Do not re-add them "
        f"without a strong justification: {regressed_plan}"
    )
    assert not regressed_allowlist, (
        "Phase 12 drained these modules from OVERSIZE_ALLOWLIST. Do not "
        f"re-add them without a strong justification: {regressed_allowlist}"
    )


def test_baseline_matches_on_disk_within_tolerance() -> None:
    """Every baseline entry must match reality within the growth tolerance."""
    baseline = getattr(budget, "BASELINE_LINE_COUNTS")
    tolerance = getattr(budget, "BASELINE_GROWTH_TOLERANCE_LINES")

    drifted: list[tuple[str, int, int, int]] = []
    for rel, pinned in sorted(baseline.items()):
        abs_path = PACKAGE_ROOT.parent / rel
        with abs_path.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        delta = actual - pinned
        if abs(delta) > tolerance:
            drifted.append((rel, pinned, actual, delta))

    assert not drifted, (
        "BASELINE_LINE_COUNTS drifted from on-disk line counts by more than "
        f"the {tolerance}-line tolerance. Update the baseline in the same "
        "commit as the intentional change, with a justification: "
        f"{drifted}"
    )


def test_all_non_allowlisted_modules_under_soft_cap() -> None:
    """Every non-allowlisted .py file under paper_chaser_mcp/ stays <= 800 LOC."""
    soft_cap = getattr(budget, "SOFT_CAP_LINES")
    allowlist = getattr(budget, "OVERSIZE_ALLOWLIST")
    all_files = getattr(budget, "_ALL_PACKAGE_FILES")

    offenders: list[tuple[str, int]] = []
    for rel in all_files:
        if rel in allowlist:
            continue
        abs_path = PACKAGE_ROOT.parent / rel
        with abs_path.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual > soft_cap:
            offenders.append((rel, actual))

    assert not offenders, (
        f"Modules exceed the {soft_cap}-line soft cap without being in "
        f"OVERSIZE_ALLOWLIST. Split them or allowlist them with a Phase-12 "
        f"note: {offenders}"
    )
