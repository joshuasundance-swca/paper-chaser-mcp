"""Phase 7c-1: dependency-graph regression tests for ``agentic``.

These tests assert that planner-layer and ranking-layer modules can be
imported without dragging in the heavy LLM provider implementations
(``provider_langchain``, ``provider_openai``). Before Phase 7c-1 the
``planner -> providers -> provider_base -> planner`` latent cycle meant
that importing ``paper_chaser_mcp.agentic.ranking`` or
``paper_chaser_mcp.agentic.planner`` transitively imported both heavy
provider modules via the ``.providers`` facade, even though those modules
only need the small ``ModelProviderBundle`` base class defined in
``provider_base``.

We use ``subprocess.run`` with a one-shot ``python -c`` script so
``sys.modules`` is clean; pytest collection in the parent process has
already imported the whole package and cannot be used as the subject.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

FORBIDDEN_MODULES = (
    "paper_chaser_mcp.agentic.provider_langchain",
    "paper_chaser_mcp.agentic.provider_openai",
)


def _run_import_probe(import_line: str) -> tuple[int, str, str]:
    """Execute ``import_line`` in a clean subprocess and report offenders."""

    script = textwrap.dedent(
        f"""
        import sys
        {import_line}
        offenders = [name for name in {FORBIDDEN_MODULES!r} if name in sys.modules]
        if offenders:
            print("OFFENDERS=" + ",".join(offenders))
            sys.exit(1)
        print("OK")
        """,
    )
    completed = subprocess.run(  # noqa: S603 -- trusted inline script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def test_importing_ranking_does_not_pull_heavy_providers() -> None:
    """``agentic.ranking`` must not transitively import provider_langchain/openai.

    ``ranking.py`` only needs the ``ModelProviderBundle`` type annotation,
    which lives in ``provider_base``. Importing the full ``.providers``
    facade is a latent cycle: planner (ranking imports planner helpers) ->
    providers -> provider_langchain -> provider_base (which lazy-imports
    planner helpers) -> planner.
    """

    returncode, stdout, stderr = _run_import_probe(
        "from paper_chaser_mcp.agentic import ranking",
    )
    assert returncode == 0, (
        "Importing paper_chaser_mcp.agentic.ranking leaked heavy provider "
        f"modules into sys.modules. stdout={stdout!r} stderr={stderr!r}"
    )
    assert "OK" in stdout


def test_importing_planner_core_does_not_pull_heavy_providers() -> None:
    """``agentic.planner._core`` must stay light at import time."""

    returncode, stdout, stderr = _run_import_probe(
        "from paper_chaser_mcp.agentic.planner import _core",
    )
    assert returncode == 0, (
        "Importing paper_chaser_mcp.agentic.planner._core leaked heavy "
        f"provider modules into sys.modules. stdout={stdout!r} "
        f"stderr={stderr!r}"
    )
    assert "OK" in stdout


def test_importing_planner_constants_does_not_pull_heavy_providers() -> None:
    """``agentic.planner.constants`` must stay light at import time."""

    returncode, stdout, stderr = _run_import_probe(
        "from paper_chaser_mcp.agentic.planner import constants",
    )
    assert returncode == 0, (
        "Importing paper_chaser_mcp.agentic.planner.constants leaked heavy "
        f"provider modules into sys.modules. stdout={stdout!r} "
        f"stderr={stderr!r}"
    )
    assert "OK" in stdout


def test_model_provider_bundle_is_reachable_from_provider_base() -> None:
    """``ModelProviderBundle`` must remain importable from its native module.

    This guards against accidental churn that moves the class back to the
    ``.providers`` facade.
    """

    returncode, stdout, stderr = _run_import_probe(
        "from paper_chaser_mcp.agentic.provider_base import ModelProviderBundle",
    )
    assert returncode == 0, (
        f"ModelProviderBundle must be importable from provider_base. stdout={stdout!r} stderr={stderr!r}"
    )
    assert "OK" in stdout
