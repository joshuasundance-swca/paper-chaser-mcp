"""Phase 2 Track C amendment 3: branch entrypoints accept ``DispatchContext``.

Phase 2 Step 2 introduced :class:`paper_chaser_mcp.dispatch.context.DispatchContext`
as the single dependency bag for ``dispatch_tool`` and its helpers, but the
inline branches inside ``dispatch_tool`` itself still unpack ~40 locals and pass
individual values to everything they call. Phase 3 will relocate these branches
into sibling submodules, and that is only safe if the branch functions already
take a ``DispatchContext`` rather than re-deriving the same state on every move.

This amendment extracts four branch entrypoints — ``research``,
``follow_up_research``, ``search_papers_smart``, ``ask_result_set`` — as module-
level async functions on ``paper_chaser_mcp.dispatch._core``:

* ``_dispatch_research(ctx: DispatchContext, arguments: dict[str, Any])``
* ``_dispatch_follow_up_research(ctx, arguments)``
* ``_dispatch_search_papers_smart(ctx, arguments)``
* ``_dispatch_ask_result_set(ctx, arguments)``

The tests here pin the signature shape (``ctx`` first, positional-or-keyword,
typed as ``DispatchContext``) so a future refactor cannot silently regress the
contract. The actual branch logic is validated end-to-end by the existing
characterization tests, which must stay green without
``PAPER_CHASER_CHAR_REGEN=1``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext

# Branch entrypoints pinned as ctx-first. Phase 2 Track C seeded the first
# four; Phase 4 extended the contract to every extracted smart + expert
# submodule so no branch can silently reintroduce per-branch kwargs.
BRANCH_ENTRYPOINTS: tuple[tuple[str, str], ...] = (
    # Guided (Phase 2/3).
    ("paper_chaser_mcp.dispatch._core", "_dispatch_research"),
    ("paper_chaser_mcp.dispatch._core", "_dispatch_follow_up_research"),
    # Smart (Phase 4).
    ("paper_chaser_mcp.dispatch.smart.search", "_dispatch_search_papers_smart"),
    ("paper_chaser_mcp.dispatch.smart.ask", "_dispatch_ask_result_set"),
    ("paper_chaser_mcp.dispatch.smart.landscape", "_dispatch_map_research_landscape"),
    ("paper_chaser_mcp.dispatch.smart.graph", "_dispatch_expand_research_graph"),
    # Expert: regulatory primary sources.
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_search_species_ecos"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_species_profile_ecos"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_list_species_documents_ecos"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_document_text_ecos"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_search_federal_register"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_federal_register_document"),
    ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_cfr_text"),
    # Expert: raw Semantic Scholar passthroughs.
    ("paper_chaser_mcp.dispatch.expert.raw", "_dispatch_search_papers"),
    ("paper_chaser_mcp.dispatch.expert.raw", "_dispatch_search_papers_match"),
    ("paper_chaser_mcp.dispatch.expert.raw", "_dispatch_get_paper_details"),
    ("paper_chaser_mcp.dispatch.expert.raw", "_dispatch_resolve_citation"),
    ("paper_chaser_mcp.dispatch.expert.raw", "_dispatch_search_papers_bulk"),
    # Expert: OpenAlex.
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_paper_autocomplete_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_papers_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_entities_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_papers_openalex_by_entity"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_papers_openalex_bulk"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_paper_details_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_paper_citations_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_paper_references_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_authors_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_author_info_openalex"),
    ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_author_papers_openalex"),
    # Expert: ScholarAPI.
    ("paper_chaser_mcp.dispatch.expert.scholarapi", "_dispatch_search_papers_scholarapi"),
    ("paper_chaser_mcp.dispatch.expert.scholarapi", "_dispatch_list_papers_scholarapi"),
    ("paper_chaser_mcp.dispatch.expert.scholarapi", "_dispatch_get_paper_text_scholarapi"),
    ("paper_chaser_mcp.dispatch.expert.scholarapi", "_dispatch_get_paper_texts_scholarapi"),
    ("paper_chaser_mcp.dispatch.expert.scholarapi", "_dispatch_get_paper_pdf_scholarapi"),
    # Expert: SerpApi.
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_search_papers_serpapi_cited_by"),
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_search_papers_serpapi_versions"),
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_get_author_profile_serpapi"),
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_get_author_articles_serpapi"),
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_get_serpapi_account_status"),
    ("paper_chaser_mcp.dispatch.expert.serpapi", "_dispatch_get_paper_citation_formats"),
    # Expert: enrichment + provider/non-search helpers.
    ("paper_chaser_mcp.dispatch.expert.enrichment", "_dispatch_get_paper_metadata_crossref"),
    ("paper_chaser_mcp.dispatch.expert.enrichment", "_dispatch_get_paper_open_access_unpaywall"),
    ("paper_chaser_mcp.dispatch.expert.enrichment", "_dispatch_enrich_paper"),
    ("paper_chaser_mcp.dispatch.expert.enrichment", "_dispatch_provider_search_tool"),
    ("paper_chaser_mcp.dispatch.expert.enrichment", "_dispatch_non_search_tool"),
)


def _load(module_path: str, name: str) -> Any:
    module = importlib.import_module(module_path)
    assert hasattr(module, name), (
        f"{module_path} must define {name!r} as a module-level entrypoint so "
        f"dispatch_tool can invoke the branch through a stable seam."
    )
    return getattr(module, name)


@pytest.mark.parametrize(("module_path", "name"), BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_exists(module_path: str, name: str) -> None:
    """Each branch entrypoint must exist as a module-level async function."""

    func = _load(module_path, name)
    assert inspect.iscoroutinefunction(func), (
        f"{module_path}.{name} must be an async function; dispatch_tool awaits the result."
    )


@pytest.mark.parametrize(("module_path", "name"), BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_first_parameter_is_ctx(module_path: str, name: str) -> None:
    """The first parameter must be ``ctx: DispatchContext``."""

    func = _load(module_path, name)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    assert params, f"{module_path}.{name} must accept at least one parameter"
    first = params[0]
    assert first.name == "ctx", (
        f"{module_path}.{name}'s first parameter must be named 'ctx' (got {first.name!r}). "
        f"This is the shared dependency-bag convention pinned by Phase 2 Step 2."
    )
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"{module_path}.{name}'s ctx parameter must be positional-or-keyword, got {first.kind!r}"


@pytest.mark.parametrize(("module_path", "name"), BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_ctx_is_dispatch_context(module_path: str, name: str) -> None:
    """The ``ctx`` parameter must be typed as ``DispatchContext``.

    This pin keeps Phase 4 honest: once branches move to sibling modules they
    must continue to accept the same frozen dependency bag rather than
    reintroducing per-branch ad-hoc kwargs.
    """

    func = _load(module_path, name)
    sig = inspect.signature(func)
    ctx_param = sig.parameters["ctx"]
    annotation = ctx_param.annotation
    # Annotations may be stringified under ``from __future__ import annotations``;
    # accept both the real class and its string name.
    assert annotation in (DispatchContext, "DispatchContext"), (
        f"{module_path}.{name}'s ctx parameter must be typed as DispatchContext, got "
        f"{annotation!r}. Use `ctx: DispatchContext` in the source to keep "
        f"Phase 4+ moves mechanical."
    )


@pytest.mark.parametrize(("module_path", "name"), BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_accepts_arguments_dict(module_path: str, name: str) -> None:
    """The entrypoint must also accept an ``arguments`` mapping parameter.

    ``dispatch_tool`` normalizes the incoming tool arguments once and then
    forwards the validated dict to the branch. Keeping the parameter name
    stable (``arguments``) lets later phases grep for call sites safely.
    """

    func = _load(module_path, name)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    assert len(params) >= 2, (
        f"{module_path}.{name} must accept at least (ctx, arguments); got {[p.name for p in params]}"
    )
    assert params[1].name == "arguments", (
        f"{module_path}.{name}'s second parameter must be named 'arguments' "
        f"(got {params[1].name!r}) to keep call sites greppable."
    )
