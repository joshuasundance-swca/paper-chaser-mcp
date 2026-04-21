"""Ctx-first entrypoints for regulatory primary-source tools (Phase 4).

Covers the seven expert regulatory branches that were previously inline in
``dispatch_tool``:

* ``search_species_ecos``
* ``get_species_profile_ecos``
* ``list_species_documents_ecos``
* ``get_document_text_ecos``
* ``search_federal_register``
* ``get_federal_register_document``
* ``get_cfr_text``

Each entrypoint keeps its pre-Phase-4 error-raising behaviour when the
underlying toggle is disabled or the client is missing, and delegates to
``_finalize_tool_result`` (imported lazily from ``.._core``) to preserve the
existing workspace-augmentation contract.
"""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS
from ...models.tools import (
    EcosSpeciesLookupArgs,
    GetCfrTextArgs,
    GetDocumentTextEcosArgs,
    GetFederalRegisterDocumentArgs,
    ListSpeciesDocumentsEcosArgs,
    SearchFederalRegisterArgs,
    SearchSpeciesEcosArgs,
)
from ..context import DispatchContext


def _finalize(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    ctx: DispatchContext,
) -> dict[str, Any]:
    """Lazy import of ``_finalize_tool_result`` avoids circular imports."""
    from .._core import _finalize_tool_result

    return _finalize_tool_result(
        tool_name,
        arguments,
        result,
        workspace_registry=ctx.workspace_registry,
    )


async def _dispatch_search_species_ecos(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_ecos or ctx.ecos_client is None:
        raise ValueError(
            "search_species_ecos requires ECOS, which is disabled. Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
        )
    ecos_args = cast(
        SearchSpeciesEcosArgs,
        TOOL_INPUT_MODELS["search_species_ecos"].model_validate(arguments),
    )
    result = await ctx.ecos_client.search_species(
        query=ecos_args.query,
        limit=ecos_args.limit,
        match_mode=ecos_args.match_mode,
    )
    return _finalize("search_species_ecos", arguments, result, ctx)


async def _dispatch_get_species_profile_ecos(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_ecos or ctx.ecos_client is None:
        raise ValueError(
            "get_species_profile_ecos requires ECOS, which is disabled. "
            "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
        )
    species_lookup_args = cast(
        EcosSpeciesLookupArgs,
        TOOL_INPUT_MODELS["get_species_profile_ecos"].model_validate(arguments),
    )
    result = await ctx.ecos_client.get_species_profile(
        species_id=species_lookup_args.species_id,
    )
    return _finalize("get_species_profile_ecos", arguments, result, ctx)


async def _dispatch_list_species_documents_ecos(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_ecos or ctx.ecos_client is None:
        raise ValueError(
            "list_species_documents_ecos requires ECOS, which is disabled. "
            "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
        )
    document_list_args = cast(
        ListSpeciesDocumentsEcosArgs,
        TOOL_INPUT_MODELS["list_species_documents_ecos"].model_validate(arguments),
    )
    result = await ctx.ecos_client.list_species_documents(
        species_id=document_list_args.species_id,
        document_kinds=document_list_args.document_kinds,
    )
    return _finalize("list_species_documents_ecos", arguments, result, ctx)


async def _dispatch_get_document_text_ecos(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_ecos or ctx.ecos_client is None:
        raise ValueError(
            "get_document_text_ecos requires ECOS, which is disabled. "
            "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
        )
    document_text_args = cast(
        GetDocumentTextEcosArgs,
        TOOL_INPUT_MODELS["get_document_text_ecos"].model_validate(arguments),
    )
    result = await ctx.ecos_client.get_document_text(url=document_text_args.url)
    return _finalize("get_document_text_ecos", arguments, result, ctx)


async def _dispatch_search_federal_register(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_federal_register or ctx.federal_register_client is None:
        raise ValueError(
            "search_federal_register requires Federal Register support, which is disabled. "
            "Set PAPER_CHASER_ENABLE_FEDERAL_REGISTER=true to use this tool."
        )
    fr_args = cast(
        SearchFederalRegisterArgs,
        TOOL_INPUT_MODELS["search_federal_register"].model_validate(arguments),
    )
    result = await ctx.federal_register_client.search_documents(
        query=fr_args.query,
        limit=fr_args.limit,
        agencies=fr_args.agencies,
        document_types=fr_args.document_types,
        publication_date_from=fr_args.publication_date_from,
        publication_date_to=fr_args.publication_date_to,
        cfr_citation=fr_args.cfr_citation,
        cfr_title=fr_args.cfr_title,
        cfr_part=fr_args.cfr_part,
        document_number=fr_args.document_number,
    )
    return _finalize("search_federal_register", arguments, result, ctx)


async def _dispatch_get_federal_register_document(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if ctx.govinfo_client is None:
        raise ValueError("get_federal_register_document requires GovInfo client initialization.")
    document_args = cast(
        GetFederalRegisterDocumentArgs,
        TOOL_INPUT_MODELS["get_federal_register_document"].model_validate(arguments),
    )
    result = await ctx.govinfo_client.get_federal_register_document(
        identifier=document_args.identifier,
        federal_register_client=ctx.federal_register_client if ctx.enable_federal_register else None,
    )
    return _finalize("get_federal_register_document", arguments, result, ctx)


async def _dispatch_get_cfr_text(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not ctx.enable_govinfo_cfr or ctx.govinfo_client is None:
        raise ValueError(
            "get_cfr_text requires GovInfo CFR support, which is disabled. "
            "Set PAPER_CHASER_ENABLE_GOVINFO_CFR=true to use this tool."
        )
    cfr_args = cast(
        GetCfrTextArgs,
        TOOL_INPUT_MODELS["get_cfr_text"].model_validate(arguments),
    )
    result = await ctx.govinfo_client.get_cfr_text(
        title_number=cfr_args.title_number,
        part_number=cfr_args.part_number,
        section_number=cfr_args.section_number,
        revision_year=cfr_args.revision_year,
        effective_date=cfr_args.effective_date,
    )
    return _finalize("get_cfr_text", arguments, result, ctx)
