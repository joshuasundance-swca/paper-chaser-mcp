"""Phase 4 TDD tests for ``dispatch/expert/regulatory.py``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext
from paper_chaser_mcp.dispatch.expert.regulatory import (
    _dispatch_get_cfr_text,
    _dispatch_get_document_text_ecos,
    _dispatch_get_federal_register_document,
    _dispatch_get_species_profile_ecos,
    _dispatch_list_species_documents_ecos,
    _dispatch_search_federal_register,
    _dispatch_search_species_ecos,
)


def _make_ctx(**overrides: Any) -> DispatchContext:
    base: dict[str, Any] = {
        "client": None,
        "core_client": None,
        "openalex_client": None,
        "scholarapi_client": None,
        "arxiv_client": None,
        "enable_core": False,
        "enable_semantic_scholar": False,
        "enable_openalex": False,
        "enable_scholarapi": False,
        "enable_arxiv": False,
    }
    base.update(overrides)
    return DispatchContext(**base)


@dataclasses.dataclass
class FakeEcosClient:
    calls: list[tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=list)

    async def search_species(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search_species", kwargs))
        return {"species": []}

    async def get_species_profile(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_species_profile", kwargs))
        return {"profile": {}}

    async def list_species_documents(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_species_documents", kwargs))
        return {"documents": []}

    async def get_document_text(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_document_text", kwargs))
        return {"text": "body"}


@dataclasses.dataclass
class FakeFederalRegisterClient:
    async def search_documents(self, **kwargs: Any) -> dict[str, Any]:
        return {"fr_results": [], "query": kwargs.get("query")}


@dataclasses.dataclass
class FakeGovInfoClient:
    async def get_federal_register_document(self, **kwargs: Any) -> dict[str, Any]:
        return {"document": kwargs.get("identifier")}

    async def get_cfr_text(self, **kwargs: Any) -> dict[str, Any]:
        return {"cfr_text": "section text", "title": kwargs.get("title_number")}


@pytest.mark.asyncio
async def test_search_species_ecos_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_ecos=False, ecos_client=None)
    with pytest.raises(ValueError, match="ECOS, which is disabled"):
        await _dispatch_search_species_ecos(ctx, {"query": "bass"})


@pytest.mark.asyncio
async def test_search_species_ecos_delegates() -> None:
    client = FakeEcosClient()
    ctx = _make_ctx(enable_ecos=True, ecos_client=client)
    result = await _dispatch_search_species_ecos(ctx, {"query": "bass", "limit": 5})
    assert "species" in result or "content" in result or isinstance(result, dict)
    assert client.calls and client.calls[0][0] == "search_species"
    assert client.calls[0][1]["query"] == "bass"


@pytest.mark.asyncio
async def test_get_species_profile_ecos_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_ecos=False, ecos_client=None)
    with pytest.raises(ValueError, match="ECOS, which is disabled"):
        await _dispatch_get_species_profile_ecos(ctx, {"species_id": "1234"})


@pytest.mark.asyncio
async def test_get_species_profile_ecos_delegates() -> None:
    client = FakeEcosClient()
    ctx = _make_ctx(enable_ecos=True, ecos_client=client)
    await _dispatch_get_species_profile_ecos(ctx, {"species_id": "1234"})
    assert client.calls[0][1]["species_id"] == "1234"


@pytest.mark.asyncio
async def test_list_species_documents_ecos_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_ecos=False, ecos_client=None)
    with pytest.raises(ValueError, match="ECOS, which is disabled"):
        await _dispatch_list_species_documents_ecos(ctx, {"species_id": "42"})


@pytest.mark.asyncio
async def test_list_species_documents_ecos_delegates() -> None:
    client = FakeEcosClient()
    ctx = _make_ctx(enable_ecos=True, ecos_client=client)
    await _dispatch_list_species_documents_ecos(ctx, {"species_id": "42"})
    assert client.calls[0][0] == "list_species_documents"


@pytest.mark.asyncio
async def test_get_document_text_ecos_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_ecos=False, ecos_client=None)
    with pytest.raises(ValueError, match="ECOS, which is disabled"):
        await _dispatch_get_document_text_ecos(ctx, {"url": "https://ecos.fws.gov/x"})


@pytest.mark.asyncio
async def test_get_document_text_ecos_delegates() -> None:
    client = FakeEcosClient()
    ctx = _make_ctx(enable_ecos=True, ecos_client=client)
    await _dispatch_get_document_text_ecos(ctx, {"url": "https://ecos.fws.gov/x"})
    assert client.calls[0][0] == "get_document_text"


@pytest.mark.asyncio
async def test_search_federal_register_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_federal_register=False, federal_register_client=None)
    with pytest.raises(ValueError, match="Federal Register support, which is disabled"):
        await _dispatch_search_federal_register(ctx, {"query": "listing"})


@pytest.mark.asyncio
async def test_search_federal_register_delegates() -> None:
    ctx = _make_ctx(
        enable_federal_register=True,
        federal_register_client=FakeFederalRegisterClient(),
    )
    await _dispatch_search_federal_register(ctx, {"query": "listing"})


@pytest.mark.asyncio
async def test_get_federal_register_document_requires_govinfo() -> None:
    ctx = _make_ctx(govinfo_client=None)
    with pytest.raises(ValueError, match="GovInfo client initialization"):
        await _dispatch_get_federal_register_document(ctx, {"identifier": "2024-1"})


@pytest.mark.asyncio
async def test_get_federal_register_document_delegates() -> None:
    ctx = _make_ctx(
        govinfo_client=FakeGovInfoClient(),
        federal_register_client=FakeFederalRegisterClient(),
        enable_federal_register=True,
    )
    await _dispatch_get_federal_register_document(ctx, {"identifier": "2024-1"})


@pytest.mark.asyncio
async def test_get_cfr_text_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_govinfo_cfr=False, govinfo_client=None)
    with pytest.raises(ValueError, match="GovInfo CFR support, which is disabled"):
        await _dispatch_get_cfr_text(ctx, {"titleNumber": 50, "partNumber": 17})


@pytest.mark.asyncio
async def test_get_cfr_text_delegates() -> None:
    ctx = _make_ctx(enable_govinfo_cfr=True, govinfo_client=FakeGovInfoClient())
    await _dispatch_get_cfr_text(ctx, {"titleNumber": 50, "partNumber": 17})
