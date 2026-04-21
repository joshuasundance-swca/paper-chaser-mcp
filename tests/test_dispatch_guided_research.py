"""Phase 3 TDD tests for ``dispatch/guided/research.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import research as research_mod


def test__guided_normalization_payload_empty() -> None:
    out = research_mod._guided_normalization_payload(
        {"repairs": [], "warnings": []}
    )
    assert out is None or isinstance(out, dict)


def test__guided_normalize_research_arguments_is_callable() -> None:
    assert callable(research_mod._guided_normalize_research_arguments)


def test__guided_normalize_follow_up_arguments_is_callable() -> None:
    assert callable(research_mod._guided_normalize_follow_up_arguments)


def test__guided_normalize_inspect_arguments_is_callable() -> None:
    assert callable(research_mod._guided_normalize_inspect_arguments)


_EXPECTED_EXPORTS = (
    "_guided_normalize_research_arguments",
    "_guided_normalize_follow_up_arguments",
    "_guided_normalize_inspect_arguments",
    "_guided_normalization_payload",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_research_submodule_exports(name: str) -> None:
    assert hasattr(research_mod, name)
