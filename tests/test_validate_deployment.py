from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_deployment

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_BICEP = REPO_ROOT / "infra" / "main.bicep"
CONTAINER_APP_BICEP = REPO_ROOT / "infra" / "modules" / "containerApp.bicep"
AZURE_DEPLOYMENT_DOC = REPO_ROOT / "docs" / "azure-deployment.md"


def test_expected_ghcr_identifier_accepts_canonical_server_name() -> None:
    assert (
        validate_deployment._expected_ghcr_identifier(
            "io.github.Owner/Repo",
            "1.2.3",
        )
        == "ghcr.io/owner/repo:1.2.3"
    )


@pytest.mark.parametrize(
    "server_name",
    [
        "not.github.owner/repo",
        "io.github.owner",
        "io.github.",
        "io.github.owner/repo/extra",
        "https://github.com/owner/repo",
    ],
)
def test_expected_ghcr_identifier_rejects_invalid_server_names(
    server_name: str,
) -> None:
    with pytest.raises(
        SystemExit,
        match=("server.json name must follow io.github.<owner>/<server>"),
    ):
        validate_deployment._expected_ghcr_identifier(server_name, "1.2.3")


def test_azure_container_app_bicep_wires_agentic_env_contract() -> None:
    main_text = MAIN_BICEP.read_text(encoding="utf-8")
    container_text = CONTAINER_APP_BICEP.read_text(encoding="utf-8")

    for expected in (
        "enableAgentic",
        "agenticProvider",
        "plannerModel",
        "synthesisModel",
        "embeddingModel",
        "disableEmbeddings",
        "agenticIndexBackend",
        "sessionTtlSeconds",
        "enableAgenticTraceLog",
        "keyVaultOpenAiApiKeySecretUri",
    ):
        assert expected in main_text

    for expected in (
        "OPENAI_API_KEY",
        "PAPER_CHASER_ENABLE_AGENTIC",
        "PAPER_CHASER_AGENTIC_PROVIDER",
        "PAPER_CHASER_PLANNER_MODEL",
        "PAPER_CHASER_SYNTHESIS_MODEL",
        "PAPER_CHASER_EMBEDDING_MODEL",
        "PAPER_CHASER_DISABLE_EMBEDDINGS",
        "PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS",
        "PAPER_CHASER_AGENTIC_INDEX_BACKEND",
        "PAPER_CHASER_SESSION_TTL_SECONDS",
        "PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG",
    ):
        assert expected in container_text


def test_azure_deployment_doc_mentions_openai_secret_and_agentic_flags() -> None:
    text = AZURE_DEPLOYMENT_DOC.read_text(encoding="utf-8")

    assert "openai-api-key" in text
    assert "enableAgentic" in text
    assert "agenticProvider" in text
