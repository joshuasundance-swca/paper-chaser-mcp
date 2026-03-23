from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_deployment

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_BICEP = REPO_ROOT / "infra" / "main.bicep"
CONTAINER_APP_BICEP = REPO_ROOT / "infra" / "modules" / "containerApp.bicep"
AZURE_DEPLOYMENT_DOC = REPO_ROOT / "docs" / "azure-deployment.md"


def test_expected_ghcr_identifier_accepts_canonical_https_github_repo_url() -> None:
    assert (
        validate_deployment._expected_ghcr_identifier(
            "https://github.com/Owner/Repo",
            "1.2.3",
        )
        == "ghcr.io/owner/repo:1.2.3"
    )


@pytest.mark.parametrize(
    "repository_url",
    [
        "http://github.com/owner/repo",
        "https://www.github.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/owner/repo/tree/main",
    ],
)
def test_expected_ghcr_identifier_rejects_non_canonical_repository_urls(
    repository_url: str,
) -> None:
    with pytest.raises(
        SystemExit,
        match=(
            "server.json repository.url must point to an "
            "https://github.com/<owner>/<repo> path"
        ),
    ):
        validate_deployment._expected_ghcr_identifier(repository_url, "1.2.3")


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
        "SCHOLAR_SEARCH_ENABLE_AGENTIC",
        "SCHOLAR_SEARCH_AGENTIC_PROVIDER",
        "SCHOLAR_SEARCH_PLANNER_MODEL",
        "SCHOLAR_SEARCH_SYNTHESIS_MODEL",
        "SCHOLAR_SEARCH_EMBEDDING_MODEL",
        "SCHOLAR_SEARCH_DISABLE_EMBEDDINGS",
        "SCHOLAR_SEARCH_AGENTIC_OPENAI_TIMEOUT_SECONDS",
        "SCHOLAR_SEARCH_AGENTIC_INDEX_BACKEND",
        "SCHOLAR_SEARCH_SESSION_TTL_SECONDS",
        "SCHOLAR_SEARCH_ENABLE_AGENTIC_TRACE_LOG",
    ):
        assert expected in container_text


def test_azure_deployment_doc_mentions_openai_secret_and_agentic_flags() -> None:
    text = AZURE_DEPLOYMENT_DOC.read_text(encoding="utf-8")

    assert "openai-api-key" in text
    assert "enableAgentic" in text
    assert "agenticProvider" in text
