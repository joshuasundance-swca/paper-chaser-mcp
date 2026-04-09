from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import validate_deployment

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_BICEP = REPO_ROOT / "infra" / "main.bicep"
CONTAINER_APP_BICEP = REPO_ROOT / "infra" / "modules" / "containerApp.bicep"
AZURE_DEPLOYMENT_DOC = REPO_ROOT / "docs" / "azure-deployment.md"
AZURE_ARCHITECTURE_DOC = REPO_ROOT / "docs" / "azure-architecture.md"


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


def test_parse_args_accepts_python_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["validate_deployment.py", "--python-version", "3.14"],
    )

    args = validate_deployment.parse_args()

    assert isinstance(args, argparse.Namespace)
    assert args.python_version == "3.14"


def test_validate_docker_passes_python_build_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    responses = iter([(403, ""), (401, ""), (200, "")])

    def fake_run(command: list[str], *, description: str):
        commands.append(command)

        class Result:
            stdout = "[]"

        return Result()

    monkeypatch.setattr(validate_deployment, "run", fake_run)
    monkeypatch.setattr(validate_deployment, "validate_image_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(validate_deployment, "free_port", lambda: 18080)
    monkeypatch.setattr(validate_deployment, "wait_for_health", lambda url: None)
    monkeypatch.setattr(validate_deployment, "request_json", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(validate_deployment.subprocess, "run", lambda *args, **kwargs: None)

    validate_deployment.validate_docker(
        "docker",
        "paper-chaser-mcp:test",
        expected_server_name="io.github.owner/repo",
        expected_server_version="1.2.3",
        expected_source_url="https://github.com/owner/repo",
        python_version="3.14",
    )

    build_command = commands[0]
    assert "--build-arg" in build_command
    assert "PYTHON_VERSION=3.14" in build_command


def test_azure_container_app_bicep_wires_agentic_env_contract() -> None:
    main_text = MAIN_BICEP.read_text(encoding="utf-8")
    container_text = CONTAINER_APP_BICEP.read_text(encoding="utf-8")

    assert "param azureOpenAiEndpoint string = ''" in main_text
    assert "param azureOpenAiApiVersion string = ''" in main_text
    assert "param openRouterBaseUrl string = 'https://openrouter.ai/api/v1'" in main_text
    assert "param openRouterHttpReferer string = ''" in main_text
    assert "param openRouterTitle string = ''" in main_text
    assert "param nvidiaNimBaseUrl string = ''" in main_text
    assert "param huggingFaceBaseUrl string = 'https://router.huggingface.co/v1'" in main_text
    assert "azureOpenAiEndpoint: azureOpenAiEndpoint" in main_text
    assert "azureOpenAiApiVersion: azureOpenAiApiVersion" in main_text
    assert "openRouterBaseUrl: openRouterBaseUrl" in main_text
    assert "openRouterHttpReferer: openRouterHttpReferer" in main_text
    assert "openRouterTitle: openRouterTitle" in main_text
    assert "nvidiaNimBaseUrl: nvidiaNimBaseUrl" in main_text
    assert "huggingFaceBaseUrl: huggingFaceBaseUrl" in main_text
    assert "azureOpenAiEndpoint: ''" not in main_text
    assert "azureOpenAiApiVersion: ''" not in main_text

    for expected in (
        "enableAgentic",
        "agenticProvider",
        "azureOpenAiEndpoint",
        "azureOpenAiApiVersion",
        "plannerModel",
        "synthesisModel",
        "embeddingModel",
        "disableEmbeddings",
        "agenticIndexBackend",
        "sessionTtlSeconds",
        "enableAgenticTraceLog",
        "keyVaultOpenAiApiKeySecretUri",
        "keyVaultNvidiaApiKeySecretUri",
        "keyVaultAzureOpenAiApiKeySecretUri",
        "keyVaultAnthropicApiKeySecretUri",
        "keyVaultGoogleApiKeySecretUri",
        "keyVaultHuggingFaceApiKeySecretUri",
        "keyVaultMistralApiKeySecretUri",
        "keyVaultOpenRouterApiKeySecretUri",
        "enableScholarApi",
        "keyVaultScholarApiKeySecretUri",
    ):
        assert expected in main_text

    for expected in (
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_NIM_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_TITLE",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_BASE_URL",
        "MISTRAL_API_KEY",
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
        "PAPER_CHASER_ENABLE_SCHOLARAPI",
        "SCHOLARAPI_API_KEY",
    ):
        assert expected in container_text

    assert "!empty(azureOpenAiEndpoint)" in container_text
    assert "!empty(azureOpenAiApiVersion)" in container_text


def test_azure_deployment_doc_mentions_openai_secret_and_agentic_flags() -> None:
    text = AZURE_DEPLOYMENT_DOC.read_text(encoding="utf-8")

    assert "openai-api-key" in text
    assert "nvidia-api-key" in text
    assert "azure-openai-api-key" in text
    assert "anthropic-api-key" in text
    assert "huggingface-api-key" in text
    assert "google-api-key" in text
    assert "mistral-api-key" in text
    assert "enableAgentic" in text
    assert "agenticProvider" in text
    assert "azureOpenAiEndpoint" in text
    assert "azureOpenAiApiVersion" in text
    assert "nvidiaNimBaseUrl" in text
    assert "huggingFaceBaseUrl" in text
    assert "enableScholarApi" in text
    assert "scholarapi-api-key" in text


def test_azure_architecture_doc_mentions_provider_specific_smart_layer_inputs() -> None:
    text = AZURE_ARCHITECTURE_DOC.read_text(encoding="utf-8")

    assert "Azure OpenAI" in text
    assert "Hugging Face" in text
    assert "NVIDIA" in text
    assert "Anthropic" in text
    assert "Google" in text
    assert "Mistral" in text
    assert "AZURE_OPENAI_ENDPOINT" in text
    assert "HUGGINGFACE_BASE_URL" in text
    assert "NVIDIA_NIM_BASE_URL" in text
