import pytest

from scholar_search_mcp import server
from scholar_search_mcp.settings import AppSettings


def test_app_settings_reads_provider_order_from_env() -> None:
    settings = AppSettings.from_env(
        {"SCHOLAR_SEARCH_PROVIDER_ORDER": "semantic_scholar,arxiv"}
    )

    assert settings.provider_order == ("semantic_scholar", "arxiv")


def test_app_settings_accepts_serpapi_alias_in_provider_order() -> None:
    settings = AppSettings.from_env(
        {"SCHOLAR_SEARCH_PROVIDER_ORDER": "semantic_scholar,serpapi,arxiv"}
    )

    assert settings.provider_order == (
        "semantic_scholar",
        "serpapi_google_scholar",
        "arxiv",
    )


def test_app_settings_rejects_duplicate_provider_order_entries() -> None:
    with pytest.raises(ValueError, match="cannot repeat providers"):
        AppSettings.from_env(
            {"SCHOLAR_SEARCH_PROVIDER_ORDER": "core,core,semantic_scholar"}
        )


def test_search_papers_args_accept_serpapi_alias() -> None:
    from scholar_search_mcp.models.tools import SearchPapersArgs

    args = SearchPapersArgs.model_validate(
        {
            "query": "fallback",
            "preferredProvider": "serpapi",
            "providerOrder": ["core", "serpapi"],
        }
    )

    assert args.preferred_provider == "serpapi_google_scholar"
    assert args.provider_order == ["core", "serpapi_google_scholar"]


def test_app_settings_serpapi_disabled_by_default() -> None:
    """SerpApi must be disabled by default to protect users from surprise costs."""

    settings = AppSettings.from_env({})  # empty env
    assert settings.enable_serpapi is False
    assert settings.enable_agentic is False
    assert settings.openai_api_key is None
    assert settings.serpapi_api_key is None
    assert settings.enable_openalex is True
    assert settings.openalex_api_key is None
    assert settings.openalex_mailto is None


def test_app_settings_serpapi_enabled_via_env() -> None:
    """SCHOLAR_SEARCH_ENABLE_SERPAPI=true must enable the provider."""

    settings = AppSettings.from_env(
        {
            "SCHOLAR_SEARCH_ENABLE_SERPAPI": "true",
            "SERPAPI_API_KEY": "my-api-key",
        }
    )
    assert settings.enable_serpapi is True
    assert settings.serpapi_api_key == "my-api-key"


def test_app_settings_openalex_enabled_with_optional_mailto_and_key() -> None:
    settings = AppSettings.from_env(
        {
            "SCHOLAR_SEARCH_ENABLE_OPENALEX": "true",
            "OPENALEX_API_KEY": "openalex-key",
            "OPENALEX_MAILTO": "team@example.com",
        }
    )

    assert settings.enable_openalex is True
    assert settings.openalex_api_key == "openalex-key"
    assert settings.openalex_mailto == "team@example.com"


def test_app_settings_normalizes_blank_optional_values_to_none() -> None:
    settings = AppSettings.from_env(
        {
            "CORE_API_KEY": "   ",
            "SEMANTIC_SCHOLAR_API_KEY": "",
            "OPENALEX_API_KEY": "   ",
            "OPENALEX_MAILTO": "",
            "SERPAPI_API_KEY": "   ",
            "OPENAI_API_KEY": "  ",
            "SCHOLAR_SEARCH_HTTP_AUTH_TOKEN": "",
        }
    )

    assert settings.core_api_key is None
    assert settings.openai_api_key is None
    assert settings.semantic_scholar_api_key is None
    assert settings.openalex_api_key is None
    assert settings.openalex_mailto is None
    assert settings.serpapi_api_key is None
    assert settings.http_auth_token is None


def test_app_settings_transport_defaults_to_stdio() -> None:
    settings = AppSettings.from_env({})

    assert settings.transport == "stdio"
    assert settings.http_host == "127.0.0.1"
    assert settings.http_port == 8000
    assert settings.http_path == "/mcp"


def test_app_settings_parses_http_transport_configuration() -> None:
    settings = AppSettings.from_env(
        {
            "SCHOLAR_SEARCH_TRANSPORT": "streamable-http",
            "SCHOLAR_SEARCH_HTTP_HOST": "0.0.0.0",
            "SCHOLAR_SEARCH_HTTP_PORT": "9000",
            "SCHOLAR_SEARCH_HTTP_PATH": "/api/mcp",
        }
    )

    assert settings.transport == "streamable-http"
    assert settings.http_host == "0.0.0.0"
    assert settings.http_port == 9000
    assert settings.http_path == "/api/mcp"


def test_app_settings_parses_agentic_configuration() -> None:
    settings = AppSettings.from_env(
        {
            "OPENAI_API_KEY": "sk-test",
            "SCHOLAR_SEARCH_ENABLE_AGENTIC": "true",
            "SCHOLAR_SEARCH_AGENTIC_PROVIDER": "openai",
            "SCHOLAR_SEARCH_PLANNER_MODEL": "gpt-5.2-mini",
            "SCHOLAR_SEARCH_SYNTHESIS_MODEL": "gpt-5.2",
            "SCHOLAR_SEARCH_EMBEDDING_MODEL": "text-embedding-3-large",
            "SCHOLAR_SEARCH_AGENTIC_INDEX_BACKEND": "memory",
            "SCHOLAR_SEARCH_SESSION_TTL_SECONDS": "900",
            "SCHOLAR_SEARCH_ENABLE_AGENTIC_TRACE_LOG": "true",
        }
    )

    assert settings.openai_api_key == "sk-test"
    assert settings.enable_agentic is True
    assert settings.agentic_provider == "openai"
    assert settings.planner_model == "gpt-5.2-mini"
    assert settings.synthesis_model == "gpt-5.2"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.agentic_index_backend == "memory"
    assert settings.session_ttl_seconds == 900
    assert settings.enable_agentic_trace_log is True


def test_app_settings_ignores_unrelated_azure_workflow_metadata() -> None:
    settings = AppSettings.from_env(
        {
            "AZURE_CLIENT_ID": "client-id",
            "AZURE_TENANT_ID": "tenant-id",
            "AZURE_SUBSCRIPTION_ID": "subscription-id",
            "AZURE_RESOURCE_GROUP": "rg-scholar-search-dev-01",
            "ACR_NAME": "acrscholarsearchdev",
            "IMAGE_REPOSITORY": "scholar-search-mcp",
            "AZURE_PRIVATE_RUNNER_LABELS_JSON": '["self-hosted","linux"]',
        }
    )

    assert settings.transport == "stdio"
    assert settings.enable_core is False
    assert settings.enable_semantic_scholar is True
    assert settings.enable_serpapi is False
    assert settings.http_auth_token is None


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_env_bool_parses_common_false_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SCHOLAR_TEST_BOOL", value)

    assert server._env_bool("SCHOLAR_TEST_BOOL", True) is False


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_env_bool_treats_other_common_values_as_true(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SCHOLAR_TEST_BOOL", value)

    assert server._env_bool("SCHOLAR_TEST_BOOL", False) is True


def test_env_bool_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCHOLAR_TEST_BOOL", raising=False)

    assert server._env_bool("SCHOLAR_TEST_BOOL", True) is True
    assert server._env_bool("SCHOLAR_TEST_BOOL", False) is False
