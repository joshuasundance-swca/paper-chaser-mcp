import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.runtime import run_server
from paper_chaser_mcp.settings import AppSettings


def test_app_settings_reads_provider_order_from_env() -> None:
    settings = AppSettings.from_env({"PAPER_CHASER_PROVIDER_ORDER": "semantic_scholar,arxiv"})

    assert settings.provider_order == ("semantic_scholar", "arxiv")


def test_app_settings_accepts_serpapi_alias_in_provider_order() -> None:
    settings = AppSettings.from_env({"PAPER_CHASER_PROVIDER_ORDER": "semantic_scholar,serpapi,arxiv"})

    assert settings.provider_order == (
        "semantic_scholar",
        "serpapi_google_scholar",
        "arxiv",
    )


def test_app_settings_rejects_duplicate_provider_order_entries() -> None:
    with pytest.raises(ValueError, match="cannot repeat providers"):
        AppSettings.from_env({"PAPER_CHASER_PROVIDER_ORDER": "core,core,semantic_scholar"})


def test_search_papers_args_accept_serpapi_alias() -> None:
    from paper_chaser_mcp.models.tools import SearchPapersArgs

    args = SearchPapersArgs.model_validate(
        {
            "query": "fallback",
            "preferredProvider": "serpapi",
            "providerOrder": ["core", "serpapi"],
        }
    )

    assert args.preferred_provider == "serpapi_google_scholar"
    assert args.provider_order == ["core", "serpapi_google_scholar"]


def test_search_papers_args_accept_scholarapi_alias() -> None:
    from paper_chaser_mcp.models.tools import SearchPapersArgs

    args = SearchPapersArgs.model_validate(
        {
            "query": "fallback",
            "preferredProvider": "scholarapi",
            "providerOrder": ["semantic_scholar", "scholarapi"],
        }
    )

    assert args.preferred_provider == "scholarapi"
    assert args.provider_order == ["semantic_scholar", "scholarapi"]


def test_app_settings_serpapi_disabled_by_default() -> None:
    """SerpApi must be disabled by default to protect users from surprise costs."""

    settings = AppSettings.from_env({})  # empty env
    assert settings.enable_serpapi is False
    assert settings.enable_scholarapi is False
    assert settings.enable_agentic is False
    assert settings.openai_api_key is None
    assert settings.serpapi_api_key is None
    assert settings.scholarapi_api_key is None
    assert settings.enable_openalex is True
    assert settings.openalex_api_key is None
    assert settings.openalex_mailto is None
    assert settings.govinfo_api_key is None
    assert settings.enable_crossref is True
    assert settings.enable_unpaywall is True
    assert settings.enable_ecos is True
    assert settings.enable_federal_register is True
    assert settings.enable_govinfo_cfr is True
    assert settings.ecos_base_url == "https://ecos.fws.gov"
    assert settings.crossref_mailto is None
    assert settings.unpaywall_email is None
    assert settings.disable_embeddings is True
    assert settings.agentic_openai_timeout_seconds == 30.0
    assert settings.crossref_timeout_seconds == 30.0
    assert settings.unpaywall_timeout_seconds == 30.0
    assert settings.ecos_timeout_seconds == 30.0
    assert settings.federal_register_timeout_seconds == 30.0
    assert settings.govinfo_timeout_seconds == 30.0
    assert settings.govinfo_document_timeout_seconds == 60.0
    assert settings.govinfo_max_document_size_mb == 25
    assert settings.ecos_document_timeout_seconds == 60.0
    assert settings.ecos_document_conversion_timeout_seconds == 60.0
    assert settings.ecos_max_document_size_mb == 25
    assert settings.ecos_verify_tls is True
    assert settings.ecos_ca_bundle is None


def test_app_settings_serpapi_enabled_via_env() -> None:
    """PAPER_CHASER_ENABLE_SERPAPI=true must enable the provider."""

    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_SERPAPI": "true",
            "SERPAPI_API_KEY": "my-api-key",
        }
    )
    assert settings.enable_serpapi is True
    assert settings.serpapi_api_key == "my-api-key"


def test_app_settings_scholarapi_enabled_via_env() -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_SCHOLARAPI": "true",
            "SCHOLARAPI_API_KEY": "sch-test-key",
        }
    )

    assert settings.enable_scholarapi is True
    assert settings.scholarapi_api_key == "sch-test-key"


def test_app_settings_openalex_enabled_with_optional_mailto_and_key() -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_OPENALEX": "true",
            "OPENALEX_API_KEY": "openalex-key",
            "OPENALEX_MAILTO": "team@example.com",
        }
    )

    assert settings.enable_openalex is True
    assert settings.openalex_api_key == "openalex-key"
    assert settings.openalex_mailto == "team@example.com"


def test_app_settings_parses_crossref_and_unpaywall_configuration() -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_CROSSREF": "true",
            "CROSSREF_MAILTO": "ops@example.com",
            "CROSSREF_TIMEOUT_SECONDS": "12.5",
            "PAPER_CHASER_ENABLE_UNPAYWALL": "true",
            "UNPAYWALL_EMAIL": "oa@example.com",
            "UNPAYWALL_TIMEOUT_SECONDS": "9",
        }
    )

    assert settings.enable_crossref is True
    assert settings.crossref_mailto == "ops@example.com"
    assert settings.crossref_timeout_seconds == 12.5
    assert settings.enable_unpaywall is True
    assert settings.unpaywall_email == "oa@example.com"
    assert settings.unpaywall_timeout_seconds == 9.0


def test_app_settings_parses_ecos_configuration() -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_ECOS": "true",
            "ECOS_BASE_URL": "https://ecos.fws.gov",
            "ECOS_TIMEOUT_SECONDS": "12",
            "ECOS_DOCUMENT_TIMEOUT_SECONDS": "75",
            "ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS": "22",
            "ECOS_MAX_DOCUMENT_SIZE_MB": "40",
            "ECOS_VERIFY_TLS": "false",
            "ECOS_CA_BUNDLE": "C:/certs/ecos-ca.pem",
        }
    )

    assert settings.enable_ecos is True
    assert settings.ecos_base_url == "https://ecos.fws.gov"
    assert settings.ecos_timeout_seconds == 12.0
    assert settings.ecos_document_timeout_seconds == 75.0
    assert settings.ecos_document_conversion_timeout_seconds == 22.0
    assert settings.ecos_max_document_size_mb == 40
    assert settings.ecos_verify_tls is False
    assert settings.ecos_ca_bundle == "C:/certs/ecos-ca.pem"


def test_app_settings_parses_regulatory_configuration() -> None:
    settings = AppSettings.from_env(
        {
            "GOVINFO_API_KEY": "gov-key",
            "PAPER_CHASER_ENABLE_FEDERAL_REGISTER": "true",
            "PAPER_CHASER_ENABLE_GOVINFO_CFR": "true",
            "FEDERAL_REGISTER_TIMEOUT_SECONDS": "11",
            "GOVINFO_TIMEOUT_SECONDS": "14",
            "GOVINFO_DOCUMENT_TIMEOUT_SECONDS": "25",
            "GOVINFO_MAX_DOCUMENT_SIZE_MB": "18",
        }
    )

    assert settings.govinfo_api_key == "gov-key"
    assert settings.enable_federal_register is True
    assert settings.enable_govinfo_cfr is True
    assert settings.federal_register_timeout_seconds == 11.0
    assert settings.govinfo_timeout_seconds == 14.0
    assert settings.govinfo_document_timeout_seconds == 25.0
    assert settings.govinfo_max_document_size_mb == 18


def test_app_settings_normalizes_blank_optional_values_to_none() -> None:
    settings = AppSettings.from_env(
        {
            "CORE_API_KEY": "   ",
            "SEMANTIC_SCHOLAR_API_KEY": "",
            "AZURE_OPENAI_API_KEY": " ",
            "AZURE_OPENAI_ENDPOINT": " ",
            "AZURE_OPENAI_API_VERSION": " ",
            "AZURE_OPENAI_PLANNER_DEPLOYMENT": " ",
            "AZURE_OPENAI_SYNTHESIS_DEPLOYMENT": " ",
            "ANTHROPIC_API_KEY": " ",
            "GOOGLE_API_KEY": " ",
            "OPENALEX_API_KEY": "   ",
            "OPENALEX_MAILTO": "",
            "SERPAPI_API_KEY": "   ",
            "SCHOLARAPI_API_KEY": "   ",
            "GOVINFO_API_KEY": " ",
            "CROSSREF_MAILTO": " ",
            "UNPAYWALL_EMAIL": " ",
            "OPENAI_API_KEY": "  ",
            "PAPER_CHASER_HTTP_AUTH_TOKEN": "",
        }
    )

    assert settings.core_api_key is None
    assert settings.openai_api_key is None
    assert settings.azure_openai_api_key is None
    assert settings.azure_openai_endpoint is None
    assert settings.azure_openai_api_version is None
    assert settings.azure_openai_planner_deployment is None
    assert settings.azure_openai_synthesis_deployment is None
    assert settings.anthropic_api_key is None
    assert settings.google_api_key is None
    assert settings.semantic_scholar_api_key is None
    assert settings.openalex_api_key is None
    assert settings.openalex_mailto is None
    assert settings.serpapi_api_key is None
    assert settings.scholarapi_api_key is None
    assert settings.govinfo_api_key is None
    assert settings.crossref_mailto is None
    assert settings.unpaywall_email is None
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
            "PAPER_CHASER_TRANSPORT": "streamable-http",
            "PAPER_CHASER_HTTP_HOST": "0.0.0.0",
            "PAPER_CHASER_HTTP_PORT": "9000",
            "PAPER_CHASER_HTTP_PATH": "/api/mcp",
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
            "PAPER_CHASER_ENABLE_AGENTIC": "true",
            "PAPER_CHASER_AGENTIC_PROVIDER": "openai",
            "PAPER_CHASER_PLANNER_MODEL": "gpt-5.4-mini",
            "PAPER_CHASER_SYNTHESIS_MODEL": "gpt-5.4",
            "PAPER_CHASER_EMBEDDING_MODEL": "text-embedding-3-large",
            "PAPER_CHASER_DISABLE_EMBEDDINGS": "true",
            "PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS": "18",
            "PAPER_CHASER_AGENTIC_INDEX_BACKEND": "memory",
            "PAPER_CHASER_SESSION_TTL_SECONDS": "900",
            "PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG": "true",
        }
    )

    assert settings.openai_api_key == "sk-test"
    assert settings.enable_agentic is True
    assert settings.agentic_provider == "openai"
    assert settings.planner_model == "gpt-5.4-mini"
    assert settings.synthesis_model == "gpt-5.4"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.disable_embeddings is True
    assert settings.agentic_openai_timeout_seconds == 18.0
    assert settings.agentic_index_backend == "memory"
    assert settings.session_ttl_seconds == 900
    assert settings.enable_agentic_trace_log is True


@pytest.mark.parametrize(
    ("provider", "env", "field_name", "field_value"),
    [
        (
            "azure-openai",
            {
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
                "AZURE_OPENAI_API_VERSION": "2024-10-21",
                "AZURE_OPENAI_PLANNER_DEPLOYMENT": "azure-planner",
                "AZURE_OPENAI_SYNTHESIS_DEPLOYMENT": "azure-synthesis",
            },
            "azure_openai_api_key",
            "azure-key",
        ),
        (
            "anthropic",
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
            "anthropic_api_key",
            "sk-ant-test",
        ),
        (
            "google",
            {"GOOGLE_API_KEY": "google-key"},
            "google_api_key",
            "google-key",
        ),
    ],
)
def test_app_settings_parses_additional_agentic_providers(
    provider: str,
    env: dict[str, str],
    field_name: str,
    field_value: str,
) -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_AGENTIC": "true",
            "PAPER_CHASER_AGENTIC_PROVIDER": provider,
            **env,
        }
    )

    assert settings.agentic_provider == provider
    assert getattr(settings, field_name) == field_value
    if provider == "azure-openai":
        assert settings.azure_openai_planner_deployment == "azure-planner"
        assert settings.azure_openai_synthesis_deployment == "azure-synthesis"
    assert settings.disable_embeddings is True


def test_app_settings_rejects_invalid_agentic_provider() -> None:
    with pytest.raises(ValueError, match="PAPER_CHASER_AGENTIC_PROVIDER"):
        AppSettings.from_env({"PAPER_CHASER_AGENTIC_PROVIDER": "unsupported"})


def test_agentic_config_tracks_azure_openai_model_sources() -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_AGENTIC": "true",
            "PAPER_CHASER_AGENTIC_PROVIDER": "azure-openai",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_PLANNER_DEPLOYMENT": "planner-deployment",
            "AZURE_OPENAI_SYNTHESIS_DEPLOYMENT": "synthesis-deployment",
        }
    )

    config = AgenticConfig.from_settings(settings)

    assert config.planner_model == "planner-deployment"
    assert config.synthesis_model == "synthesis-deployment"
    assert config.planner_model_source == "azure_deployment"
    assert config.synthesis_model_source == "azure_deployment"


@pytest.mark.parametrize(
    ("provider", "api_key_env", "expected_model"),
    [
        ("anthropic", {"ANTHROPIC_API_KEY": "sk-ant-test"}, "claude-sonnet-4-5"),
        ("google", {"GOOGLE_API_KEY": "google-key"}, "gemini-2.5-flash"),
    ],
)
def test_agentic_config_tracks_provider_default_model_sources(
    provider: str,
    api_key_env: dict[str, str],
    expected_model: str,
) -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_AGENTIC": "true",
            "PAPER_CHASER_AGENTIC_PROVIDER": provider,
            **api_key_env,
        }
    )

    config = AgenticConfig.from_settings(settings)

    assert config.planner_model == expected_model
    assert config.synthesis_model == expected_model
    assert config.planner_model_source == "provider_default"
    assert config.synthesis_model_source == "provider_default"


def test_run_server_logs_embedding_flag(caplog: pytest.LogCaptureFixture) -> None:
    settings = AppSettings.from_env(
        {
            "PAPER_CHASER_ENABLE_AGENTIC": "true",
            "PAPER_CHASER_DISABLE_EMBEDDINGS": "true",
        }
    )

    class _App:
        def run(self, **kwargs: object) -> None:
            del kwargs

    caplog.set_level("INFO", logger="paper-chaser-mcp")

    run_server(app=_App(), logger=server.logger, settings=settings)

    messages = [record.getMessage() for record in caplog.records]
    assert any("embeddings=disabled" in message for message in messages)


def test_app_settings_ignores_unrelated_azure_workflow_metadata() -> None:
    settings = AppSettings.from_env(
        {
            "AZURE_CLIENT_ID": "client-id",
            "AZURE_TENANT_ID": "tenant-id",
            "AZURE_SUBSCRIPTION_ID": "subscription-id",
            "AZURE_RESOURCE_GROUP": "rg-paper-chaser-dev-01",
            "ACR_NAME": "acrscholarsearchdev",
            "IMAGE_REPOSITORY": "paper-chaser-mcp",
            "AZURE_PRIVATE_RUNNER_LABELS_JSON": '["self-hosted","linux"]',
        }
    )

    assert settings.transport == "stdio"
    assert settings.enable_core is False
    assert settings.enable_semantic_scholar is True
    assert settings.enable_serpapi is False
    assert settings.http_auth_token is None


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_env_bool_parses_common_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SCHOLAR_TEST_BOOL", value)

    assert server._env_bool("SCHOLAR_TEST_BOOL", True) is False


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_env_bool_treats_other_common_values_as_true(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SCHOLAR_TEST_BOOL", value)

    assert server._env_bool("SCHOLAR_TEST_BOOL", False) is True


def test_env_bool_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCHOLAR_TEST_BOOL", raising=False)

    assert server._env_bool("SCHOLAR_TEST_BOOL", True) is True
    assert server._env_bool("SCHOLAR_TEST_BOOL", False) is False
