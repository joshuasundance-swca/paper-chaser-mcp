"""LangChain-based provider adapters (Anthropic, Google, NVIDIA, Mistral, HuggingFace, OpenRouter).

Extracted from ``paper_chaser_mcp.agentic.provider_langchain`` during
Phase 8d of the modularization refactor. Each adapter is a thin subclass
of :class:`LangChainChatProviderBundle` that wires a provider-specific
LangChain chat model into the shared structured-output machinery.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from ....provider_runtime import ProviderDiagnosticsRegistry
from ...config import AgenticConfig
from .bundle import LangChainChatProviderBundle

__all__ = [
    "AnthropicProviderBundle",
    "GoogleProviderBundle",
    "HuggingFaceProviderBundle",
    "MistralProviderBundle",
    "NvidiaProviderBundle",
    "OpenRouterProviderBundle",
]


class AnthropicProviderBundle(LangChainChatProviderBundle):
    """Anthropic smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="anthropic",
            api_key=api_key,
            provider_registry=provider_registry,
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_anthropic import ChatAnthropic

        chat_anthropic: Any = ChatAnthropic
        kwargs = {
            "model_name": model_name,
            "api_key": SecretStr(self._api_key or ""),
            "stop": None,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        return chat_anthropic(**kwargs)


class GoogleProviderBundle(LangChainChatProviderBundle):
    """Google Gemini smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="google",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self._api_key,
            temperature=0,
        )


class NvidiaProviderBundle(LangChainChatProviderBundle):
    """NVIDIA NIM smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="nvidia",
            api_key=api_key,
            provider_registry=provider_registry,
        )
        self._base_url = base_url

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": self._api_key,
            "temperature": 0,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        model = ChatNVIDIA(**kwargs)
        client = getattr(model, "_client", None)
        if client is not None and hasattr(client, "timeout"):
            client.timeout = self._timeout_seconds
        return model


class MistralProviderBundle(LangChainChatProviderBundle):
    """Mistral smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="mistral",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model_name=model_name,
            api_key=SecretStr(self._api_key or ""),
            temperature=0,
            max_retries=0,
            timeout=int(self._timeout_seconds),
        )


class HuggingFaceProviderBundle(LangChainChatProviderBundle):
    """Hugging Face router smart-layer adapter via OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str = "https://router.huggingface.co/v1",
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="huggingface",
            api_key=api_key,
            provider_registry=provider_registry,
        )
        self._base_url = base_url

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self._api_key or ""),
            base_url=self._base_url,
            temperature=0,
            max_retries=0,
            timeout=self._timeout_seconds,
        )


class OpenRouterProviderBundle(LangChainChatProviderBundle):
    """OpenRouter smart-layer adapter via OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        title: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="openrouter",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )
        self._base_url = base_url
        self._http_referer = (http_referer or "").strip() or None
        self._title = (title or "").strip() or None

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_openai import ChatOpenAI

        default_headers: dict[str, str] = {}
        if self._http_referer:
            default_headers["HTTP-Referer"] = self._http_referer
        if self._title:
            default_headers["X-OpenRouter-Title"] = self._title

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": SecretStr(self._api_key or ""),
            "base_url": self._base_url,
            "temperature": 0,
            "max_retries": 0,
            "timeout": self._timeout_seconds,
            "extra_body": {"provider": {"require_parameters": True}},
        }
        if default_headers:
            kwargs["default_headers"] = default_headers
        return ChatOpenAI(**kwargs)
