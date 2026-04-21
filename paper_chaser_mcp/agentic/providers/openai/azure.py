"""Azure OpenAI provider bundle (``AzureOpenAIProviderBundle``).

Extracted from ``paper_chaser_mcp.agentic.provider_openai`` during
Phase 8c of the modularization refactor. Inherits the OpenAI-compatible
Responses API path from :mod:`.bundle` and only overrides the client
and LangChain model loaders.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import SecretStr

from ....provider_runtime import ProviderDiagnosticsRegistry
from ...config import AgenticConfig
from .bundle import OpenAIProviderBundle

logger = logging.getLogger("paper-chaser-mcp")

__all__ = ["AzureOpenAIProviderBundle"]


class AzureOpenAIProviderBundle(OpenAIProviderBundle):
    """Azure OpenAI adapter that reuses the OpenAI-compatible response path."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        azure_endpoint: str | None,
        api_version: str | None,
        *,
        azure_planner_deployment: str | None = None,
        azure_synthesis_deployment: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(config, api_key, provider_registry=provider_registry)
        self._provider_name = "azure-openai"
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._azure_planner_deployment = azure_planner_deployment
        self._azure_synthesis_deployment = azure_synthesis_deployment
        if azure_planner_deployment:
            self.planner_model_name = azure_planner_deployment
        if azure_synthesis_deployment:
            self.synthesis_model_name = azure_synthesis_deployment

    def is_available(self) -> bool:
        return bool(self._api_key and self._azure_endpoint)

    def supports_embeddings(self) -> bool:
        return (not self._disable_embeddings) and bool(self._api_key and self._azure_endpoint)

    def _allow_langchain_chat_fallback(self) -> bool:
        """Azure sync parity should prefer responses and then deterministic fallback."""
        return False

    def _load_openai_client(self) -> Any | None:
        if self._openai_client is not None:
            return self._openai_client
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from openai import AzureOpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to LangChain and deterministic smart-provider adapters.")
            return None
        self._openai_client = AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._openai_client

    def _load_async_openai_client(self) -> Any | None:
        if self._async_openai_client is not None:
            return self._async_openai_client
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to deterministic smart-provider adapters.")
            return None
        self._async_openai_client = AsyncAzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._async_openai_client

    def _load_models(self) -> tuple[Any | None, Any | None]:
        if self._planner is not None and self._synthesizer is not None:
            return self._planner, self._synthesizer
        if not self._api_key or not self._azure_endpoint:
            return None, None
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            logger.info("langchain-openai is not installed; falling back to deterministic smart planning.")
            return None, None

        azure_chat_model: Any = AzureChatOpenAI
        planner_kwargs = {
            "azure_endpoint": self._azure_endpoint,
            "api_key": SecretStr(self._api_key),
            "api_version": self._api_version,
            "azure_deployment": self.planner_model_name,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        synthesis_kwargs = {
            "azure_endpoint": self._azure_endpoint,
            "api_key": SecretStr(self._api_key),
            "api_version": self._api_version,
            "azure_deployment": self.synthesis_model_name,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        self._planner = azure_chat_model(**planner_kwargs)
        self._synthesizer = azure_chat_model(**synthesis_kwargs)
        return self._planner, self._synthesizer

    def _load_embeddings(self) -> Any | None:
        if self._embeddings is not None:
            return self._embeddings
        if self._disable_embeddings:
            return None
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError:
            logger.info(
                "langchain-openai is not installed; falling back to lexical similarity for Azure OpenAI smart ranking."
            )
            return None

        self._embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_model_name,
            azure_deployment=self.embedding_model_name,
            api_key=SecretStr(self._api_key),
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            max_retries=0,
            timeout=self._timeout_seconds,
        )
        return self._embeddings
