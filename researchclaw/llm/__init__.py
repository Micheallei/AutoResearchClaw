"""LLM integration — OpenAI-compatible, Azure OpenAI AAD, OpenRouter, and ACP agent clients."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from researchclaw.config import RCConfig
    from researchclaw.llm.acp_client import ACPClient
    from researchclaw.llm.azure_openai_aad_client import AzureOpenAIAADClient
    from researchclaw.llm.client import LLMClient

# Provider presets for common LLM services
PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
    },
    "openai-compatible": {
        "base_url": None,  # Use user-provided base_url
    },
}


def create_llm_client(config: "RCConfig") -> "LLMClient | ACPClient | AzureOpenAIAADClient":
    """Factory: return the right LLM client based on ``config.llm.provider``.

    Supported providers:
    - ``"acp"`` → ACP-compatible local agent
    - ``"azure-openai-aad"`` → Azure OpenAI via Azure AD bearer tokens (e.g. ``az login``)
    - ``"openrouter"`` → OpenRouter base URL preset
    - ``"openai"`` → OpenAI base URL preset
    - ``"deepseek"`` → DeepSeek base URL preset
    - ``"openai-compatible"`` → user-provided OpenAI-compatible endpoint
    """
    if config.llm.provider == "acp":
        from researchclaw.llm.acp_client import ACPClient as _ACP

        return _ACP.from_rc_config(config)

    if config.llm.provider == "azure-openai-aad":
        from researchclaw.llm.azure_openai_aad_client import AzureOpenAIAADClient as _AZURE_AAD

        return _AZURE_AAD.from_rc_config(config)

    from researchclaw.llm.client import LLMClient as _LLM
    from researchclaw.llm.client import LLMConfig

    preset = PROVIDER_PRESETS.get(config.llm.provider, {})
    preset_base_url = preset.get("base_url")
    base_url = preset_base_url if preset_base_url else config.llm.base_url

    return _LLM(
        LLMConfig(
            base_url=base_url,
            api_key=(
                config.llm.api_key
                or os.environ.get(config.llm.api_key_env, "")
                or ""
            ),
            primary_model=config.llm.primary_model or "gpt-4o",
            fallback_models=list(config.llm.fallback_models or []),
        )
    )
