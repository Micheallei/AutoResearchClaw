"""Azure OpenAI client using Azure AD bearer tokens (via az login / Azure CLI)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from researchclaw.llm.client import LLMResponse, _NEW_PARAM_MODELS

logger = logging.getLogger(__name__)


@dataclass
class AzureOpenAIAADConfig:
    azure_endpoint: str
    api_version: str = "2024-12-01-preview"
    token_scope: str = "https://cognitiveservices.azure.com/.default"
    credential: str = "azure-cli"
    primary_model: str = "gpt-4o"
    fallback_models: list[str] | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_sec: int = 300
    max_retries: int = 5
    retry_delay_sec: int = 60


class AzureOpenAIAADClient:
    """LLM client compatible with LLMClient, backed by Azure AD auth."""

    def __init__(self, config: AzureOpenAIAADConfig) -> None:
        self.config = config
        self._model_chain = [config.primary_model] + list(config.fallback_models or [])
        self._client: AzureOpenAI | None = None

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> "AzureOpenAIAADClient":
        llm = rc_config.llm
        return cls(
            AzureOpenAIAADConfig(
                azure_endpoint=llm.azure_endpoint,
                api_version=llm.azure_api_version,
                token_scope=llm.azure_token_scope,
                credential=llm.azure_credential,
                primary_model=llm.primary_model or "gpt-4o",
                fallback_models=list(llm.fallback_models or []),
            )
        )

    def _build_credential(self):
        kind = (self.config.credential or "azure-cli").strip().lower()
        if kind == "azure-cli":
            return AzureCliCredential()
        if kind == "default":
            return DefaultAzureCredential()
        raise ValueError(f"Unsupported azure credential mode: {self.config.credential}")

    def _ensure_client(self) -> AzureOpenAI:
        if self._client is not None:
            return self._client
        credential = self._build_credential()
        token_provider = get_bearer_token_provider(credential, self.config.token_scope)
        self._client = AzureOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.config.api_version,
            timeout=self.config.timeout_sec,
        )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
    ) -> LLMResponse:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        models = [model] if model else self._model_chain
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        last_error: Exception | None = None
        client = self._ensure_client()
        extra_args: dict[str, Any] = {}
        if json_mode:
            extra_args["response_format"] = {"type": "json_object"}

        for m in models:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    request_args: dict[str, Any] = {
                        'model': m,
                        'messages': messages,
                        'temperature': temp,
                        **extra_args,
                    }
                    if any(m.startswith(prefix) for prefix in _NEW_PARAM_MODELS):
                        request_args['max_completion_tokens'] = max(max_tok, 64)
                    else:
                        request_args['max_tokens'] = max_tok
                    resp = client.chat.completions.create(**request_args)
                    choice = resp.choices[0]
                    content = choice.message.content or ""
                    usage = resp.usage
                    return LLMResponse(
                        content=content,
                        model=getattr(resp, "model", m) or m,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        finish_reason=getattr(choice, "finish_reason", "") or "",
                        truncated=(getattr(choice, "finish_reason", "") == "length"),
                        raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < self.config.max_retries:
                        logger.warning(
                            "Azure model %s failed on attempt %d/%d: %s. Retrying in %ds.",
                            m,
                            attempt,
                            self.config.max_retries,
                            exc,
                            self.config.retry_delay_sec,
                        )
                        time.sleep(self.config.retry_delay_sec)
                    else:
                        logger.warning(
                            "Azure model %s failed after %d attempts: %s. Trying next model.",
                            m,
                            self.config.max_retries,
                            exc,
                        )

        raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error

    def preflight(self) -> tuple[bool, str]:
        if not self.config.azure_endpoint.strip():
            return False, "Missing llm.azure_endpoint"
        try:
            _ = self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=16,
                temperature=0,
            )
            return True, f"OK - Azure OpenAI responding ({self.config.primary_model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"Azure OpenAI preflight failed: {exc}"
