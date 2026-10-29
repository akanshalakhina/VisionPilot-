"""A provider-neutral LLM client.

Any service that speaks the OpenAI chat-completions format works: OpenRouter, Groq,
Google's OpenAI-compatible Gemini endpoint, a local Ollama server, and others. You
bring your OWN key. Nothing here ships a key, and the key never appears in logs or reprs.

Not yet tested against a live provider. The request and response handling is covered by
unit tests with a mock transport. Check your provider's current model names and URLs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx

PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}


class LLMError(RuntimeError):
    """The LLM could not be reached or returned something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(base_url={self.base_url!r}, model={self.model!r})"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            response = self._http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM request failed with HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from None
        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError):
            raise LLMError("the LLM returned an unexpected response shape") from None

        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (ValueError, TypeError):
                args = {"_unparsed_arguments": str(raw_args)[:200]}
            if not isinstance(args, dict):
                args = {"_unparsed_arguments": str(args)[:200]}
            calls.append(
                ToolCall(id=raw_call.get("id") or f"call_{len(calls)}", name=function.get("name", ""), arguments=args)
            )
        return LLMResponse(content=message.get("content"), tool_calls=calls)


def build_llm(settings) -> OpenAICompatibleClient | None:
    """Return a client for the configured provider, or None for the offline rule-based planner."""
    provider = settings.llm_provider
    if provider in ("", "none"):
        return None
    base_url = settings.llm_base_url or PROVIDER_URLS.get(provider, "")
    if not base_url:
        raise ValueError(f"unknown LLM_PROVIDER '{provider}'. Use one of {sorted(PROVIDER_URLS)} or set LLM_BASE_URL")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL must be set when LLM_PROVIDER is set (model names change, so none is hardcoded)")
    if provider != "ollama" and not settings.llm_api_key:
        raise ValueError("LLM_API_KEY must be set. Use your own key, never someone else's")
    return OpenAICompatibleClient(base_url, settings.llm_api_key, settings.llm_model)
