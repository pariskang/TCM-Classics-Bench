"""Provider-agnostic LLM clients for the LLM-backed generators.

A client only needs to implement ``complete(system, prompt) -> str``.  Four
providers are bundled; each lazily imports its SDK so the rest of the package
runs without any of them installed:

    anthropic   Anthropic Messages API            ANTHROPIC_API_KEY
    azure       Azure OpenAI (chat completions)   AZURE_OPENAI_API_KEY / _ENDPOINT
    poe         Poe OpenAI-compatible endpoint    POE_API_KEY
    litellm     LiteLLM router (100+ providers)   provider-specific env vars

Use :func:`make_client` to construct one by name, or :func:`client_from_env`
to pick the provider from ``TCM_BENCH_LLM_PROVIDER``.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface the generators depend on."""

    model: str

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.0
    ) -> str: ...


def _require(env: str) -> str:
    val = os.environ.get(env)
    if not val:
        raise RuntimeError(f"environment variable {env} is required for this provider")
    return val


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None, **_):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from e
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system, prompt, *, max_tokens=2048, temperature=0.0) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


# --------------------------------------------------------------------------
# OpenAI-compatible chat (shared by Azure and Poe)
# --------------------------------------------------------------------------
def _chat_complete(client, model, system, prompt, max_tokens, temperature) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


class AzureOpenAIClient:
    """Azure OpenAI.  ``model`` is the *deployment* name."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        **_,
    ):
        try:
            from openai import AzureOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e
        self.model = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
        self._client = AzureOpenAI(
            api_key=api_key or _require("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or _require("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )

    def complete(self, system, prompt, *, max_tokens=2048, temperature=0.0) -> str:
        return _chat_complete(self._client, self.model, system, prompt, max_tokens, temperature)


class PoeClient:
    """Poe via its OpenAI-compatible endpoint (https://api.poe.com/v1)."""

    def __init__(
        self,
        model: str = "Claude-Sonnet-4",
        api_key: str | None = None,
        base_url: str = "https://api.poe.com/v1",
        **_,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e
        self.model = model
        self._client = OpenAI(api_key=api_key or _require("POE_API_KEY"), base_url=base_url)

    def complete(self, system, prompt, *, max_tokens=2048, temperature=0.0) -> str:
        return _chat_complete(self._client, self.model, system, prompt, max_tokens, temperature)


# --------------------------------------------------------------------------
# LiteLLM (universal router)
# --------------------------------------------------------------------------
class LiteLLMClient:
    """LiteLLM router; ``model`` uses LiteLLM naming, e.g. ``azure/gpt-4o``,
    ``anthropic/claude-opus-4-8``, ``gemini/gemini-1.5-pro``."""

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4-8",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install litellm") from e
        self._litellm = litellm
        self.model = model
        self._extra = {k: v for k, v in {"api_key": api_key, "api_base": api_base, **kwargs}.items() if v is not None}

    def complete(self, system, prompt, *, max_tokens=2048, temperature=0.0) -> str:
        resp = self._litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **self._extra,
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicClient,
    "azure": AzureOpenAIClient,
    "poe": PoeClient,
    "litellm": LiteLLMClient,
}


def make_client(provider: str = "anthropic", model: str | None = None, **kwargs) -> LLMClient:
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    cls = PROVIDERS[provider]
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)


def client_from_env() -> LLMClient:
    """Build a client from ``TCM_BENCH_LLM_PROVIDER`` / ``TCM_BENCH_LLM_MODEL``."""
    provider = os.environ.get("TCM_BENCH_LLM_PROVIDER", "anthropic")
    model = os.environ.get("TCM_BENCH_LLM_MODEL")
    return make_client(provider, model)
