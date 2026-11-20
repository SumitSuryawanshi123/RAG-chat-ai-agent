"""Streaming OpenRouter client with ordered fallback across free models."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger("chatbot.llm")


class LLMError(Exception):
    """A single model failed in a way that another model might survive."""


class LLMAuthError(LLMError):
    """The API key is missing/invalid - falling back to other models won't help."""


class LLMUnavailable(LLMError):
    """Every configured model failed before producing output."""


class LLMStreamInterrupted(LLMError):
    """A model died after it had already streamed some of its answer."""


class _ThinkFilter:
    """Drop ``<think>...</think>`` blocks that some reasoning models put in content."""

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end == -1:
                    self.buf = self.buf[-8:]  # keep a tail in case the tag is split
                    break
                self.buf = self.buf[end + len("</think>"):]
                self.inside = False
            else:
                start = self.buf.find("<think>")
                if start == -1:
                    # Hold back a possible partial "<think>" prefix.
                    keep = 0
                    for i in range(1, min(len("<think>"), len(self.buf)) + 1):
                        if "<think>".startswith(self.buf[-i:]):
                            keep = i
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                out.append(self.buf[:start])
                self.buf = self.buf[start + len("<think>"):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


class OpenRouterClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self.settings = settings
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=settings.llm_timeout_seconds, write=10.0, pool=10.0)
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    async def aclose(self) -> None:
        await self._http.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.app_url,
            "X-Title": self.settings.app_name,
        }

    async def stream(self, messages: list[dict]) -> AsyncIterator[tuple[str, str]]:
        """Yield ``(model, token)`` pairs, trying each configured model in turn."""
        if not self.configured:
            raise LLMAuthError("OPENROUTER_API_KEY is not set")
        failures: list[str] = []
        for model in self.settings.models:
            emitted = False
            try:
                async for token in self._stream_one(model, messages):
                    emitted = True
                    yield model, token
                if emitted:
                    return
                failures.append(f"{model}: empty response")
                log.warning("Model %s returned an empty response, trying next", model)
            except LLMAuthError:
                raise
            except (LLMError, httpx.HTTPError) as exc:
                if emitted:
                    raise LLMStreamInterrupted(f"{model}: {exc}") from exc
                failures.append(f"{model}: {exc}")
                log.warning("Model %s failed (%s), trying next", model, exc)
        raise LLMUnavailable("; ".join(failures) or "no models configured")

    async def _stream_one(self, model: str, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
        }
        url = f"{self.settings.openrouter_base_url}/chat/completions"
        think = _ThinkFilter()
        async with self._http.stream("POST", url, headers=self._headers(), json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                detail = _error_message(body) or f"HTTP {resp.status_code}"
                if resp.status_code == 401:
                    raise LLMAuthError(f"OpenRouter rejected the API key: {detail}")
                raise LLMError(f"HTTP {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):  # SSE comments / keep-alives
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    err = obj["error"]
                    raise LLMError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content")  # "reasoning" deltas are intentionally ignored
                    if text:
                        text = think.feed(text)
                        if text:
                            yield text
                    if choice.get("finish_reason") == "error":
                        raise LLMError("model reported finish_reason=error")
        tail = think.flush()
        if tail:
            yield tail


def _error_message(body: str) -> str:
    try:
        err = json.loads(body).get("error", {})
        return err.get("message", "") if isinstance(err, dict) else str(err)
    except (json.JSONDecodeError, AttributeError):
        return body[:200]
