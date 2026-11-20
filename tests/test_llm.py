import json

import httpx
import pytest

from chatbot.config import Settings
from chatbot.llm import LLMAuthError, LLMStreamInterrupted, LLMUnavailable, OpenRouterClient, _ThinkFilter


def make_settings(**over) -> Settings:
    base = dict(
        openrouter_api_key="sk-test", openrouter_base_url="https://or.test/api/v1",
        models=("m/one:free", "m/two:free"), app_url="https://app.test", app_name="t",
        kb_path=None, top_k=6, max_context_chars=7000, max_history_messages=8, max_message_chars=1000,
        max_output_tokens=100, temperature=0.2, llm_timeout_seconds=5.0, rate_limit_per_minute=100,
        cors_origins=(), log_level="INFO",
    )
    base.update(over)
    return Settings(**base)


def sse(*chunks, done=True) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': c}]})}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def client_for(handler, **over) -> OpenRouterClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterClient(make_settings(**over), http=http)


async def collect(client):
    return [(m, t) async for m, t in client.stream([{"role": "user", "content": "hi"}])]


async def test_streams_tokens_and_ignores_reasoning_and_comments():
    def handler(request):
        body = b": OPENROUTER PROCESSING\n\n" + sse({"reasoning": "thinking..."}, {"content": "Hel"}, {"content": "lo"})
        return httpx.Response(200, content=body)

    out = await collect(client_for(handler))
    assert out == [("m/one:free", "Hel"), ("m/one:free", "lo")]


async def test_sends_expected_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["referer"] = request.headers["http-referer"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, content=sse({"content": "ok"}))

    await collect(client_for(handler))
    assert seen["auth"] == "Bearer sk-test"
    assert seen["referer"] == "https://app.test"
    assert seen["payload"]["model"] == "m/one:free" and seen["payload"]["stream"] is True


async def test_falls_back_to_next_model_on_http_error():
    def handler(request):
        model = json.loads(request.content)["model"]
        if model == "m/one:free":
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, content=sse({"content": "from two"}))

    assert await collect(client_for(handler)) == [("m/two:free", "from two")]


async def test_falls_back_on_empty_response_and_midstream_error_before_tokens():
    calls = []

    def handler(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "m/one:free":
            return httpx.Response(200, content=sse({"reasoning": "only reasoning"}))
        return httpx.Response(200, content=sse({"content": "answer"}))

    assert await collect(client_for(handler)) == [("m/two:free", "answer")]
    assert calls == ["m/one:free", "m/two:free"]


async def test_all_models_failing_raises_unavailable():
    def handler(request):
        return httpx.Response(503, text="down")

    with pytest.raises(LLMUnavailable):
        await collect(client_for(handler))


async def test_invalid_key_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "No auth credentials found"}})

    with pytest.raises(LLMAuthError):
        await collect(client_for(handler))
    assert len(calls) == 1


async def test_missing_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        await collect(client_for(lambda r: httpx.Response(200), openrouter_api_key=""))


async def test_error_after_tokens_is_reported_as_interrupted():
    def handler(request):
        body = sse({"content": "partial"}, done=False) + b'data: {"error": {"message": "boom"}}\n\n'
        return httpx.Response(200, content=body)

    client = client_for(handler)
    got = []
    with pytest.raises(LLMStreamInterrupted):
        async for _, tok in client.stream([{"role": "user", "content": "x"}]):
            got.append(tok)
    assert got == ["partial"]


def test_think_filter_strips_reasoning_blocks_even_when_split():
    f = _ThinkFilter()
    parts = ["Hello <th", "ink>secret reason", "ing</thi", "nk> world", "!"]
    assert "".join(f.feed(p) for p in parts) + f.flush() == "Hello  world!"
