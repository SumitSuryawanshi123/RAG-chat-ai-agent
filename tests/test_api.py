import json

import pytest
from fastapi.testclient import TestClient

from chatbot import main
from chatbot.llm import LLMAuthError, LLMUnavailable
from chatbot.rate_limit import RateLimiter
from tests.test_llm import make_settings


class FakeLLM:
    configured = True

    def __init__(self, tokens=("Painite villas ", "are by Lamborghini [1]."), error=None):
        self.tokens, self.error, self.calls = tokens, error, []

    async def stream(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for t in self.tokens:
            yield "fake/model:free", t

    async def aclose(self):
        pass


@pytest.fixture
def make_client(kb_path, monkeypatch):
    def _make(llm=None, limit=100):
        settings = make_settings(kb_path=kb_path, rate_limit_per_minute=limit)
        for fn in (main.get_knowledge_base, main.get_rag, main.get_llm, main.get_rate_limiter):
            fn.cache_clear()  # drop services cached by other tests
        # Everything in chatbot.main resolves settings via this module-level name.
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        app = main.create_app()
        fake = llm or FakeLLM()
        limiter = RateLimiter(limit)
        app.dependency_overrides[main.get_llm] = lambda: fake
        app.dependency_overrides[main.get_rate_limiter] = lambda: limiter
        return TestClient(app), fake

    yield _make
    for fn in (main.get_knowledge_base, main.get_rag, main.get_llm, main.get_rate_limiter):
        fn.cache_clear()


def parse_sse(text):
    return [json.loads(f[5:]) for f in text.strip().split("\n\n") if f.startswith("data:")]


def test_health(make_client):
    client, _ = make_client()
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["knowledge_base"]["chunks"] == 4


def test_chat_streams_sources_tokens_done(make_client):
    client, fake = make_client()
    r = client.post("/api/chat", json={"message": "Which brand partners with DarGlobal on Painite villas?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["url"].endswith("/partners/lamborghini")
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Painite villas are by Lamborghini [1]."
    assert events[-1] == {"type": "done", "model": "fake/model:free"}
    prompt = fake.calls[0]
    assert prompt[0]["role"] == "system" and "Painite" in prompt[-1]["content"]


def test_chat_non_streaming_json(make_client):
    client, _ = make_client()
    r = client.post("/api/chat", json={"message": "How do Wasalt auctions work?", "stream": False})
    body = r.json()
    assert r.status_code == 200 and body["model"] == "fake/model:free" and body["sources"]


def test_history_is_forwarded_and_short_followups_use_previous_turn(make_client):
    client, fake = make_client()
    client.post("/api/chat", json={
        "message": "and the golf course?", "stream": False,
        "history": [{"role": "user", "content": "Tell me about AIDA villas"},
                    {"role": "assistant", "content": "AIDA is in Oman."}],
    })
    msgs = fake.calls[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "AIDA" in msgs[-1]["content"]  # retrieval used the previous user turn


def test_greeting_skips_llm_and_retrieval(make_client):
    client, fake = make_client()
    events = parse_sse(client.post("/api/chat", json={"message": "hello"}).text)
    assert events[0]["sources"] == [] and events[-1]["model"] == "canned"
    assert fake.calls == []


def test_unknown_topic_returns_no_context_reply_without_llm(make_client):
    client, fake = make_client()
    body = client.post("/api/chat", json={"message": "quantum chromodynamics lagrangian", "stream": False}).json()
    assert "couldn't find" in body["answer"] and fake.calls == []


@pytest.mark.parametrize("error", [LLMUnavailable("all down"), LLMAuthError("bad key")])
def test_llm_outage_falls_back_to_extractive_answer(make_client, error):
    client, _ = make_client(llm=FakeLLM(error=error))
    body = client.post("/api/chat", json={"message": "Tell me about AIDA Trump villas", "stream": False}).json()
    assert body["model"] == "extractive-fallback"
    assert "AIDA Trump Villas" in body["answer"]


def test_validation(make_client):
    client, _ = make_client()
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "history": [{"role": "system", "content": "x"}]}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_rate_limit(make_client):
    client, _ = make_client(limit=2)
    codes = [client.post("/api/chat", json={"message": "hello"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/api/chat", json={"message": "hello"})
    assert r.headers["retry-after"].isdigit()


def test_security_headers_and_ui(make_client):
    client, _ = make_client()
    r = client.get("/")
    assert r.status_code == 200 and "DarGlobal" in r.text
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/health").headers["cache-control"] == "no-store"
