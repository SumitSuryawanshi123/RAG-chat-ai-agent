"""FastAPI application: chat API (SSE streaming) + static single-page UI."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import ROOT, Settings, get_settings
from .llm import LLMAuthError, LLMError, LLMStreamInterrupted, LLMUnavailable, OpenRouterClient
from .rag import Prepared, RagService
from .rate_limit import RateLimiter
from .retrieval import KnowledgeBase

log = logging.getLogger("chatbot")
PUBLIC_DIR = ROOT / "public"


# --------------------------------------------------------------------- schemas
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)
    stream: bool = True


# ------------------------------------------------------------------- services
@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.from_jsonl(get_settings().kb_path)


@lru_cache(maxsize=1)
def get_rag() -> RagService:
    return RagService(get_knowledge_base(), get_settings())


@lru_cache(maxsize=1)
def get_llm() -> OpenRouterClient:
    return OpenRouterClient(get_settings())


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


def client_ip(request: Request) -> str:
    # Behind Vercel / a reverse proxy the real client is the first X-Forwarded-For hop.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------------ app
def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            kb = get_knowledge_base()
            log.info("Ready: %s", kb.stats())
        except FileNotFoundError as exc:  # keep /api/health informative instead of crashing
            log.error("%s", exc)
        if not settings.openrouter_api_key:
            log.warning("OPENROUTER_API_KEY is not set - chat will fall back to extractive answers")
        yield
        if get_llm.cache_info().currsize:
            await get_llm().aclose()

    app = FastAPI(title="DarGlobal x Wasalt RAG Chatbot", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def security_and_logging(request: Request, call_next):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path.startswith("/api/") and request.url.path != "/api/docs":
            response.headers["Cache-Control"] = "no-store"
        log.info("%s %s %s %.0fms rid=%s", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - started) * 1000, request_id)
        return response

    # ---------------------------------------------------------------- routes
    @app.get("/api/health")
    async def health(settings: Settings = Depends(get_settings)):
        try:
            kb_stats = get_knowledge_base().stats()
            status = "ok"
        except FileNotFoundError:
            kb_stats, status = None, "knowledge_base_missing"
        return {"status": status, "llm_configured": bool(settings.openrouter_api_key),
                "models": list(settings.models), "knowledge_base": kb_stats}

    @app.get("/api/info")
    async def info():
        """Lightweight stats for the UI footer."""
        try:
            return get_knowledge_base().stats()
        except FileNotFoundError:
            return {"chunks": 0, "chunks_per_site": {}, "pages_per_site": {}, "last_scraped": ""}

    @app.post("/api/chat")
    async def chat(
        body: ChatRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        limiter: RateLimiter = Depends(get_rate_limiter),
        llm: OpenRouterClient = Depends(get_llm),
    ):
        allowed, retry_after = limiter.check(client_ip(request))
        if not allowed:
            raise HTTPException(429, "Too many requests - please slow down.",
                                headers={"Retry-After": str(retry_after)})

        message = body.message.strip()
        if not message:
            raise HTTPException(422, "Message must not be empty.")
        if len(message) > settings.max_message_chars:
            raise HTTPException(422, f"Message is too long (max {settings.max_message_chars} characters).")

        history = [{"role": m.role, "content": m.content[: settings.max_message_chars * 3]}
                   for m in body.history]
        try:
            prepared = get_rag().prepare(message, history)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

        events = _answer_events(prepared, llm, get_rag())
        if body.stream:
            return StreamingResponse(
                _sse_stream(events),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        return await _collect(events)

    # ------------------------------------------------------------ static UI
    @app.get("/", include_in_schema=False)
    async def index():
        page = PUBLIC_DIR / "index.html"
        if not page.exists():
            return JSONResponse({"message": "UI not found. See /api/docs."}, status_code=404)
        return FileResponse(page, headers={"Cache-Control": "public, max-age=300"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    return app


# ----------------------------------------------------------- answer generation
async def _answer_events(prepared: Prepared, llm: OpenRouterClient, rag: RagService) -> AsyncIterator[dict]:
    """The single source of truth for what a chat turn emits (stream & non-stream)."""
    yield {"type": "sources", "sources": [s.public() for s in prepared.sources]}

    if prepared.canned_answer is not None:
        yield {"type": "token", "text": prepared.canned_answer}
        yield {"type": "done", "model": "canned"}
        return

    model_used = ""
    try:
        async for model, token in llm.stream(prepared.messages):
            model_used = model
            yield {"type": "token", "text": token}
    except LLMAuthError as exc:
        log.error("OpenRouter auth problem: %s", exc)
        yield {"type": "token", "text": rag.extractive_answer(prepared.sources)}
        model_used = "extractive-fallback"
    except LLMUnavailable as exc:
        log.error("All models failed: %s", exc)
        yield {"type": "token", "text": rag.extractive_answer(prepared.sources)}
        model_used = "extractive-fallback"
    except (LLMStreamInterrupted, LLMError) as exc:
        log.error("Stream interrupted: %s", exc)
        yield {"type": "error", "message": "The response was interrupted. Please try again."}
        return
    yield {"type": "done", "model": model_used}


async def _sse_stream(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield sse(event)
    except Exception:  # never leak internals into the stream
        log.exception("Unhandled error while streaming")
        yield sse({"type": "error", "message": "Something went wrong. Please try again."})


async def _collect(events: AsyncIterator[dict]) -> dict:
    answer, sources, model, error = [], [], "", None
    async for event in events:
        if event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "token":
            answer.append(event["text"])
        elif event["type"] == "done":
            model = event["model"]
        elif event["type"] == "error":
            error = event["message"]
    if error and not answer:
        raise HTTPException(502, error)
    return {"answer": "".join(answer), "sources": sources, "model": model}


app = create_app()
