"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Free OpenRouter models, tried in order until one answers. Free endpoints are
# frequently rate-limited or rotated out, so the last entry is OpenRouter's own
# free-tier router which survives model churn. Override with OPENROUTER_MODELS.
DEFAULT_MODELS = (
    "google/gemma-4-31b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/free",
)


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a local .env (dev convenience, no dependency).

    Real environment variables always win, so on Vercel / Docker (where the
    platform injects them and there is no .env file) this is a no-op.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    models: tuple[str, ...]
    app_url: str
    app_name: str
    kb_path: Path
    top_k: int
    max_context_chars: int
    max_history_messages: int
    max_message_chars: int
    max_output_tokens: int
    temperature: float
    llm_timeout_seconds: float
    rate_limit_per_minute: int
    cors_origins: tuple[str, ...]
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        models_env = os.getenv("OPENROUTER_MODELS", "")
        models = tuple(m.strip() for m in models_env.split(",") if m.strip()) or DEFAULT_MODELS
        cors = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip())
        kb = Path(os.getenv("KB_PATH", ROOT / "data" / "knowledge_base.jsonl"))
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            models=models,
            app_url=os.getenv("APP_URL", "https://github.com/SumitSuryawanshi123/RAG-chat-ai-agent"),
            app_name=os.getenv("APP_NAME", "DarGlobal x Wasalt Assistant"),
            kb_path=kb if kb.is_absolute() else ROOT / kb,
            top_k=_int("TOP_K", 6),
            max_context_chars=_int("MAX_CONTEXT_CHARS", 7000),
            max_history_messages=_int("MAX_HISTORY_MESSAGES", 8),
            max_message_chars=_int("MAX_MESSAGE_CHARS", 1000),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 800),
            temperature=_float("LLM_TEMPERATURE", 0.2),
            llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 40.0),
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 20),
            cors_origins=cors,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
