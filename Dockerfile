# syntax=docker/dockerfile:1.7
# Runtime image for the chatbot. Scraping is done offline (see README); the image
# only ships the pre-built knowledge base, so it stays small (~200 MB, no browser).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser

COPY chatbot ./chatbot
COPY public ./public
COPY data/knowledge_base.jsonl data/knowledge_base.meta.json ./data/

USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'), timeout=4)" || exit 1

# Shell form so $PORT is honoured on platforms that inject it (Render, Railway, Fly...).
CMD uvicorn chatbot.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'
