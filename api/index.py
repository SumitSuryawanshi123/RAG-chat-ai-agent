"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI/WSGI object called ``app`` in
``api/index.py``. ``vercel.json`` rewrites every ``/api/*`` request here and the
static UI in ``public/`` is served from Vercel's CDN.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.main import app  # noqa: E402,F401  (re-exported for Vercel)
