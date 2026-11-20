"""Vercel serverless entrypoint.

Vercel's FastAPI support looks for an ASGI object called ``app`` in
``api/index.py`` and routes *every* path to it (``/``, ``/api/chat``, ...).
Do not add ``rewrites`` to ``vercel.json``: they change the path FastAPI sees.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.main import app  # noqa: E402,F401  (re-exported for Vercel)
