"""Tiny in-memory sliding-window rate limiter.

On serverless platforms each warm instance keeps its own counters, so this is a
best-effort abuse guard (it stops a single client hammering one instance and
protects the free OpenRouter quota), not a hard global limit. For a hard limit,
put Vercel's Firewall / WAF rate limiting in front of the function.
"""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 5000):
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit. Returns ``(allowed, retry_after_seconds)``."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        if len(self._hits) > self.max_keys:
            self._prune(now)
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(1, int(self.window - (now - q[0])) + 1)
        q.append(now)
        return True, 0

    def _prune(self, now: float) -> None:
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] > self.window]:
            del self._hits[key]
