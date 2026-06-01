"""Lightweight in-process rate limiting for the expensive QA endpoint.

A single uvicorn process serves one Ollama runner, so a per-process sliding
window is enough — no Redis/external store needed. Requests are keyed by client
IP (X-Forwarded-For first, since the public deployment sits behind a reverse
proxy / tunnel that hides the socket peer), which bounds how fast any one
client can spend GPU time. Behind a proxy that does not set X-Forwarded-For,
this degrades to a single global bucket — still a useful cap on total load.
"""
from __future__ import annotations

import time
from collections import deque

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` per key."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> None:
        """Record one request for ``key``; raise HTTP 429 if over the limit."""
        now = time.monotonic()
        cutoff = now - self._window
        dq = self._hits.setdefault(key, deque())
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= self._max:
            retry_after = int(self._window - (now - dq[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests; please slow down.",
                headers={"Retry-After": str(max(1, retry_after))},
            )
        dq.append(now)


def _client_key(request: Request) -> str:
    """Best-effort client identity: prefer the first X-Forwarded-For hop (set
    by the reverse proxy / tunnel), else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


_qa_limiter = SlidingWindowRateLimiter(
    max_requests=get_settings().RATE_LIMIT_QA_PER_MIN,
    window_seconds=60.0,
)


async def rate_limit_qa(request: Request) -> None:
    """FastAPI dependency: throttle POST /qa/stream per client."""
    _qa_limiter.hit(_client_key(request))
