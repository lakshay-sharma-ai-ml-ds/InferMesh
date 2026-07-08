"""
InferMesh API Middleware
=========================
Request ID injection, rate limiting, and timing middleware.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID into every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding window rate limiter.
    Limits requests per client IP to `max_requests` per `window_s` seconds.
    """

    def __init__(self, app, max_requests: int = 1000, window_s: float = 1.0):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_s = window_s
        self._client_requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip metrics and health endpoints
        if request.url.path in {"/metrics", "/health", "/ready"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old timestamps
        window_start = now - self.window_s
        timestamps = self._client_requests[client_ip]
        self._client_requests[client_ip] = [t for t in timestamps if t > window_start]

        if len(self._client_requests[client_ip]) >= self.max_requests:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        self._client_requests[client_ip].append(now)
        response = await call_next(request)
        return response
