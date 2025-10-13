"""Shared network helpers for lpbot."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Optional

import httpx

RETRY_JITTER_MAX_FRACTION = 0.25


def parse_retry_after(response: Optional[httpx.Response]) -> Optional[float]:
    """Return wait seconds derived from a response's Retry-After header.

    Both numeric seconds and HTTP-date values are accepted per RFC 7231. The
    lookup is case-insensitive. ``None`` is returned when the header is absent
    or cannot be parsed.
    """
    if response is None:
        return None

    header_value: Optional[str] = None
    for key, value in response.headers.items():
        if key.lower() == "retry-after":
            header_value = value
            break

    if not header_value:
        return None

    stripped = header_value.strip()
    if not stripped:
        return None

    try:
        seconds = float(stripped)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None

        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = (parsed - now).total_seconds()
        return max(0.0, delta)

    return max(0.0, seconds)


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    attempts: int = 3,
    base_backoff: float = 0.5,
    timeout: Optional[httpx.Timeout] = None,
    retry_on_status: Optional[Iterable[int]] = None,
) -> httpx.Response:
    """Issue an HTTP request with retry/backoff handling.

    Responses with configured status codes (default: ``429`` and all ``5xx``)
    are retried with exponential backoff and jitter. Retry-After headers take
    precedence over the calculated delay. All exceptions raised by httpx after
    exhausting attempts propagate to the caller.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if base_backoff <= 0:
        raise ValueError("base_backoff must be > 0")

    retry_statuses = {429} if retry_on_status is None else set(retry_on_status)
    retry_statuses.update({status for status in range(500, 600)})

    for attempt in range(1, attempts + 1):
        response: Optional[httpx.Response] = None
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as status_exc:
            response = status_exc.response
            status = response.status_code if response is not None else None
            should_retry = (
                status is not None
                and (status in retry_statuses or 500 <= status < 600)
            )
            if not should_retry or attempt == attempts:
                raise
            wait_seconds = parse_retry_after(response)
            if wait_seconds is None:
                wait_seconds = base_backoff * (2 ** (attempt - 1))
        except httpx.HTTPError:
            if attempt == attempts:
                raise
            wait_seconds = base_backoff * (2 ** (attempt - 1))
            response = None

        jitter_cap = max(0.0, wait_seconds * RETRY_JITTER_MAX_FRACTION)
        jitter = random.uniform(0.0, jitter_cap)
        await asyncio.sleep(wait_seconds + jitter)

    raise httpx.HTTPError("request_with_retries exhausted attempts")
