"""Shared network helpers."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

# Tunables
DEFAULT_MAX_BACKOFF = 60.0  # Seconds
DEFAULT_RETRY_AFTER_EPS = 0.10  # Up to +10% multiplicative jitter above Retry-After


def parse_retry_after(response: httpx.Response | None) -> float | None:
    """Return wait seconds derived from a response's Retry-After header.

    Accepts both numeric seconds and HTTP-date per RFC 7231 (case-insensitive lookup).
    Returns None when the header is absent or cannot be parsed.
    """
    if response is None:
        return None

    header_value: str | None = None
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
            parsed = parsed.replace(tzinfo=UTC)

        now = datetime.now(UTC)
        delta = (parsed - now).total_seconds()
        return max(0.0, delta)

    return max(0.0, seconds)


def _full_jitter_delay(
    attempt: int,
    *,
    base: float,
    max_backoff: float,
    rng: random.Random,
) -> float:
    """Exponential backoff with full jitter: Uniform(0, min(max_backoff, base * 2**(n-1)))."""
    expo = min(max_backoff, base * (2 ** (attempt - 1)))
    return rng.uniform(0.0, expo)


def _apply_retry_after_floor(
    retry_after_floor: float | None,
    backoff: float,
    *,
    eps: float,
    rng: random.Random,
) -> float:
    """Honor Retry-After as a floor; add a small multiplicative jitter above it."""
    if retry_after_floor is None:
        return backoff
    floor = max(0.0, retry_after_floor)
    jitter = rng.uniform(0.0, floor * max(0.0, eps))  # Keeps us strictly >= floor
    return max(floor, backoff) + jitter


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    attempts: int = 3,
    base_backoff: float = 0.5,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    timeout: httpx.Timeout | None = None,
    retry_on_status: Iterable[int] | None = None,
    retry_after_jitter_eps: float = DEFAULT_RETRY_AFTER_EPS,
    rng: random.Random | None = None,
    **request_kwargs: Any,
) -> httpx.Response:
    """Issue an HTTP request with retry/backoff handling.

    Behavior:
      • Retries on configured status codes (default: 429 and all 5xx) and on transient httpx errors.
      • Exponential backoff with FULL jitter.
      • Retry-After (RFC 7231) is treated as a FLOOR with small multiplicative jitter.
      • After exhausting attempts, the last httpx exception is propagated.

    Args:
      client: An httpx.AsyncClient.
      method: HTTP method.
      url: Target URL.
      headers, params, timeout: Passed to httpx.
      attempts: Total attempts including the first (must be >= 1).
      base_backoff: Initial backoff in seconds (must be > 0).
      max_backoff: Maximum backoff cap in seconds.
      retry_on_status: Iterable of HTTP status codes to retry (in addition to 5xx).
      retry_after_jitter_eps: Fractional jitter applied on top of Retry-After (e.g., 0.10 → up to +10%).
      rng: Optional Random instance (useful for tests); defaults to module-level RNG.
      **request_kwargs: Forwarded to httpx.AsyncClient.request (e.g., json=, data=, files=, etc.).

    Returns:
      httpx.Response on success.

    Raises:
      httpx.HTTPError: When attempts are exhausted or on non-retryable errors.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if base_backoff <= 0:
        raise ValueError("base_backoff must be > 0")
    if max_backoff <= 0:
        raise ValueError("max_backoff must be > 0")

    retry_statuses = {429} if retry_on_status is None else set(retry_on_status)
    retry_statuses.update({status for status in range(500, 600)})

    rng = rng or random

    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                **request_kwargs,
            )
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as status_exc:
            response = status_exc.response
            status = response.status_code if response is not None else None
            should_retry = status is not None and (status in retry_statuses or 500 <= status < 600)
            if not should_retry or attempt == attempts:
                raise

            retry_after = parse_retry_after(response)
            delay = _full_jitter_delay(
                attempt,
                base=base_backoff,
                max_backoff=max_backoff,
                rng=rng,
            )
            delay = _apply_retry_after_floor(
                retry_after,
                delay,
                eps=retry_after_jitter_eps,
                rng=rng,
            )
            await asyncio.sleep(delay)

        except httpx.HTTPError:
            if attempt == attempts:
                raise
            delay = _full_jitter_delay(
                attempt,
                base=base_backoff,
                max_backoff=max_backoff,
                rng=rng,
            )
            await asyncio.sleep(delay)

    # Should be unreachable; kept for explicitness.
    raise httpx.HTTPError("request_with_retries exhausted attempts")
