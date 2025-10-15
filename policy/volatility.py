"""Volatility fetching and bucketing utilities.

This module computes realized volatility for SOL/USDT using Binance 1m klines.
It fetches closes, derives log returns r_t = ln(c_t / c_{t-1}) over the latest
60 one-minute intervals, and scales the sample standard deviation by sqrt(60)
and *100 to express a 1-hour sigma in percent. The value is then bucketed into
risk bands: low (<0.6%), mid (0.6%-1.2%), and high (>1.2%).

Consumers should pass in an ``httpx.AsyncClient``. Results are cached per
(base_url, symbol) to avoid redundant network calls, with support for serving
stale-but-bounded data when live fetches fail.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx
import structlog

import net
from shared_types import Bucket, UnixTs

log = structlog.get_logger("volatility")

DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_SYMBOL = "SOLUSDT"
DEFAULT_LIMIT = 120
DEFAULT_CACHE_TTL = 60
DEFAULT_MAX_STALE = 7200

MAX_FETCH_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
DEFAULT_USER_AGENT = "outtaband-volatility/0.1 (+https://github.com/dave/outtaband)"
REQUEST_TIMEOUT = httpx.Timeout(7.5, connect=5.0, read=7.5, write=7.5, pool=5.0)


@dataclass
class VolReading:
    """Snapshot of realized volatility derived from Binance minute candles."""

    sigma_pct: float
    bucket: Bucket
    window_minutes: int = 60
    as_of_ts: UnixTs = 0
    sample_count: int = 0
    stale: bool = False


_cache_lock = asyncio.Lock()
_cache: dict[tuple[str, str], tuple[VolReading, float]] = {}


async def fetch_sigma_1h(
    client: httpx.AsyncClient,
    *,
    base_url: str = DEFAULT_BASE_URL,
    symbol: str = DEFAULT_SYMBOL,
    limit: int = DEFAULT_LIMIT,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    max_stale: int = DEFAULT_MAX_STALE,
    user_agent: str | None = None,
) -> VolReading | None:
    """Fetch 1-hour realized volatility for the given market.

    Results are cached for ``cache_ttl`` seconds. If live fetching fails, the
    function may return a stale reading as long as it is newer than
    ``max_stale`` seconds. Callers receive ``None`` when no data is available.
    """

    key = (base_url, symbol)
    observed_ts: float | None = None
    now = time.time()

    async with _cache_lock:
        cached = _cache.get(key)
        if cached and _is_fresh(cached, now, cache_ttl):
            return replace(cached[0], stale=False)
        observed_ts = cached[1] if cached else None

    payload = await _fetch_klines_with_retries(
        client,
        base_url=base_url,
        symbol=symbol,
        limit=limit,
        user_agent=user_agent,
    )
    if payload is None:
        return await _return_stale_if_available(key, max_stale)

    closes, close_times = _extract_closes(payload)
    sigma_data = compute_sigma_from_closes(closes)
    if sigma_data is None:
        first_close = closes[0] if closes else None
        last_close = closes[-1] if closes else None
        log.debug(
            "sigma_compute_failed",
            close_count=len(closes),
            first_close=first_close,
            last_close=last_close,
            limit=limit,
        )
        return await _return_stale_if_available(key, max_stale)

    if len(close_times) < 61:
        return await _return_stale_if_available(key, max_stale)

    sigma_pct, sample_count = sigma_data
    as_of_ts = int(close_times[-1] // 1000) if close_times else int(time.time())

    reading = VolReading(
        sigma_pct=sigma_pct,
        bucket=_bucket(sigma_pct),
        as_of_ts=as_of_ts,
        sample_count=sample_count,
        stale=False,
    )

    fetched_at = time.time()
    async with _cache_lock:
        current = _cache.get(key)
        if (
            current
            and (observed_ts is None or current[1] != observed_ts)
            and current[1] >= fetched_at
        ):
            return replace(current[0], stale=False)
        _cache[key] = (reading, fetched_at)
        return reading


# The observed_ts / double-check protects against a race: another coroutine may
# have already refreshed the cache while we were awaiting Binance, so we only
# overwrite if we are still the latest producer.


def _bucket(sig_pct: float) -> Bucket:
    if sig_pct < 0.6:
        return "low"
    if sig_pct <= 1.2:
        return "mid"
    return "high"


def compute_sigma_from_closes(
    closes: Sequence[float],
) -> tuple[float, int] | None:
    if len(closes) < 61:
        return None

    window = closes[-61:]
    if any(price <= 0 or not math.isfinite(price) for price in window):
        return None

    returns = [math.log(window[idx] / window[idx - 1]) for idx in range(1, len(window))]
    if len(returns) < 2 or any(not math.isfinite(r) for r in returns):
        return None

    try:
        stdev = statistics.stdev(returns)
    except statistics.StatisticsError:
        return None

    if not math.isfinite(stdev):
        return None

    sigma_pct = stdev * math.sqrt(60) * 100.0
    if not math.isfinite(sigma_pct):
        return None

    return sigma_pct, len(returns)


async def clear_cache() -> None:
    async with _cache_lock:
        _cache.clear()


async def get_cache_age(base_url: str, symbol: str) -> float | None:
    """Return the age in seconds of the cached reading for the given market."""
    key = (base_url, symbol)
    async with _cache_lock:
        cached = _cache.get(key)
    if not cached:
        return None
    _reading, fetched_ts = cached
    age = time.time() - fetched_ts
    if age < 0:
        age = 0.0
    return age


def _extract_closes(payload: Any) -> tuple[list[float], list[int]]:
    closes: list[float] = []
    close_times: list[int] = []

    if not isinstance(payload, list):
        return closes, close_times

    for entry in payload:
        if not isinstance(entry, (list, tuple)):
            continue
        try:
            close_price = float(entry[4])
            close_time = int(entry[6])
        except (IndexError, TypeError, ValueError):
            continue
        if not math.isfinite(close_price) or close_price <= 0:
            continue
        closes.append(close_price)
        close_times.append(close_time)

    return closes, close_times


async def _return_stale_if_available(key: tuple[str, str], max_stale: int) -> VolReading | None:
    async with _cache_lock:
        cached = _cache.get(key)
        current_ts = time.time()
        result = _maybe_return_stale(cached, current_ts, max_stale)
    return result


async def _fetch_klines_with_retries(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    symbol: str,
    limit: int,
    user_agent: str | None,
) -> Any | None:
    url = f"{base_url}/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    ua = user_agent or DEFAULT_USER_AGENT

    try:
        response = await net.request_with_retries(
            client,
            "GET",
            url,
            headers={"user-agent": ua},
            params=params,
            attempts=MAX_FETCH_ATTEMPTS,
            base_backoff=BACKOFF_BASE_SECONDS,
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        log.warning(
            "klines_fetch_failed",
            url=url,
            attempts=MAX_FETCH_ATTEMPTS,
            user_agent=ua,
            err=str(exc),
        )
        return None

    try:
        return response.json()
    except ValueError:
        log.debug("klines_json_invalid")
        return None


def _is_fresh(cached: tuple[VolReading, float] | None, current_ts: float, ttl: int) -> bool:
    if not cached:
        return False
    if ttl <= 0:
        return False
    _, fetched_ts = cached
    return (current_ts - fetched_ts) <= ttl


def _maybe_return_stale(
    cached: tuple[VolReading, float] | None, current_ts: float, max_stale: int
) -> VolReading | None:
    if not cached:
        return None
    if max_stale <= 0:
        return None
    reading, fetched_ts = cached
    if (current_ts - fetched_ts) > max_stale:
        return None
    return replace(reading, stale=True)
