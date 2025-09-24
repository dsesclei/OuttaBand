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
import random
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import httpx

DEFAULT_BASE_URL = "https://api.binance.us"
DEFAULT_SYMBOL = "SOLUSDT"
DEFAULT_LIMIT = 120
DEFAULT_CACHE_TTL = 60
DEFAULT_MAX_STALE = 7200

MAX_FETCH_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_JITTER_SECONDS = 0.25
REQUEST_USER_AGENT = "lpbot-volatility/0.1 (+https://github.com/dave/lpbot)"
REQUEST_TIMEOUT = httpx.Timeout(7.5, connect=5.0, read=7.5, write=7.5, pool=5.0)
REQUEST_HEADERS = {"user-agent": REQUEST_USER_AGENT}


@dataclass
class VolReading:
    """Snapshot of realized volatility derived from Binance minute candles."""

    sigma_pct: float
    bucket: str
    window_minutes: int = 60
    as_of_ts: int = 0
    sample_count: int = 0
    stale: bool = False


_cache_lock = asyncio.Lock()
_cache: Dict[Tuple[str, str], Tuple[VolReading, float]] = {}


async def fetch_sigma_1h(
    client: httpx.AsyncClient,
    *,
    base_url: str = DEFAULT_BASE_URL,
    symbol: str = DEFAULT_SYMBOL,
    limit: int = DEFAULT_LIMIT,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    max_stale: int = DEFAULT_MAX_STALE,
) -> Optional[VolReading]:
    """Fetch 1-hour realized volatility for the given market.

    Results are cached for ``cache_ttl`` seconds. If live fetching fails, the
    function may return a stale reading as long as it is newer than
    ``max_stale`` seconds. Callers receive ``None`` when no data is available.
    """

    key = (base_url, symbol)
    observed_ts: Optional[float] = None
    now = time.time()

    async with _cache_lock:
        cached = _cache.get(key)
        if _is_fresh(cached, now, cache_ttl):
            return replace(cached[0], stale=False)
        observed_ts = cached[1] if cached else None

    payload = await _fetch_klines_with_retries(
        client,
        base_url=base_url,
        symbol=symbol,
        limit=limit,
    )
    if payload is None:
        return await _return_stale_if_available(key, max_stale)

    closes, close_times = _extract_closes(payload)
    sigma_data = compute_sigma_from_closes(closes)
    if sigma_data is None or len(close_times) < 61:
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
        if current and (observed_ts is None or current[1] != observed_ts):
            if current[1] >= fetched_at:
                return replace(current[0], stale=False)
        _cache[key] = (reading, fetched_at)
        return reading


# The observed_ts / double-check protects against a race: another coroutine may
# have already refreshed the cache while we were awaiting Binance, so we only
# overwrite if we are still the latest producer.


def _bucket(sig_pct: float) -> str:
    if sig_pct < 0.6:
        return "low"
    if sig_pct <= 1.2:
        return "mid"
    return "high"


def compute_sigma_from_closes(
    closes: Sequence[float],
) -> Optional[Tuple[float, int]]:
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


def _extract_closes(payload: Any) -> Tuple[list[float], list[int]]:
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


async def _return_stale_if_available(
    key: Tuple[str, str], max_stale: int
) -> Optional[VolReading]:
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
) -> Optional[Any]:
    url = f"{base_url}/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        response: Optional[httpx.Response] = None
        try:
            response = await client.get(
                url,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            response = exc.response
        except httpx.HTTPError as exc:
            response = getattr(exc, "response", None)
        except ValueError:
            pass

        if attempt == MAX_FETCH_ATTEMPTS:
            break

        delay = _compute_retry_delay(attempt, response)
        await asyncio.sleep(delay)

    return None


def _compute_retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return retry_after

    base = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0, BACKOFF_JITTER_SECONDS)
    return base + jitter


def _retry_after_seconds(response: Optional[httpx.Response]) -> Optional[float]:
    if response is None:
        return None

    header = response.headers.get("Retry-After")
    if not header:
        return None

    try:
        return max(0.0, float(header))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None

    if parsed is None:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return max(0.0, (parsed - now).total_seconds())


def _is_fresh(
    cached: Optional[Tuple[VolReading, float]], current_ts: float, ttl: int
) -> bool:
    if not cached:
        return False
    if ttl <= 0:
        return False
    _, fetched_ts = cached
    return (current_ts - fetched_ts) <= ttl


def _maybe_return_stale(
    cached: Optional[Tuple[VolReading, float]], current_ts: float, max_stale: int
) -> Optional[VolReading]:
    if not cached:
        return None
    if max_stale <= 0:
        return None
    reading, fetched_ts = cached
    if (current_ts - fetched_ts) > max_stale:
        return None
    return replace(reading, stale=True)
