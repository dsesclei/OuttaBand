from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
import pytest

from outtaband.policy import volatility
from tests.conftest import json_response, make_mock_transport, now_time


def _make_kline_payload(count: int = 120) -> list[list[Any]]:
    payload: list[list[Any]] = []
    base_time = 1_700_000_000_000
    for idx in range(count):
        close = 100.0 + idx * 0.2 + (idx % 5) * 0.05
        open_time = base_time + idx * 60_000
        close_time = open_time + 59_000
        payload.append(
            [
                open_time,
                str(close - 0.1),
                str(close + 0.1),
                str(close - 0.2),
                str(close),
                "100",
                close_time,
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
        )
    return payload


@pytest.fixture(autouse=True)
async def clear_vol_cache() -> None:
    await volatility.clear_cache()
    yield
    await volatility.clear_cache()


def test_compute_sigma_from_closes_min_sample_guard() -> None:
    assert volatility.compute_sigma_from_closes([100.0] * 60) is None


def test_compute_sigma_from_closes_happy_path() -> None:
    closes = [100.0 + idx * 0.5 + ((idx % 3) * 0.1) for idx in range(70)]
    result = volatility.compute_sigma_from_closes(closes)
    assert result is not None
    sigma, sample_count = result
    assert math.isfinite(sigma)
    assert sigma >= 0.0
    assert sample_count == 60


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "low"),
        (0.59, "low"),
        (0.6, "mid"),
        (1.2, "mid"),
        (1.21, "high"),
    ],
)
def test_bucket_boundaries(value: float, expected: volatility.Bucket) -> None:
    assert volatility._bucket(value) == expected  # type: ignore[attr-defined]


def test_extract_closes_filters_garbage() -> None:
    payload = [
        ["bad"],  # malformed
        [0, "open", "high", "low", "not-a-number", "vol", 0],  # invalid close
        [0, "open", "high", "low", "-5", "vol", 0],  # non-positive
        [0, "open", "high", "low", "100.5", "vol", "bad-time"],  # invalid time
        [0, "open", "high", "low", "101.5", "vol", 2000],  # valid
        [0, "open", "high", "low", 102.5, "vol", 3000],  # valid float
    ]

    closes, times = volatility._extract_closes(payload)  # type: ignore[attr-defined]
    assert closes == [101.5, 102.5]
    assert times == [2000, 3000]


@pytest.mark.asyncio
async def test_fetch_sigma_1h_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    now_time(monkeypatch, 1_000.0)
    payload = _make_kline_payload()
    transport = make_mock_transport(json_response(json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        first = await volatility.fetch_sigma_1h(
            client,
            base_url="http://example.invalid",
            symbol="SOLUSDT",
            cache_ttl=120,
            max_stale=3600,
        )
        assert first is not None
        assert first.stale is False

        second = await volatility.fetch_sigma_1h(
            client,
            base_url="http://example.invalid",
            symbol="SOLUSDT",
            cache_ttl=120,
            max_stale=3600,
        )
        assert second == first
        assert second.stale is False


@pytest.mark.asyncio
async def test_fetch_sigma_1h_returns_stale_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "http://example.invalid"
    symbol = "SOLUSDT"
    payload = _make_kline_payload()

    now_time(monkeypatch, 1_000.0)
    async with httpx.AsyncClient(
        transport=make_mock_transport(json_response(json=payload))
    ) as client:
        fresh = await volatility.fetch_sigma_1h(
            client,
            base_url=base_url,
            symbol=symbol,
            cache_ttl=60,
            max_stale=3_600,
        )
        assert fresh is not None
        assert fresh.stale is False

    now_time(monkeypatch, 1_000.0 + 120.0)

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPError(f"boom: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing_handler)) as client:
        stale = await volatility.fetch_sigma_1h(
            client,
            base_url=base_url,
            symbol=symbol,
            cache_ttl=60,
            max_stale=3_600,
        )
        assert stale is not None
        assert stale.stale is True
        assert stale.sigma_pct == pytest.approx(fresh.sigma_pct)
        assert stale.bucket == fresh.bucket


@pytest.mark.asyncio
async def test_get_cache_age(monkeypatch: pytest.MonkeyPatch) -> None:
    base_url = "http://example.invalid/cache-age"
    symbol = "SOLUSDT"
    now_time(monkeypatch, 5_000.0)
    assert await volatility.get_cache_age(base_url, symbol) is None

    payload = _make_kline_payload()
    async with httpx.AsyncClient(
        transport=make_mock_transport(json_response(json=payload))
    ) as client:
        reading = await volatility.fetch_sigma_1h(
            client,
            base_url=base_url,
            symbol=symbol,
            cache_ttl=300,
            max_stale=3_600,
        )
        assert reading is not None

    now_time(monkeypatch, 5_020.0)
    age = await volatility.get_cache_age(base_url, symbol)
    assert age is not None
    assert age >= 20.0
    assert age < 21.0


@pytest.mark.asyncio
async def test_fetch_sigma_1h_handles_http_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_request(*_: Any, **__: Any) -> httpx.Response:
        raise httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("GET", "http://example.invalid"),
            response=httpx.Response(500),
        )

    monkeypatch.setattr(volatility.net, "request_with_retries", failing_request)
    now_time(monkeypatch, 10_000.0)

    async with httpx.AsyncClient() as client:
        result = await volatility.fetch_sigma_1h(
            client, base_url="http://example.invalid", symbol="SOLUSDT"
        )
        assert result is None


@pytest.mark.asyncio
async def test_fetch_sigma_1h_sets_as_of_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _make_kline_payload(80)
    close_times = [entry[6] for entry in payload]

    now_time(monkeypatch, 1_000.0)
    async with httpx.AsyncClient(
        transport=make_mock_transport(json_response(json=payload))
    ) as client:
        reading = await volatility.fetch_sigma_1h(
            client,
            base_url="http://example.invalid/as-of",
            symbol="SOLUSDT",
            cache_ttl=60,
            max_stale=3_600,
        )

    assert reading is not None
    expected_ts = int(close_times[-1] // 1000)
    assert reading.as_of_ts == expected_ts
    assert reading.sample_count == 60


@pytest.mark.asyncio
async def test_fetch_sigma_1h_cache_stays_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure concurrent refreshes keep the freshest result."""

    payload = _make_kline_payload()
    now_time(monkeypatch, 2_000.0)
    transport = make_mock_transport(json_response(json=payload), json_response(json=payload))
    base_url = "http://example.invalid/concurrent"

    async with httpx.AsyncClient(transport=transport) as client:
        results = await asyncio.gather(
            volatility.fetch_sigma_1h(
                client,
                base_url=base_url,
                symbol="SOLUSDT",
                cache_ttl=60,
                max_stale=3_600,
            ),
            volatility.fetch_sigma_1h(
                client,
                base_url=base_url,
                symbol="SOLUSDT",
                cache_ttl=60,
                max_stale=3_600,
            ),
        )

    assert results[0] == results[1]
    assert results[0] is not None
