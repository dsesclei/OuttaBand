from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import net


def _response_with_retry_after(value: str) -> httpx.Response:
    request = httpx.Request("GET", "https://example.invalid")
    return httpx.Response(429, headers={"Retry-After": value}, request=request)


def test_parse_retry_after_numeric_seconds() -> None:
    response = _response_with_retry_after("12")
    assert net.parse_retry_after(response) == pytest.approx(12.0)


def test_parse_retry_after_http_date() -> None:
    target = datetime.now(UTC) + timedelta(seconds=15)
    header = target.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response = _response_with_retry_after(header)
    delay = net.parse_retry_after(response)
    assert delay is not None
    assert 0.0 <= delay <= 15.0


def test_parse_retry_after_missing() -> None:
    response = httpx.Response(200, headers={}, request=httpx.Request("GET", "https://example.invalid"))
    assert net.parse_retry_after(response) is None


def test_full_jitter_delay_bounds() -> None:
    rng = random.Random(0)
    delay = net._full_jitter_delay(3, base=0.5, max_backoff=10.0, rng=rng)  # type: ignore[attr-defined]
    upper = min(10.0, 0.5 * (2 ** (3 - 1)))
    assert 0.0 <= delay <= upper


def test_apply_retry_after_floor_enforces_bounds() -> None:
    rng = random.Random(0)
    value = net._apply_retry_after_floor(2.0, 1.0, eps=0.05, rng=rng)  # type: ignore[attr-defined]
    assert value >= 2.0
    assert value < 2.0 * (1.0 + 0.05) + 1e-6


@pytest.mark.asyncio
async def test_request_with_retries_retries_on_429_and_5xx_only() -> None:
    statuses = [500, 502, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        return httpx.Response(status, json={"attempt": status}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await net.request_with_retries(
            client,
            "GET",
            "https://example.invalid/resource",
            attempts=3,
            base_backoff=0.01,
            rng=random.Random(0),
        )
        assert response.status_code == 200

    # Ensure 404 does not retry.
    calls = 0

    def handler_404(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_404)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await net.request_with_retries(
                client,
                "GET",
                "https://example.invalid/not-found",
                attempts=3,
                base_backoff=0.01,
                rng=random.Random(0),
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_request_with_retries_raises_after_exhaustion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await net.request_with_retries(
                client,
                "GET",
                "https://example.invalid",
                attempts=2,
                base_backoff=0.01,
                rng=random.Random(0),
            )


@pytest.mark.asyncio
async def test_request_with_retries_validates_args() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError):
            await net.request_with_retries(client, "GET", "https://example.invalid", attempts=0)
        with pytest.raises(ValueError):
            await net.request_with_retries(client, "GET", "https://example.invalid", base_backoff=0.0)


@pytest.mark.asyncio
async def test_request_with_retries_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    # First response instructs Retry-After: 2 seconds, second succeeds.
    first_delay = 2.0
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": str(first_delay)}, request=request)
        return httpx.Response(200, request=request)

    sleep_calls: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await net.request_with_retries(
            client,
            "GET",
            "https://example.invalid",
            attempts=2,
            base_backoff=0.5,
            rng=random.Random(0),
        )
        assert response.status_code == 200

    assert sleep_calls, "expected retry delay"
    for delay in sleep_calls:
        assert delay >= first_delay
        assert delay <= first_delay * (1 + net.DEFAULT_RETRY_AFTER_EPS) + 1e-6
