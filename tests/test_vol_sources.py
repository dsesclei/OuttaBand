from __future__ import annotations

import httpx
import pytest

from policy import vol_sources, volatility


@pytest.mark.asyncio
async def test_binance_vol_source_read_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    reading = volatility.VolReading(sigma_pct=0.75, bucket="mid", as_of_ts=1, sample_count=60)

    async def fake_fetch(*_, **__) -> volatility.VolReading:
        return reading

    monkeypatch.setattr(volatility, "fetch_sigma_1h", fake_fetch)

    async with httpx.AsyncClient() as client:
        source = vol_sources.BinanceVolSource(
            client=client,
            base_url="http://example.invalid",
            symbol="SOLUSDT",
            cache_ttl=30,
            max_stale=300,
            user_agent="lpbot-tests",
        )
        result = await source.read()
        assert result == reading


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        httpx.HTTPError("boom"),
        RuntimeError("oops"),
    ],
)
async def test_binance_vol_source_read_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch, exception: Exception
) -> None:
    async def failing_fetch(*_, **__) -> volatility.VolReading:
        raise exception

    monkeypatch.setattr(volatility, "fetch_sigma_1h", failing_fetch)

    async with httpx.AsyncClient() as client:
        source = vol_sources.BinanceVolSource(
            client=client,
            base_url="http://example.invalid",
            symbol="SOLUSDT",
            cache_ttl=30,
            max_stale=300,
            user_agent="lpbot-tests",
        )
        result = await source.read()
        assert result is None
