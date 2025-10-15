from __future__ import annotations

import httpx
import pytest

from price_sources import MeteoraPriceSource


def _make_source(transport: httpx.BaseTransport) -> MeteoraPriceSource:
    client = httpx.AsyncClient(transport=transport)
    return MeteoraPriceSource(
        client=client,
        pair_address="pair123",
        base_url="http://example.invalid/api",
        user_agent="outtaband-test",
    )


@pytest.mark.asyncio
async def test_read_happy_path() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"current_price": "123.45"}, request=request)
    )
    source = _make_source(transport)
    async with source.client:
        price = await source.read()
    assert price == pytest.approx(123.45)


@pytest.mark.asyncio
async def test_read_invalid_json_or_missing_key_returns_none() -> None:
    # Invalid JSON path
    transport_invalid = httpx.MockTransport(
        lambda request: httpx.Response(200, text="{not-json}", request=request)
    )
    source_invalid = _make_source(transport_invalid)
    async with source_invalid.client:
        assert await source_invalid.read() is None

    # Missing key path
    transport_missing = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"price": "1.23"}, request=request)
    )
    source_missing = _make_source(transport_missing)
    async with source_missing.client:
        assert await source_missing.read() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_value", [0, -1, "nan", "inf"])
async def test_read_nonfinite_or_nonpositive_returns_none(payload_value: object) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"current_price": payload_value}, request=request)
    )
    source = _make_source(transport)
    async with source.client:
        assert await source.read() is None
