"""Swappable data source implementations for price and volatility."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

import net
import volatility

log = structlog.get_logger("sources")


class PriceSource:
    async def read(self) -> float | None:  # pragma: no cover - interface
        raise NotImplementedError


class VolSource:
    async def read(self) -> volatility.VolReading | None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(slots=True)
class MeteoraPriceSource(PriceSource):
    client: httpx.AsyncClient
    pair_address: str
    base_url: str
    user_agent: str

    async def read(self) -> float | None:
        url = f"{self.base_url.rstrip('/')}/pair/{self.pair_address}"
        try:
            response = await net.request_with_retries(
                self.client,
                "GET",
                url,
                headers={"user-agent": self.user_agent},
                attempts=3,
                base_backoff=0.5,
            )
        except httpx.HTTPError as exc:
            log.debug("meteora_price_fetch_failed", error=str(exc))
            return None

        try:
            payload: Any = response.json()
        except ValueError as exc:
            log.debug("meteora_price_json_error", error=str(exc))
            return None

        current_price = payload.get("current_price") if isinstance(payload, dict) else None
        try:
            price = float(current_price)
        except (TypeError, ValueError):
            return None

        if price <= 0 or not math.isfinite(price):
            return None
        return price


@dataclass(slots=True)
class BinanceVolSource(VolSource):
    client: httpx.AsyncClient
    base_url: str
    symbol: str
    cache_ttl: int
    max_stale: int
    user_agent: str

    async def read(self) -> volatility.VolReading | None:
        try:
            return await volatility.fetch_sigma_1h(
                self.client,
                base_url=self.base_url,
                symbol=self.symbol,
                cache_ttl=max(5, self.cache_ttl),
                max_stale=self.max_stale,
                user_agent=self.user_agent,
            )
        except httpx.HTTPError as exc:
            log.debug("binance_vol_fetch_failed", error=str(exc))
            return None
        except Exception as exc:  # Defensive: ensure callers see None on unexpected errors
            log.warning("binance_vol_unexpected_error", error=str(exc))
            return None


__all__ = [
    "BinanceVolSource",
    "MeteoraPriceSource",
    "PriceSource",
    "VolSource",
]
