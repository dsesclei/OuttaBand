"""Volatility source implementations grouped under the policy package."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

from . import volatility

log = structlog.get_logger("policy.sources")


class VolSource:
    async def read(self) -> volatility.VolReading | None:  # pragma: no cover - interface
        raise NotImplementedError


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


__all__ = ["BinanceVolSource", "VolSource"]
