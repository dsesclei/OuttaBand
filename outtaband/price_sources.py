from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, SupportsFloat, SupportsIndex, cast

import httpx
import structlog

from . import net

log = structlog.get_logger("price_sources")


class PriceSource:
    async def read(self) -> float | None:  # pragma: no cover - interface
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

        current_price_obj = payload.get("current_price") if isinstance(payload, dict) else None
        if current_price_obj is None:
            return None
        try:
            price = float(cast(SupportsFloat | SupportsIndex | str, current_price_obj))
        except (TypeError, ValueError):
            return None

        if price <= 0 or not math.isfinite(price):
            return None
        return price


__all__ = [
    "MeteoraPriceSource",
    "PriceSource",
]
