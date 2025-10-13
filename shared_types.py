from __future__ import annotations

from typing import Literal

BandRange = tuple[float, float]
BandMap = dict[str, BandRange]
Side = Literal["low", "high"]

__all__ = ["BandRange", "BandMap", "Side"]
