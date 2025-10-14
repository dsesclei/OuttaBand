"""Pure policy helpers for band breach evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import policy.band_advisor as band_policy
from shared_types import BAND_ORDER, BandMap, BandName, BandRange, Bucket, Side


@dataclass(slots=True)
class BreachSuggestion:
    band: BandName
    side: Side
    current: BandRange
    suggested: BandRange
    policy_meta: tuple[Bucket, float] | None


def suggest_ranges(price: float, bucket: Bucket, include_a_on_high: bool) -> BandMap:
    """Return policy ranges for ``price`` under ``bucket``."""

    return band_policy.ranges_for_price(price, bucket, include_a_on_high=include_a_on_high)


def compute_breaches(
    price: float,
    bands: BandMap,
    bucket: Bucket,
    include_a_on_high: bool,
) -> list[BreachSuggestion]:
    """Compare stored bands with policy suggestions and flag breaches."""

    suggestions = suggest_ranges(price, bucket, include_a_on_high)
    widths, _ = band_policy.widths_for_bucket(bucket)
    breaches: list[BreachSuggestion] = []

    for band in BAND_ORDER:
        if band not in bands:
            continue

        current_range = bands[band]
        suggested = suggestions.get(band)
        if suggested is None:
            continue

        lo, hi = current_range
        if price < lo:
            side: Side = "low"
        elif price > hi:
            side = "high"
        else:
            continue

        width = widths.get(band)
        policy_meta = (bucket, width) if width is not None else None

        breaches.append(
            BreachSuggestion(
                band=band,
                side=side,
                current=current_range,
                suggested=suggested,
                policy_meta=policy_meta,
            )
        )

    return breaches


__all__ = ["BreachSuggestion", "compute_breaches", "suggest_ranges"]
