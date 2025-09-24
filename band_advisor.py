"""Band policy helpers.

This module centralises the band policy so all features reuse the same
decisions. It exposes helpers for the bucket → width table, sigma-based split
recommendations, range building, and simple quantisation.
"""
from __future__ import annotations

import math

from typing import Any, Dict, Optional, Set, Tuple


_BUCKET_WIDTHS: Dict[str, Dict[str, float]] = {
    "low": {
        "a": 0.0035,
        "b": 0.011,
        "c": 0.018,
    },
    "mid": {
        "a": 0.006,
        "b": 0.018,
        "c": 0.030,
    },
    "high": {
        "a": 0.009,
        "b": 0.025,
        "c": 0.040,
    },
}


def widths_for_bucket(bucket: str) -> Tuple[Dict[str, float], bool]:
    """Return percent widths for the requested sigma bucket.

    The table encodes the policy mapping σ bucket → percent ranges. Buckets are
    the strings ``"low"``, ``"mid"``, and ``"high"``. The accompanying boolean
    indicates whether band ``"a"`` should be skipped by default for the bucket.
    Currently only the ``"high"`` bucket sets ``skip_a`` to ``True`` so callers
    can opt-in when they truly need band ``a`` during elevated volatility.
    Width values are stored as fractions of price (e.g. ``0.006`` == ``0.6%``).
    """

    if bucket not in _BUCKET_WIDTHS:
        raise ValueError(f"Unknown bucket: {bucket}")

    widths = _BUCKET_WIDTHS[bucket]
    skip_a = bucket == "high"
    return widths, skip_a


def split_for_sigma(sigma_pct: float | None) -> Tuple[int, int, int]:
    """Return the advisory split for the supplied sigma percentage.

    ``sigma_pct`` is expected as a percentage (e.g. ``0.6`` for 0.6%). ``None``
    falls back to the default distribution. Values below 0.6% use a 60/20/20
    split, while values at or above the threshold tighten the first band to a
    50/30/20 split. The tuple is intended for display only, so it remains in
    integers.
    """

    if sigma_pct is None or sigma_pct < 0.6:
        return (60, 20, 20)
    return (50, 30, 20)


def split_for_bucket(bucket: str) -> Tuple[int, int, int]:
    """Return the advisory split for a bucket identifier."""

    if bucket == "low":
        return (60, 20, 20)
    if bucket in {"mid", "high"}:
        return (50, 30, 20)
    raise ValueError(f"unknown bucket: {bucket}")


def compute_amounts(
    price: float,
    split: Tuple[int, int, int],
    ranges: Dict[str, Tuple[float, float]],
    notional_usd: Optional[float],
    tilt_sol_frac: float,
    *,
    present_bands: Optional[Set[str]] = None,
    redistribute_skipped: bool = False,
    sol_decimals: int = 6,
    usdc_decimals: int = 2,
) -> Tuple[Dict[str, Tuple[float, float]], float]:
    """Compute per-band token amounts using a simple tilt heuristic."""

    if (
        notional_usd is None
        or notional_usd <= 0
        or price <= 0
        or not math.isfinite(price)
        or not math.isfinite(tilt_sol_frac)
    ):
        return {}, 0.0

    tilt = min(max(tilt_sol_frac, 0.0), 1.0)
    band_order = ("a", "b", "c")
    split_map = {band: pct for band, pct in zip(band_order, split)}

    present = set(ranges.keys()) if present_bands is None else set(present_bands)
    present &= set(band_order)

    if not present:
        return {}, 0.0

    total_present_pct = sum(split_map.get(band, 0) for band in present)
    total_pct = sum(split_map.values())
    skipped_pct = max(total_pct - total_present_pct, 0)

    if redistribute_skipped and total_present_pct:
        scale = total_pct / total_present_pct
    else:
        scale = 1.0

    amounts: Dict[str, Tuple[float, float]] = {}
    for band in present:
        pct = split_map.get(band, 0) * scale
        per_band_usd = (notional_usd * pct) / 100.0
        sol_amt = round((per_band_usd * tilt) / price, sol_decimals)
        usdc_usd = round(per_band_usd * (1.0 - tilt), usdc_decimals)
        amounts[band] = (sol_amt, usdc_usd)

    unallocated_usd = 0.0 if redistribute_skipped else round((notional_usd * skipped_pct) / 100.0, usdc_decimals)
    return amounts, unallocated_usd


def ranges_for_price(
    price: float,
    bucket: str,
    *,
    include_a_on_high: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Build absolute ranges for ``price`` under the given ``bucket`` policy.

    The helper uses ``widths_for_bucket`` to determine the percent widths. When
    the bucket is ``"high"`` it omits band ``"a"`` unless ``include_a_on_high``
    is set, which lets callers opt in explicitly without mutating stored bands.
    Ranges are returned as raw floats so formatting layers can choose how to
    present the numbers.
    """

    widths, skip_a = widths_for_bucket(bucket)
    ranges: Dict[str, Tuple[float, float]] = {}
    for band, width in widths.items():
        if skip_a and band == "a" and not include_a_on_high:
            continue
        delta = price * width
        ranges[band] = (price - delta, price + delta)
    return ranges


def build_advisory(
    price: float,
    sigma_pct: Optional[float],
    bucket: str,
    *,
    include_a_on_high: bool = False,
) -> Dict[str, Any]:
    """Construct a structured advisory payload for the supplied state.

    The payload keeps all inputs plus the derived policy pieces so downstream
    consumers (e.g. Telegram formatting, HTTP responses) can reuse the same
    data without needing to recompute anything. Callers can opt in to include
    band ``"a"`` during ``"high"`` bucket volatility via ``include_a_on_high``.
    """

    return {
        "price": price,
        "sigma_pct": sigma_pct,
        "bucket": bucket,
        "split": split_for_sigma(sigma_pct),
        "ranges": ranges_for_price(
            price,
            bucket,
            include_a_on_high=include_a_on_high,
        ),
    }
