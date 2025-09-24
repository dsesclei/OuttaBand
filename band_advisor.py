"""Band policy helpers.

This module centralises the band policy so all features reuse the same
decisions. It exposes helpers for the bucket → width table, sigma-based split
recommendations, range building, and simple quantisation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


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
) -> Dict[str, object]:
    """Construct a structured advisory payload for the supplied state.

    The payload keeps all inputs plus the derived policy pieces so downstream
    consumers (e.g. Telegram formatting, HTTP responses) can reuse the same
    data without needing to recompute anything.
    """

    return {
        "price": price,
        "sigma_pct": sigma_pct,
        "bucket": bucket,
        "split": split_for_sigma(sigma_pct),
        "ranges": ranges_for_price(
            price,
            bucket,
            include_a_on_high=False,
        ),
    }
