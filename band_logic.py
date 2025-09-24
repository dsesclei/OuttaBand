from __future__ import annotations

from html import escape
from typing import Dict, Optional, Set, Tuple

import band_advisor


def fmt_price(x: float) -> str:
    return f"{x:.2f}"


def fmt_range(lo: float, hi: float) -> str:
    return f"{fmt_price(lo)}–{fmt_price(hi)}"  # en dash


def format_alert_message(
    band_name: str,
    side: str,
    price: float,
    source_label: Optional[str],
    bands: Dict[str, Tuple[float, float]],
) -> str:
    side_label = side.title()
    lines = ["⚠️ <b>Range Breach</b>"]
    lines.append(
        f"<b>Band</b>: {escape(band_name.upper())} ({escape(side_label)})"
    )

    price_line = f"<b>Price</b>: {fmt_price(price)}"
    if source_label:
        price_line += f" ({escape(source_label)})"
    lines.append(price_line)

    for name in sorted(bands.keys()):
        lo, hi = bands[name]
        lines.append(f"{escape(name.upper())}: {fmt_range(lo, hi)}")
    return "\n".join(lines)


def broken_bands(p: float, bands: Dict[str, Tuple[float, float]]) -> Set[str]:
    return {name for name, (lo, hi) in bands.items() if p < lo or p > hi}


def suggest_with_policy(
    price: float,
    bands: Dict[str, Tuple[float, float]],
    broken: Set[str],
    bucket: str,
) -> Dict[str, Tuple[float, float]]:
    """Return updated band suggestions while respecting high-vol skip rules.

    The helper builds fresh ranges via ``band_advisor.ranges_for_price`` with
    ``include_a_on_high=False`` so band ``"a"`` is omitted during high
    volatility buckets unless explicitly requested by the caller.
    """
    suggested = band_advisor.ranges_for_price(
        price,
        bucket,
        include_a_on_high=False,
    )
    new_bands = dict(bands)
    for name in broken:
        if name not in suggested:
            continue
        new_bands[name] = suggested[name]
    return new_bands
def format_advisory_card(
    price: float,
    sigma_pct: Optional[float],
    bucket: str,
    ranges: Dict[str, Tuple[float, float]],
    split: Tuple[int, int, int],
    *,
    stale: bool = False,
) -> str:
    sigma_display = "–"
    if sigma_pct is not None:
        sigma_display = f"{sigma_pct:.2f}%"

    bucket_label_raw = bucket or "Unknown"
    bucket_label = bucket_label_raw.title()
    header = (
        f"<b>Bands</b> at P=<b>{price:.2f}</b> | σ=<b>{sigma_display}</b> ({escape(bucket_label)}) "
        f"→ Split <b>{split[0]}/{split[1]}/{split[2]}</b>"
    )
    if stale:
        header = f"{header} [<i>Stale</i>]"

    widths, _ = band_advisor.widths_for_bucket(bucket)
    lines = [header]
    for name in ("a", "b", "c"):
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        width = widths.get(name)
        if width is None:
            continue
        pct_display = f"±{width * 100:.2f}%"
        lines.append(
            f"<b>{escape(name.upper())}</b> ({pct_display}): {fmt_range(lo, hi)}"
        )

    return "\n".join(lines)
