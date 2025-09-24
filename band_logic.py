from __future__ import annotations

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
    lines = []
    lines.append("◧ hard break")
    lines.append(f"band: {band_name.upper()} ({side})")

    price_line = f"price: {fmt_price(price)}"
    if source_label:
        price_line += f" ({source_label})"
    lines.append(price_line)

    for name in sorted(bands.keys()):
        lo, hi = bands[name]
        lines.append(f"{name}: {fmt_range(lo, hi)}")
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


def suggest_new_bands(
    price: float,
    bands: Dict[str, Tuple[float, float]],
    broken: Set[str],
) -> Dict[str, Tuple[float, float]]:
    raise NotImplementedError(
        "use suggest_with_policy(price, bands, broken, bucket)"
    )


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
    if stale:
        sigma_display = f"{sigma_display} (STALE)"

    header = (
        f"bands @ p={price:.2f} | σ={sigma_display} ({bucket}) "
        f"→ split {split[0]}/{split[1]}/{split[2]}"
    )

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
        lines.append(f"{name} {pct_display} : {fmt_range(lo, hi)}")

    return "\n".join(lines)
