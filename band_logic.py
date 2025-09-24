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
    # Deprecated: migrate callers to suggest_with_policy so they can supply a bucket.
    return suggest_with_policy(price, bands, broken, bucket="mid")


def format_advisory_card(advisory: Dict[str, object]) -> str:
    """Render the advisory payload into a multi-line human message."""

    price = float(advisory["price"])
    bucket = str(advisory["bucket"])
    sigma_pct = advisory.get("sigma_pct")
    split = advisory["split"]
    ranges = advisory["ranges"]

    sigma_display = "–"
    if sigma_pct is not None:
        sigma_display = f"{float(sigma_pct):.1f}%"

    split_str = "/".join(str(part) for part in split)
    header = (
        "bands @ p="
        f"{fmt_price(price)} | σ={sigma_display} ({bucket}) → split {split_str}"
    )

    widths, _ = band_advisor.widths_for_bucket(bucket)
    lines = [header]
    for name in sorted(ranges.keys()):
        lo, hi = ranges[name]
        width = widths.get(name)
        if width is None:
            continue
        pct_display = f"±{width * 100:.1f}%"
        lines.append(f"  {name} {pct_display} : {fmt_range(lo, hi)}")

    return "\n".join(lines)
