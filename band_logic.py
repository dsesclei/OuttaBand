from __future__ import annotations

from html import escape
from typing import Dict, Optional, Set, Tuple

import band_advisor


def fmt_price(x: float) -> str:
    return f"{x:.2f}"


def fmt_range(lo: float, hi: float) -> str:
    return f"{fmt_price(lo)}–{fmt_price(hi)}"  # en dash


def broken_bands(p: float, bands: Dict[str, Tuple[float, float]]) -> Set[str]:
    return {name for name, (lo, hi) in bands.items() if p < lo or p > hi}

def format_advisory_card(
    price: float,
    sigma_pct: Optional[float],
    bucket: str,
    ranges: Dict[str, Tuple[float, float]],
    split: Tuple[int, int, int],
    *,
    stale: bool = False,
    amounts: Optional[Dict[str, Tuple[float, float]]] = None,
    unallocated_usd: Optional[float] = None,
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
        line = f"<b>{escape(name.upper())}</b> ({pct_display}): {fmt_range(lo, hi)}"
        if amounts and name in amounts:
            sol_amt, usdc_amt = amounts[name]
            line = (
                f"{line} → {sol_amt:.6f} SOL / ${usdc_amt:.2f} USDC"
            )
        lines.append(line)

    if unallocated_usd is not None and unallocated_usd > 0.005:
        lines.append(f"<i>Unallocated</i>: ${unallocated_usd:.2f}")

    return "\n".join(lines)
