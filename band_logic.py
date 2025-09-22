from __future__ import annotations

from typing import Dict, Optional, Set, Tuple


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


def suggest_new_bands_stub(
    price: float,
    bands: Dict[str, Tuple[float, float]],
    broken: Set[str],
) -> Optional[Dict[str, Tuple[float, float]]]:
    from main import jlog  # local import to avoid circular dependency

    jlog("info", "suggest_new_bands_stub", price=price, bands=len(bands), broken=sorted(broken))
    return None
