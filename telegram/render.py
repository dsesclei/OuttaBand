from __future__ import annotations

from html import escape

from band_logic import fmt_range, format_advisory_card
from policy import VolReading
from shared_types import BAND_ORDER, AmountsMap, BandMap, BandName, Bucket, BucketSplit
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def sigma_summary(sigma: VolReading | None) -> tuple[str, Bucket, float | None]:
    bucket: Bucket = sigma.bucket if sigma else "mid"
    label = escape(bucket.title())
    pct = float(sigma.sigma_pct) if sigma and sigma.sigma_pct is not None else None
    display = f"{pct:.2f}%" if pct is not None else "–"
    stale = " [<i>Stale</i>]" if sigma and sigma.stale else ""
    return f"<b>Volatility</b>: {display} ({label}){stale}", bucket, pct


def bands_lines(bands: BandMap) -> list[str]:
    if not bands:
        return ["(No bands configured)"]
    return [f"{escape(name.upper())}: {fmt_range(*rng)}" for name, rng in sorted(bands.items())]


def advisory_text(
    price: float,
    sigma_pct: float | None,
    bucket: Bucket,
    ranges: BandMap,
    split: BucketSplit,
    *,
    stale: bool = False,
    amounts: AmountsMap | None = None,
    unallocated_usd: float | None = None,
) -> str:
    return format_advisory_card(
        price,
        sigma_pct,
        bucket,
        ranges,
        split,
        stale=stale,
        amounts=amounts,
        unallocated_usd=unallocated_usd,
    )


def drift_summary(
    baseline: tuple[float, float, int] | None,
    latest: tuple[int, float, float, float, float] | None,
    price: float | None,
) -> str | None:
    if price and baseline and latest:
        base_sol, base_usdc, _ = baseline
        _, snap_sol, snap_usdc, _, _ = latest
        base_val = max(base_sol, 0.0) * price + max(base_usdc, 0.0)
        cur_val = max(snap_sol, 0.0) * price + max(snap_usdc, 0.0)
        if base_val > 0:
            drift = cur_val - base_val
            pct = (drift / base_val) * 100.0
            return f"<b>Drift</b>: ${drift:+.2f} ({pct:+.2f}%)"
    if latest:
        _, _, _, snap_price, snap_drift = latest
        return f"<b>Drift</b> (Last @ <b>{snap_price:.2f}</b>): ${snap_drift:+.2f}"
    return None


# Keyboards


def adv_kb(token: str) -> InlineKeyboardMarkup:
    from .callbacks import AdvAction, encode

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Apply All", callback_data=encode(AdvAction(a="apply", t=token))
                ),
                InlineKeyboardButton(
                    "Set Exact", callback_data=encode(AdvAction(a="set", t=token))
                ),
                InlineKeyboardButton(
                    "Ignore", callback_data=encode(AdvAction(a="ignore", t=token))
                ),
            ]
        ]
    )


def alert_kb(band: BandName, token: str) -> InlineKeyboardMarkup:
    from .callbacks import AlertAction, encode

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Apply", callback_data=encode(AlertAction(a="accept", band=band, t=token))
                ),
                InlineKeyboardButton(
                    "Ignore", callback_data=encode(AlertAction(a="ignore", band=band, t=token))
                ),
                InlineKeyboardButton(
                    "Set Exact", callback_data=encode(AlertAction(a="set", band=band, t=token))
                ),
            ]
        ]
    )


def bands_menu_text(bands: BandMap) -> str:
    if not bands:
        return "(No bands configured)"
    return "\n".join(
        ["<b>Configured Bands</b>:"]
        + [f"{escape(name.upper())}: {fmt_range(*rng)}" for name, rng in sorted(bands.items())]
    )


def bands_menu_kb() -> InlineKeyboardMarkup:
    from .callbacks import BandsAction, encode

    buttons = [
        [
            InlineKeyboardButton(
                f"Edit {band.upper()}", callback_data=encode(BandsAction(a="edit", band=band))
            )
        ]
        for band in BAND_ORDER
    ]
    buttons.append([InlineKeyboardButton("Back", callback_data=encode(BandsAction(a="back")))])
    return InlineKeyboardMarkup(buttons)
