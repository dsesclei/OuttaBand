"""Orchestration helpers for price checks and advisories."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import band_advisor
import policy_engine
from db_repo import DBRepo
from shared_types import BandMap, Bucket, Side
from sources import PriceSource, VolSource
from structlog.typing import FilteringBoundLogger
from telegram import TelegramApp


@dataclass(slots=True)
class JobSettings:
    check_every_minutes: int
    cooldown_minutes: int
    include_a_on_high: bool
    price_label: str = "meteora"


@dataclass(slots=True)
class AppContext:
    repo: DBRepo
    tg: TelegramApp
    price: PriceSource
    vol: VolSource
    job: JobSettings
    log: FilteringBoundLogger


def floor_to_slot(ts: int, slot_seconds: int) -> int:
    """Snap a timestamp down to the nearest slot boundary."""

    if slot_seconds <= 0:
        raise ValueError("slot_seconds must be positive")
    return (ts // slot_seconds) * slot_seconds


async def check_once(ctx: AppContext, *, now_ts: int | None = None) -> None:
    """Run a single price check cycle."""

    price = await ctx.price.read()
    if price is None:
        ctx.log.warning("price_missing", source=ctx.job.price_label)
        return
    if not math.isfinite(price) or price <= 0:
        ctx.log.warning("price_invalid", price=price, source=ctx.job.price_label)
        return

    ctx.log.info("price_ok", source=ctx.job.price_label, price=price)

    sigma = await ctx.vol.read()
    sigma_pct_log: float | None = None
    if sigma is not None:
        try:
            sigma_pct_raw = float(sigma.sigma_pct)
        except (TypeError, ValueError):
            sigma_pct_raw = None
        else:
            if math.isfinite(sigma_pct_raw):
                sigma_pct_log = round(sigma_pct_raw, 3)

    log_kwargs = {
        "sigma_pct": sigma_pct_log,
        "bucket": sigma.bucket if sigma else None,
        "stale": sigma.stale if sigma else None,
        "sample_count": sigma.sample_count if sigma else None,
        "as_of_ts": sigma.as_of_ts if sigma else None,
    }
    ctx.log.info("sigma_ok" if sigma else "sigma_miss", **log_kwargs)

    bucket: Bucket = sigma.bucket if sigma else "mid"
    if sigma is None:
        ctx.log.warning(
            "bucket_missing",
            fallback="mid",
            include_a_on_high=ctx.job.include_a_on_high,
            source=ctx.job.price_label,
        )

    slot_seconds = max(60, ctx.job.check_every_minutes * 60)
    base_ts = int(now_ts if now_ts is not None else time.time())
    now_aligned = floor_to_slot(base_ts, slot_seconds)
    cooldown_secs = max(60, ctx.job.cooldown_minutes * 60)

    bands: BandMap = await ctx.repo.get_bands()
    suggestions = policy_engine.compute_breaches(
        price,
        bands,
        bucket,
        include_a_on_high=ctx.job.include_a_on_high,
    )

    sent = 0
    for suggestion in suggestions:
        side: Side = suggestion.side

        last = await ctx.repo.get_last_alert(suggestion.band, side)
        last_aligned = floor_to_slot(last, slot_seconds) if last is not None else None
        if last_aligned is not None:
            delta = now_aligned - last_aligned
            if delta < cooldown_secs:
                ctx.log.info(
                    "cooldown_skip",
                    band=suggestion.band,
                    side=side,
                    seconds_remaining=cooldown_secs - delta,
                    now_aligned=now_aligned,
                    last_aligned=last_aligned,
                )
                continue

        ctx.log.info(
            "breach_offer",
            band=suggestion.band,
            side=side,
            price=price,
            bucket=bucket,
            suggested_lo=suggestion.suggested[0],
            suggested_hi=suggestion.suggested[1],
        )
        await ctx.tg.send_breach_offer(
            band=suggestion.band,
            price=price,
            src_label=ctx.job.price_label,
            bands=bands,
            suggested_range=suggestion.suggested,
            policy_meta=suggestion.policy_meta,
        )
        await ctx.repo.set_last_alert(suggestion.band, side, now_aligned)
        sent += 1

    if sent == 0:
        ctx.log.info("no_breach", price=price, source=ctx.job.price_label)


async def send_daily_advisory(ctx: AppContext) -> None:
    """Send the daily policy advisory card."""

    price = await ctx.price.read()
    if price is None:
        ctx.log.warning("advisory_price_missing")
        return
    if not math.isfinite(price) or price <= 0:
        ctx.log.warning("advisory_price_invalid", price=price)
        return

    sigma = await ctx.vol.read()
    if sigma is None:
        ctx.log.warning("sigma_miss_daily")

    bucket: Bucket = sigma.bucket if sigma else "mid"

    sigma_pct: float | None = None
    if sigma is not None:
        try:
            sigma_pct_raw = float(sigma.sigma_pct)
        except (TypeError, ValueError):
            sigma_pct_raw = None
        else:
            if math.isfinite(sigma_pct_raw):
                sigma_pct = sigma_pct_raw

    advisory = band_advisor.build_advisory(
        price,
        sigma_pct,
        bucket,
        include_a_on_high=ctx.job.include_a_on_high,
    )
    if sigma:
        advisory["stale"] = bool(sigma.stale)

    baseline = await ctx.repo.get_baseline()
    latest = await ctx.repo.get_latest_snapshot()

    drift_line: str | None = None
    if baseline and latest:
        base_sol, base_usdc, _ = baseline
        _snap_ts, snap_sol, snap_usdc, _snap_price, _snap_drift = latest
        base_val_now = max(base_sol, 0.0) * price + max(base_usdc, 0.0)
        cur_val_now = max(snap_sol, 0.0) * price + max(snap_usdc, 0.0)
        if math.isfinite(base_val_now) and math.isfinite(cur_val_now):
            drift_now = cur_val_now - base_val_now
            if math.isfinite(drift_now):
                drift_pct = (drift_now / base_val_now) if base_val_now > 0 else 0.0
                if not math.isfinite(drift_pct):
                    drift_pct = 0.0
                drift_line = f"<b>Drift</b>: ${drift_now:+.2f} ({drift_pct * 100:+.2f}%)"

    await ctx.tg.send_advisory_card(advisory, drift_line=drift_line)
    ctx.log.info(
        "advisory_sent",
        price=price,
        bucket=bucket,
        sigma_pct=sigma_pct,
        stale=bool(advisory.get("stale")),
    )


__all__ = [
    "AppContext",
    "JobSettings",
    "check_once",
    "floor_to_slot",
    "send_daily_advisory",
]
