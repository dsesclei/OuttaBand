from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from html import escape
from typing import Any

from policy.band_advisor import compute_amounts, ranges_for_price, split_for_sigma
from db_repo import DBRepo
from shared_types import BAND_ORDER
from policy import VolReading

from .render import bands_lines, bands_menu_kb, bands_menu_text, drift_summary, sigma_summary

_USAGE_SETBASELINE = "[<i>Usage</i>] <code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code> (e.g. <code>/setbaseline 1.0 sol 200 usdc</code>)"
_USAGE_UPDATEBAL = "[<i>Usage</i>] <code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code> (e.g. <code>/updatebalances 1.0 sol 200 usdc</code>)"
_USAGE_SETNOTIONAL = "[<i>Usage</i>] <code>/setnotional &lt;usd&gt;</code> (e.g. <code>/setnotional 2500</code>)"
_USAGE_SETTILT = "[<i>Usage</i>] <code>/settilt &lt;sol&gt;:&lt;usdc&gt;</code> (e.g. <code>/settilt 60:40</code> or <code>/settilt 0.6</code>)"

class BotCtx:
    """Minimal façade for sending or editing messages via TelegramApp."""

    def __init__(
        self,
        send: Callable[[str, Any | None], Awaitable[None]],
        edit_or_send: Callable[[Any, str, Any | None], Awaitable[None]],
        edit_by_id: Callable[[int, str], Awaitable[bool]],
    ):
        self.send = send
        self.edit_or_send = edit_or_send
        self.edit_by_id = edit_by_id


class Providers:
    def __init__(
        self,
        price_provider: Callable[[], Awaitable[float | None]] | None = None,
        sigma_provider: Callable[[], Awaitable[VolReading | None]] | None = None,
    ):
        self.price_provider = price_provider
        self.sigma_provider = sigma_provider


class Handlers:
    def __init__(self, repo: DBRepo, providers: Providers):
        self.repo = repo
        self.providers = providers

    # Helpers (parsing/formatting)

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            val = float(value)
        except Exception:
            return 0.0
        return min(1.0, max(0.0, val))

    @staticmethod
    def _extract_numbers(text: str, count: int, *, ignore_labels: set[str] | None = None) -> list[float]:
        labels = {label.lower() for label in (ignore_labels or set())}
        tokens = text.replace(":", " ").replace("/", " ").replace(",", " ").split()
        numbers: list[float] = []
        for token in tokens:
            if token.startswith("/"):
                continue
            if token.strip().lower() in labels:
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            numbers.append(value)
            if len(numbers) == count:
                break
        return numbers

    def _parse_first_number(self, text: str) -> float | None:
        numbers = self._extract_numbers(text, 1)
        return numbers[0] if numbers else None

    def _parse_two_numbers(self, text: str) -> tuple[float, float] | None:
        numbers = self._extract_numbers(text, 2, ignore_labels={"sol", "usdc"})
        if len(numbers) < 2:
            return None
        return numbers[0], numbers[1]

    def _parse_tilt(self, text: str) -> tuple[float, float] | None:
        numbers = self._extract_numbers(text, 2, ignore_labels={"sol", "usdc"})
        if not numbers:
            return None
        if len(numbers) == 1:
            sol_raw = numbers[0]
            sol_frac = sol_raw / (100.0 if sol_raw > 1 else 1.0)
            sol_frac = self._clamp01(sol_frac)
            return sol_frac, 1.0 - sol_frac
        sol_raw, usdc_raw = numbers[0], numbers[1]
        if not (math.isfinite(sol_raw) and math.isfinite(usdc_raw)):
            return None
        total = sol_raw + usdc_raw
        if not math.isfinite(total) or total <= 0:
            return None
        sol_frac = self._clamp01(sol_raw / total)
        return sol_frac, 1.0 - sol_frac

    @staticmethod
    def _pct_str(frac: float) -> str:
        try:
            pct = round(float(frac) * 100.0, 1)
        except Exception:
            pct = 0.0
        if not math.isfinite(pct):
            pct = 0.0
        return str(int(pct)) if pct.is_integer() else f"{pct:.1f}"

    # Command handlers

    async def cmd_start(self, bctx: BotCtx) -> None:
        await bctx.send("[<i>Online</i>] Bot ready. Use <code>/status</code> for details.", None)

    async def cmd_help(self, bctx: BotCtx) -> None:
        lines = [
            "<b>Commands</b>:",
            "<code>/status</code> — latest price, σ, bands, policy split",
            "<code>/bands</code> — edit band ranges",
            "<code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code>",
            "<code>/setnotional &lt;usd&gt;</code> — total usd to allocate across bands",
            "<code>/settilt &lt;sol&gt;:&lt;usdc&gt;</code> — inside-band split, sol first (e.g. 50:50, 60:40)",
            "<code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code>",
            "<code>/help</code> — show this list",
            "Token amounts are advisory (tilt-based), not exact DLMM quotes.",
        ]
        await bctx.send("\n".join(lines), None)

    async def cmd_status(self, bctx: BotCtx) -> None:
        repo = self.repo
        price = await (self.providers.price_provider() if self.providers.price_provider else None)
        sigma = await (self.providers.sigma_provider() if self.providers.sigma_provider else None)
        bands = await repo.get_bands()
        latest = await repo.get_latest_snapshot()
        baseline = await repo.get_baseline()

        lines: list[str] = []
        lines.append(f"<b>Price</b>: {price:.2f}" if (price and math.isfinite(price)) else "<b>Price</b>: Unknown")

        sigma_line, bucket, sigma_pct = sigma_summary(sigma)
        lines.append(sigma_line)

        lines.append("<b>Configured Bands</b>:")
        lines.extend(bands_lines(bands))

        split = split_for_sigma(sigma_pct)
        lines.append(f"<b>Advisory Split</b>: {split[0]}/{split[1]}/{split[2]} ({escape(bucket.title())})")

        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()
        sol_pct = self._pct_str(tilt_sol_frac)
        usdc_pct = self._pct_str(1.0 - tilt_sol_frac)
        notional_display = "unset" if not (notional and notional > 0) else f"${notional:.2f}"
        lines.append(f"notional: {escape(notional_display)} | tilt sol/usdc: {escape(sol_pct)}/{escape(usdc_pct)}")

        if latest:
            _snap_ts, snap_sol, snap_usdc, _snap_price, _snap_drift = latest
            lines.append(f"<b>Balances</b>: {snap_sol:g} SOL, {snap_usdc:g} USDC")
        else:
            lines.append("<b>Balances</b>: (none; run /updatebalances)")

        if price and (notional and notional > 0):
            planned = {}
            with suppress(ValueError):
                planned = ranges_for_price(price, bucket or "mid", include_a_on_high=False)
            if planned:
                amounts_map, unallocated = compute_amounts(price, split, planned, notional, tilt_sol_frac)
                for band in BAND_ORDER:
                    if band in amounts_map:
                        sol_amt, usdc_amt = amounts_map[band]
                        lines.append(f"{band.upper()} amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f} USDC")
                if unallocated > 0.005:
                    lines.append(f"unallocated: ${unallocated:.2f}")

        drift_line = drift_summary(baseline, latest, price)
        if drift_line:
            lines.append(drift_line)

        await bctx.send("\n".join(lines), None)

    async def cmd_bands(self, bctx: BotCtx) -> None:
        bands = await self.repo.get_bands()
        await bctx.send(bands_menu_text(bands), bands_menu_kb())

    async def cmd_setbaseline(self, bctx: BotCtx, text: str) -> None:
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await bctx.send(_USAGE_SETBASELINE, None)
            return
        sol, usdc = parsed
        sol = max(sol, 0.0)
        usdc = max(usdc, 0.0)
        if not (math.isfinite(sol) and math.isfinite(usdc)):
            await bctx.send(_USAGE_SETBASELINE, None)
            return
        ts = int(time.time())
        await self.repo.set_baseline(sol, usdc, ts)
        await bctx.send(f"[<i>Applied</i>] Baseline → {sol:g} SOL, {usdc:g} USDC", None)

    async def cmd_updatebalances(self, bctx: BotCtx, text: str) -> None:
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await bctx.send(_USAGE_UPDATEBAL, None)
            return
        sol_amt, usdc_amt = parsed
        sol_amt = max(sol_amt, 0.0)
        usdc_amt = max(usdc_amt, 0.0)
        if not (math.isfinite(sol_amt) and math.isfinite(usdc_amt)):
            await bctx.send(_USAGE_UPDATEBAL, None)
            return

        price = await (self.providers.price_provider() if self.providers.price_provider else None)
        if price is None or not math.isfinite(price) or price <= 0:
            await bctx.send("[<i>Error</i>] Price unavailable. Try again later.", None)
            return

        baseline = await self.repo.get_baseline()
        if baseline is None:
            await bctx.send("[<i>Error</i>] Baseline not set. Run <code>/setbaseline</code> first.", None)
            return

        base_sol, base_usdc, _ = baseline
        base_val = max(base_sol, 0.0) * price + max(base_usdc, 0.0)
        cur_val = max(sol_amt, 0.0) * price + max(usdc_amt, 0.0)

        drift = cur_val - base_val
        drift_pct = (drift / base_val) if base_val > 0 else 0.0
        if not math.isfinite(drift_pct):
            drift_pct = 0.0

        ts = int(time.time())
        await self.repo.insert_snapshot(ts, sol_amt, usdc_amt, price, drift)
        await bctx.send(
            (
                f"[<i>Applied</i>] <b>Drift</b>: ${drift:+.2f} ({drift_pct * 100:+.2f}%) | "
                f"<b>Base</b> ${base_val:.2f} → <b>Now</b> ${cur_val:.2f} @ Price <b>{price:.2f}</b>"
            ),
            None,
        )

    async def cmd_setnotional(self, bctx: BotCtx, text: str) -> None:
        value = self._parse_first_number(text)
        if value is None or not math.isfinite(value) or value < 0:
            await bctx.send(_USAGE_SETNOTIONAL, None)
            return
        try:
            await self.repo.set_notional_usd(float(value))
        except ValueError:
            await bctx.send("[<i>Error</i>] Notional must be finite and non-negative.", None)
            return
        await bctx.send(f"[<i>Applied</i>] Notional → ${float(value):.2f}", None)

    async def cmd_settilt(self, bctx: BotCtx, text: str) -> None:
        parsed = self._parse_tilt(text)
        if parsed is None:
            await bctx.send(_USAGE_SETTILT, None)
            return
        sol_frac, usdc_frac = parsed
        if not (math.isfinite(sol_frac) and math.isfinite(usdc_frac)):
            await bctx.send("[<i>Error</i>] Invalid tilt values.", None)
            return
        total = sol_frac + usdc_frac
        if not math.isfinite(total) or total <= 0:
            await bctx.send("[<i>Error</i>] Tilt must allocate some share to SOL or USDC.", None)
            return
        sol_frac = self._clamp01(sol_frac)
        usdc_frac = 1.0 - sol_frac
        await self.repo.set_tilt_sol_frac(sol_frac)
        await bctx.send(
            f"[<i>Applied</i>] Tilt → SOL/USDC = {escape(self._pct_str(sol_frac))}/{escape(self._pct_str(usdc_frac))}",
            None,
        )
