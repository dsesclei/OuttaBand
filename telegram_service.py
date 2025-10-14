from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from functools import wraps
from html import escape
from typing import Any, cast

from structlog.typing import FilteringBoundLogger
from telegram import CallbackQuery, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    filters,
)

from band_advisor import compute_amounts, ranges_for_price, split_for_bucket, split_for_sigma
from band_logic import fmt_range, format_advisory_card
from db_repo import DBRepo
from shared_types import (
    BAND_ORDER,
    AdvisoryPayload,
    AlertPayload,
    BandMap,
    BandName,
    BandRange,
    Baseline,
    Bucket,
    BucketSplit,
    PendingKind,
    PendingPayload,
    Snapshot,
)
from volatility import VolReading


def _auth_cmd(fn):
    @wraps(fn)
    async def wrapper(self, update, context, *args, **kwargs):
        if not self._is_authorized(update):
            await self._reply_unauthorized(update, context)
            return None
        return await fn(self, update, context, *args, **kwargs)

    return wrapper


def _auth_cb(fn):
    @wraps(fn)
    async def wrapper(self, update, context, *args, **kwargs):
        query = update.callback_query
        if query is None:
            return None
        if not self._is_authorized(update):
            await self._alert_unauthorized(query)
            return None
        return await fn(self, update, context, *args, **kwargs)

    return wrapper


class TelegramSvc:
    MAX_PENDING = int(os.getenv("LPBOT_PENDING_CAP", "100"))
    _UNAUTHORIZED_HTML = "[<i>Unauthorized</i>]"
    _USAGE_SETBASELINE = "[<i>Usage</i>] <code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code> (e.g. <code>/setbaseline 1.0 sol 200 usdc</code>)"
    _USAGE_UPDATEBAL = "[<i>Usage</i>] <code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code> (e.g. <code>/updatebalances 1.0 sol 200 usdc</code>)"
    _USAGE_SETNOTIONAL = "[<i>Usage</i>] <code>/setnotional &lt;usd&gt;</code> (e.g. <code>/setnotional 2500</code>)"
    _USAGE_SETTILT = "[<i>Usage</i>] <code>/settilt &lt;sol&gt;:&lt;usdc&gt;</code> (e.g. <code>/settilt 60:40</code> or <code>/settilt 0.6</code>)"
    _STALE_ALERT = "[<i>Stale</i>] This alert is no longer active."
    _STALE_ADV = "[<i>Stale</i>] This advisory is no longer active."
    _STALE_GENERIC = "[<i>Stale</i>] No longer active."
    _PENDING_ALERT: PendingKind = "alert"
    _PENDING_ADV: PendingKind = "adv"

    def __init__(self, token: str, chat_id: int, logger: FilteringBoundLogger | None = None) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: Application | None = None
        self._repo: DBRepo | None = None
        self._ready: asyncio.Event = asyncio.Event()
        self._pending: dict[int, tuple[PendingKind, PendingPayload]] = {}
        self._log: FilteringBoundLogger | None = logger
        self._price_provider: Callable[[], Awaitable[float | None]] | None = None
        self._sigma_provider: Callable[[], Awaitable[VolReading | None]] | None = None

    def _prune_pending(self) -> None:
        cap = max(1, self.MAX_PENDING)
        if len(self._pending) <= cap:
            return
        # prune oldest message ids (Telegram message IDs increase monotonically per chat)
        for mid in sorted(self._pending.keys()):
            if len(self._pending) <= cap:
                break
            del self._pending[mid]

    async def _send(self, text: str, reply_markup: Any | None = None) -> None:
        app = self._ensure_app()
        await self._ready.wait()
        await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=reply_markup,
        )

    async def _edit_or_send(
        self,
        query: CallbackQuery,
        text: str,
        reply_markup: Any | None = None,
    ) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except Exception:
            await self._send(text, reply_markup=reply_markup)

    async def _edit_by_id(self, message_id: int, text: str) -> bool:
        try:
            await self._ensure_app().bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=message_id,
                text=text,
            )
            return True
        except Exception:
            return False

    async def _pop_pending(
        self,
        query: CallbackQuery,
        expected_kind: PendingKind,
        *,
        stale_text: str | None = None,
    ) -> tuple[int, PendingPayload] | None:
        message = query.message
        mid = message.message_id if message else None
        if mid is None:
            await query.answer(stale_text or self._STALE_GENERIC, show_alert=True)
            return None
        entry = self._pending.get(mid)
        if not entry or entry[0] != expected_kind:
            await query.answer(stale_text or self._STALE_GENERIC, show_alert=True)
            return None
        return mid, entry[1]

    def _clamp01(self, value: float) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        return min(1.0, max(0.0, v))

    def _extract_numbers(
        self,
        text: str,
        count: int,
        *,
        ignore_labels: set[str] | None = None,
    ) -> list[float]:
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

    def _bands_lines(self, bands: BandMap) -> list[str]:
        if not bands:
            return ["(No bands configured)"]
        return [f"{escape(name.upper())}: {fmt_range(*rng)}" for name, rng in sorted(bands.items())]

    def _sigma_summary(self, sigma: VolReading | None) -> tuple[str, Bucket, float | None]:
        bucket: Bucket = sigma.bucket if sigma else "mid"
        label = escape(bucket.title())
        pct = float(sigma.sigma_pct) if sigma and math.isfinite(sigma.sigma_pct) else None
        display = f"{pct:.2f}%" if pct is not None else "–"
        stale = " [<i>Stale</i>]" if sigma and sigma.stale else ""
        return f"<b>Volatility</b>: {display} ({label}){stale}", bucket, pct

    def _amounts_lines(
        self,
        price: float | None,
        bucket: Bucket,
        notional: float | None,
        tilt_sol_frac: float,
        split: BucketSplit,
    ) -> list[str]:
        if not (notional and notional > 0 and price and math.isfinite(price) and price > 0):
            return []
        planned: BandMap = {}
        with suppress(ValueError):
            planned = ranges_for_price(price, bucket, include_a_on_high=False)
        if not planned:
            return []
        amounts_map, unallocated = compute_amounts(
            price,
            split,
            planned,
            notional,
            tilt_sol_frac,
        )
        if not amounts_map:
            return []
        lines: list[str] = []
        for band in BAND_ORDER:
            if band in amounts_map:
                sol_amt, usdc_amt = amounts_map[band]
                lines.append(f"{band.upper()} amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f} USDC")
        if unallocated > 0.005:
            lines.append(f"unallocated: ${unallocated:.2f}")
        return lines

    def _drift_summary(
        self,
        baseline: Baseline | None,
        latest: Snapshot | None,
        price: float | None,
    ) -> str | None:
        if price and math.isfinite(price) and price > 0 and baseline and latest:
            base_sol, base_usdc, _ = baseline
            _, snap_sol, snap_usdc, _, _ = latest
            base_val = max(base_sol, 0.0) * price + max(base_usdc, 0.0)
            cur_val = max(snap_sol, 0.0) * price + max(snap_usdc, 0.0)
            if math.isfinite(base_val) and math.isfinite(cur_val):
                drift = cur_val - base_val
                if math.isfinite(drift):
                    pct = (drift / base_val) if base_val > 0 else 0.0
                    pct = pct if math.isfinite(pct) else 0.0
                    return f"<b>Drift</b>: ${drift:+.2f} ({pct * 100:+.2f}%)"
        if latest:
            _, _, _, snap_price, snap_drift = latest
            if math.isfinite(snap_price) and math.isfinite(snap_drift):
                return f"<b>Drift</b> (Last @ <b>{snap_price:.2f}</b>): ${snap_drift:+.2f}"
        return None

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return

        self._repo = repo
        self._ready.clear()
        if self._log is None:
            candidate = getattr(repo, "_log", None)
            if candidate is not None:
                self._log = cast(FilteringBoundLogger, candidate).bind(module="telegram")
        app = (
            Application.builder()
            .token(self._token)
            .defaults(Defaults(parse_mode=ParseMode.HTML))
            .build()
        )
        self._add_handlers(app)

        self._app = app
        await app.initialize()
        await app.start()
        if app.updater is not None:
            await app.updater.start_polling(drop_pending_updates=True)
        else:
            raise RuntimeError("Application updater not available; cannot start polling")
        self._ready.set()

    def _add_handlers(self, app: Application) -> None:
        for name, fn in [
            ("start", self._on_start),
            ("help", self._on_help),
            ("status", self._on_status),
            ("bands", self._on_bands),
            ("setbaseline", self._on_setbaseline),
            ("setnotional", self._on_setnotional),
            ("settilt", self._on_settilt),
            ("updatebalances", self._on_updatebalances),
        ]:
            app.add_handler(CommandHandler(name, fn))

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_any))

        for pattern, fn in [
            (r"^alert:(accept|ignore|set):[abc]$", self._on_alert_action),
            (r"^adv:(apply|ignore|set)$", self._on_adv_action),
            (r"^b:[abc]$", self._on_bands_pick),
            (r"^b:back$", self._on_bands_back),
        ]:
            app.add_handler(CallbackQueryHandler(fn, pattern=pattern))

        app.add_handler(CallbackQueryHandler(self._on_callback))

    async def stop(self) -> None:
        if self._app is None:
            return

        try:
            if self._app.updater is not None:
                await self._app.updater.stop()
        except Exception:
            pass

        await self._app.stop()
        await self._app.shutdown()

        self._app = None
        self._repo = None
        self._ready.clear()
        self._pending.clear()
        self._log = None

    async def send_text(self, text: str) -> None:
        await self._send(text)

    async def send_advisory_card(
        self, advisory: AdvisoryPayload, drift_line: str | None = None
    ) -> None:
        app = self._ensure_app()
        await self._ready.wait()

        repo = self._ensure_repo()

        price = float(advisory["price"])
        sigma_pct = advisory.get("sigma_pct")
        bucket = advisory["bucket"]
        ranges = advisory["ranges"]

        split: BucketSplit | None = None
        raw_split = advisory.get("split")
        if isinstance(raw_split, (list, tuple)) and len(raw_split) == 3:
            with suppress(Exception):
                split = tuple(int(float(item)) for item in raw_split)  # type: ignore[misc]
        if not split:
            with suppress(ValueError):
                split = split_for_bucket(bucket)
        if not split:
            split = split_for_sigma(sigma_pct)

        notional = await repo.get_notional_usd()
        tilt = await repo.get_tilt_sol_frac()

        amounts_map, unallocated_usd = compute_amounts(
            price,
            split,
            ranges,
            notional,
            tilt,
        )

        text = format_advisory_card(
            price,
            sigma_pct,
            bucket,
            ranges,
            split,
            stale=bool(advisory.get("stale")),
            amounts=amounts_map or None,
            unallocated_usd=unallocated_usd if unallocated_usd > 0 else None,
        )
        if drift_line is not None:
            text = f"{text}\n{drift_line}"

        buttons = [
            [
                InlineKeyboardButton("Apply All", callback_data="adv:apply"),
                InlineKeyboardButton("Set Exact", callback_data="adv:set"),
                InlineKeyboardButton("Ignore", callback_data="adv:ignore"),
            ]
        ]

        message = await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        self._pending[message.message_id] = (
            self._PENDING_ADV,
            cast(BandMap, dict(ranges)),
        )
        self._prune_pending()

    async def send_breach_offer(
        self,
        band: BandName,
        price: float,
        src_label: str | None,
        bands: BandMap,
        suggested_range: BandRange,
        policy_meta: tuple[Bucket, float] | None = None,
    ) -> None:
        app = self._ensure_app()
        await self._ready.wait()

        current = bands.get(band)
        if current is None:
            raise ValueError(f"unknown band '{band}'")

        side = "Below" if price < current[0] else "Above"
        source = f" ({escape(src_label)})" if src_label else ""
        band_label = escape(band.upper())
        text = (
            "⚠️ <b>Range Breach</b>\n"
            f"<b>Band</b>: {band_label} ({escape(side)})\n"
            f"<b>Price</b>: {price:.2f}{source}\n"
            f"<b>Current {band_label}</b>: {fmt_range(*current)}\n"
            f"<b>Suggested {band_label}</b>: {fmt_range(*suggested_range)}"
        )
        bucket_meta: Bucket | None = None
        if policy_meta is not None:
            bucket_meta, width = policy_meta
            bucket_label = escape(bucket_meta.title())
            text = (
                f"{text}\n<i>Policy</i>: {bucket_label} (±{width * 100:.2f}%)"
            )

        repo = self._ensure_repo()
        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()

        amount_line: str | None = None
        if (
            bucket_meta is not None
            and notional is not None
            and notional > 0
            and price > 0
            and math.isfinite(price)
        ):
            try:
                split = split_for_bucket(bucket_meta)
            except ValueError:
                split = None
            if split:
                pct = next((share for name, share in zip(BAND_ORDER, split, strict=False) if name == band), None)
                if pct is not None and pct > 0:
                    per_band_usd = (notional * pct) / 100.0
                    tilt = min(max(tilt_sol_frac, 0.0), 1.0)
                    sol_amt = round((per_band_usd * tilt) / price, 6)
                    usdc_amt = round(per_band_usd * (1.0 - tilt), 2)
                    amount_line = (
                        f"amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f}"
                    )

        if amount_line:
            text = f"{text}\n{amount_line}"
        buttons = [
            [
                InlineKeyboardButton("Apply", callback_data=f"alert:accept:{band}"),
                InlineKeyboardButton("Ignore", callback_data=f"alert:ignore:{band}"),
                InlineKeyboardButton("Set Exact", callback_data=f"alert:set:{band}"),
            ]
        ]
        message = await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        self._pending[message.message_id] = (self._PENDING_ALERT, AlertPayload(band, suggested_range))
        self._prune_pending()

    def _ensure_app(self) -> Application:
        if self._app is None:
            raise RuntimeError("Telegram service has not been started")
        return self._app

    def set_price_provider(self, fn: Callable[[], Awaitable[float | None]]) -> None:
        self._price_provider = fn

    def set_sigma_provider(self, fn: Callable[[], Awaitable[VolReading | None]]) -> None:
        self._sigma_provider = fn

    async def _get_price(self) -> float | None:
        if not self._price_provider:
            return None
        try:
            return await self._price_provider()
        except Exception:
            return None

    async def _get_sigma(self) -> VolReading | None:
        if not self._sigma_provider:
            return None
        try:
            return await self._sigma_provider()
        except Exception:
            return None

    @_auth_cmd
    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._send("[<i>Online</i>] Bot ready. Use <code>/status</code> for details.")

    @_auth_cmd
    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_lines = [
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
        await self._send("\n".join(help_lines))

    @_auth_cmd
    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        repo = self._ensure_repo()
        price = await self._get_price()
        sigma = await self._get_sigma()
        bands = await repo.get_bands()
        latest = await repo.get_latest_snapshot()
        baseline = await repo.get_baseline()

        lines: list[str] = []
        lines.append(f"<b>Price</b>: {price:.2f}" if (price and math.isfinite(price)) else "<b>Price</b>: Unknown")

        sigma_line, bucket, sigma_pct = self._sigma_summary(sigma)
        lines.append(sigma_line)

        lines.append("<b>Configured Bands</b>:")
        lines.extend(self._bands_lines(bands))

        split = split_for_sigma(sigma_pct)
        bucket_label = escape(bucket.title())
        lines.append(f"<b>Advisory Split</b>: {split[0]}/{split[1]}/{split[2]} ({bucket_label})")

        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()
        sol_pct = self._format_pct(tilt_sol_frac)
        usdc_pct = self._format_pct(1.0 - tilt_sol_frac)
        notional_display = "unset" if not (notional and notional > 0) else f"${notional:.2f}"
        lines.append(
            f"notional: {escape(notional_display)} | tilt sol/usdc: {escape(sol_pct)}/{escape(usdc_pct)}"
        )

        if latest:
            _snap_ts, snap_sol, snap_usdc, _snap_price, _snap_drift = latest
            lines.append(f"<b>Balances</b>: {snap_sol:g} SOL, {snap_usdc:g} USDC")
        else:
            lines.append("<b>Balances</b>: (none; run /updatebalances)")

        lines.extend(self._amounts_lines(price, bucket, notional, tilt_sol_frac, split))

        drift_line = self._drift_summary(baseline, latest, price)
        if drift_line:
            lines.append(drift_line)

        await self._send("\n".join(lines))

    @_auth_cmd
    async def _on_setbaseline(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await self._send(self._USAGE_SETBASELINE)
            return

        sol, usdc = parsed
        sol = max(sol, 0.0)
        usdc = max(usdc, 0.0)
        if not (math.isfinite(sol) and math.isfinite(usdc)):
            await self._send(self._USAGE_SETBASELINE)
            return

        repo = self._ensure_repo()
        ts = int(time.time())
        await repo.set_baseline(sol, usdc, ts)
        await self._send(f"[<i>Applied</i>] Baseline → {sol:g} SOL, {usdc:g} USDC")

    @_auth_cmd
    async def _on_updatebalances(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await self._send(self._USAGE_UPDATEBAL)
            return

        sol_amt, usdc_amt = parsed
        sol_amt = max(sol_amt, 0.0)
        usdc_amt = max(usdc_amt, 0.0)
        if not (math.isfinite(sol_amt) and math.isfinite(usdc_amt)):
            await self._send(self._USAGE_UPDATEBAL)
            return

        price = await self._get_price()
        if price is None or not math.isfinite(price) or price <= 0:
            await self._send("[<i>Error</i>] Price unavailable. Try again later.")
            return

        repo = self._ensure_repo()
        baseline = await repo.get_baseline()
        if baseline is None:
            await self._send("[<i>Error</i>] Baseline not set. Run <code>/setbaseline</code> first.")
            return

        base_sol, base_usdc, _ = baseline
        base_sol = max(float(base_sol), 0.0)
        base_usdc = max(float(base_usdc), 0.0)

        base_val = base_sol * price + base_usdc
        cur_val = sol_amt * price + usdc_amt
        if not (math.isfinite(base_val) and math.isfinite(cur_val)):
            await self._send("[<i>Error</i>] Unable to compute drift.")
            return

        drift = cur_val - base_val
        if not math.isfinite(drift):
            await self._send("[<i>Error</i>] Unable to compute drift.")
            return

        drift_pct = (drift / base_val) if base_val > 0 else 0.0
        if not math.isfinite(drift_pct):
            drift_pct = 0.0

        ts = int(time.time())
        await repo.insert_snapshot(ts, sol_amt, usdc_amt, price, drift)

        await self._send(
            f"[<i>Applied</i>] <b>Drift</b>: ${drift:+.2f} ({drift_pct * 100:+.2f}%) | "
            f"<b>Base</b> ${base_val:.2f} → <b>Now</b> ${cur_val:.2f} @ "
            f"Price <b>{price:.2f}</b>"
        )

    @_auth_cmd
    async def _on_setnotional(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        message = update.message
        text = message.text if message and message.text else ""
        value = self._parse_first_number(text)
        if value is None or not math.isfinite(value) or value < 0:
            await self._send(self._USAGE_SETNOTIONAL)
            return

        repo = self._ensure_repo()
        try:
            await repo.set_notional_usd(float(value))
        except ValueError:
            await self._send("[<i>Error</i>] Notional must be finite and non-negative.")
            return

        await self._send(f"[<i>Applied</i>] Notional → ${float(value):.2f}")

    @_auth_cmd
    async def _on_settilt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_tilt_sol_usdc(text)
        if parsed is None:
            await self._send(self._USAGE_SETTILT)
            return

        sol_frac, usdc_frac = parsed
        if not (math.isfinite(sol_frac) and math.isfinite(usdc_frac)):
            await self._send("[<i>Error</i>] Invalid tilt values.")
            return

        total = sol_frac + usdc_frac
        if not math.isfinite(total) or total <= 0:
            await self._send("[<i>Error</i>] Tilt must allocate some share to SOL or USDC.")
            return

        sol_frac = self._clamp01(sol_frac)
        usdc_frac = self._clamp01(usdc_frac)
        total = sol_frac + usdc_frac
        if total == 0:
            await self._send("[<i>Error</i>] Tilt must allocate some share to SOL or USDC.")
            return

        sol_frac /= total
        usdc_frac = 1.0 - sol_frac

        repo = self._ensure_repo()
        await repo.set_tilt_sol_frac(sol_frac)

        sol_pct = self._format_pct(sol_frac)
        usdc_pct = self._format_pct(usdc_frac)
        await self._send(f"[<i>Applied</i>] Tilt → SOL/USDC = {escape(sol_pct)}/{escape(usdc_pct)}")

    def _bands_menu_text(self, bands: BandMap) -> str:
        if not bands:
            return "(No bands configured)"
        return "\n".join(
            ["<b>Configured Bands</b>:"] + [f"{escape(n.upper())}: {fmt_range(*rng)}" for n, rng in sorted(bands.items())]
        )

    def _bands_menu_kb(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton(f"Edit {c}", callback_data=f"b:{c.lower()}")] for c in "ABC"]
        )

    @_auth_cmd
    async def _on_bands(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        text = self._bands_menu_text(bands)
        await self._send(text, reply_markup=self._bands_menu_kb())

    @_auth_cb
    async def _on_bands_pick(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = (query.data or "").split(":")
        if len(data) != 2:
            await query.answer()
            return

        band = data[1]
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        current = bands.get(band)
        if current is None:
            await query.answer("[<i>Error</i>] Unknown band.", show_alert=True)
            return

        band_label = escape(band.upper())
        editor_text = (
            f"<b>Edit Band</b> {band_label}\n"
            f"<b>Current</b>: {fmt_range(*current)}\n"
            "Send <code>low high</code> or tap Back."
        )
        back_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="b:back")]])
        await self._edit_or_send(query, editor_text, reply_markup=back_markup)

        mid = query.message.message_id if query.message else None
        context.chat_data["await_exact"] = {"mid": mid, "band": band}
        await self._send(
            f"Enter <code>low high</code> for <b>{band_label}</b>.",
            reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
        )

    @_auth_cb
    async def _on_bands_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        await query.answer()
        await self._edit_or_send(
            query,
            self._bands_menu_text(bands),
            reply_markup=self._bands_menu_kb(),
        )

    @_auth_cb
    async def _on_alert_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 3:
            await query.answer()
            return

        action = parts[1]
        band_raw = parts[2]
        if band_raw not in BAND_ORDER:
            await query.answer("[<i>Error</i>] Unknown band.", show_alert=True)
            return
        band = cast(BandName, band_raw)
        pending = await self._pop_pending(query, self._PENDING_ALERT, stale_text=self._STALE_ALERT)
        if not pending:
            return

        mid, payload = pending
        if not isinstance(payload, AlertPayload):
            await query.answer(self._STALE_ALERT, show_alert=True)
            return
        pending_band, rng = payload
        if pending_band != band:
            await query.answer(self._STALE_ALERT, show_alert=True)
            return

        await query.answer()
        band_label = escape(band.upper())

        if action == "accept":
            repo = self._ensure_repo()
            await repo.upsert_band(band, rng[0], rng[1])
            if self._log is not None:
                with suppress(Exception):
                    self._log.info("breach_applied", band=band, low=rng[0], high=rng[1])
            del self._pending[mid]
            await self._edit_or_send(query, f"[<i>Applied</i>] {band_label} → {fmt_range(*rng)}")
            return

        if action == "ignore":
            del self._pending[mid]
            await self._edit_or_send(query, "[<i>Dismissed</i>]")
            return

        if action == "set":
            context.chat_data["await_exact"] = {"mid": mid, "band": band}
            prompt_text = (
                f"Send <code>low high</code> for band <b>{band_label}</b> "
                f"(e.g. <code>141.25 159.75</code>)."
            )
            await self._edit_or_send(query, prompt_text)
            await self._send(
                f"Enter <code>low high</code> for <b>{band_label}</b>.",
                reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
            )

    @_auth_cb
    async def _on_adv_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 2:
            await query.answer()
            return

        action = parts[1]
        pending = await self._pop_pending(query, self._PENDING_ADV, stale_text=self._STALE_ADV)
        if not pending:
            return

        mid, payload = pending
        if not isinstance(payload, dict):
            await query.answer(self._STALE_ADV, show_alert=True)
            return
        ranges = cast(BandMap, payload)

        await query.answer()

        if action == "apply":
            repo = self._ensure_repo()
            try:
                existing = (await repo.get_bands()).keys()
                filtered = {name: rng for name, rng in ranges.items() if name in existing}
                await repo.upsert_many(filtered)
            except Exception as exc:
                if self._log is not None:
                    with suppress(Exception):
                        self._log.error("advisory_apply_failed", error=str(exc))
                await self._send("[<i>Error</i>] Apply failed. Please retry.")
                return

            del self._pending[mid]
            if self._log is not None:
                with suppress(Exception):
                    self._log.info(
                        "advisory_applied",
                        ranges={
                            name: (rng[0], rng[1]) for name, rng in sorted(filtered.items())
                        },
                    )
            summary = ", ".join(
                f"{escape(name.upper())}→{fmt_range(*rng)}" for name, rng in sorted(filtered.items())
            )
            await self._edit_or_send(
                query,
                f"[<i>Applied</i>] {summary}" if summary else "[<i>Applied</i>]",
            )
            return

        if action == "ignore":
            del self._pending[mid]
            await self._edit_or_send(query, "[<i>Dismissed</i>]")
            return

        if action == "set":
            del self._pending[mid]
            await self._edit_or_send(
                query,
                "Select a band to edit.",
                reply_markup=self._bands_menu_kb(),
            )

    @_auth_cb
    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.callback_query.answer()

    @_auth_cmd
    async def _on_text_any(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return

        state = context.chat_data.get("await_exact")
        if not state:
            return

        payload = (message.text or "").replace(",", " ").split()
        try:
            if len(payload) != 2:
                raise ValueError
            low, high = float(payload[0]), float(payload[1])
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError
        except ValueError:
            await self._send("[<i>Invalid</i>] Expected two numbers: <code>low high</code>.")
            return

        band = state.get("band")
        mid = state.get("mid")
        if not isinstance(band, str):
            context.chat_data.pop("await_exact", None)
            return

        repo = self._ensure_repo()
        await repo.upsert_band(band, low, high)

        if isinstance(mid, int) and mid in self._pending:
            del self._pending[mid]
        band_label = escape(band.upper())
        applied_text = f"[<i>Applied</i>] {band_label} → {fmt_range(low, high)}"
        sent = isinstance(mid, int) and await self._edit_by_id(mid, applied_text)
        if not sent:
            await self._send(applied_text)

        if self._log is not None:
            with suppress(Exception):
                self._log.info("bands_set_exact", band=band, low=low, high=high)

        bands = await repo.get_bands()
        await self._send(
            self._bands_menu_text(bands),
            reply_markup=self._bands_menu_kb(),
        )

        context.chat_data.pop("await_exact", None)

    def _parse_first_number(self, text: str) -> float | None:
        nums = self._extract_numbers(text, 1, ignore_labels=set())
        return nums[0] if nums else None

    def _parse_tilt_sol_usdc(self, text: str) -> tuple[float, float] | None:
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

    def _format_pct(self, fraction: float) -> str:
        try:
            pct_value = round(float(fraction) * 100.0, 1)
        except Exception:
            pct_value = 0.0
        if not math.isfinite(pct_value):
            pct_value = 0.0
        return str(int(pct_value)) if pct_value.is_integer() else f"{pct_value:.1f}"

    def _parse_two_numbers(self, text: str) -> tuple[float, float] | None:
        numbers = self._extract_numbers(text, 2, ignore_labels={"sol", "usdc"})
        if len(numbers) < 2:
            return None
        return numbers[0], numbers[1]

    def _is_authorized(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == self._chat_id

    async def _reply_unauthorized(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if chat is None:
            return
        await context.bot.send_message(
            chat_id=chat.id,
            text=self._UNAUTHORIZED_HTML,
        )

    async def _alert_unauthorized(self, query: CallbackQuery) -> None:
        await query.answer(self._UNAUTHORIZED_HTML, show_alert=True)

    def _ensure_repo(self) -> DBRepo:
        if self._repo is None:
            raise RuntimeError("Telegram service repository not set. Call start() first.")
        return self._repo

    def is_ready(self) -> bool:
        return self._ready.is_set()
