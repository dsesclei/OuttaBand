from __future__ import annotations

import asyncio
import math
import os
import time
from html import escape
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, cast

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from db_repo import DBRepo
from band_logic import fmt_range, format_advisory_card
from band_advisor import compute_amounts, ranges_for_price, split_for_bucket, split_for_sigma
from structlog.typing import FilteringBoundLogger


class TelegramSvc:
    MAX_PENDING = int(os.getenv("LPBOT_PENDING_CAP", "100"))

    def __init__(self, token: str, chat_id: int, logger: Optional[FilteringBoundLogger] = None) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: Optional[Application] = None
        self._repo: Optional[DBRepo] = None
        self._ready: asyncio.Event = asyncio.Event()
        self._pending: Dict[int, Tuple[str, object]] = {}
        self._log: Optional[FilteringBoundLogger] = logger
        self._price_provider: Optional[Callable[[], Awaitable[Optional[float]]]] = None
        self._sigma_provider: Optional[Callable[[], Awaitable[Optional[Dict[str, Any]]]]] = None

    def _prune_pending(self) -> None:
        cap = max(1, self.MAX_PENDING)
        if len(self._pending) <= cap:
            return
        # prune oldest message ids (Telegram message IDs increase monotonically per chat)
        for mid in sorted(self._pending.keys()):
            if len(self._pending) <= cap:
                break
            del self._pending[mid]

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return

        self._repo = repo
        if self._log is None:
            candidate = getattr(repo, "_log", None)
            if candidate is not None:
                self._log = cast(FilteringBoundLogger, candidate).bind(module="telegram")
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._on_start))
        app.add_handler(CommandHandler("help", self._on_help))
        app.add_handler(CommandHandler("status", self._on_status))
        app.add_handler(CommandHandler("bands", self._on_bands))
        app.add_handler(CommandHandler("setbaseline", self._on_setbaseline))
        app.add_handler(CommandHandler("setnotional", self._on_setnotional))
        app.add_handler(CommandHandler("settilt", self._on_settilt))
        app.add_handler(CommandHandler("updatebalances", self._on_updatebalances))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_any))
        app.add_handler(CallbackQueryHandler(self._on_alert_action, pattern=r"^alert:(accept|ignore|set):[abc]$"))
        app.add_handler(CallbackQueryHandler(self._on_adv_action, pattern=r"^adv:(apply|ignore|set)$"))
        app.add_handler(CallbackQueryHandler(self._on_bands_pick, pattern=r"^b:[abc]$"))
        app.add_handler(CallbackQueryHandler(self._on_bands_back, pattern=r"^b:back$"))
        app.add_handler(CallbackQueryHandler(self._on_callback))

        self._app = app
        await app.initialize()
        await app.start()
        if app.updater is not None:
            await app.updater.start_polling(drop_pending_updates=True)
        else:
            raise RuntimeError("Application updater not available; cannot start polling")
        self._ready.set()

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
        self._ready = asyncio.Event()
        self._pending.clear()
        self._log = None

    async def send_text(self, text: str) -> None:
        app = self._ensure_app()
        await self._ready.wait()
        await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            parse_mode="HTML",
        )

    async def send_advisory_card(
        self, advisory: Dict[str, Any], drift_line: Optional[str] = None
    ) -> None:
        app = self._ensure_app()
        await self._ready.wait()

        repo = self._ensure_repo()

        price = float(advisory["price"])
        sigma_pct = cast(Optional[float], advisory.get("sigma_pct"))
        bucket = str(advisory["bucket"])
        ranges = cast(Dict[str, Tuple[float, float]], cast(Any, advisory["ranges"]))

        split_raw = advisory.get("split")
        split: Optional[Tuple[int, int, int]] = None
        if isinstance(split_raw, (list, tuple)) and len(split_raw) == 3:
            try:
                split = (
                    int(float(split_raw[0])),
                    int(float(split_raw[1])),
                    int(float(split_raw[2])),
                )
            except (TypeError, ValueError):
                split = None
        if split is None:
            try:
                split = split_for_bucket(bucket)
            except ValueError:
                split = split_for_sigma(sigma_pct)
        if split is None:
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
            parse_mode="HTML",
        )
        self._pending[message.message_id] = ("adv", dict(ranges))
        self._prune_pending()

    async def send_breach_offer(
        self,
        band: str,
        price: float,
        src_label: Optional[str],
        bands: dict[str, Tuple[float, float]],
        suggested_range: Tuple[float, float],
        policy_meta: Optional[Tuple[str, float]] = None,
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
        bucket = None
        if policy_meta is not None:
            bucket, width = policy_meta
            bucket_label = escape(bucket.title())
            text = (
                f"{text}\n<i>Policy</i>: {bucket_label} (±{width * 100:.2f}%)"
            )

        repo = self._ensure_repo()
        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()

        amount_line: Optional[str] = None
        if (
            bucket is not None
            and notional is not None
            and notional > 0
            and price > 0
            and math.isfinite(price)
        ):
            try:
                split = split_for_bucket(bucket)
            except ValueError:
                split = None
            if split:
                bands_order = ("a", "b", "c")
                pct = next((share for name, share in zip(bands_order, split) if name == band), None)
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
            parse_mode="HTML",
        )
        self._pending[message.message_id] = ("alert", (band, suggested_range))
        self._prune_pending()

    def _ensure_app(self) -> Application:
        if self._app is None:
            raise RuntimeError("Telegram service has not been started")
        return self._app

    def set_price_provider(self, fn: Callable[[], Awaitable[Optional[float]]]) -> None:
        self._price_provider = fn

    def set_sigma_provider(self, fn: Callable[[], Awaitable[Optional[Dict[str, Any]]]]) -> None:
        self._sigma_provider = fn

    async def _get_price(self) -> Optional[float]:
        if not self._price_provider:
            return None
        try:
            return await self._price_provider()
        except Exception:
            return None

    async def _get_sigma(self) -> Optional[Dict[str, Any]]:
        if not self._sigma_provider:
            return None
        try:
            return await self._sigma_provider()
        except Exception:
            return None

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return
        await context.bot.send_message(
            chat_id=self._chat_id,
            text="[<i>Online</i>] Bot ready. Use <code>/status</code> for details.",
            parse_mode="HTML",
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return
        help_lines = [
            "<b>Commands</b>:",
            "<code>/status</code> — latest price, σ, bands, policy split",
            "<code>/bands</code> — edit band ranges",
            "<code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code>",
            "<code>/setnotional &lt;usd&gt;</code> — total usd to allocate across bands",
            "<code>/settilt &lt;sol&gt;:&lt;usdc&gt;</code> — inside-band split, sol first (e.g. 50:50, 60:40)",
            "<code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code>",
            "<code>/help</code> — show this list",
        ]
        await context.bot.send_message(
            chat_id=self._chat_id,
            text="\n".join(help_lines),
            parse_mode="HTML",
        )

    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return
        repo = self._ensure_repo()
        price = await self._get_price()
        sigma = await self._get_sigma()
        bands = await repo.get_bands()
        latest = await repo.get_latest_snapshot()
        baseline = await repo.get_baseline()

        lines = []
        if price is not None and math.isfinite(price):
            lines.append(f"<b>Price</b>: {price:.2f}")
        else:
            lines.append("<b>Price</b>: Unknown")

        sigma_bucket = "mid"
        sigma_display = "–"
        sigma_stale = False
        sigma_pct_value: Optional[float] = None
        if sigma:
            bucket_raw = sigma.get("bucket")
            if isinstance(bucket_raw, str) and bucket_raw:
                sigma_bucket = bucket_raw
            sigma_pct_val = sigma.get("sigma_pct")
            try:
                if sigma_pct_val is not None and math.isfinite(float(sigma_pct_val)):
                    sigma_pct_value = float(sigma_pct_val)
                    sigma_display = f"{sigma_pct_value:.2f}%"
            except (TypeError, ValueError):
                sigma_display = "–"
            sigma_stale = bool(sigma.get("stale"))
        sigma_bucket_label = sigma_bucket.title()
        sigma_line = f"<b>Volatility</b>: {sigma_display} ({escape(sigma_bucket_label)})"
        if sigma_stale:
            sigma_line = f"{sigma_line} [<i>Stale</i>]"
        lines.append(sigma_line)

        lines.append("<b>Configured Bands</b>:")
        if bands:
            for name in sorted(bands.keys()):
                lo, hi = bands[name]
                lines.append(f"{escape(name.upper())}: {fmt_range(lo, hi)}")
        else:
            lines.append("(No bands configured)")

        split = split_for_sigma(sigma_pct_value)
        lines.append(
            f"<b>Advisory Split</b>: {split[0]}/{split[1]}/{split[2]} ({escape(sigma_bucket_label)})"
        )

        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()
        usdc_frac = 1.0 - tilt_sol_frac
        sol_pct = self._format_pct(tilt_sol_frac)
        usdc_pct = self._format_pct(usdc_frac)
        if notional is None or notional <= 0:
            notional_display = "unset"
        else:
            notional_display = f"${notional:.2f}"
        lines.append(
            f"notional: {escape(notional_display)} | tilt sol/usdc: {escape(sol_pct)}/{escape(usdc_pct)}"
        )

        if latest:
            _snap_ts, snap_sol, snap_usdc, _snap_price, _snap_drift = latest
            lines.append(f"<b>Balances</b>: {snap_sol:g} SOL, {snap_usdc:g} USDC")
        else:
            lines.append("<b>Balances</b>: (none; run /updatebalances)")

        if (
            notional is not None
            and notional > 0
            and price is not None
            and math.isfinite(price)
            and price > 0
        ):
            try:
                planned_ranges = ranges_for_price(
                    price,
                    sigma_bucket,
                    include_a_on_high=False,
                )
            except ValueError:
                planned_ranges = {}

            if planned_ranges:
                amounts_map, unallocated_usd = compute_amounts(
                    price,
                    split,
                    planned_ranges,
                    notional,
                    tilt_sol_frac,
                )
                if amounts_map:
                    for band_name in ("a", "b", "c"):
                        if band_name not in amounts_map:
                            continue
                        sol_amt, usdc_amt = amounts_map[band_name]
                        lines.append(
                            f"{escape(band_name.upper())} amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f} USDC"
                        )
                    if unallocated_usd > 0.005:
                        lines.append(f"unallocated: ${unallocated_usd:.2f}")

        if price is not None and math.isfinite(price) and price > 0 and baseline and latest:
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
                    lines.append(
                        f"<b>Drift</b>: ${drift_now:+.2f} ({drift_pct * 100:+.2f}%)"
                    )
                else:
                    lines.append("<b>Drift</b>: Unavailable")
            else:
                lines.append("<b>Drift</b>: Unavailable")
        elif latest:
            _snap_ts, _snap_sol, _snap_usdc, snap_price, snap_drift = latest
            if math.isfinite(snap_price) and math.isfinite(snap_drift):
                lines.append(
                    f"<b>Drift</b> (Last @ <b>{snap_price:.2f}</b>): ${snap_drift:+.2f}"
                )

        await context.bot.send_message(
            chat_id=self._chat_id,
            text="\n".join(lines),
            parse_mode="HTML",
        )

    async def _on_setbaseline(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Usage</i>] <code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code> "
                    "(e.g. <code>/setbaseline 1.0 sol 200 usdc</code>)"
                ),
                parse_mode="HTML",
            )
            return

        sol, usdc = parsed
        sol = max(sol, 0.0)
        usdc = max(usdc, 0.0)
        if not (math.isfinite(sol) and math.isfinite(usdc)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Usage</i>] <code>/setbaseline &lt;sol&gt; sol &lt;usdc&gt; usdc</code> "
                    "(e.g. <code>/setbaseline 1.0 sol 200 usdc</code>)"
                ),
                parse_mode="HTML",
            )
            return

        repo = self._ensure_repo()
        ts = int(time.time())
        await repo.set_baseline(sol, usdc, ts)
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=(
                f"[<i>Applied</i>] Baseline → {sol:g} SOL, {usdc:g} USDC"
            ),
            parse_mode="HTML",
        )

    async def _on_updatebalances(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Usage</i>] <code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code> "
                    "(e.g. <code>/updatebalances 1.0 sol 200 usdc</code>)"
                ),
                parse_mode="HTML",
            )
            return

        sol_amt, usdc_amt = parsed
        sol_amt = max(sol_amt, 0.0)
        usdc_amt = max(usdc_amt, 0.0)
        if not (math.isfinite(sol_amt) and math.isfinite(usdc_amt)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Usage</i>] <code>/updatebalances &lt;sol&gt; sol &lt;usdc&gt; usdc</code> "
                    "(e.g. <code>/updatebalances 1.0 sol 200 usdc</code>)"
                ),
                parse_mode="HTML",
            )
            return

        price = await self._get_price()
        if price is None or not math.isfinite(price) or price <= 0:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Price unavailable. Try again later.",
                parse_mode="HTML",
            )
            return

        repo = self._ensure_repo()
        baseline = await repo.get_baseline()
        if baseline is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Error</i>] Baseline not set. Run <code>/setbaseline</code> first."
                ),
                parse_mode="HTML",
            )
            return

        base_sol, base_usdc, _ = baseline
        base_sol = max(float(base_sol), 0.0)
        base_usdc = max(float(base_usdc), 0.0)

        base_val = base_sol * price + base_usdc
        cur_val = sol_amt * price + usdc_amt
        if not (math.isfinite(base_val) and math.isfinite(cur_val)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Unable to compute drift.",
                parse_mode="HTML",
            )
            return

        drift = cur_val - base_val
        if not math.isfinite(drift):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Unable to compute drift.",
                parse_mode="HTML",
            )
            return

        drift_pct = (drift / base_val) if base_val > 0 else 0.0
        if not math.isfinite(drift_pct):
            drift_pct = 0.0

        ts = int(time.time())
        await repo.insert_snapshot(ts, sol_amt, usdc_amt, price, drift)

        await context.bot.send_message(
            chat_id=self._chat_id,
            text=(
                f"[<i>Applied</i>] <b>Drift</b>: ${drift:+.2f} ({drift_pct * 100:+.2f}%) | "
                f"<b>Base</b> ${base_val:.2f} → <b>Now</b> ${cur_val:.2f} @ "
                f"Price <b>{price:.2f}</b>"
            ),
            parse_mode="HTML",
        )

    async def _on_setnotional(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return

        message = update.message
        text = message.text if message and message.text else ""
        value = self._parse_first_number(text)
        if value is None or not math.isfinite(value) or value < 0:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Usage</i>] <code>/setnotional &lt;usd&gt;</code> (e.g. <code>/setnotional 2500</code>)",
                parse_mode="HTML",
            )
            return

        repo = self._ensure_repo()
        try:
            await repo.set_notional_usd(float(value))
        except ValueError:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Notional must be finite and non-negative.",
                parse_mode="HTML",
            )
            return

        await context.bot.send_message(
            chat_id=self._chat_id,
            text=f"[<i>Applied</i>] Notional → ${float(value):.2f}",
            parse_mode="HTML",
        )

    async def _on_settilt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_tilt_sol_usdc(text)
        if parsed is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=(
                    "[<i>Usage</i>] <code>/settilt &lt;sol&gt;:&lt;usdc&gt;</code> "
                    "(e.g. <code>/settilt 60:40</code> or <code>/settilt 0.6</code>)"
                ),
                parse_mode="HTML",
            )
            return

        sol_frac, usdc_frac = parsed
        if not (math.isfinite(sol_frac) and math.isfinite(usdc_frac)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Invalid tilt values.",
                parse_mode="HTML",
            )
            return

        total = sol_frac + usdc_frac
        if total <= 0:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Tilt must allocate some share to SOL or USDC.",
                parse_mode="HTML",
            )
            return

        sol_frac = min(max(sol_frac, 0.0), 1.0)
        usdc_frac = min(max(usdc_frac, 0.0), 1.0)
        total = sol_frac + usdc_frac
        if total == 0:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Error</i>] Tilt must allocate some share to SOL or USDC.",
                parse_mode="HTML",
            )
            return

        sol_frac /= total
        usdc_frac /= total

        repo = self._ensure_repo()
        await repo.set_tilt_sol_frac(sol_frac)

        sol_pct = self._format_pct(sol_frac)
        usdc_pct = self._format_pct(usdc_frac)
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=(
                f"[<i>Applied</i>] Tilt → SOL/USDC = {escape(sol_pct)}/{escape(usdc_pct)}"
            ),
            parse_mode="HTML",
        )

    def _bands_menu_text(self, bands: Dict[str, Tuple[float, float]]) -> str:
        if not bands:
            return "(No bands configured)"
        lines = ["<b>Configured Bands</b>:"]
        for name in sorted(bands.keys()):
            lo, hi = bands[name]
            lines.append(f"{escape(name.upper())}: {fmt_range(lo, hi)}")
        return "\n".join(lines)

    def _bands_menu_kb(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Edit A", callback_data="b:a")],
                [InlineKeyboardButton("Edit B", callback_data="b:b")],
                [InlineKeyboardButton("Edit C", callback_data="b:c")],
            ]
        )

    async def _on_bands(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
            return
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        text = self._bands_menu_text(bands)
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=self._bands_menu_kb(),
            parse_mode="HTML",
        )

    async def _on_bands_pick(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("[<i>Unauthorized</i>]", show_alert=True)
            return

        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 2:
            await query.answer()
            return

        band = parts[1]
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
        try:
            await query.edit_message_text(
                editor_text,
                reply_markup=back_markup,
                parse_mode="HTML",
            )
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=editor_text,
                reply_markup=back_markup,
                parse_mode="HTML",
            )

        mid = query.message.message_id if query.message else None
        context.chat_data["await_exact"] = {"mid": mid, "band": band}
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=f"Enter <code>low high</code> for <b>{band_label}</b>.",
            reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
            parse_mode="HTML",
        )

    async def _on_bands_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("[<i>Unauthorized</i>]", show_alert=True)
            return

        repo = self._ensure_repo()
        bands = await repo.get_bands()
        await query.answer()
        try:
            await query.edit_message_text(
                self._bands_menu_text(bands),
                reply_markup=self._bands_menu_kb(),
                parse_mode="HTML",
            )
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=self._bands_menu_text(bands),
                reply_markup=self._bands_menu_kb(),
                parse_mode="HTML",
            )

    async def _on_alert_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("[<i>Unauthorized</i>]", show_alert=True)
            return

        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 3:
            await query.answer()
            return

        action, band = parts[1], parts[2]
        message = query.message
        mid = message.message_id if message else None
        if mid is None or mid not in self._pending:
            await query.answer("[<i>Stale</i>] This alert is no longer active.", show_alert=True)
            return

        kind, payload = self._pending[mid]
        if kind != "alert":
            await query.answer("[<i>Stale</i>] This alert is no longer active.", show_alert=True)
            return

        pending_band, rng = cast(Tuple[str, Tuple[float, float]], payload)
        if pending_band != band:
            await query.answer("[<i>Stale</i>] This alert is no longer active.", show_alert=True)
            return

        await query.answer()
        band_label = escape(band.upper())

        if action == "accept":
            repo = self._ensure_repo()
            await repo.upsert_band(band, rng[0], rng[1])
            if self._log is not None:
                try:
                    self._log.info("breach_applied", band=band, low=rng[0], high=rng[1])
                except Exception:
                    pass
            del self._pending[mid]
            applied_text = f"[<i>Applied</i>] {band_label} → {fmt_range(*rng)}"
            try:
                await query.edit_message_text(
                    applied_text,
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=applied_text,
                    parse_mode="HTML",
                )
        elif action == "ignore":
            del self._pending[mid]
            dismissed_text = "[<i>Dismissed</i>]"
            try:
                await query.edit_message_text(
                    dismissed_text,
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=dismissed_text,
                    parse_mode="HTML",
                )
        elif action == "set":
            context.chat_data["await_exact"] = {"mid": mid, "band": band}
            prompt_text = (
                f"Send <code>low high</code> for band <b>{band_label}</b> "
                f"(e.g. <code>141.25 159.75</code>)."
            )
            try:
                await query.edit_message_text(
                    prompt_text,
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=prompt_text,
                    parse_mode="HTML",
                )
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=f"Enter <code>low high</code> for <b>{band_label}</b>.",
                reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
                parse_mode="HTML",
            )

    async def _on_adv_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("[<i>Unauthorized</i>]", show_alert=True)
            return

        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 2:
            await query.answer()
            return

        action = parts[1]
        message = query.message
        mid = message.message_id if message else None
        if mid is None or mid not in self._pending:
            await query.answer(
                "[<i>Stale</i>] This advisory is no longer active.",
                show_alert=True,
            )
            return

        kind, payload = self._pending[mid]
        if kind != "adv":
            await query.answer(
                "[<i>Stale</i>] This advisory is no longer active.",
                show_alert=True,
            )
            return

        ranges = cast(Dict[str, Tuple[float, float]], payload)

        await query.answer()

        if action == "apply":
            repo = self._ensure_repo()
            try:
                existing = (await repo.get_bands()).keys()
                filtered = {name: rng for name, rng in ranges.items() if name in existing}
                await repo.upsert_many(filtered)
            except Exception as exc:
                if self._log is not None:
                    try:
                        self._log.error(
                            "advisory_apply_failed",
                            error=str(exc),
                        )
                    except Exception:
                        pass
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text="[<i>Error</i>] Apply failed. Please retry.",
                    parse_mode="HTML",
                )
                return

            del self._pending[mid]
            if self._log is not None:
                try:
                    self._log.info(
                        "advisory_applied",
                        ranges={
                            name: (rng[0], rng[1]) for name, rng in sorted(filtered.items())
                        },
                    )
                except Exception:
                    pass
            summary = ", ".join(
                f"{escape(name.upper())}→{fmt_range(*rng)}" for name, rng in sorted(filtered.items())
            )
            message_text = (
                f"[<i>Applied</i>] {summary}" if summary else "[<i>Applied</i>]"
            )
            try:
                await query.edit_message_text(
                    message_text,
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=message_text,
                    parse_mode="HTML",
                )
        elif action == "ignore":
            del self._pending[mid]
            dismissed_text = "[<i>Dismissed</i>]"
            try:
                await query.edit_message_text(
                    dismissed_text,
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=dismissed_text,
                    parse_mode="HTML",
                )
        elif action == "set":
            del self._pending[mid]
            try:
                await query.edit_message_text(
                    "Select a band to edit.",
                    reply_markup=self._bands_menu_kb(),
                    parse_mode="HTML",
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text="Select a band to edit.",
                    reply_markup=self._bands_menu_kb(),
                    parse_mode="HTML",
                )

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.callback_query is None:
            return
        if not self._is_authorized(update):
            await update.callback_query.answer("[<i>Unauthorized</i>]", show_alert=True)
            return
        await update.callback_query.answer()

    async def _on_text_any(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="[<i>Unauthorized</i>]",
                    parse_mode="HTML",
                )
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
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="[<i>Invalid</i>] Expected two numbers: <code>low high</code>.",
                parse_mode="HTML",
            )
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
        try:
            if isinstance(mid, int):
                await context.bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=mid,
                    text=applied_text,
                    parse_mode="HTML",
                )
            else:
                raise ValueError
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=applied_text,
                parse_mode="HTML",
            )

        if self._log is not None:
            try:
                self._log.info("bands_set_exact", band=band, low=low, high=high)
            except Exception:
                pass

        bands = await repo.get_bands()
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=self._bands_menu_text(bands),
            reply_markup=self._bands_menu_kb(),
            parse_mode="HTML",
        )

        context.chat_data.pop("await_exact", None)

    def _parse_first_number(self, text: str) -> Optional[float]:
        tokens = text.split()
        for token in tokens:
            if token.startswith("/"):
                continue
            try:
                candidate = float(token.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(candidate):
                return candidate
        return None

    def _parse_tilt_sol_usdc(self, text: str) -> Optional[Tuple[float, float]]:
        normalized = text.replace(":", " ").replace("/", " ")
        tokens = normalized.split()
        numbers: list[float] = []
        for token in tokens:
            if token.startswith("/"):
                continue
            label = token.strip().lower()
            if label in {"sol", "usdc"}:
                continue
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            numbers.append(value)
            if len(numbers) == 2:
                break

        if not numbers:
            return None

        if len(numbers) == 1:
            sol_raw = numbers[0]
            if not math.isfinite(sol_raw):
                return None
            if sol_raw > 1:
                sol_frac = sol_raw / 100.0
            else:
                sol_frac = sol_raw
            if not math.isfinite(sol_frac):
                return None
            sol_frac = min(max(sol_frac, 0.0), 1.0)
            return sol_frac, 1.0 - sol_frac

        sol_raw, usdc_raw = numbers[0], numbers[1]
        if not (math.isfinite(sol_raw) and math.isfinite(usdc_raw)):
            return None
        total = sol_raw + usdc_raw
        if total <= 0 or not math.isfinite(total):
            return None
        sol_frac = sol_raw / total
        usdc_frac = usdc_raw / total
        sol_frac = min(max(sol_frac, 0.0), 1.0)
        usdc_frac = min(max(usdc_frac, 0.0), 1.0)
        total = sol_frac + usdc_frac
        if total == 0 or not math.isfinite(total):
            return None
        sol_frac /= total
        usdc_frac /= total
        return sol_frac, usdc_frac

    def _format_pct(self, fraction: float) -> str:
        pct_value = round(float(fraction) * 100.0, 1)
        if not math.isfinite(pct_value):
            pct_value = 0.0
        if pct_value.is_integer():
            return str(int(pct_value))
        return f"{pct_value:.1f}"

    def _parse_two_numbers(self, text: str) -> Optional[Tuple[float, float]]:
        tokens = text.split()
        numbers: list[float] = []
        for token in tokens:
            if token.startswith("/"):
                continue
            label = token.strip().lower()
            if label in {"sol", "usdc"}:
                continue
            try:
                token_clean = token.replace(",", "")
                value = float(token_clean)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            numbers.append(value)
            if len(numbers) == 2:
                break
        if len(numbers) < 2:
            return None
        return numbers[0], numbers[1]

    def _is_authorized(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == self._chat_id

    def _ensure_repo(self) -> DBRepo:
        if self._repo is None:
            raise RuntimeError("Telegram service repository not set. Call start() first.")
        return self._repo


telegramsvc = TelegramSvc
