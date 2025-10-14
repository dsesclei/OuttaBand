from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from html import escape
from typing import Any, cast

from band_advisor import compute_amounts, split_for_bucket
from band_logic import fmt_range
from db_repo import DBRepo
from shared_types import AdvisoryPayload, BandMap, BandRange
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
from volatility import VolReading

from .callbacks import AdvAction, AlertAction, BandsAction, decode, encode
from .handlers import BotCtx, Handlers, Providers
from .pending import PendingStore
from .render import adv_kb, advisory_text, alert_kb, bands_menu_kb, bands_menu_text


class TelegramApp:
    """Public façade matching the old TelegramSvc surface."""

    MAX_PENDING = int(os.getenv("LPBOT_PENDING_CAP", "100"))

    def __init__(self, token: str, chat_id: int, logger: Any | None = None) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: Application | None = None
        self._repo: DBRepo | None = None
        self._ready: asyncio.Event = asyncio.Event()
        self._log = logger
        self._price_provider: Callable[[], Awaitable[float | None]] | None = None
        self._sigma_provider: Callable[[], Awaitable[VolReading | None]] | None = None
        self._pending = PendingStore(cap=self.MAX_PENDING, ttl_s=3600)
        self._handlers: Handlers | None = None

    # Provider setters

    def set_price_provider(self, fn: Callable[[], Awaitable[float | None]]) -> None:
        self._price_provider = fn

    def set_sigma_provider(self, fn: Callable[[], Awaitable[VolReading | None]]) -> None:
        self._sigma_provider = fn

    # Lifecycle

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return
        self._repo = repo
        self._ready.clear()
        self._handlers = Handlers(
            repo=repo,
            providers=Providers(self._price_provider, self._sigma_provider),
        )
        app = (
            Application.builder()
            .token(self._token)
            .defaults(Defaults(parse_mode=ParseMode.HTML))
            .build()
        )
        self._wire(app)
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
        self._handlers = None
        self._ready.clear()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    # Public send helpers

    async def send_text(self, text: str) -> None:
        await self._send(text, None)

    async def send_advisory_card(self, advisory: AdvisoryPayload, drift_line: str | None = None) -> None:
        app = self._ensure_app()
        await self._ready.wait()
        repo = self._ensure_repo()

        price = advisory["price"]
        sigma_pct = advisory["sigma_pct"]
        bucket = advisory["bucket"]
        ranges = advisory["ranges"]
        split = advisory["split"]

        # Compute allocation figures
        notional = await repo.get_notional_usd()
        tilt = await repo.get_tilt_sol_frac()
        amounts_map, unallocated_usd = compute_amounts(
            price,
            split,
            ranges,
            notional,
            tilt,
        )

        # Render advisory text
        text = advisory_text(
            price,
            sigma_pct,
            bucket,
            ranges,
            split,
            stale=bool(advisory.get("stale")),
            amounts=amounts_map or None,
            unallocated_usd=unallocated_usd if unallocated_usd > 0 else None,
        )
        if drift_line:
            text = f"{text}\n{drift_line}"

        # Stash pending payload (BandMap) and send
        token = self._pending.put("adv", dict(ranges))
        keyboard = adv_kb(token)
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=keyboard)
        
    async def send_breach_offer(
        self,
        band: str,
        price: float,
        src_label: str | None,
        bands: BandMap,
        suggested_range: BandRange,
        policy_meta: tuple[str, float] | None = None,
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
        if policy_meta is not None:
            bucket, width = policy_meta
            text = f"{text}\n<i>Policy</i>: {escape(bucket.title())} (±{width * 100:.2f}%)"

        repo = self._ensure_repo()
        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()
        amount_line: str | None = None
        try:
            split = split_for_bucket(policy_meta[0]) if policy_meta else None  # type: ignore[index]
        except ValueError:
            split = None
        if split and notional and price > 0 and math.isfinite(price):
            bands_order = ("a", "b", "c")
            pct = next((share for name, share in zip(bands_order, split, strict=False) if name == band), None)
            if pct and pct > 0:
                per_band_usd = (notional * pct) / 100.0
                tilt = min(max(tilt_sol_frac, 0.0), 1.0)
                sol_amt = round((per_band_usd * tilt) / price, 6)
                usdc_amt = round(per_band_usd * (1.0 - tilt), 2)
                amount_line = f"amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f}"
        if amount_line:
            text = f"{text}\n{amount_line}"

        token = self._pending.put("alert", (band, suggested_range))
        keyboard = alert_kb(band, token)
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=keyboard)

    # Wiring

    def _wire(self, app: Application) -> None:
        handlers = self._ensure_handlers()
        bot_ctx = BotCtx(self._send, self._edit_or_send, self._edit_by_id)

        def auth_gate(fn):
            async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
                chat = update.effective_chat
                if chat is None or chat.id != self._chat_id:
                    if update.callback_query:
                        await update.callback_query.answer("[<i>Unauthorized</i>]", show_alert=True)
                    elif chat:
                        await context.bot.send_message(chat_id=chat.id, text="[<i>Unauthorized</i>]")
                    return
                return await fn(update, context)

            return wrapped

        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await handlers.cmd_start(bot_ctx)

        async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await handlers.cmd_help(bot_ctx)

        async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await handlers.cmd_status(bot_ctx)

        async def cmd_bands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await handlers.cmd_bands(bot_ctx)

        async def cmd_setbaseline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            text = update.message.text if update.message and update.message.text else ""
            await handlers.cmd_setbaseline(bot_ctx, text)

        async def cmd_setnotional(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            text = update.message.text if update.message and update.message.text else ""
            await handlers.cmd_setnotional(bot_ctx, text)

        async def cmd_settilt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            text = update.message.text if update.message and update.message.text else ""
            await handlers.cmd_settilt(bot_ctx, text)

        async def cmd_updatebalances(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            text = update.message.text if update.message and update.message.text else ""
            await handlers.cmd_updatebalances(bot_ctx, text)

        commands = [
            ("start", cmd_start),
            ("help", cmd_help),
            ("status", cmd_status),
            ("bands", cmd_bands),
            ("setbaseline", cmd_setbaseline),
            ("setnotional", cmd_setnotional),
            ("settilt", cmd_settilt),
            ("updatebalances", cmd_updatebalances),
        ]

        for name, fn in commands:
            app.add_handler(CommandHandler(name, auth_gate(fn)))

        async def on_text_any(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
                await self._send("[<i>Invalid</i>] Expected two numbers: <code>low high</code>.", None)
                return
            band = state.get("band")
            mid = state.get("mid")
            if not isinstance(band, str):
                context.chat_data.pop("await_exact", None)
                return
            await self._ensure_repo().upsert_band(band, low, high)
            if isinstance(mid, int):
                await self._edit_by_id(mid, f"[<i>Applied</i>] {escape(band.upper())} → {fmt_range(low, high)}")
            bands = await self._ensure_repo().get_bands()
            await self._send(bands_menu_text(bands), bands_menu_kb())
            context.chat_data.pop("await_exact", None)

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auth_gate(on_text_any)))

        async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            if not query:
                return
            payload = decode(query.data or "")
            if payload is None:
                await query.answer()
                return

            if isinstance(payload, BandsAction):
                if payload.a == "back":
                    await query.answer()
                    bands = await self._ensure_repo().get_bands()
                    await self._edit_or_send(query, bands_menu_text(bands), bands_menu_kb())
                elif payload.a == "edit":
                    bands = await self._ensure_repo().get_bands()
                    band = (payload.band or "").lower()
                    current = bands.get(band)
                    if current is None:
                        await query.answer("[<i>Error</i>] Unknown band.", show_alert=True)
                        return
                    band_label = escape(band.upper())
                    text = (
                        f"<b>Edit Band</b> {band_label}\n"
                        f"<b>Current</b>: {fmt_range(*current)}\n"
                        "Send <code>low high</code> or tap Back."
                    )
                    back_markup = InlineKeyboardMarkup(
                        [[InlineKeyboardButton("Back", callback_data=encode(BandsAction(a="back")))]]
                    )
                    await self._edit_or_send(query, text, back_markup)
                    mid = query.message.message_id if query.message else None
                    context.chat_data["await_exact"] = {"mid": mid, "band": band}
                    await self._send(
                        f"Enter <code>low high</code> for <b>{band_label}</b>.",
                        ForceReply(selective=True, input_field_placeholder="low high"),
                    )
                return

            if isinstance(payload, AlertAction):
                pending = self._pending.pop("alert", payload.t)
                if not pending:
                    await query.answer("[<i>Stale</i>] This alert is no longer active.", show_alert=True)
                    return
                pend_band, rng = cast(tuple[str, BandRange], pending.payload)
                if pend_band != payload.band:
                    await query.answer("[<i>Stale</i>] This alert is no longer active.", show_alert=True)
                    return
                await query.answer()
                band_label = escape(payload.band.upper())
                if payload.a == "accept":
                    await self._ensure_repo().upsert_band(payload.band, rng[0], rng[1])
                    with suppress(Exception):
                        if self._log:
                            self._log.info("breach_applied", band=payload.band, low=rng[0], high=rng[1])
                    await self._edit_or_send(query, f"[<i>Applied</i>] {band_label} → {fmt_range(*rng)}", None)
                    return
                if payload.a == "ignore":
                    await self._edit_or_send(query, "[<i>Dismissed</i>]", None)
                    return
                if payload.a == "set":
                    context.chat_data["await_exact"] = {"mid": query.message.message_id if query.message else None, "band": payload.band}
                    prompt = f"Send <code>low high</code> for band <b>{band_label}</b> (e.g. <code>141.25 159.75</code>)."
                    await self._edit_or_send(query, prompt, None)
                    await self._send(
                        f"Enter <code>low high</code> for <b>{band_label}</b>.",
                        ForceReply(selective=True, input_field_placeholder="low high"),
                    )
                return

            if isinstance(payload, AdvAction):
                pending = self._pending.pop("adv", payload.t)
                if not pending:
                    await query.answer("[<i>Stale</i>] This advisory is no longer active.", show_alert=True)
                    return
                ranges = cast(BandMap, pending.payload)
                await query.answer()
                if payload.a == "apply":
                    try:
                        existing = (await self._ensure_repo().get_bands()).keys()
                        filtered = {name: rng for name, rng in ranges.items() if name in existing}
                        await self._ensure_repo().upsert_many(filtered)
                    except Exception as exc:
                        with suppress(Exception):
                            if self._log:
                                self._log.error("advisory_apply_failed", error=str(exc))
                        await self._send("[<i>Error</i>] Apply failed. Please retry.", None)
                        return
                    with suppress(Exception):
                        if self._log:
                            self._log.info(
                                "advisory_applied",
                                ranges={name: (rng[0], rng[1]) for name, rng in sorted(filtered.items())},
                            )
                    summary = ", ".join(f"{escape(name.upper())}→{fmt_range(*rng)}" for name, rng in sorted(filtered.items()))
                    await self._edit_or_send(query, f"[<i>Applied</i>] {summary}" if summary else "[<i>Applied</i>]", None)
                    return
                if payload.a == "ignore":
                    await self._edit_or_send(query, "[<i>Dismissed</i>]", None)
                    return
                if payload.a == "set":
                    await self._edit_or_send(query, "Select a band to edit.", bands_menu_kb())
                return

        app.add_handler(CallbackQueryHandler(auth_gate(on_callback)))

    # Low-level helpers

    async def _send(self, text: str, reply_markup: Any | None) -> None:
        app = self._ensure_app()
        await self._ready.wait()
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=reply_markup)

    async def _edit_or_send(self, query: CallbackQuery, text: str, reply_markup: Any | None) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except Exception:
            await self._send(text, reply_markup)

    async def _edit_by_id(self, message_id: int, text: str) -> bool:
        try:
            await self._ensure_app().bot.edit_message_text(chat_id=self._chat_id, message_id=message_id, text=text)
            return True
        except Exception:
            return False

    # Guards

    def _ensure_app(self) -> Application:
        if self._app is None:
            raise RuntimeError("Telegram app has not been started")
        return self._app

    def _ensure_repo(self) -> DBRepo:
        if self._repo is None:
            raise RuntimeError("Telegram app repository not set. Call start() first.")
        return self._repo

    def _ensure_handlers(self) -> Handlers:
        if self._handlers is None:
            raise RuntimeError("Telegram handlers are not initialized")
        return self._handlers
