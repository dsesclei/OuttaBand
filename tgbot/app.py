from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from html import escape
from typing import Any, Final, Literal, cast

from telegram import (
    CallbackQuery,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
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

from band_logic import fmt_range
from db_repo import DBRepo
from policy import VolReading
from policy.band_advisor import compute_amounts, split_for_bucket
from shared_types import (
    BAND_ORDER,
    AdvisoryPayload,
    AdvPayload,
    AlertPayload,
    BandMap,
    BandName,
    BandRange,
    Bucket,
    PendingKind,
)

from .callbacks import AdvAction, AlertAction, BandsAction, decode, encode
from .handlers import BotCtx, Handlers, Providers
from .pending import PendingStore
from .render import adv_kb, advisory_text, alert_kb, bands_menu_kb, bands_menu_text

ApplicationType = Application[Any, Any, Any, Any, Any, Any]
HandlerFn = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]

# ---------- Constants ----------

# Pending kinds
PENDING_KIND_ADV: PendingKind = "adv"
PENDING_KIND_ALERT: PendingKind = "alert"

# ChatData keys
CHATKEY_AWAIT_EXACT: Final[str] = "await_exact"
CHATKEY_MID: Final[str] = "mid"
CHATKEY_BAND: Final[str] = "band"

# Callback action tokens
BANDS_ACT_BACK: Literal["back"] = "back"
BANDS_ACT_EDIT: Literal["edit"] = "edit"

ALERT_ACT_ACCEPT: Literal["accept"] = "accept"
ALERT_ACT_IGNORE: Literal["ignore"] = "ignore"
ALERT_ACT_SET: Literal["set"] = "set"

ADV_ACT_APPLY: Literal["apply"] = "apply"
ADV_ACT_IGNORE: Literal["ignore"] = "ignore"
ADV_ACT_SET: Literal["set"] = "set"

# UI tags / fragments
TAG_UNAUTHORIZED = "[<i>Unauthorized</i>]"
TAG_ERROR = "[<i>Error</i>]"
TAG_APPLIED = "[<i>Applied</i>]"
TAG_DISMISSED = "[<i>Dismissed</i>]"
TAG_STALE = "[<i>Stale</i>]"
TAG_INVALID = "[<i>Invalid</i>]"

PROMPT_LOW_HIGH = "low high"
LABEL_BACK = "Back"


class TelegramApp:
    """Public facade matching the old TelegramSvc surface."""

    MAX_PENDING: Final[int] = int(os.getenv("LPBOT_PENDING_CAP", "100"))

    def __init__(self, token: str, chat_id: int, logger: Any | None = None) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: ApplicationType | None = None
        self._repo: DBRepo | None = None
        self._ready: asyncio.Event = asyncio.Event()
        self._log = logger
        self._price_provider: Callable[[], Awaitable[float | None]] | None = None
        self._sigma_provider: Callable[[], Awaitable[VolReading | None]] | None = None
        self._pending = PendingStore(cap=self.MAX_PENDING, ttl_s=3600)
        self._handlers: Handlers | None = None

    # ---------- Provider Setters ----------

    def set_price_provider(self, fn: Callable[[], Awaitable[float | None]]) -> None:
        self._price_provider = fn

    def set_sigma_provider(self, fn: Callable[[], Awaitable[VolReading | None]]) -> None:
        self._sigma_provider = fn

    # ---------- Lifecycle ----------

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return
        self._repo = repo
        self._ready.clear()
        self._handlers = Handlers(
            repo=repo,
            providers=Providers(self._price_provider, self._sigma_provider),
        )
        app = cast(
            ApplicationType,
            Application.builder()
            .token(self._token)
            .defaults(Defaults(parse_mode=ParseMode.HTML))
            .build(),
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

    # ---------- Public Send Helpers ----------

    async def send_text(self, text: str) -> None:
        await self._send(text, None)

    async def send_advisory_card(
        self, advisory: AdvisoryPayload, drift_line: str | None = None
    ) -> None:
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
        adv_payload: AdvPayload = dict(ranges)
        token = self._pending.put(PENDING_KIND_ADV, adv_payload)
        keyboard = adv_kb(token)
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=keyboard)

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
        if policy_meta is not None:
            bucket, width = policy_meta
            text = f"{text}\n<i>Policy</i>: {escape(bucket.title())} (±{width * 100:.2f}%)"

        repo = self._ensure_repo()
        notional = await repo.get_notional_usd()
        tilt_sol_frac = await repo.get_tilt_sol_frac()
        amount_line: str | None = None
        try:
            split = split_for_bucket(policy_meta[0]) if policy_meta else None
        except ValueError:
            split = None
        if split and notional and price > 0 and math.isfinite(price):
            pct = next(
                (share for name, share in zip(BAND_ORDER, split, strict=False) if name == band),
                None,
            )
            if pct and pct > 0:
                per_band_usd = (notional * pct) / 100.0
                tilt = min(max(tilt_sol_frac, 0.0), 1.0)
                sol_amt = round((per_band_usd * tilt) / price, 6)
                usdc_amt = round(per_band_usd * (1.0 - tilt), 2)
                amount_line = f"amount: {sol_amt:.6f} SOL / ${usdc_amt:.2f}"
        if amount_line:
            text = f"{text}\n{amount_line}"

        token = self._pending.put(PENDING_KIND_ALERT, AlertPayload(band, suggested_range))
        keyboard = alert_kb(band, token)
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=keyboard)

    # ---------- Wiring ----------

    def _wire(self, app: ApplicationType) -> None:
        handlers = self._ensure_handlers()
        bot_ctx = BotCtx(self._send, self._edit_or_send, self._edit_by_id)

        def auth_gate(fn: HandlerFn) -> HandlerFn:
            async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                chat = update.effective_chat
                if chat is None or chat.id != self._chat_id:
                    if update.callback_query:
                        await update.callback_query.answer(TAG_UNAUTHORIZED, show_alert=True)
                    elif chat:
                        await context.bot.send_message(chat_id=chat.id, text=TAG_UNAUTHORIZED)
                    return
                await fn(update, context)

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

        async def on_text_any(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            message = update.message
            if message is None:
                return
            chat_data_raw = context.chat_data
            if chat_data_raw is None:
                return
            chat_data = cast(dict[str, Any], chat_data_raw)
            state = chat_data.get(CHATKEY_AWAIT_EXACT)
            if not isinstance(state, dict):
                return
            payload = (message.text or "").replace(",", " ").split()
            try:
                if len(payload) != 2:
                    raise ValueError
                low, high = float(payload[0]), float(payload[1])
                if not (math.isfinite(low) and math.isfinite(high) and low < high):
                    raise ValueError
            except ValueError:
                await self._send(
                    f"{TAG_INVALID} Expected two numbers: <code>{PROMPT_LOW_HIGH}</code>.", None
                )
                return
            band = state.get(CHATKEY_BAND)
            mid = state.get(CHATKEY_MID)
            if not isinstance(band, str) or band not in BAND_ORDER:
                chat_data.pop(CHATKEY_AWAIT_EXACT, None)
                return
            band_name = cast(BandName, band)
            await self._ensure_repo().upsert_band(band_name, low, high)
            if isinstance(mid, int):
                await self._edit_by_id(
                    mid, f"{TAG_APPLIED} {escape(band_name.upper())} → {fmt_range(low, high)}"
                )
            bands = await self._ensure_repo().get_bands()
            await self._send(bands_menu_text(bands), bands_menu_kb())
            chat_data.pop(CHATKEY_AWAIT_EXACT, None)

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auth_gate(on_text_any)))

        async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            query = update.callback_query
            if query is None:
                return
            payload = decode(query.data or "")
            if payload is None:
                await query.answer()
                return

            if isinstance(payload, BandsAction):
                if payload.a == BANDS_ACT_BACK:
                    await query.answer()
                    bands = await self._ensure_repo().get_bands()
                    await self._edit_or_send(query, bands_menu_text(bands), bands_menu_kb())
                elif payload.a == BANDS_ACT_EDIT:
                    bands = await self._ensure_repo().get_bands()
                    band_name = payload.band
                    if band_name is None:
                        await query.answer(f"{TAG_ERROR} Unknown band.", show_alert=True)
                        return
                    current = bands.get(band_name)
                    if current is None:
                        await query.answer(f"{TAG_ERROR} Unknown band.", show_alert=True)
                        return
                    band_label = escape(band_name.upper())
                    text = (
                        f"<b>Edit Band</b> {band_label}\n"
                        f"<b>Current</b>: {fmt_range(*current)}\n"
                        f"Send <code>{PROMPT_LOW_HIGH}</code> or tap {LABEL_BACK}."
                    )
                    back_markup = InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    LABEL_BACK,
                                    callback_data=encode(BandsAction(a=BANDS_ACT_BACK)),
                                )
                            ]
                        ]
                    )
                    await self._edit_or_send(query, text, back_markup)
                    mid = query.message.message_id if query.message else None
                    chat_data_raw = context.chat_data
                    if chat_data_raw is None:
                        await query.answer(f"{TAG_ERROR} Unable to track state.", show_alert=True)
                        return
                    chat_data = cast(dict[str, Any], chat_data_raw)
                    chat_data[CHATKEY_AWAIT_EXACT] = {
                        CHATKEY_MID: mid,
                        CHATKEY_BAND: band_name,
                    }
                    await self._send(
                        f"Enter <code>{PROMPT_LOW_HIGH}</code> for <b>{band_label}</b>.",
                        ForceReply(selective=True, input_field_placeholder=PROMPT_LOW_HIGH),
                    )
                return

            if isinstance(payload, AlertAction):
                pending = self._pending.pop(PENDING_KIND_ALERT, payload.t)
                if not pending:
                    await query.answer(
                        f"{TAG_STALE} This alert is no longer active.", show_alert=True
                    )
                    return
                pend_payload = pending.payload
                if not isinstance(pend_payload, AlertPayload):
                    await query.answer(f"{TAG_ERROR} Malformed alert payload.", show_alert=True)
                    return
                if pend_payload.band != payload.band:
                    await query.answer(
                        f"{TAG_STALE} This alert is no longer active.", show_alert=True
                    )
                    return
                rng = pend_payload.suggested_range
                await query.answer()
                band_label = escape(payload.band.upper())
                if payload.a == ALERT_ACT_ACCEPT:
                    await self._ensure_repo().upsert_band(payload.band, rng[0], rng[1])
                    with suppress(Exception):
                        if self._log:
                            self._log.info(
                                "breach_applied", band=payload.band, low=rng[0], high=rng[1]
                            )
                    await self._edit_or_send(
                        query, f"{TAG_APPLIED} {band_label} → {fmt_range(*rng)}", None
                    )
                    return
                if payload.a == ALERT_ACT_IGNORE:
                    await self._edit_or_send(query, TAG_DISMISSED, None)
                    return
                if payload.a == ALERT_ACT_SET:
                    chat_data_raw = context.chat_data
                    if chat_data_raw is None:
                        await query.answer(f"{TAG_ERROR} Unable to track state.", show_alert=True)
                        return
                    chat_data = cast(dict[str, Any], chat_data_raw)
                    chat_data[CHATKEY_AWAIT_EXACT] = {
                        CHATKEY_MID: query.message.message_id if query.message else None,
                        CHATKEY_BAND: payload.band,
                    }
                    prompt = (
                        f"Send <code>{PROMPT_LOW_HIGH}</code> for band <b>{band_label}</b> "
                        f"(e.g. <code>141.25 159.75</code>)."
                    )
                    await self._edit_or_send(query, prompt, None)
                    await self._send(
                        f"Enter <code>{PROMPT_LOW_HIGH}</code> for <b>{band_label}</b>.",
                        ForceReply(selective=True, input_field_placeholder=PROMPT_LOW_HIGH),
                    )
                return

            if isinstance(payload, AdvAction):
                pending = self._pending.pop(PENDING_KIND_ADV, payload.t)
                if not pending:
                    await query.answer(
                        f"{TAG_STALE} This advisory is no longer active.", show_alert=True
                    )
                    return
                ranges = cast(AdvPayload, pending.payload)
                await query.answer()
                if payload.a == ADV_ACT_APPLY:
                    try:
                        existing = (await self._ensure_repo().get_bands()).keys()
                        filtered = {name: rng for name, rng in ranges.items() if name in existing}
                        await self._ensure_repo().upsert_many(filtered)
                    except Exception as exc:
                        with suppress(Exception):
                            if self._log:
                                self._log.error("advisory_apply_failed", error=str(exc))
                        await self._send(f"{TAG_ERROR} Apply failed. Please retry.", None)
                        return
                    with suppress(Exception):
                        if self._log:
                            self._log.info(
                                "advisory_applied",
                                ranges={
                                    name: (rng[0], rng[1]) for name, rng in sorted(filtered.items())
                                },
                            )
                    summary = ", ".join(
                        f"{escape(name.upper())}→{fmt_range(*rng)}"
                        for name, rng in sorted(filtered.items())
                    )
                    await self._edit_or_send(
                        query, f"{TAG_APPLIED} {summary}" if summary else TAG_APPLIED, None
                    )
                    return
                if payload.a == ADV_ACT_IGNORE:
                    await self._edit_or_send(query, TAG_DISMISSED, None)
                    return
                if payload.a == ADV_ACT_SET:
                    await self._edit_or_send(query, "Select a band to edit.", bands_menu_kb())
                return

        app.add_handler(CallbackQueryHandler(auth_gate(on_callback)))

    # ---------- Low-level Helpers ----------

    async def _send(self, text: str, reply_markup: Any | None) -> None:
        app = self._ensure_app()
        await self._ready.wait()
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=reply_markup)

    async def _edit_or_send(
        self, query: CallbackQuery, text: str, reply_markup: Any | None
    ) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except Exception:
            await self._send(text, reply_markup)

    async def _edit_by_id(self, message_id: int, text: str) -> bool:
        try:
            await self._ensure_app().bot.edit_message_text(
                chat_id=self._chat_id, message_id=message_id, text=text
            )
            return True
        except Exception:
            return False

    # ---------- Guards ----------

    def _ensure_app(self) -> ApplicationType:
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
