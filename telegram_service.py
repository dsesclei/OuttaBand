from __future__ import annotations

import asyncio
import math
from typing import Callable, Dict, Optional, Tuple

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
from band_logic import fmt_range


class TelegramSvc:
    def __init__(self, token: str, chat_id: int) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: Optional[Application] = None
        self._repo: Optional[DBRepo] = None
        self._ready: asyncio.Event = asyncio.Event()
        self._pending: Dict[int, Tuple[str, Tuple[float, float]]] = {}
        self._log: Optional[Callable[..., None]] = None

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return

        self._repo = repo
        self._log = getattr(repo, "_log", None)
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._on_start))
        app.add_handler(CommandHandler("status", self._on_status))
        app.add_handler(CommandHandler("bands", self._on_bands))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_any))
        app.add_handler(CallbackQueryHandler(self._on_alert_action, pattern=r"^alert:(accept|ignore|set):[abc]$"))
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
        await app.bot.send_message(chat_id=self._chat_id, text=text)

    async def send_breach_offer(
        self,
        band: str,
        price: float,
        src_label: Optional[str],
        bands: dict[str, Tuple[float, float]],
        suggested_range: Tuple[float, float],
    ) -> None:
        app = self._ensure_app()
        await self._ready.wait()

        current = bands.get(band)
        if current is None:
            raise ValueError(f"unknown band '{band}'")

        side = "below" if price < current[0] else "above"
        source = f" ({src_label})" if src_label else ""
        text = (
            "◧ hard break\n"
            f"band: {band.upper()} ({side})\n"
            f"price: {price:.2f}{source}\n"
            f"current {band}: {fmt_range(*current)}\n"
            f"suggested {band}: {fmt_range(*suggested_range)}"
        )
        buttons = [
            [
                InlineKeyboardButton("accept", callback_data=f"alert:accept:{band}"),
                InlineKeyboardButton("ignore", callback_data=f"alert:ignore:{band}"),
                InlineKeyboardButton("set exact", callback_data=f"alert:set:{band}"),
            ]
        ]
        message = await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        self._pending[message.message_id] = (band, suggested_range)

    def _ensure_app(self) -> Application:
        if self._app is None:
            raise RuntimeError("Telegram service has not been started")
        return self._app

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(chat_id=chat.id, text="unauthorized")
            return
        await context.bot.send_message(chat_id=self._chat_id, text="Watcher online. Use /status for bands.")

    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(chat_id=chat.id, text="unauthorized")
            return
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        if not bands:
            body = "(no bands configured)"
        else:
            lines = ["Configured bands:"]
            for name in sorted(bands.keys()):
                lo, hi = bands[name]
                lines.append(f"{name}: {fmt_range(lo, hi)}")
            body = "\n".join(lines)
        await context.bot.send_message(chat_id=self._chat_id, text=body)

    def _bands_menu_text(self, bands: Dict[str, Tuple[float, float]]) -> str:
        if not bands:
            return "(no bands configured)"
        lines = ["configured bands:"]
        for name in sorted(bands.keys()):
            lo, hi = bands[name]
            lines.append(f"{name}: {fmt_range(lo, hi)}")
        return "\n".join(lines)

    def _bands_menu_kb(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("edit a", callback_data="b:a")],
                [InlineKeyboardButton("edit b", callback_data="b:b")],
                [InlineKeyboardButton("edit c", callback_data="b:c")],
            ]
        )

    async def _on_bands(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        text = self._bands_menu_text(bands)
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=self._bands_menu_kb(),
        )

    async def _on_bands_pick(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("unauthorized", show_alert=True)
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
            await query.answer("unknown band", show_alert=True)
            return

        editor_text = (
            f"◧ edit band {band.upper()}\n"
            f"current: {fmt_range(*current)}\n"
            "send \"low high\" or tap back"
        )
        back_markup = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data="b:back")]])
        try:
            await query.edit_message_text(editor_text, reply_markup=back_markup)
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=editor_text,
                reply_markup=back_markup,
            )

        mid = query.message.message_id if query.message else None
        context.chat_data["await_exact"] = {"mid": mid, "band": band}
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=f"enter low high for {band.upper()}",
            reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
        )

    async def _on_bands_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("unauthorized", show_alert=True)
            return

        repo = self._ensure_repo()
        bands = await repo.get_bands()
        await query.answer()
        try:
            await query.edit_message_text(
                self._bands_menu_text(bands),
                reply_markup=self._bands_menu_kb(),
            )
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=self._bands_menu_text(bands),
                reply_markup=self._bands_menu_kb(),
            )

    async def _on_alert_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer("unauthorized", show_alert=True)
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
            await query.answer("stale or unknown alert", show_alert=True)
            return

        pending_band, rng = self._pending[mid]
        if pending_band != band:
            await query.answer("stale or unknown alert", show_alert=True)
            return

        await query.answer()

        if action == "accept":
            repo = self._ensure_repo()
            await repo.upsert_band(band, rng[0], rng[1])
            del self._pending[mid]
            try:
                await query.edit_message_text(f"applied: {band} → {fmt_range(*rng)}")
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=f"applied: {band} → {fmt_range(*rng)}",
                )
        elif action == "ignore":
            del self._pending[mid]
            try:
                await query.edit_message_text("dismissed")
            except Exception:
                await context.bot.send_message(chat_id=self._chat_id, text="dismissed")
        elif action == "set":
            context.chat_data["await_exact"] = {"mid": mid, "band": band}
            try:
                await query.edit_message_text(
                    f"send \"low high\" for band {band.upper()} (e.g., \"141.25 159.75\")"
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text=f"send \"low high\" for band {band.upper()} (e.g., \"141.25 159.75\")",
                )
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=f"enter low high for {band.upper()}",
                reply_markup=ForceReply(selective=True, input_field_placeholder="low high"),
            )

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.callback_query is None:
            return
        if not self._is_authorized(update):
            await update.callback_query.answer("unauthorized", show_alert=True)
            return
        await update.callback_query.answer()

    async def _on_text_any(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(chat_id=chat.id, text="unauthorized")
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
                text="invalid format, expected two numbers: low high",
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
        try:
            if isinstance(mid, int):
                await context.bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=mid,
                    text=f"applied: {band} → {fmt_range(low, high)}",
                )
            else:
                raise ValueError
        except Exception:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text=f"applied: {band} → {fmt_range(low, high)}",
            )

        if self._log is not None:
            try:
                self._log("info", "bands_set_exact", band=band, low=low, high=high)
            except Exception:
                pass

        bands = await repo.get_bands()
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=self._bands_menu_text(bands),
            reply_markup=self._bands_menu_kb(),
        )

        context.chat_data.pop("await_exact", None)

    def _is_authorized(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == self._chat_id

    def _ensure_repo(self) -> DBRepo:
        if self._repo is None:
            raise RuntimeError("Telegram service repository not set. Call start() first.")
        return self._repo


telegramsvc = TelegramSvc
