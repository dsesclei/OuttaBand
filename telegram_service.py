from __future__ import annotations

import asyncio
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from db_repo import DBRepo


class TelegramSvc:
    def __init__(self, token: str, chat_id: int) -> None:
        self._token = token
        self._chat_id = chat_id
        self._app: Optional[Application] = None
        self._repo: Optional[DBRepo] = None
        self._polling_task: Optional[asyncio.Task[None]] = None

    async def start(self, repo: DBRepo) -> None:
        if self._app is not None:
            return

        self._repo = repo
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._on_start))
        app.add_handler(CommandHandler("status", self._on_status))
        app.add_handler(CallbackQueryHandler(self._on_callback))

        self._app = app
        self._polling_task = asyncio.create_task(
            app.run_polling(allowed_updates=None)
        )

    async def stop(self) -> None:
        if self._app is None:
            return

        await self._app.stop()
        await self._app.shutdown()

        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._polling_task = None
        self._app = None
        self._repo = None

    async def send_text(self, text: str) -> None:
        app = self._ensure_app()
        await app.bot.send_message(chat_id=self._chat_id, text=text)

    async def send_alert_with_buttons(
        self,
        text: str,
        buttons: list[list[InlineKeyboardButton]],
    ) -> None:
        app = self._ensure_app()
        markup = InlineKeyboardMarkup(buttons)
        await app.bot.send_message(chat_id=self._chat_id, text=text, reply_markup=markup)

    def _ensure_app(self) -> Application:
        if self._app is None:
            raise RuntimeError("Telegram service has not been started")
        return self._app

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await context.bot.send_message(chat_id=self._chat_id, text="Watcher online. Use /status for bands.")

    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        repo = self._ensure_repo()
        bands = await repo.get_bands()
        if not bands:
            body = "(no bands configured)"
        else:
            lines = ["Configured bands:"]
            for name in sorted(bands.keys()):
                lo, hi = bands[name]
                lines.append(f"{name}: {lo:.2f} – {hi:.2f}")
            body = "\n".join(lines)
        await context.bot.send_message(chat_id=self._chat_id, text=body)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.callback_query is None:
            return
        if not self._is_authorized(update):
            await update.callback_query.answer()
            return
        await update.callback_query.answer()

    def _is_authorized(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == self._chat_id

    def _ensure_repo(self) -> DBRepo:
        if self._repo is None:
            raise RuntimeError("Telegram service repository not set. Call start() first.")
        return self._repo
