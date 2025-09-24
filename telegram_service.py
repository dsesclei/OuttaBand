from __future__ import annotations

import asyncio
import math
import os
import time
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
from band_advisor import split_for_sigma
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
        app.add_handler(CommandHandler("status", self._on_status))
        app.add_handler(CommandHandler("bands", self._on_bands))
        app.add_handler(CommandHandler("setbaseline", self._on_setbaseline))
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
        await app.bot.send_message(chat_id=self._chat_id, text=text)

    async def send_advisory_card(self, advisory: Dict[str, object]) -> None:
        app = self._ensure_app()
        await self._ready.wait()

        price = float(advisory["price"])
        sigma_pct = cast(Optional[float], advisory.get("sigma_pct"))
        bucket = str(advisory["bucket"])
        split = cast(Tuple[int, int, int], advisory["split"])
        ranges = cast(Dict[str, Tuple[float, float]], advisory["ranges"])

        text = format_advisory_card(
            price,
            sigma_pct,
            bucket,
            ranges,
            split,
            stale=bool(advisory.get("stale")),
        )

        buttons = [
            [
                InlineKeyboardButton("apply all", callback_data="adv:apply"),
                InlineKeyboardButton("set exact", callback_data="adv:set"),
                InlineKeyboardButton("ignore", callback_data="adv:ignore"),
            ]
        ]

        message = await app.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
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
        price = await self._get_price()
        sigma = await self._get_sigma()
        bands = await repo.get_bands()
        latest = await repo.get_latest_snapshot()
        baseline = await repo.get_baseline()

        lines = []
        if price is not None and math.isfinite(price):
            lines.append(f"p={price:.2f}")
        else:
            lines.append("p=unknown")

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
        lines.append(
            f"σ={sigma_display} ({sigma_bucket})" + (" (STALE)" if sigma_stale else "")
        )

        lines.append("bands:")
        if bands:
            for name in sorted(bands.keys()):
                lo, hi = bands[name]
                lines.append(f"{name}: {fmt_range(lo, hi)}")
        else:
            lines.append("(none)")

        split = split_for_sigma(sigma_pct_value)
        lines.append(f"advisory split: {split[0]}/{split[1]}/{split[2]}")

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
                        f"drift now: ${drift_now:+.2f} ({drift_pct * 100:+.2f}%)"
                    )
                else:
                    lines.append("drift now: unavailable")
            else:
                lines.append("drift now: unavailable")
        elif latest:
            _snap_ts, _snap_sol, _snap_usdc, snap_price, snap_drift = latest
            if math.isfinite(snap_price) and math.isfinite(snap_drift):
                lines.append(f"drift (last @ {snap_price:.2f}): ${snap_drift:+.2f}")

        await context.bot.send_message(chat_id=self._chat_id, text="\n".join(lines))

    async def _on_setbaseline(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(chat_id=chat.id, text="unauthorized")
            return

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="usage: /setbaseline n sol m usdc",
            )
            return

        sol, usdc = parsed
        sol = max(sol, 0.0)
        usdc = max(usdc, 0.0)
        if not (math.isfinite(sol) and math.isfinite(usdc)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="usage: /setbaseline n sol m usdc",
            )
            return

        repo = self._ensure_repo()
        ts = int(time.time())
        await repo.set_baseline(sol, usdc, ts)
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=f"baseline set: {sol:g} sol, {usdc:g} usdc",
        )

    async def _on_updatebalances(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_message(chat_id=chat.id, text="unauthorized")
            return

        message = update.message
        text = message.text if message and message.text else ""
        parsed = self._parse_two_numbers(text)
        if parsed is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="usage: /updatebalances x sol y usdc",
            )
            return

        sol_amt, usdc_amt = parsed
        sol_amt = max(sol_amt, 0.0)
        usdc_amt = max(usdc_amt, 0.0)
        if not (math.isfinite(sol_amt) and math.isfinite(usdc_amt)):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="usage: /updatebalances x sol y usdc",
            )
            return

        price = await self._get_price()
        if price is None or not math.isfinite(price) or price <= 0:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="price unavailable, try again later",
            )
            return

        repo = self._ensure_repo()
        baseline = await repo.get_baseline()
        if baseline is None:
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="baseline not set. run /setbaseline first",
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
                text="unable to compute drift",
            )
            return

        drift = cur_val - base_val
        if not math.isfinite(drift):
            await context.bot.send_message(
                chat_id=self._chat_id,
                text="unable to compute drift",
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
                f"drift: ${drift:+.2f} ({drift_pct * 100:+.2f}%) | "
                f"base ${base_val:.2f} → now ${cur_val:.2f} @ p={price:.2f}"
            ),
        )

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

        kind, payload = self._pending[mid]
        if kind != "alert":
            await query.answer("stale or unknown alert", show_alert=True)
            return

        pending_band, rng = cast(Tuple[str, Tuple[float, float]], payload)
        if pending_band != band:
            await query.answer("stale or unknown alert", show_alert=True)
            return

        await query.answer()

        if action == "accept":
            repo = self._ensure_repo()
            await repo.upsert_band(band, rng[0], rng[1])
            if self._log is not None:
                try:
                    self._log.info("breach_applied", band=band, low=rng[0], high=rng[1])
                except Exception:
                    pass
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

    async def _on_adv_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        action = parts[1]
        message = query.message
        mid = message.message_id if message else None
        if mid is None or mid not in self._pending:
            await query.answer("stale or unknown advisory", show_alert=True)
            return

        kind, payload = self._pending[mid]
        if kind != "adv":
            await query.answer("stale or unknown advisory", show_alert=True)
            return

        ranges = cast(Dict[str, Tuple[float, float]], payload)

        await query.answer()

        if action == "apply":
            repo = self._ensure_repo()
            try:
                await repo.upsert_many(ranges)
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
                    text="apply failed, please retry",
                )
                return

            del self._pending[mid]
            if self._log is not None:
                try:
                    self._log.info(
                        "advisory_applied",
                        ranges={name: (rng[0], rng[1]) for name, rng in sorted(ranges.items())},
                    )
                except Exception:
                    pass
            summary = ", ".join(
                f"{name}→{fmt_range(*rng)}" for name, rng in sorted(ranges.items())
            )
            message_text = f"applied: {summary}" if summary else "applied"
            try:
                await query.edit_message_text(message_text)
            except Exception:
                await context.bot.send_message(chat_id=self._chat_id, text=message_text)
        elif action == "ignore":
            del self._pending[mid]
            try:
                await query.edit_message_text("dismissed")
            except Exception:
                await context.bot.send_message(chat_id=self._chat_id, text="dismissed")
        elif action == "set":
            del self._pending[mid]
            try:
                await query.edit_message_text(
                    "tap a band to edit",
                    reply_markup=self._bands_menu_kb(),
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=self._chat_id,
                    text="tap a band to edit",
                    reply_markup=self._bands_menu_kb(),
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
                self._log.info("bands_set_exact", band=band, low=low, high=high)
            except Exception:
                pass

        bands = await repo.get_bands()
        await context.bot.send_message(
            chat_id=self._chat_id,
            text=self._bands_menu_text(bands),
            reply_markup=self._bands_menu_kb(),
        )

        context.chat_data.pop("await_exact", None)

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
