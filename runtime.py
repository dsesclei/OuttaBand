from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast
from zoneinfo import ZoneInfo

import aiosqlite
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]

import jobs
from config import Settings
from db_repo import DBRepo
from policy import volatility as vol
from policy.vol_sources import BinanceVolSource
from price_sources import MeteoraPriceSource

DEFAULT_TIMEOUT = httpx.Timeout(7.5, read=7.5, write=7.5, connect=5.0, pool=5.0)


def _load_tgbot_app() -> type[Any]:
    module = import_module("tgbot.app")
    telegram_app = getattr(module, "TelegramApp", None)
    if telegram_app is None:
        raise ImportError("tgbot.app.TelegramApp not found")
    return cast(type[Any], telegram_app)


class _DisabledTelegram:
    def __init__(self, logger: Any) -> None:
        self._log = logger

    async def start(self, repo: DBRepo) -> None:
        return

    async def stop(self) -> None:
        return

    def set_price_provider(self, fn: Callable[[], Any]) -> None:
        return

    def set_sigma_provider(self, fn: Callable[[], Any]) -> None:
        return

    async def send_breach_offer(self, **kwargs: Any) -> None:
        return

    async def send_advisory_card(
        self, advisory: dict[str, Any], drift_line: str | None = None
    ) -> None:
        return

    def is_ready(self) -> bool:
        return False


@dataclass
class Runtime:
    settings: Settings
    tz: ZoneInfo
    log: Any  # structlog bound logger

    http: httpx.AsyncClient | None = None
    db: aiosqlite.Connection | None = None
    repo: DBRepo | None = None
    tg: Any | None = None
    scheduler: AsyncIOScheduler | None = None
    ctx: jobs.AppContext | None = None

    last_run_ts_check: int | None = None
    last_run_ts_daily: int | None = None
    next_run_ts_check: int | None = None
    next_run_ts_daily: int | None = None
    slot_seconds: int = field(default=60)

    # ---------- Lifecycle ----------

    async def start(self) -> None:
        s, base_log = self.settings, self.log
        self.slot_seconds = max(60, s.CHECK_EVERY_MINUTES * 60)
        db = await aiosqlite.connect(s.DB_PATH)
        self.db = db
        self.repo = DBRepo(db, base_log.bind(module="db"))
        await self.repo.init(s)

        self.http = httpx.AsyncClient(http2=False, timeout=DEFAULT_TIMEOUT)

        tg_logger = base_log.bind(module="tgbot")
        if s.TELEGRAM_ENABLED:
            TelegramApp = _load_tgbot_app()
            self.tg = TelegramApp(
                s.TELEGRAM_BOT_TOKEN or "", int(s.TELEGRAM_CHAT_ID or 0), logger=tg_logger
            )
            await self.tg.start(self.repo)
        else:
            self.tg = _DisabledTelegram(tg_logger)

        price_src = MeteoraPriceSource(
            self.http, s.METEORA_PAIR_ADDRESS, s.METEORA_BASE_URL, s.HTTP_UA_MAIN
        )
        vol_src = BinanceVolSource(
            self.http,
            s.BINANCE_BASE_URL,
            s.BINANCE_SYMBOL,
            s.VOL_CACHE_TTL_SECONDS,
            s.VOL_MAX_STALE_SECONDS,
            s.HTTP_UA_VOL,
        )
        self.tg.set_price_provider(price_src.read)
        self.tg.set_sigma_provider(vol_src.read)

        job_settings = jobs.JobSettings(
            check_every_minutes=s.CHECK_EVERY_MINUTES,
            cooldown_minutes=s.COOLDOWN_MINUTES,
            include_a_on_high=s.INCLUDE_A_ON_HIGH,
            price_label="meteora",
        )
        self.ctx = jobs.AppContext(
            repo=self.repo,
            tg=self.tg,
            price=price_src,
            vol=vol_src,
            job=job_settings,
            log=base_log.bind(module="jobs"),
        )

        self._start_scheduler()

        self.log.info(
            "app_start",
            interval_min=s.CHECK_EVERY_MINUTES,
            check_trigger=self._check_trigger_desc(),
            slot_seconds=self.slot_seconds,
            tz=s.LOCAL_TZ,
            daily_hour=s.DAILY_LOCAL_HOUR,
            daily_minute=s.DAILY_LOCAL_MINUTE,
        )

    async def stop(self) -> None:
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.log.info("scheduler_stopped")
            self.scheduler = None
        if self.tg:
            if self.settings.TELEGRAM_ENABLED:
                await self.tg.stop()
            self.tg = None
        if self.http:
            await self.http.aclose()
            self.log.info("http_client_closed")
            self.http = None
        if self.db:
            await self.db.close()
            self.log.info("db_closed")
            self.repo = None
            self.db = None
        self.ctx = None
        self.log.info("app_stop")

    # ---------- Public facades for routes ----------

    async def sigma_payload(self) -> dict[str, Any]:
        data = None
        if self.ctx is not None:
            try:
                data = await self.ctx.vol.read()
            except Exception as exc:
                self.log.error("sigma_read_failed", err=str(exc))
        return {"ok": data is not None, "data": data}

    async def health_payload(self) -> dict[str, Any]:
        db_ok = False
        if self.db is not None:
            try:
                async with self.db.execute("SELECT 1") as cur:
                    await cur.fetchone()
                db_ok = True
            except Exception as exc:
                self.log.warning("health_db_check_failed", err=str(exc))
        http_client_ok = self.http is not None
        telegram_ready = self.tg.is_ready() if self.tg is not None else False
        scheduler_ok = False
        if self.scheduler is not None:
            try:
                _jobs = self.scheduler.get_jobs()
                scheduler_ok = bool(_jobs)
            except Exception as exc:
                self.log.warning("health_scheduler_inspect_failed", err=str(exc))
        cache_age = await vol.get_cache_age(
            self.settings.BINANCE_BASE_URL, self.settings.BINANCE_SYMBOL
        )
        return {
            "ok": bool(db_ok and http_client_ok and telegram_ready and scheduler_ok),
            "db_ok": db_ok,
            "http_client_ok": http_client_ok,
            "telegram_ready": telegram_ready,
            "scheduler_ok": scheduler_ok,
            "last_run_ts_check": self.last_run_ts_check,
            "last_run_ts_daily": self.last_run_ts_daily,
            "next_run_ts_check": self.next_run_ts_check,
            "next_run_ts_daily": self.next_run_ts_daily,
            "volatility_cache_age_s": cache_age,
        }

    # ---------- Scheduler wiring ----------

    def _start_scheduler(self) -> None:
        s = self.settings
        if not s.TELEGRAM_ENABLED:
            self.log.info("scheduler_disabled", reason="telegram disabled")
            return

        sched = AsyncIOScheduler(timezone=self.tz)
        self.scheduler = sched

        async def run_check_once() -> None:
            if not (self.repo and self.ctx and self.scheduler):
                return
            try:
                got = await self.repo.acquire_lock(
                    "job:check-once", ttl_s=max(60, self.slot_seconds // 2)
                )
            except Exception as exc:
                self.log.warning("job_lock_error", job="check-once", err=str(exc))
                return
            if not got:
                self.log.info("job_lock_skip", job="check-once")
                return
            try:
                await jobs.check_once(self.ctx, now_ts=int(time.time()))
            except Exception as exc:
                self.log.error("job_exception", err=str(exc))
            finally:
                self.last_run_ts_check = int(time.time())
                job = self.scheduler.get_job("check-once")
                self.next_run_ts_check = (
                    int(job.next_run_time.timestamp()) if job and job.next_run_time else None
                )
                self.log.info(
                    "job_next_run", job_id="check-once", next_run_ts=self.next_run_ts_check
                )

        async def run_daily_advisory() -> None:
            if not (self.repo and self.ctx and self.scheduler):
                return
            try:
                got = await self.repo.acquire_lock("job:daily-advisory", ttl_s=3600)
            except Exception as exc:
                self.log.warning("job_lock_error", job="daily-advisory", err=str(exc))
                return
            if not got:
                self.log.info("job_lock_skip", job="daily-advisory")
                return
            try:
                await jobs.send_daily_advisory(self.ctx)
            except Exception as exc:
                self.log.error("advisory_failed", err=str(exc))
            finally:
                self.last_run_ts_daily = int(time.time())
                job = self.scheduler.get_job("daily-advisory")
                self.next_run_ts_daily = (
                    int(job.next_run_time.timestamp()) if job and job.next_run_time else None
                )
                self.log.info(
                    "job_next_run", job_id="daily-advisory", next_run_ts=self.next_run_ts_daily
                )

        interval_minutes = max(1, s.CHECK_EVERY_MINUTES)
        if 60 % interval_minutes == 0:
            minutes_str = ",".join(str(m) for m in range(0, 60, interval_minutes))
            sched.add_job(
                run_check_once,
                "cron",
                id="check-once",
                minute=minutes_str,
                second=0,
                timezone=self.tz,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=120,
            )
            self._check_trigger = f"cron:{minutes_str}"
        else:
            interval_seconds = interval_minutes * 60
            sched.add_job(
                run_check_once,
                IntervalTrigger(seconds=interval_seconds, timezone=self.tz),
                id="check-once",
                coalesce=True,
                max_instances=1,
                misfire_grace_time=120,
            )
            self._check_trigger = f"interval:{interval_seconds}s"

        if s.DAILY_ENABLED:
            sched.add_job(
                run_daily_advisory,
                "cron",
                id="daily-advisory",
                hour=s.DAILY_LOCAL_HOUR,
                minute=s.DAILY_LOCAL_MINUTE,
                second=0,
                timezone=self.tz,
                coalesce=True,
                misfire_grace_time=1800,
                max_instances=1,
            )

        sched.start()
        cj = sched.get_job("check-once")
        dj = sched.get_job("daily-advisory") if s.DAILY_ENABLED else None
        self.next_run_ts_check = (
            int(cj.next_run_time.timestamp()) if cj and cj.next_run_time else None
        )
        self.next_run_ts_daily = (
            int(dj.next_run_time.timestamp()) if dj and dj.next_run_time else None
        )
        if cj:
            self.log.info("job_next_run", job_id=cj.id, next_run_ts=self.next_run_ts_check)
        if dj:
            self.log.info("job_next_run", job_id=dj.id, next_run_ts=self.next_run_ts_daily)
        self.log.info("cooldown_quantization", slot_seconds=self.slot_seconds)

    def _check_trigger_desc(self) -> str:
        return getattr(self, "_check_trigger", "disabled")
