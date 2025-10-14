# main.py
# mvp: sol/usdc band watcher with telegram alerts
# stack: fastapi, apscheduler (asyncio), python-telegram-bot (async), httpx, aiosqlite, pydantic v2

from __future__ import annotations

import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite
import httpx
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import FilteringBoundLogger

import jobs
import volatility as vol
from db_repo import DBRepo
from shared_types import BAND_ORDER
from sources import BinanceVolSource, MeteoraPriceSource
from telegram import TelegramApp

# ----------------------------
# Settings
# ----------------------------

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: int | None = None
    TELEGRAM_ENABLED: bool = True

    METEORA_PAIR_ADDRESS: str
    METEORA_BASE_URL: str = "https://dlmm-api.meteora.ag"

    CHECK_EVERY_MINUTES: int = 15
    COOLDOWN_MINUTES: int = 60
    DB_PATH: str = "./app.db"
    HTTP_UA_MAIN: str = "outtaband/0.1 (+https://github.com/desclei/OuttaBand)"
    HTTP_UA_VOL: str = "outtaband-volatility/0.1 (+https://github.com/dsesclei/OuttaBand)"

    # Optional band seeds (low-high). If provided, they will be upserted at startup.
    BAND_A: str | None = None
    BAND_B: str | None = None
    BAND_C: str | None = None

    BINANCE_BASE_URL: str = "https://api.binance.us"
    BINANCE_SYMBOL: str = "SOLUSDT"
    VOL_CACHE_TTL_SECONDS: int = 60
    VOL_MAX_STALE_SECONDS: int = 7200

    LOCAL_TZ: str = "America/New_York"
    DAILY_LOCAL_HOUR: int = 8
    DAILY_LOCAL_MINUTE: int = 0
    DAILY_ENABLED: bool = True
    INCLUDE_A_ON_HIGH: bool = False

    # pydantic v2 config: .env file, case-insensitive for ops sanity
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if not (0 <= self.DAILY_LOCAL_HOUR <= 23):
            raise ValueError("DAILY_LOCAL_HOUR must be between 0 and 23 inclusive")
        if not (0 <= self.DAILY_LOCAL_MINUTE <= 59):
            raise ValueError("DAILY_LOCAL_MINUTE must be between 0 and 59 inclusive")
        if self.VOL_CACHE_TTL_SECONDS < 5:
            raise ValueError("VOL_CACHE_TTL_SECONDS must be at least 5 seconds")
        if self.VOL_MAX_STALE_SECONDS < self.VOL_CACHE_TTL_SECONDS:
            raise ValueError("VOL_MAX_STALE_SECONDS must be >= VOL_CACHE_TTL_SECONDS")
        if self.COOLDOWN_MINUTES < 1:
            raise ValueError("COOLDOWN_MINUTES must be at least 1")
        if self.CHECK_EVERY_MINUTES < 1:
            raise ValueError("CHECK_EVERY_MINUTES must be at least 1")
        try:
            ZoneInfo(self.LOCAL_TZ)
        except Exception as exc:  # pragma: no cover - defensive, requires malformed tz name
            raise ValueError(f"Invalid LOCAL_TZ '{self.LOCAL_TZ}'") from exc
        if self.TELEGRAM_ENABLED:
            if not self.TELEGRAM_BOT_TOKEN or self.TELEGRAM_CHAT_ID is None:
                raise ValueError(
                    "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required when TELEGRAM_ENABLED=true"
                )
        return self


settings = Settings()
local_tz = ZoneInfo(settings.LOCAL_TZ)

SLOT_SECONDS = max(60, settings.CHECK_EVERY_MINUTES * 60)

last_run_ts_check: int | None = None
last_run_ts_daily: int | None = None
next_run_ts_check: int | None = None
next_run_ts_daily: int | None = None


# ----------------------------
# Globals
# ----------------------------

http_client: httpx.AsyncClient | None = None
db_conn: aiosqlite.Connection | None = None
repo: DBRepo | None = None
tg: TelegramApp | None = None
scheduler: AsyncIOScheduler | None = None
ctx: jobs.AppContext | None = None


# ----------------------------
# Logging
# ----------------------------

SERVICE_NAME = os.getenv("SERVICE_NAME", "lpbot")
GIT_SHA = os.getenv("GIT_SHA")
SERVICE_VERSION = os.getenv("SERVICE_VERSION") or GIT_SHA or "dev"


def configure_logging() -> FilteringBoundLogger:
    level_name = os.getenv("LPBOT_LOG_LEVEL", "INFO").upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    level = level_map.get(level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", key="ts", utc=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                    "foreign_pre_chain": [
                        structlog.processors.add_log_level,
                        timestamper,
                    ],
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "structlog",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": level,
                    "propagate": True,
                }
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("lpbot")


base_log = configure_logging().bind(service=SERVICE_NAME, version=SERVICE_VERSION)
if GIT_SHA:
    base_log = base_log.bind(git_sha=GIT_SHA)
log = base_log.bind(module="main")


DEFAULT_TIMEOUT = httpx.Timeout(7.5, read=7.5, write=7.5, connect=5.0, pool=5.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize core services and configure scheduler semantics.

    The recurring price check job uses cron-style scheduling when
    ``CHECK_EVERY_MINUTES`` divides 60 so runs stay aligned to wall-clock
    boundaries (e.g., :00, :15). Otherwise an interval trigger preserves the
    user-specified cadence without silently rounding to 15-minute buckets.
    """
    global SLOT_SECONDS
    global http_client, db_conn, repo, tg, scheduler, ctx
    global last_run_ts_check, last_run_ts_daily, next_run_ts_check, next_run_ts_daily

    db_conn = await aiosqlite.connect(settings.DB_PATH)
    repo = DBRepo(db_conn, base_log.bind(module="db"))
    await repo.init(settings)

    http_client = httpx.AsyncClient(http2=True, timeout=DEFAULT_TIMEOUT)

    tg = TelegramApp(
        settings.TELEGRAM_BOT_TOKEN or "",
        int(settings.TELEGRAM_CHAT_ID or 0),
        logger=base_log.bind(module="telegram"),
    )
    if settings.TELEGRAM_ENABLED:
        await tg.start(repo)

    price_src = MeteoraPriceSource(
        client=http_client,
        pair_address=settings.METEORA_PAIR_ADDRESS,
        base_url=settings.METEORA_BASE_URL,
        user_agent=settings.HTTP_UA_MAIN,
    )
    vol_src = BinanceVolSource(
        client=http_client,
        base_url=settings.BINANCE_BASE_URL,
        symbol=settings.BINANCE_SYMBOL,
        cache_ttl=settings.VOL_CACHE_TTL_SECONDS,
        max_stale=settings.VOL_MAX_STALE_SECONDS,
        user_agent=settings.HTTP_UA_VOL,
    )

    tg.set_price_provider(price_src.read)
    tg.set_sigma_provider(vol_src.read)

    SLOT_SECONDS = max(60, settings.CHECK_EVERY_MINUTES * 60)
    log.info("cooldown_quantization", slot_seconds=SLOT_SECONDS)

    band_seed_values = (settings.BAND_A, settings.BAND_B, settings.BAND_C)
    band_seeds = {
        name: value
        for name, value in zip(BAND_ORDER, band_seed_values, strict=False)
        if value is not None
    }

    job_settings = jobs.JobSettings(
        check_every_minutes=settings.CHECK_EVERY_MINUTES,
        cooldown_minutes=settings.COOLDOWN_MINUTES,
        include_a_on_high=settings.INCLUDE_A_ON_HIGH,
        price_label="meteora",
    )
    ctx = jobs.AppContext(
        repo=repo,
        tg=tg,
        price=price_src,
        vol=vol_src,
        job=job_settings,
        log=base_log.bind(module="jobs"),
    )

    log.info(
        "config_ok",
        meteora_url=settings.METEORA_BASE_URL,
        binance_url=settings.BINANCE_BASE_URL,
        binance_symbol=settings.BINANCE_SYMBOL,
        db_path=settings.DB_PATH,
        local_tz=settings.LOCAL_TZ,
        daily_hour=settings.DAILY_LOCAL_HOUR,
        daily_minute=settings.DAILY_LOCAL_MINUTE,
        include_a_on_high=settings.INCLUDE_A_ON_HIGH,
        band_seeds=band_seeds,
        http_ua_main=settings.HTTP_UA_MAIN,
        http_ua_vol=settings.HTTP_UA_VOL,
    )

    interval_minutes = max(1, settings.CHECK_EVERY_MINUTES)

    async def run_check_once() -> None:
        global last_run_ts_check, next_run_ts_check
        if repo is None or scheduler is None or ctx is None:
            return
        try:
            got = await repo.acquire_lock("job:check-once", ttl_s=max(60, SLOT_SECONDS // 2))
        except Exception as exc:
            log.warning("job_lock_error", job="check-once", err=str(exc))
            return
        if not got:
            log.info("job_lock_skip", job="check-once")
            return

        try:
            await jobs.check_once(ctx, now_ts=int(time.time()))
        except Exception as exc:
            log.error("job_exception", err=str(exc))
        finally:
            last_run_ts_check = int(time.time())
            if scheduler is not None:
                job = scheduler.get_job("check-once")
                next_run_ts_check = (
                    int(job.next_run_time.timestamp()) if job and job.next_run_time else None
                )
                log.info(
                    "job_next_run",
                    job_id="check-once",
                    next_run_ts=next_run_ts_check,
                )

    async def run_daily_advisory() -> None:
        global last_run_ts_daily, next_run_ts_daily
        if repo is None or scheduler is None or ctx is None:
            return
        try:
            got = await repo.acquire_lock("job:daily-advisory", ttl_s=3600)
        except Exception as exc:
            log.warning("job_lock_error", job="daily-advisory", err=str(exc))
            return
        if not got:
            log.info("job_lock_skip", job="daily-advisory")
            return

        try:
            await jobs.send_daily_advisory(ctx)
        except Exception as exc:
            log.error("advisory_failed", err=str(exc))
        finally:
            last_run_ts_daily = int(time.time())
            if scheduler is not None:
                job = scheduler.get_job("daily-advisory")
                next_run_ts_daily = (
                    int(job.next_run_time.timestamp()) if job and job.next_run_time else None
                )
                log.info(
                    "job_next_run",
                    job_id="daily-advisory",
                    next_run_ts=next_run_ts_daily,
                )

    global next_run_ts_check, next_run_ts_daily
    next_run_ts_check = None
    next_run_ts_daily = None

    check_trigger_desc = "disabled"
    scheduler = None

    if settings.TELEGRAM_ENABLED:
        scheduler = AsyncIOScheduler(timezone=local_tz)
        check_job_id = "check-once"

        if 60 % interval_minutes == 0:
            minutes_str = ",".join(str(minute) for minute in range(0, 60, interval_minutes))
            scheduler.add_job(
                run_check_once,
                "cron",
                id=check_job_id,
                minute=minutes_str,
                second=0,
                timezone=local_tz,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=120,
            )
            check_trigger_desc = f"cron:{minutes_str}"
        else:
            interval_seconds = interval_minutes * 60
            scheduler.add_job(
                run_check_once,
                IntervalTrigger(seconds=interval_seconds, timezone=local_tz),
                id=check_job_id,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=120,
            )
            check_trigger_desc = f"interval:{interval_seconds}s"

        if settings.DAILY_ENABLED:
            scheduler.add_job(
                run_daily_advisory,
                "cron",
                id="daily-advisory",
                hour=settings.DAILY_LOCAL_HOUR,
                minute=settings.DAILY_LOCAL_MINUTE,
                second=0,
                timezone=local_tz,
                coalesce=True,
                misfire_grace_time=1800,
                max_instances=1,
            )

        scheduler.start()

        check_job = scheduler.get_job(check_job_id)
        daily_job = scheduler.get_job("daily-advisory") if settings.DAILY_ENABLED else None
        next_run_ts_check = (
            int(check_job.next_run_time.timestamp()) if check_job and check_job.next_run_time else None
        )
        next_run_ts_daily = (
            int(daily_job.next_run_time.timestamp()) if daily_job and daily_job.next_run_time else None
        )
        if check_job:
            log.info(
                "job_next_run",
                job_id=check_job.id,
                next_run_ts=next_run_ts_check,
            )
        if daily_job:
            log.info(
                "job_next_run",
                job_id=daily_job.id,
                next_run_ts=next_run_ts_daily,
            )
    else:
        log.info("scheduler_disabled", reason="telegram disabled")

    log.info(
        "app_start",
        interval_min=settings.CHECK_EVERY_MINUTES,
        check_trigger=check_trigger_desc,
        slot_seconds=SLOT_SECONDS,
        tz=settings.LOCAL_TZ,
        daily_hour=settings.DAILY_LOCAL_HOUR,
        daily_minute=settings.DAILY_LOCAL_MINUTE,
    )

    try:
        yield
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)
            log.info("scheduler_stopped")
            scheduler = None
        if tg:
            if settings.TELEGRAM_ENABLED:
                await tg.stop()
            tg = None
        if http_client:
            await http_client.aclose()
            log.info("http_client_closed")
            http_client = None
        if db_conn:
            await db_conn.close()
            log.info("db_closed")
            repo = None
            db_conn = None
        ctx = None
        log.info("app_stop")


app = FastAPI(lifespan=lifespan)
# expose prometheus metrics; don't crash if unavailable
try:
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
except Exception:
    pass


@app.get("/sigma")
async def sigma() -> dict[str, Any]:
    data = None
    if ctx is not None:
        try:
            data = await ctx.vol.read()
        except Exception as exc:
            log.error("sigma_read_failed", err=str(exc))
    return {"ok": data is not None, "data": data}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    db_ok = False
    if db_conn is not None:
        try:
            async with db_conn.execute("SELECT 1") as cur:
                await cur.fetchone()
            db_ok = True
        except Exception as exc:
            log.warning("health_db_check_failed", err=str(exc))

    http_client_ok = http_client is not None
    telegram_ready = tg.is_ready() if tg is not None else False

    scheduler_ok = False
    if scheduler is not None:
        try:
            jobs = scheduler.get_jobs()
            scheduler_ok = bool(jobs)
        except Exception as exc:
            log.warning("health_scheduler_inspect_failed", err=str(exc))

    cache_age = await vol.get_cache_age(settings.BINANCE_BASE_URL, settings.BINANCE_SYMBOL)
    health: dict[str, Any] = {
        "ok": bool(db_ok and http_client_ok and telegram_ready and scheduler_ok),
        "db_ok": db_ok,
        "http_client_ok": http_client_ok,
        "telegram_ready": telegram_ready,
        "scheduler_ok": scheduler_ok,
        "last_run_ts_check": last_run_ts_check,
        "last_run_ts_daily": last_run_ts_daily,
        "next_run_ts_check": next_run_ts_check,
        "next_run_ts_daily": next_run_ts_daily,
        "volatility_cache_age_s": cache_age,
    }
    return health


@app.get("/version")
async def version() -> dict[str, str | None]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "git_sha": GIT_SHA,
    }


# ----------------------------
# Local run helper
# ----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
