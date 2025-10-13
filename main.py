# main.py
# mvp: sol/usdc band watcher with telegram alerts
# stack: fastapi, apscheduler (asyncio), python-telegram-bot (async), httpx, aiosqlite, pydantic v2

from __future__ import annotations

import asyncio
import logging
import logging.config
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aiosqlite
import httpx
import structlog
from structlog.typing import FilteringBoundLogger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from zoneinfo import ZoneInfo

from db_repo import DBRepo
from band_logic import broken_bands
import band_advisor
import volatility as vol
from telegram_service import TelegramSvc
import net


# ----------------------------
# Settings
# ----------------------------

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int

    METEORA_PAIR_ADDRESS: str
    METEORA_BASE_URL: str = "https://dlmm-api.meteora.ag"

    CHECK_EVERY_MINUTES: int = 15
    COOLDOWN_MINUTES: int = 60
    DB_PATH: str = "./app.db"
    HTTP_UA_MAIN: str = "lpbot/0.1 (+https://github.com/dave/lpbot)"
    HTTP_UA_VOL: str = "lpbot-volatility/0.1 (+https://github.com/dave/lpbot)"

    # Optional band seeds (low-high). If provided, they will be upserted at startup.
    BAND_A: Optional[str] = None
    BAND_B: Optional[str] = None
    BAND_C: Optional[str] = None

    BINANCE_BASE_URL: str = "https://api.binance.com"
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
    def _validate(self) -> "Settings":
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
        return self


settings = Settings()
local_tz = ZoneInfo(settings.LOCAL_TZ)

SLOT_SECONDS = max(60, settings.CHECK_EVERY_MINUTES * 60)

last_run_ts_check: Optional[int] = None
last_run_ts_daily: Optional[int] = None
next_run_ts_check: Optional[int] = None
next_run_ts_daily: Optional[int] = None


def floor_to_slot(ts: int) -> int:
    """Snap a unix timestamp down to the current scheduling slot boundary."""
    return (ts // SLOT_SECONDS) * SLOT_SECONDS


# ----------------------------
# Globals
# ----------------------------

http_client: Optional[httpx.AsyncClient] = None
db_conn: Optional[aiosqlite.Connection] = None
repo: Optional[DBRepo] = None
tg: Optional[TelegramSvc] = None
scheduler: Optional[AsyncIOScheduler] = None


# ----------------------------
# Logging
# ----------------------------

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


base_log = configure_logging()
log = base_log.bind(module="main")


# ----------------------------
# HTTP helpers
# ----------------------------

DEFAULT_TIMEOUT = httpx.Timeout(7.5, read=7.5, write=7.5, connect=5.0, pool=5.0)

async def fetch_json_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    attempts: int = 3,
    base_backoff: float = 0.5,
) -> Optional[Any]:
    merged_headers = {"user-agent": settings.HTTP_UA_MAIN, **(headers or {})}
    try:
        response = await net.request_with_retries(
            client,
            method,
            url,
            headers=merged_headers,
            params=params,
            attempts=attempts,
            base_backoff=base_backoff,
            timeout=DEFAULT_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        log.error("http_failed", url=url, err=str(exc))
        return None

    try:
        return response.json()
    except ValueError:
        log.warning("json_parse_failed", url=url)
        return response.content


# ----------------------------
# Price fetchers + decision
# ----------------------------

async def fetch_meteora_price(client: httpx.AsyncClient, pair_address: str) -> Optional[float]:
    base_url = settings.METEORA_BASE_URL.rstrip("/")
    url = f"{base_url}/pair/{pair_address}"
    data = await fetch_json_with_retries(client, "GET", url)
    if data is None:
        return None

    try:
        raw = data["current_price"]
        price = float(raw)
    except Exception:
        return None

    if not math.isfinite(price) or price <= 0:
        return None

    return price


async def decide_price(client: httpx.AsyncClient) -> Optional[float]:
    price = await fetch_meteora_price(client, settings.METEORA_PAIR_ADDRESS)
    if price is None:
        log.error("price_unavailable", source="meteora")
        return None

    log.info("price_ok", source="meteora", price=price)
    return price


async def process_breaches(
    price: float,
    src_label: str,
    bucket: Optional[str] = None,
    *,
    now_slot_ts: Optional[int] = None,
) -> None:
    assert repo is not None, "db repo not initialized"
    assert tg is not None, "telegram service not initialized"

    base_ts = int(time.time()) if now_slot_ts is None else int(now_slot_ts)
    now_aligned = floor_to_slot(base_ts)
    bands = await repo.get_bands()
    broken = broken_bands(price, bands)
    cooldown_secs = settings.COOLDOWN_MINUTES * 60
    sent = 0

    effective_bucket = bucket or "mid"
    warned = False

    suggested_map = band_advisor.ranges_for_price(
        price,
        effective_bucket,
        include_a_on_high=False,
    )
    widths, _ = band_advisor.widths_for_bucket(effective_bucket)
    for name, (lo, hi) in bands.items():
        if name not in broken:
            continue

        side = "below" if price < lo else "above"

        # Persisted rows pre-date slot quantization, so floor before comparing.
        last = await repo.get_last_alert(name, side)
        last_aligned = floor_to_slot(last) if last is not None else None
        if last_aligned is not None:
            delta = now_aligned - last_aligned
            if delta < cooldown_secs:
                seconds_remaining = cooldown_secs - delta
                log.info(
                    "cooldown_skip",
                    band=name,
                    side=side,
                    seconds_remaining=seconds_remaining,
                    now_aligned=now_aligned,
                    last_aligned=last_aligned,
                )
                continue

        if bucket is None and not warned:
            log.warning("bucket_missing", band=name, side=side, fallback="mid")
            warned = True

        rng = suggested_map.get(name)
        if rng is None:
            continue

        log.info(
            "breach_offer",
            band=name,
            side=side,
            price=price,
            bucket=effective_bucket,
            suggested_lo=rng[0],
            suggested_hi=rng[1],
        )
        width = widths.get(name)
        await tg.send_breach_offer(
            band=name,
            price=price,
            src_label=src_label,
            bands=bands,
            suggested_range=rng,
            policy_meta=(effective_bucket, width) if width is not None else None,
        )
        await repo.set_last_alert(name, side, now_aligned)
        sent += 1

    if sent == 0:
        log.info("no_breach", price=price, source=src_label)


async def get_sigma_reading() -> Optional[Dict[str, Any]]:
    if http_client is None:
        return None

    reading = await vol.fetch_sigma_1h(
        http_client,
        base_url=settings.BINANCE_BASE_URL,
        symbol=settings.BINANCE_SYMBOL,
        cache_ttl=max(5, settings.VOL_CACHE_TTL_SECONDS),
        max_stale=settings.VOL_MAX_STALE_SECONDS,
        user_agent=settings.HTTP_UA_VOL,
    )
    if reading is None:
        return None

    return {
        "sigma_pct": reading.sigma_pct,
        "bucket": reading.bucket,
        "window_min": reading.window_minutes,
        "as_of_ts": reading.as_of_ts,
        "sample_count": reading.sample_count,
        "stale": reading.stale,
    }


# ----------------------------
# Telegram
# ----------------------------


# ----------------------------
# Scheduler job
# ----------------------------

async def check_once() -> None:
    global last_run_ts_check, next_run_ts_check
    assert http_client is not None, "http client not initialized"
    assert repo is not None, "db repo not initialized"
    assert tg is not None, "telegram service not initialized"

    started_ts = int(time.time())
    last_run_ts_check = started_ts

    try:
        price = await decide_price(http_client)
        if price is None:
            return
        if not math.isfinite(price) or price <= 0:
            log.warning("advisory_price_invalid", price=price)
            return
        sigma = await get_sigma_reading()
        val = sigma.get("sigma_pct") if sigma else None
        sigma_pct_log: Optional[float]
        if isinstance(val, (int, float)):
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                sigma_pct_log = None
            else:
                sigma_pct_log = round(val_float, 3) if math.isfinite(val_float) else None
        else:
            sigma_pct_log = None

        bucket = sigma.get("bucket") if sigma else None
        stale = bool(sigma.get("stale")) if sigma else None

        sample_count = None
        if sigma and sigma.get("sample_count") is not None:
            try:
                sample_count = int(sigma.get("sample_count"))
            except (TypeError, ValueError):
                sample_count = None

        as_of_ts = None
        if sigma and sigma.get("as_of_ts") is not None:
            try:
                as_of_ts = int(sigma.get("as_of_ts"))
            except (TypeError, ValueError):
                as_of_ts = None

        log_kwargs = {
            "sigma_pct": sigma_pct_log,
            "bucket": bucket,
            "stale": stale,
            "sample_count": sample_count,
            "as_of_ts": as_of_ts,
        }
        log.info("sigma_ok" if sigma else "sigma_miss", **log_kwargs)
        bucket = sigma["bucket"] if sigma else None
        current_slot_ts = floor_to_slot(int(time.time()))
        await process_breaches(
            price,
            "meteora",
            bucket=bucket,
            now_slot_ts=current_slot_ts,
        )
    except Exception as e:
        log.error("job_exception", err=str(e))
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


async def send_daily_advisory() -> None:
    global last_run_ts_daily, next_run_ts_daily
    assert http_client is not None, "http client not initialized"
    assert tg is not None, "telegram service not initialized"
    assert repo is not None, "db repo not initialized"

    if not settings.DAILY_ENABLED:
        log.info("advisory_skip_disabled")
        return

    started_ts = int(time.time())
    last_run_ts_daily = started_ts

    try:
        price = await decide_price(http_client)
        if price is None:
            return
        if not math.isfinite(price) or price <= 0:
            log.warning("advisory_price_invalid", price=price)
            return

        sigma = await get_sigma_reading()
        if sigma is None:
            log.warning("sigma_miss_daily")
        bucket = (sigma.get("bucket") if sigma else None) or "mid"
        sigma_pct_raw = sigma.get("sigma_pct") if sigma else None

        sigma_pct: Optional[float]
        if sigma_pct_raw is None:
            sigma_pct = None
        else:
            try:
                sigma_pct = float(sigma_pct_raw)
            except (TypeError, ValueError):
                sigma_pct = None

        baseline = await repo.get_baseline()
        latest = await repo.get_latest_snapshot()

        advisory = band_advisor.build_advisory(
            price,
            sigma_pct,
            bucket,
            include_a_on_high=settings.INCLUDE_A_ON_HIGH,
        )
        advisory["stale"] = bool(sigma.get("stale")) if sigma else False

        drift_line: Optional[str] = None
        if baseline and latest:
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
                    drift_line = (
                        f"<b>Drift</b>: ${drift_now:+.2f} ({drift_pct * 100:+.2f}%)"
                    )

        await tg.send_advisory_card(advisory, drift_line=drift_line)
        log.info(
            "advisory_sent",
            price=price,
            bucket=bucket,
            sigma_pct=sigma_pct,
            stale=advisory["stale"],
        )
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


# ----------------------------
# FastAPI app + lifespan
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_conn, repo, tg, scheduler

    # open db connection and init
    db_conn = await aiosqlite.connect(settings.DB_PATH)
    repo = DBRepo(db_conn, base_log.bind(module="db"))
    await repo.init(settings)

    # one httpx client (http/2), shared
    http_client = httpx.AsyncClient(http2=True, timeout=DEFAULT_TIMEOUT)

    tg = TelegramSvc(
        settings.TELEGRAM_BOT_TOKEN,
        settings.TELEGRAM_CHAT_ID,
        logger=base_log.bind(module="telegram"),
    )
    await tg.start(repo)

    band_seeds = {
        name: value
        for name, value in {
            "a": settings.BAND_A,
            "b": settings.BAND_B,
            "c": settings.BAND_C,
        }.items()
        if value is not None
    }
    interval_minutes = max(1, settings.CHECK_EVERY_MINUTES)
    slot_seconds = max(60, interval_minutes * 60)

    global SLOT_SECONDS
    SLOT_SECONDS = slot_seconds
    log.info("cooldown_quantization", slot_seconds=SLOT_SECONDS)

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

    async def _price_provider() -> Optional[float]:
        if http_client is None:
            return None
        return await decide_price(http_client)

    tg.set_price_provider(_price_provider)
    tg.set_sigma_provider(get_sigma_reading)

    # scheduler with explicit timezone and sane semantics
    scheduler = AsyncIOScheduler(timezone=local_tz)
    check_job_id = "check-once"
    if 60 % interval_minutes == 0:
        minutes_str = ",".join(
            str(minute) for minute in range(0, 60, interval_minutes)
        )
        scheduler.add_job(
            check_once,
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
            check_once,
            IntervalTrigger(seconds=interval_seconds, timezone=local_tz),
            id=check_job_id,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=120,
        )
        check_trigger_desc = f"interval:{interval_seconds}s"
    if settings.DAILY_ENABLED:
        scheduler.add_job(
            send_daily_advisory,
            "cron",
            id="daily-advisory",
            hour=settings.DAILY_LOCAL_HOUR,
            minute=settings.DAILY_LOCAL_MINUTE,
            timezone=local_tz,
            coalesce=True,
            second=0,
            misfire_grace_time=1800,
            max_instances=1,
        )
    scheduler.start()
    check_job = scheduler.get_job(check_job_id)
    daily_job = scheduler.get_job("daily-advisory") if settings.DAILY_ENABLED else None
    global next_run_ts_check, next_run_ts_daily
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
        log.info("app_stop")


app = FastAPI(lifespan=lifespan)


@app.get("/sigma")
async def sigma() -> Dict[str, Any]:
    data = await get_sigma_reading()
    return {"ok": data is not None, "data": data}


@app.get("/healthz")
async def healthz() -> Dict[str, bool]:
    return {"ok": True}


# ----------------------------
# Local run helper
# ----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
