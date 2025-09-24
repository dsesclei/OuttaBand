# main.py
# mvp: sol/usdc band watcher with telegram alerts
# stack: fastapi, apscheduler (asyncio), python-telegram-bot (async), httpx, aiosqlite, pydantic v2

from __future__ import annotations

import asyncio
import logging
import logging.config
import math
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import timezone
from typing import Any, Dict, Optional

import aiosqlite
import httpx
import structlog
from structlog.typing import FilteringBoundLogger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from db_repo import dbrepo
from band_logic import broken_bands, suggest_new_bands
from telegram_service import telegramsvc


# ----------------------------
# Settings
# ----------------------------

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int

    METEORA_PAIR_ADDRESS: str

    CHECK_EVERY_MINUTES: int = 15
    COOLDOWN_MINUTES: int = 60

    # Optional band seeds (low-high). If provided, they will be upserted at startup.
    BAND_A: Optional[str] = None
    BAND_B: Optional[str] = None
    BAND_C: Optional[str] = None

    # pydantic v2 config: .env file, case-insensitive for ops sanity
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()


# ----------------------------
# Globals
# ----------------------------

DB_PATH = "./app.db"

http_client: Optional[httpx.AsyncClient] = None
db_conn: Optional[aiosqlite.Connection] = None
repo: Optional[dbrepo] = None
tg: Optional[telegramsvc] = None
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
UA = "sol-band-watch/0.1 (+github.com/you) httpx"

async def fetch_json_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    attempts: int = 3,
    base_backoff: float = 0.5,
) -> Optional[Dict[str, Any]]:
    for i in range(attempts):
        try:
            r = await client.request(
                method,
                url,
                headers={"user-agent": UA, **(headers or {})},
                params=params,
                timeout=DEFAULT_TIMEOUT,
            )
            # treat 429/5xx as retryable; others raise
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            # honor Retry-After if present on 429/5xx; else exponential backoff with jitter
            retry_after = None
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                ra = e.response.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        retry_after = None
            wait = (retry_after if retry_after is not None else base_backoff * (2 ** i)) + random.uniform(0, 0.2)
            if i == attempts - 1:
                log.error("http_failed", url=url, err=str(e))
                return None
            await asyncio.sleep(wait)
    return None


# ----------------------------
# Price fetchers + decision
# ----------------------------

async def fetch_meteora_price(client: httpx.AsyncClient, pair_address: str) -> Optional[float]:
    url = f"https://dlmm-api.meteora.ag/pair/{pair_address}"
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


async def process_breaches(price: float, src_label: str) -> None:
    assert repo is not None, "db repo not initialized"
    assert tg is not None, "telegram service not initialized"

    now_ts = int(time.time())
    bands = await repo.get_bands()
    broken = broken_bands(price, bands)
    cooldown_secs = settings.COOLDOWN_MINUTES * 60
    sent = 0

    for name, (lo, hi) in bands.items():
        if name not in broken:
            continue

        side = "below" if price < lo else "above"

        last = await repo.get_last_alert(name, side)
        if last is not None and (now_ts - last) < cooldown_secs:
            log.info(
                "cooldown_skip",
                band=name,
                side=side,
                seconds_remaining=cooldown_secs - (now_ts - last),
            )
            continue

        suggested = suggest_new_bands(price, bands, {name})
        rng = suggested.get(name)
        if rng is None:
            rng = bands.get(name)
        if rng is None:
            rng = (price, price)
        await tg.send_breach_offer(
            band=name,
            price=price,
            src_label=src_label,
            bands=bands,
            suggested_range=rng,
        )
        await repo.set_last_alert(name, side, now_ts)
        sent += 1

    if sent == 0:
        log.info("no_breach", price=price, source=src_label)


# ----------------------------
# Telegram
# ----------------------------


# ----------------------------
# Scheduler job
# ----------------------------

async def check_once() -> None:
    assert http_client is not None, "http client not initialized"
    assert repo is not None, "db repo not initialized"
    assert tg is not None, "telegram service not initialized"

    try:
        price = await decide_price(http_client)
        if price is None:
            return
        await process_breaches(price, "meteora")
    except Exception as e:
        log.error("job_exception", err=str(e))


# ----------------------------
# FastAPI app + lifespan
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_conn, repo, tg, scheduler

    # open db connection and init
    db_conn = await aiosqlite.connect(DB_PATH)
    repo = dbrepo(db_conn, base_log.bind(module="db"))
    await repo.init(settings)

    tg = telegramsvc(
        settings.TELEGRAM_BOT_TOKEN,
        settings.TELEGRAM_CHAT_ID,
        logger=base_log.bind(module="telegram"),
    )
    await tg.start(repo)

    # one httpx client (http/2), shared
    http_client = httpx.AsyncClient(http2=True, timeout=DEFAULT_TIMEOUT)

    # scheduler with explicit timezone and sane semantics
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.add_job(
        check_once,
        "interval",
        minutes=max(1, settings.CHECK_EVERY_MINUTES),
        jitter=30,            # up to +30s
        coalesce=True,        # collapse missed runs to one
        max_instances=1,      # never overlap
        misfire_grace_time=60 # small grace
    )
    scheduler.start()
    log.info("app_start", interval_min=settings.CHECK_EVERY_MINUTES)

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

@app.get("/healthz")
async def healthz() -> Dict[str, bool]:
    return {"ok": True}


# ----------------------------
# Local run helper
# ----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
