# main.py
# mvp: sol/usdc band watcher with telegram alerts
# stack: fastapi, apscheduler (asyncio), python-telegram-bot (async), httpx, aiosqlite, pydantic v2

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict
from telegram import Bot


# ----------------------------
# Settings
# ----------------------------

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int

    CMC_API_KEY: Optional[str] = None
    CG_API_KEY: Optional[str] = None

    CHECK_EVERY_MINUTES: int = 15
    COOLDOWN_MINUTES: int = 60
    SPREAD_MAX_PCT: float = 0.5

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
scheduler: Optional[AsyncIOScheduler] = None


# ----------------------------
# Logging (structured json)
# ----------------------------

def jlog(level: str, event: str, **kwargs: Any) -> None:
    payload = {"ts": int(time.time()), "level": level, "event": event}
    if kwargs:
        payload.update(kwargs)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


# ----------------------------
# DB
# ----------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS bands (
    name TEXT PRIMARY KEY,
    low REAL NOT NULL,
    high REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    band TEXT,
    side TEXT,
    last_sent_ts INTEGER,
    PRIMARY KEY (band, side)
);
"""

async def init_db(conn: aiosqlite.Connection) -> None:
    await conn.executescript(CREATE_TABLES_SQL)
    await conn.commit()
    await seed_bands_if_missing(conn)
    await upsert_bands_from_env(conn, settings)

async def seed_bands_if_missing(conn: aiosqlite.Connection) -> None:
    defaults: List[Tuple[str, float, float]] = [("a", 0.0, 100.0), ("b", 0.0, 100.0), ("c", 0.0, 100.0)]
    for name, low, high in defaults:
        await conn.execute(
            "INSERT OR IGNORE INTO bands(name, low, high) VALUES(?,?,?)",
            (name, low, high),
        )
    await conn.commit()

def _parse_band_spec(spec: str) -> Optional[Tuple[float, float]]:
    try:
        lo_str, hi_str = spec.strip().split("-", 1)
        lo, hi = float(lo_str), float(hi_str)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        if lo >= hi:
            return None
        return (lo, hi)
    except Exception:
        return None

async def upsert_bands_from_env(conn: aiosqlite.Connection, settings: Settings) -> None:
    for name, spec in (("a", settings.BAND_A), ("b", settings.BAND_B), ("c", settings.BAND_C)):
        if not spec:
            continue
        parsed = _parse_band_spec(spec)
        if parsed is None:
            jlog("warn", "band_env_invalid", band=name, spec=spec)
            continue
        lo, hi = parsed
        await conn.execute(
            "INSERT INTO bands(name, low, high) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET low=excluded.low, high=excluded.high",
            (name, lo, hi),
        )
        jlog("info", "band_env_upserted", band=name, low=lo, high=hi)
    await conn.commit()

async def get_bands(conn: aiosqlite.Connection) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    async with conn.execute("SELECT name, low, high FROM bands ORDER BY name ASC") as cur:
        async for name, low, high in cur:
            out[name] = (float(low), float(high))
    return out

async def get_last_alert(conn: aiosqlite.Connection, band: str, side: str) -> Optional[int]:
    async with conn.execute(
        "SELECT last_sent_ts FROM alerts WHERE band=? AND side=?", (band, side)
    ) as cur:
        row = await cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

async def set_last_alert(conn: aiosqlite.Connection, band: str, side: str, ts: int) -> None:
    await conn.execute(
        "INSERT INTO alerts(band, side, last_sent_ts) VALUES(?,?,?) "
        "ON CONFLICT(band, side) DO UPDATE SET last_sent_ts=excluded.last_sent_ts",
        (band, side, ts),
    )
    await conn.commit()


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
                jlog("error", "http_failed", url=url, err=str(e))
                return None
            await asyncio.sleep(wait)
    return None


# ----------------------------
# Price fetchers + decision
# ----------------------------

async def fetch_cg_price_usd(client: httpx.AsyncClient, api_key: Optional[str]) -> Optional[float]:
    if not api_key:
        return None
    url = "https://api.coingecko.com/api/v3/simple/price"
    headers: Dict[str, str] = {"x-cg-demo-api-key": api_key}
    params = {"ids": "solana", "vs_currencies": "usd"}
    data = await fetch_json_with_retries(client, "GET", url, headers=headers, params=params)
    try:
        val = float(data["solana"]["usd"])
        return val if math.isfinite(val) and val > 0 else None
    except Exception:
        return None

async def fetch_cmc_price_usd(client: httpx.AsyncClient, api_key: Optional[str]) -> Optional[float]:
    if not api_key:
        return None
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}
    params = {"symbol": "SOL", "convert": "USD"}  # returns data["SOL"]["quote"]["USD"]["price"]
    data = await fetch_json_with_retries(client, "GET", url, headers=headers, params=params)
    try:
        val = float(data["data"]["SOL"]["quote"]["USD"]["price"])
        return val if math.isfinite(val) and val > 0 else None
    except Exception:
        return None

def pct_spread(a: float, b: float) -> float:
    mid = (a + b) / 2.0
    if mid == 0:
        return float("inf")
    return abs(a - b) / mid * 100.0

@dataclass
class PriceDecision:
    price: Optional[float]
    label: Optional[str]
    cg: Optional[float]
    cmc: Optional[float]
    suspect: bool  # true means both sources present but divergence > threshold

async def decide_price(client: httpx.AsyncClient, spread_max_pct: float) -> PriceDecision:
    cg_task = asyncio.create_task(fetch_cg_price_usd(client, settings.CG_API_KEY))
    cmc_task = asyncio.create_task(fetch_cmc_price_usd(client, settings.CMC_API_KEY))
    cg, cmc = await asyncio.gather(cg_task, cmc_task)

    if cg is not None and cmc is not None:
        spread = pct_spread(cg, cmc)
        if spread > spread_max_pct:
            jlog("warn", "divergence_guard", cg=cg, cmc=cmc, spread_pct=round(spread, 4), max_pct=spread_max_pct)
            return PriceDecision(None, None, cg, cmc, True)
        price = (cg + cmc) / 2.0
        jlog("info", "price_ok_mid", cg=cg, cmc=cmc, spread_pct=round(spread, 4), price=price)
        return PriceDecision(price, None, cg, cmc, False)
    elif cg is not None:
        jlog("info", "price_ok_cg_only", cg=cg)
        return PriceDecision(cg, "cg only", cg, None, False)
    elif cmc is not None:
        jlog("info", "price_ok_cmc_only", cmc=cmc)
        return PriceDecision(cmc, "cmc only", None, cmc, False)
    else:
        jlog("error", "price_unavailable")
        return PriceDecision(None, None, None, None, False)


# ----------------------------
# Band logic
# ----------------------------

def fmt_price(x: float) -> str:
    return f"{x:.2f}"

def fmt_range(lo: float, hi: float) -> str:
    return f"{fmt_price(lo)}–{fmt_price(hi)}"  # en dash

def format_message(
    band_name: str,
    side: str,
    price: float,
    source_label: Optional[str],
    bands: Dict[str, Tuple[float, float]],
) -> str:
    lines: List[str] = []
    lines.append("◧ hard break")
    lines.append(f"band: {band_name.upper()} ({side})")

    price_line = f"price: {fmt_price(price)}"
    if source_label:  # only show on single-source fallback
        price_line += f" ({source_label})"
    lines.append(price_line)

    for name in sorted(bands.keys()):
        lo, hi = bands[name]
        lines.append(f"{name}: {fmt_range(lo, hi)}")
    return "\n".join(lines)

async def process_breaches(conn: aiosqlite.Connection, price: float, src_label: Optional[str]) -> None:
    now_ts = int(time.time())
    bands = await get_bands(conn)
    cooldown_secs = settings.COOLDOWN_MINUTES * 60
    sent = 0

    for name, (lo, hi) in bands.items():
        side: Optional[str] = None
        if price < lo:
            side = "below"
        elif price > hi:
            side = "above"
        if side is None:
            continue

        last = await get_last_alert(conn, name, side)
        if last is not None and (now_ts - last) < cooldown_secs:
            jlog("info", "cooldown_skip", band=name, side=side, seconds_remaining=cooldown_secs - (now_ts - last))
            continue

        text = format_message(name, side, price, src_label, bands)
        await send_telegram(text)
        await set_last_alert(conn, name, side, now_ts)
        sent += 1

    if sent == 0:
        jlog("info", "no_breach", price=price, source=src_label)


# ----------------------------
# Telegram
# ----------------------------

async def send_telegram(text: str) -> None:
    try:
        async with Bot(settings.TELEGRAM_BOT_TOKEN) as bot:
            await bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=text)
        jlog("info", "telegram_sent", bytes=len(text))
    except Exception as e:
        jlog("error", "telegram_failed", err=str(e))


# ----------------------------
# Scheduler job
# ----------------------------

async def check_once() -> None:
    assert http_client is not None, "http client not initialized"
    assert db_conn is not None, "db connection not initialized"

    try:
        decision = await decide_price(http_client, settings.SPREAD_MAX_PCT)
        if decision.suspect or decision.price is None:
            return  # already logged
        await process_breaches(db_conn, decision.price, decision.label)
    except Exception as e:
        jlog("error", "job_exception", err=str(e))


# ----------------------------
# FastAPI app + lifespan
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_conn, scheduler

    # open db connection and init
    db_conn = await aiosqlite.connect(DB_PATH)
    await init_db(db_conn)

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
    jlog("info", "app_start", interval_min=settings.CHECK_EVERY_MINUTES)

    try:
        yield
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)
            jlog("info", "scheduler_stopped")
        if http_client:
            await http_client.aclose()
            jlog("info", "http_client_closed")
        if db_conn:
            await db_conn.close()
            jlog("info", "db_closed")
        jlog("info", "app_stop")


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
