# lpbot — SOL/USDC band watcher

An async FastAPI service that tracks the Meteora SOL/USDC DLMM price, computes realized volatility, and coordinates Telegram alerts when configured bands are breached. It keeps state in SQLite, quantizes alert cooldowns to scheduling slots, and exposes lightweight operational endpoints.

## Quickstart

### Local

Python 3.11 is the target runtime. A minimal run looks like:

```bash
python3.11 -m pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

Creating a virtual environment first (`python3.11 -m venv .venv && source .venv/bin/activate`) keeps dependencies isolated. The `.env` file will contain real Telegram tokens—it's already gitignored, but double-check before committing or screen sharing.

### Docker

```bash
docker build -t lpbot:local .
mkdir -p data
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/data" \
  -p 8000:8000 \
  lpbot:local
```

The image runs as a non-root user and persists its SQLite database under `/data`. Mounting `$(pwd)/data` keeps `app.db` between runs. The built-in health check polls `http://127.0.0.1:8000/healthz`; wait for that to go green before trusting alerts.

### Docker Compose

```bash
docker compose up --build
```

Compose uses the same Dockerfile and automatically loads `.env` via `env_file`. A `./data` volume is mounted into `/data`, so `data/app.db` survives restarts. Stream logs with `docker compose logs -f lpbot` and verify `/healthz` returns `ok=true` before relying on Telegram alerts.

## Continuous integration

GitHub Actions runs pytest on Python 3.11 and builds the Docker image on every push and PR targeting `main`.

## Configuration

All settings are provided via environment variables (pydantic-settings, case-insensitive):

| Variable | Default | Notes |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | — | Required bot token. |
| `TELEGRAM_CHAT_ID` | — | Target chat/channel id. |
| `METEORA_PAIR_ADDRESS` | — | Meteora pool address (required). |
| `CHECK_EVERY_MINUTES` | `15` | Scheduler cadence. If the value divides 60, APScheduler uses a cron trigger aligned to minute boundaries; otherwise an `IntervalTrigger` in seconds preserves the exact cadence. |
| `COOLDOWN_MINUTES` | `60` | Cooldown gate between alerts per band/side pair. Converted to `SLOT_SECONDS = max(60, CHECK_EVERY_MINUTES * 60)` so persisted alerts snap to slot boundaries. |
| `DB_PATH` | `./app.db` | SQLite file location; change for persistent volumes. |
| `HTTP_UA_MAIN` | `lpbot/0.1 (+https://github.com/dave/lpbot)` | User-Agent header for general HTTP calls (Meteora). |
| `HTTP_UA_VOL` | `lpbot-volatility/0.1 (+https://github.com/dave/lpbot)` | User-Agent for Binance volatility fetches. |
| `BINANCE_BASE_URL` | `https://api.binance.com` | Override if using Binance US or a mock. |
| `VOL_CACHE_TTL_SECONDS` | `60` | Freshness window before refetching sigma. |
| `VOL_MAX_STALE_SECONDS` | `7200` | Max age to serve cached sigma when live fetches fail; beyond this, sigma is treated as unavailable. |
| `LOCAL_TZ` | `America/New_York` | Scheduler timezone. Controls cron alignment for the check and daily advisory jobs. |
| `DAILY_*` vars | `8:00` / `True` | Daily advisory timing. |
| `INCLUDE_A_ON_HIGH` | `False` | Whether policy suggestions include band A in high-vol buckets. |

Optional band seed variables `BAND_A`, `BAND_B`, `BAND_C` accept `"low-high"` strings and seed the database at boot.

The range notation in this README uses an en dash (–); rendering assumes UTF-8.

## Volatility buckets & staleness

`volatility.fetch_sigma_1h` computes σ from 60 Binance 1m closes:

- `< 0.6%` → `low`
- `0.6% – 1.2%` (inclusive) → `mid`
- `> 1.2%` → `high`

Buckets map to band widths of roughly ±0.35%/±1.1%/±1.8% (`low`), ±0.6%/±1.8%/±3.0% (`mid`), and ±0.9%/±2.5%/±4.0% (`high`). Advisory splits stay 60/20/20 in `low` volatility and tighten to 50/30/20 for `mid` and `high`. When the bucket is `high`, band `a` is suppressed unless `INCLUDE_A_ON_HIGH=true`, so alerts and advisories line up with the more defensive policy.

Readings are cached by `(base_url, symbol)` for `VOL_CACHE_TTL_SECONDS`. When live fetches fail, cached data younger than `VOL_MAX_STALE_SECONDS` is returned with `stale=True`; older entries result in `None`, causing Sigma-dependent decisions to fall back to defaults.

## Operational notes

- **Scheduler**: APScheduler runs inside the app lifespan. When `CHECK_EVERY_MINUTES` divides 60 the job fires at deterministic minute offsets (`0, interval, …`). Otherwise an interval trigger steps in `interval_minutes * 60` seconds to avoid silently “rounding” to 15 minutes. All timestamps are stored in UTC seconds.
- **Cooldown slots**: Alert timestamps are floored to `SLOT_SECONDS = max(60, CHECK_EVERY_MINUTES * 60)` before comparisons. Cooldown enforcement is per `(band, side)` and stored in the `alerts` table.
- **Alert throttling**: If an alert fired within the last `COOLDOWN_MINUTES`, the service logs `cooldown_skip` including the aligned now/last slots. Suggestions draw from the active policy bucket and skip band “a” during high volatility unless explicitly configured to include it.
- **Health check**: `GET /healthz` returns a JSON struct detailing `ok`, `db_ok`, `http_client_ok`, `telegram_ready`, `scheduler_ok`, scheduler run timestamps, and `volatility_cache_age_s`. Database health is probed via a lightweight `SELECT 1`.
- **Metrics**: A `/metrics` endpoint is planned; until then rely on structured logs (sigma, advisory, and Telegram send events) plus `/healthz` for readiness.

## Telegram bot commands

The Telegram bot only responds to the configured chat and exposes:

- `/status` — latest price, σ, bucket, and policy split.
- `/bands` — view or edit stored band ranges.
- `/setbaseline <sol> sol <usdc> usdc` — record baseline balances for tilt calculations.
- `/setnotional <usd>` — set the notional USD budget for allocations.
- `/settilt <sol>:<usdc>` — adjust the in-band SOL/USDC mix (e.g., `60:40`).
- `/updatebalances <sol> sol <usdc> usdc` — refresh balances without replacing the baseline.
- `/help` — show the command list plus policy caveats.

## Testing

A lightweight test suite focuses on math and data edges (`volatility`, `band_advisor`, `db_repo`). Run with:

```bash
pytest
```
