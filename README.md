# OuttaBand

Telegram bot built for a friend to monitor Meteora liquidity‑providing ranges.  
It watches configured bands, reports when positions drift out of range, and offers suggestions based on a policy (here, live Binance data) for rebalancing.

I later turned it into an exercise in productionizing a small async service using best practices, with LLM assistance.

#### Development
Requires `just`, `uv`, and Python 3.11+.

```bash
cp .env.example .env
uv sync --dev
just check      # run format/lint/type/test/actionlint/hadolint
just run        # start dev server (http://127.0.0.1:8000)
```

#### Docker / Compose

```bash
just build
just up
```
