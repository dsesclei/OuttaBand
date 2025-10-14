# Outtaband

A small asynchronous service that watches SOL/USDC ranges and sends Telegram advisories.  
It’s built for reliability: structured logs, jittered retries, SQLite + WAL, and deterministic CI/CD.

## Development

- `uv sync`  
- `uv run pre-commit install`  
- `uv run uvicorn main:app --reload`

## Quality

- Ruff (lint + format), Mypy (strict), Pytest (unit + property).  
- `uv run ruff check . ; uv run mypy . ; uv run pytest -q`

## Operations

- `/healthz` for liveness  
- `/metrics` for Prometheus  
- Docker:  
  `docker buildx build -t ghcr.io/dsesclei/outtaband:dev .`

## CI/CD (GitHub Actions)

- **Pull Requests:** lint → type → test → docker build → SBOM + scan → smoke.  
- **Tags:** push image (GHCR) with OCI labels and attach SBOM.

## Notes

- Singleton job lock prevents duplicate sends when scaling out.  
- Telegram can be disabled (`TELEGRAM_ENABLED=false`) to run in CI or headless mode.
