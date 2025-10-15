default: check

actionlint:
    uv run actionlint -color

hadolint:
    docker run --rm -v $(pwd):/work -w /work hadolint/hadolint:latest-debian \
        hadolint --failure-threshold error Dockerfile

fmt:
    uv run ruff format .

lint:
    uv run ruff check . --fix

type:
    uv run mypy .

test:
    uv run pytest -q

check: fmt lint type test actionlint hadolint

run:
    uv run uvicorn outtaband.main:app --reload

build:
    docker buildx build -t outtaband:local .

smoke:
    @set -eu -o pipefail; \
    name=outtaband-smoke; image=outtaband:local; \
    docker rm -f "$name" >/dev/null 2>&1 || true; \
    cid=$(docker run -d --name "$name" -p 8000:8000 "$image"); \
    for i in $(seq 1 30); do \
      code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/healthz || true); \
      if [ "$code" = "200" ]; then \
        echo "✓ smoke passed — server responded 200 on /healthz"; \
        docker rm -f "$name" >/dev/null; exit 0; \
      fi; \
      sleep 1; \
    done; \
    echo "✗ smoke failed — /healthz never returned 200"; \
    docker logs "$name" || true; \
    docker rm -f "$name" >/dev/null || true; exit 1

ci: check build smoke

# ------- Compose -------

compose_project := "outtaband"
compose_file   := "docker-compose.yml"

dc *args:
    @docker compose -p {{compose_project}} -f {{compose_file}} {{args}}

up *services:
    @docker compose -p {{compose_project}} -f {{compose_file}} up -d --remove-orphans {{services}}

down:
    @docker compose -p {{compose_project}} -f {{compose_file}} down --remove-orphans

logs *services:
    @docker compose -p {{compose_project}} -f {{compose_file}} logs -f {{services}}

ps:
    @docker compose -p {{compose_project}} -f {{compose_file}} ps
    
reload service:
    @docker compose -p {{compose_project}} -f {{compose_file}} up -d --no-deps --build {{service}}
