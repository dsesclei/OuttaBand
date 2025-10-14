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
    uv run uvicorn main:app --reload

build:
    docker buildx build -t lpbot:local .

smoke:
    docker run -d --rm --name lpbot -p 8000:8000 lpbot:dev
    @for i in {1..30}; do \
      code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/healthz || true); \
      [ "$$code" = "200" ] && docker stop lpbot && exit 0; \
      sleep 1; \
    done; \
    echo "healthz never returned 200"; docker logs lpbot; docker stop lpbot; exit 1

ci: check build smoke
