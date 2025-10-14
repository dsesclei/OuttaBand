default: check

actionlint:
    uv run actionlint -color

hadolint:
    docker run --rm -v $(pwd):/work -w /work hadolint/hadolint:latest-debian hadolint Dockerfile

fmt:
    uv run ruff format .

lint:
    uv run ruff check .

type:
    uv run mypy .

test:
    uv run pytest -q

check: fmt lint type test actionlint hadolint

run:
    uv run uvicorn main:app --reload

build:
    docker buildx build -t lpbot:dev .
