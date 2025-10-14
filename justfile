default: check

fmt:
    uv run ruff format .

lint:
    uv run ruff check .

type:
    uv run mypy .

test:
    uv run pytest -q

check: fmt lint type test

run:
    uv run uvicorn main:app --reload

build:
    docker buildx build -t lpbot:dev .
