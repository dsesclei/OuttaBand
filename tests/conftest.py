from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterator

import httpx
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Provide an asyncio event loop for pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture()
def fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Populate environment variables so tests stay offline and isolated."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    monkeypatch.setenv("METEORA_PAIR_ADDRESS", "dummy_pair")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("BINANCE_BASE_URL", "http://example.invalid")


def make_mock_transport(*responses: httpx.Response | Exception) -> httpx.MockTransport:
    """Build a transport that replays canned responses or raises errors."""
    queue: Deque[httpx.Response | Exception] = deque(responses)

    def handler(_: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError("Unexpected request with no queued response")
        next_item = queue.popleft()
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    return httpx.MockTransport(handler)


def json_response(*, status: int = 200, json: dict[str, Any] | None = None) -> httpx.Response:
    """Convenience helper to build JSON responses for MockTransport."""
    return httpx.Response(status_code=status, json=json or {})


def now_time(monkeypatch: pytest.MonkeyPatch, t: float) -> None:
    """Freeze time.time() to a specific value."""
    monkeypatch.setattr(time, "time", lambda: float(t))


class Capture:
    """Record telegram-style messages for assertion in tests."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any | None]] = []

    def send(self, text: str, markup: Any | None = None) -> None:
        self.messages.append((text, markup))
