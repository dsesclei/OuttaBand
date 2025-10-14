from __future__ import annotations

import importlib

import pytest

from config import Settings, configure_logging, tz
from runtime import Runtime


async def _start_runtime(settings: Settings) -> Runtime:
    log = configure_logging().bind(module="test-runtime")
    rt = Runtime(settings=settings, tz=tz(settings), log=log)
    await rt.start()
    return rt


@pytest.mark.asyncio
async def test_runtime_start_and_stop_without_telegram(fake_env) -> None:
    settings = Settings(METEORA_PAIR_ADDRESS="dummy_pair")
    assert settings.TELEGRAM_ENABLED is False

    rt = await _start_runtime(settings)
    try:
        assert rt.http is not None
        assert rt.db is not None
        assert rt.repo is not None
        assert rt.scheduler is None
        assert rt.tg is not None

        payload = await rt.health_payload()
        expected_keys = {
            "ok",
            "db_ok",
            "http_client_ok",
            "scheduler_ok",
            "telegram_ready",
            "last_run_ts_check",
            "last_run_ts_daily",
            "next_run_ts_check",
            "next_run_ts_daily",
            "volatility_cache_age_s",
        }
        assert expected_keys <= payload.keys()
        assert payload["scheduler_ok"] is False
        assert payload["telegram_ready"] is False
        assert payload["ok"] is False
    finally:
        await rt.stop()
        await rt.stop()


@pytest.mark.asyncio
async def test_sigma_payload_handles_none(fake_env) -> None:
    settings = Settings(METEORA_PAIR_ADDRESS="dummy_pair")
    rt = await _start_runtime(settings)

    class DummyVol:
        def __init__(self, result=None, error: Exception | None = None) -> None:
            self._result = result
            self._error = error

        async def read(self):
            if self._error:
                raise self._error
            return self._result

    try:
        assert rt.ctx is not None
        rt.ctx.vol = DummyVol(None)
        payload = await rt.sigma_payload()
        assert payload == {"ok": False, "data": None}

        rt.ctx.vol = DummyVol(error=RuntimeError("read failed"))
        payload_err = await rt.sigma_payload()
        assert payload_err["ok"] is False
        assert payload_err["data"] is None
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_api_endpoints(fake_env) -> None:
    settings = Settings(METEORA_PAIR_ADDRESS="dummy_pair")
    assert settings.TELEGRAM_ENABLED is False

    main = importlib.import_module("main")
    main = importlib.reload(main)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        health_payload = health.json()
        assert "ok" in health_payload
        assert "db_ok" in health_payload
        assert "telegram_ready" in health_payload

        sigma = client.get("/sigma")
        assert sigma.status_code == 200
        sigma_payload = sigma.json()
        assert sigma_payload.keys() == {"ok", "data"}

        version = client.get("/version")
        assert version.status_code == 200
        version_payload = version.json()
        assert {"service", "version", "git_sha"} <= version_payload.keys()
