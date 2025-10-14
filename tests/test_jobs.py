from __future__ import annotations

from typing import Any

import pytest
import policy.band_advisor as band_advisor
from jobs import AppContext, JobSettings, check_once, floor_to_slot, send_daily_advisory
from policy.volatility import VolReading


class FakeRepo:
    def __init__(self) -> None:
        self.bands: dict[str, tuple[float, float]] = {
            "a": (90.0, 110.0),
            "b": (95.0, 115.0),
            "c": (100.0, 120.0),
        }
        self.alerts: dict[tuple[str, str], int] = {}
        self.baseline: tuple[float, float, int] | None = None
        self.snapshot: tuple[int, float, float, float, float] | None = None
        self.notional: float | None = 1000.0
        self.tilt: float = 0.5

    async def get_bands(self) -> dict[str, tuple[float, float]]:
        return dict(self.bands)

    async def upsert_band(self, name: str, low: float, high: float) -> None:
        self.bands[name] = (low, high)

    async def get_last_alert(self, band: str, side: str) -> int | None:
        return self.alerts.get((band, side))

    async def set_last_alert(self, band: str, side: str, ts: int) -> None:
        self.alerts[(band, side)] = ts

    async def get_baseline(self) -> tuple[float, float, int] | None:
        return self.baseline

    async def set_baseline(self, sol: float, usdc: float, ts: int) -> None:
        self.baseline = (sol, usdc, ts)

    async def get_latest_snapshot(self) -> tuple[int, float, float, float, float] | None:
        return self.snapshot

    async def insert_snapshot(self, ts: int, sol: float, usdc: float, price: float, drift: float) -> None:
        self.snapshot = (ts, sol, usdc, price, drift)

    async def get_notional_usd(self) -> float | None:
        return self.notional

    async def set_notional_usd(self, value: float) -> None:
        self.notional = value

    async def get_tilt_sol_frac(self) -> float:
        return self.tilt

    async def set_tilt_sol_frac(self, value: float) -> None:
        self.tilt = max(0.0, min(1.0, value))


class FakeTG:
    def __init__(self) -> None:
        self.breaches: list[dict[str, Any]] = []
        self.advisories: list[dict[str, Any]] = []

    async def send_breach_offer(
        self,
        *,
        band: str,
        price: float,
        src_label: str | None,
        bands: dict[str, tuple[float, float]],
        suggested_range: tuple[float, float],
        policy_meta: tuple[str, float] | None,
    ) -> None:
        self.breaches.append(
            {
                "band": band,
                "price": price,
                "src_label": src_label,
                "suggested_range": suggested_range,
                "policy_meta": policy_meta,
            }
        )

    async def send_advisory_card(self, advisory: dict[str, Any], drift_line: str | None = None) -> None:
        self.advisories.append({"advisory": advisory, "drift_line": drift_line})


class FakePrice:
    def __init__(self, value: float | None) -> None:
        self.value = value
        self.calls = 0

    async def read(self) -> float | None:
        self.calls += 1
        return self.value


class FakeVol:
    def __init__(self, reading: VolReading | None) -> None:
        self.reading = reading
        self.calls = 0

    async def read(self) -> VolReading | None:
        self.calls += 1
        return self.reading


class CaptureLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def bind(self, **kwargs: Any) -> "CaptureLog":
        return self

    def info(self, event: str, **kwargs: Any) -> None:
        self.records.append((event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.records.append((event, kwargs))


@pytest.fixture()
def default_ctx() -> tuple[AppContext, FakeRepo, FakeTG]:
    repo = FakeRepo()
    tg = FakeTG()
    price = FakePrice(105.0)
    sigma = VolReading(sigma_pct=0.7, bucket="mid", sample_count=60)
    vol = FakeVol(sigma)
    log = CaptureLog()
    job = JobSettings(check_every_minutes=5, cooldown_minutes=10, include_a_on_high=False)
    ctx = AppContext(repo=repo, tg=tg, price=price, vol=vol, job=job, log=log)
    return ctx, repo, tg


def test_floor_to_slot_quantizes_correctly() -> None:
    assert floor_to_slot(123, 60) == 120
    assert floor_to_slot(3600, 600) == 3600
    assert floor_to_slot(3599, 600) == 3000
    with pytest.raises(ValueError):
        floor_to_slot(10, 0)


@pytest.mark.asyncio
async def test_check_once_skips_on_missing_or_invalid_price(default_ctx: tuple[AppContext, FakeRepo, FakeTG]) -> None:
    ctx, repo, tg = default_ctx
    ctx.price = FakePrice(None)
    await check_once(ctx, now_ts=1_000)
    assert not tg.breaches

    ctx.price = FakePrice(float("nan"))
    await check_once(ctx, now_ts=1_100)
    assert not tg.breaches

    ctx.price = FakePrice(-5.0)
    await check_once(ctx, now_ts=1_200)
    assert not tg.breaches


@pytest.mark.asyncio
async def test_check_once_generates_breach_and_enforces_cooldown(default_ctx: tuple[AppContext, FakeRepo, FakeTG]) -> None:
    ctx, repo, tg = default_ctx
    ctx.job.cooldown_minutes = 5
    ctx.job.check_every_minutes = 1

    # Breach price below band a
    repo.bands = {"a": repo.bands["a"]}
    ctx.price = FakePrice(89.0)
    await check_once(ctx, now_ts=600)
    assert len(tg.breaches) == 1
    first = tg.breaches[0]
    assert first["band"] == "a"
    assert first["policy_meta"][0] == "mid"
    assert len(tg.breaches) == 1
    assert repo.alerts[("a", "low")] == 600

    # Within cooldown
    ctx.price = FakePrice(82.0)
    await check_once(ctx, now_ts=660)
    assert len(tg.breaches) == 1

    # After cooldown
    ctx.price = FakePrice(83.0)
    await check_once(ctx, now_ts=1_000)
    assert len(tg.breaches) == 2


@pytest.mark.asyncio
async def test_check_once_bucket_from_sigma_and_include_a_flag() -> None:
    repo = FakeRepo()
    tg = FakeTG()
    price = FakePrice(123.0)
    sigma = VolReading(sigma_pct=2.0, bucket="high", sample_count=60)
    vol = FakeVol(sigma)
    log = CaptureLog()
    job = JobSettings(check_every_minutes=5, cooldown_minutes=5, include_a_on_high=True)
    ctx = AppContext(repo=repo, tg=tg, price=price, vol=vol, job=job, log=log)

    await check_once(ctx, now_ts=600)

    assert tg.breaches
    policy_meta = tg.breaches[0]["policy_meta"]
    assert policy_meta is not None
    bucket, _width = policy_meta
    assert bucket == "high"
    suggested_range = tg.breaches[0]["suggested_range"]
    ranges = band_advisor.ranges_for_price(123.0, "high", include_a_on_high=True)
    assert "a" in ranges
    assert suggested_range == ranges["a"]


@pytest.mark.asyncio
async def test_check_once_no_breach_logs(default_ctx: tuple[AppContext, FakeRepo, FakeTG]) -> None:
    ctx, repo, tg = default_ctx
    ctx.price = FakePrice(107.0)
    await check_once(ctx, now_ts=600)
    assert not tg.breaches
    assert any(event == "no_breach" for event, _ in ctx.log.records)


@pytest.mark.asyncio
async def test_send_daily_advisory_includes_amounts_and_drift(default_ctx: tuple[AppContext, FakeRepo, FakeTG]) -> None:
    ctx, repo, tg = default_ctx

    # Set prefs for advisory amounts
    await repo.set_notional_usd(500.0)
    await repo.set_tilt_sol_frac(0.6)
    await repo.set_baseline(1.0, 100.0, 0)
    repo.snapshot = (1, 1.5, 80.0, 105.0, 0.0)
    await repo.set_last_alert("a", "low", 0)  # Ensure repo is used
    await repo.insert_snapshot(2, 1.4, 82.0, 106.0, 0.0)
    repo.snapshot = (2, 1.4, 82.0, 106.0, 0.0)

    ctx.price = FakePrice(110.0)
    ctx.vol = FakeVol(VolReading(sigma_pct=0.5, bucket="low"))

    await send_daily_advisory(ctx)

    assert len(tg.advisories) == 1
    payload = tg.advisories[0]
    advisory = payload["advisory"]
    assert advisory["price"] == pytest.approx(110.0)
    assert "ranges" in advisory
    assert "stale" in advisory
    drift_line = payload["drift_line"]
    assert drift_line is not None and "Drift" in drift_line
