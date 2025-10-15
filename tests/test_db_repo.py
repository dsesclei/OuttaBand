from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from outtaband.config import Settings
from outtaband.db_repo import DBRepo
from outtaband.shared_types import Baseline
from tests.conftest import now_time


class DummyLogger:
    def bind(self, *args, **kwargs):  # type: ignore[override]
        return self

    def warning(self, *args, **kwargs) -> None:
        return None

    def info(self, *args, **kwargs) -> None:
        return None

    def debug(self, *args, **kwargs) -> None:
        return None


@pytest_asyncio.fixture()
async def repo(fake_env, tmp_path: Path) -> DBRepo:
    db_path = tmp_path / "db.sqlite"
    conn = await aiosqlite.connect(db_path)
    logger = DummyLogger()
    repository = DBRepo(conn, logger)
    settings = Settings(DB_PATH=str(db_path))
    await repository.init(settings)
    yield repository
    await conn.close()


@pytest.mark.asyncio
async def test_upsert_band_normalizes_and_persists(repo: DBRepo) -> None:
    await repo.upsert_band("a", "1", "2")
    bands = await repo.get_bands()
    assert bands["a"] == (1.0, 2.0)


@pytest.mark.asyncio
async def test_upsert_many_aggregates_errors_and_is_atomic(repo: DBRepo) -> None:
    original = await repo.get_bands()
    with pytest.raises(ValueError):
        await repo.upsert_many({"a": (1.0, 2.0), "b": (5.0, 4.0)})
    assert await repo.get_bands() == original


def test_parse_band_spec_accepts_variants() -> None:
    assert DBRepo._parse_band_spec("1.0-2.0") == (1.0, 2.0)
    assert DBRepo._parse_band_spec("1.0 : 2.0") == (1.0, 2.0)
    assert DBRepo._parse_band_spec("nan-2") is None
    assert DBRepo._parse_band_spec("5-5") is None


@pytest.mark.asyncio
async def test_notional_roundtrip_and_validation(repo: DBRepo) -> None:
    await repo.set_notional_usd(123.45)
    assert await repo.get_notional_usd() == pytest.approx(123.45)
    with pytest.raises(ValueError):
        await repo.set_notional_usd(-1.0)


@pytest.mark.asyncio
async def test_get_notional_usd_handles_junk_values_gracefully(repo: DBRepo) -> None:
    await repo._conn.execute(
        "INSERT OR REPLACE INTO prefs(key, value) VALUES('notional_usd', 'abc')"
    )
    await repo._conn.commit()
    assert await repo.get_notional_usd() is None


@pytest.mark.asyncio
async def test_tilt_defaults_and_clamping(repo: DBRepo) -> None:
    assert await repo.get_tilt_sol_frac() == pytest.approx(0.5)
    await repo.set_tilt_sol_frac(-1.0)
    assert await repo.get_tilt_sol_frac() == pytest.approx(0.0)
    await repo.set_tilt_sol_frac(2.0)
    assert await repo.get_tilt_sol_frac() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_insert_and_get_latest_snapshot(repo: DBRepo) -> None:
    await repo.insert_snapshot(10, 1.0, 2.0, 3.0, 0.1)
    await repo.insert_snapshot(20, 1.5, 2.5, 3.5, 0.2)
    snapshot = await repo.get_latest_snapshot()
    assert snapshot is not None
    assert snapshot.ts == 20
    assert snapshot.sol == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_set_get_baseline_roundtrip(repo: DBRepo) -> None:
    assert await repo.get_baseline() is None
    await repo.set_baseline(10.0, 20.0, 30)
    baseline = await repo.get_baseline()
    assert baseline == Baseline(10.0, 20.0, 30)


@pytest.mark.asyncio
async def test_acquire_lock_first_then_false_until_ttl(
    repo: DBRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_time(monkeypatch, 1_000.0)
    assert await repo.acquire_lock("job", ttl_s=60) is True
    assert await repo.acquire_lock("job", ttl_s=60) is False
    now_time(monkeypatch, 1_000.0 + 61.0)
    assert await repo.acquire_lock("job", ttl_s=60) is True


@pytest.mark.asyncio
async def test_ensure_snapshots_schema_creates_when_missing(fake_env, tmp_path: Path) -> None:
    db_path = tmp_path / "schema.sqlite"
    conn = await aiosqlite.connect(db_path)
    repo = DBRepo(conn, DummyLogger())
    settings = Settings(DB_PATH=str(db_path))
    await repo.init(settings)
    async with conn.execute("PRAGMA table_info(snapshots)") as cur:
        columns = [row["name"] for row in await cur.fetchall()]
    assert "id" in columns
    async with conn.execute("PRAGMA index_list('snapshots')") as cur:
        indexes = [row["name"] for row in await cur.fetchall()]
    assert any(idx == "snapshots_ts_idx" for idx in indexes)
    await conn.close()


@pytest.mark.asyncio
async def test_migrates_from_old_layout(fake_env, tmp_path: Path) -> None:
    db_path = tmp_path / "migrate.sqlite"
    conn = await aiosqlite.connect(db_path)
    repo = DBRepo(conn, DummyLogger())
    await conn.execute("CREATE TABLE prefs (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    await conn.execute(
        """
        CREATE TABLE snapshots (
            ts INTEGER NOT NULL,
            sol REAL NOT NULL,
            usdc REAL NOT NULL,
            price REAL NOT NULL,
            drift REAL NOT NULL
        )
        """
    )
    await conn.execute(
        "INSERT INTO snapshots(ts, sol, usdc, price, drift) VALUES(1, 2.0, 3.0, 4.0, 0.1)"
    )
    await conn.commit()
    await repo._ensure_snapshots_schema()
    async with conn.execute("PRAGMA table_info(snapshots)") as cur:
        columns = [row["name"] for row in await cur.fetchall()]
    assert "id" in columns
    async with conn.execute("SELECT COUNT(1) FROM snapshots") as cur:
        count = (await cur.fetchone())[0]
    assert count == 1
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots_old'"
    ) as cur:
        assert await cur.fetchone() is None
    await conn.close()


@pytest.mark.asyncio
async def test_cleanup_legacy_snapshots_table_backfills_when_empty(repo: DBRepo) -> None:
    await repo._conn.execute("DELETE FROM snapshots")
    await repo._conn.execute(
        """
        CREATE TABLE snapshots_old (
            ts INTEGER,
            sol REAL,
            usdc REAL,
            price REAL,
            drift REAL
        )
        """
    )
    await repo._conn.execute(
        "INSERT INTO snapshots_old(ts, sol, usdc, price, drift) VALUES(5, 1.1, 2.2, 3.3, 0.4)"
    )
    await repo._conn.commit()
    await repo._cleanup_legacy_snapshots_table()
    async with repo._conn.execute("SELECT COUNT(1) FROM snapshots") as cur:
        assert (await cur.fetchone())[0] == 1
    async with repo._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots_old'"
    ) as cur:
        assert await cur.fetchone() is None


@pytest.mark.asyncio
async def test_upsert_bands_from_env_applies_defaults(repo: DBRepo) -> None:
    for name, low, high in DBRepo._DEFAULT_BANDS:
        await repo.upsert_band(name, low, high)
    custom = Settings(
        BAND_A="10-11",
        BAND_B="20-21",
        BAND_C="30-31",
        METEORA_PAIR_ADDRESS="dummy_pair",
    )
    await repo._upsert_bands_from_env(custom)
    bands = await repo.get_bands()
    assert bands["a"] == (10.0, 11.0)
    assert bands["b"] == (20.0, 21.0)
    assert bands["c"] == (30.0, 31.0)


@pytest.mark.asyncio
async def test_upsert_bands_from_env_skips_when_non_default(repo: DBRepo) -> None:
    for name, low, high in DBRepo._DEFAULT_BANDS:
        await repo.upsert_band(name, low, high)
    await repo.upsert_band("b", 5.0, 6.0)
    custom = Settings(
        BAND_A="10-11",
        BAND_B="20-21",
        BAND_C="30-31",
        METEORA_PAIR_ADDRESS="dummy_pair",
    )
    await repo._upsert_bands_from_env(custom)
    bands = await repo.get_bands()
    assert bands["b"] == (5.0, 6.0)
    assert bands["a"] == (10.0, 11.0)
    assert bands["c"] == (30.0, 31.0)
