import asyncio
import math

import aiosqlite
import pytest
import structlog

from db_repo import DBRepo
from shared_types import BAND_ORDER


def test_upsert_band_rejects_invalid_range() -> None:
    async def runner() -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.executescript(DBRepo.CREATE_TABLES_SQL)
            await conn.commit()
            repo = DBRepo(conn, structlog.get_logger("test"))

            with pytest.raises(ValueError):
                await repo.upsert_band(BAND_ORDER[0], 2.0, 1.0)

            with pytest.raises(ValueError):
                await repo.upsert_band(BAND_ORDER[0], math.nan, 2.0)
        finally:
            await conn.close()

    asyncio.run(runner())


def test_upsert_many_rolls_back_on_invalid_entries() -> None:
    async def runner() -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.executescript(DBRepo.CREATE_TABLES_SQL)
            await conn.commit()
            repo = DBRepo(conn, structlog.get_logger("test"))

            valid_item = (BAND_ORDER[0], (1.0, 2.0))
            invalid_item = (BAND_ORDER[1], (3.0, 1.0))
            items = dict([valid_item, invalid_item])

            with pytest.raises(ValueError):
                await repo.upsert_many(items)

            async with conn.execute("SELECT COUNT(*) FROM bands") as cur:
                count_row = await cur.fetchone()
            assert count_row is not None
            assert count_row[0] == 0

            await repo.upsert_many({BAND_ORDER[0]: (1.5, 2.5)})

            async with conn.execute(
                "SELECT low, high FROM bands WHERE name=?", (BAND_ORDER[0],)
            ) as cur:
                row = await cur.fetchone()
            assert row is not None
            assert (row["low"], row["high"]) == (1.5, 2.5)
        finally:
            await conn.close()

    asyncio.run(runner())
