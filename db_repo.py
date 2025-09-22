from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from main import Settings


class DBRepo:
    CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS bands (
        name TEXT PRIMARY KEY,
        low REAL NOT NULL,
        high REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS alerts (
        band TEXT,
        side TEXT,
        last_sent_ts INTEGER,
        PRIMARY KEY (band, side)
    );
    """

    _DEFAULT_BANDS = (
        ("a", 0.0, 100.0),
        ("b", 0.0, 100.0),
        ("c", 0.0, 100.0),
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def init(self, settings: "Settings") -> None:
        await self._conn.executescript(self.CREATE_TABLES_SQL)
        await self._conn.commit()
        await self._seed_bands_if_missing()
        await self._upsert_bands_from_env(settings)

    async def get_bands(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        async with self._conn.execute("SELECT name, low, high FROM bands ORDER BY name ASC") as cur:
            async for name, low, high in cur:
                out[name] = (float(low), float(high))
        return out

    async def upsert_band(self, name: str, low: float, high: float) -> None:
        await self._conn.execute(
            "INSERT INTO bands(name, low, high) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET low=excluded.low, high=excluded.high",
            (name, low, high),
        )
        await self._conn.commit()

    async def get_last_alert(self, band: str, side: str) -> Optional[int]:
        async with self._conn.execute(
            "SELECT last_sent_ts FROM alerts WHERE band=? AND side=?",
            (band, side),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None

    async def set_last_alert(self, band: str, side: str, ts: int) -> None:
        await self._conn.execute(
            "INSERT INTO alerts(band, side, last_sent_ts) VALUES(?,?,?) "
            "ON CONFLICT(band, side) DO UPDATE SET last_sent_ts=excluded.last_sent_ts",
            (band, side, ts),
        )
        await self._conn.commit()

    async def _seed_bands_if_missing(self) -> None:
        for name, low, high in self._DEFAULT_BANDS:
            await self._conn.execute(
                "INSERT OR IGNORE INTO bands(name, low, high) VALUES(?,?,?)",
                (name, low, high),
            )
        await self._conn.commit()

    async def _upsert_bands_from_env(self, settings: "Settings") -> None:
        for name, spec in (("a", settings.BAND_A), ("b", settings.BAND_B), ("c", settings.BAND_C)):
            if not spec:
                continue
            parsed = self._parse_band_spec(spec)
            if parsed is None:
                self._log("warn", "band_env_invalid", band=name, spec=spec)
                continue
            lo, hi = parsed
            await self.upsert_band(name, lo, hi)
            self._log("info", "band_env_upserted", band=name, low=lo, high=hi)

    @staticmethod
    def _parse_band_spec(spec: str) -> Optional[Tuple[float, float]]:
        try:
            lo_str, hi_str = spec.strip().split("-", 1)
            lo, hi = float(lo_str), float(hi_str)
            if not math.isfinite(lo) or not math.isfinite(hi):
                return None
            if lo >= hi:
                return None
            return (lo, hi)
        except Exception:
            return None

    @staticmethod
    def _log(level: str, event: str, **kwargs: object) -> None:
        from main import jlog  # imported lazily to avoid circular import

        jlog(level, event, **kwargs)


dbrepo = DBRepo
