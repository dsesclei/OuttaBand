from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import aiosqlite
from structlog.typing import FilteringBoundLogger

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
    CREATE TABLE IF NOT EXISTS baseline (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        sol REAL NOT NULL,
        usdc REAL NOT NULL,
        ts INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        sol REAL NOT NULL,
        usdc REAL NOT NULL,
        price REAL NOT NULL,
        drift REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS snapshots_ts_idx ON snapshots(ts);
    CREATE TABLE IF NOT EXISTS prefs (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """

    _DEFAULT_BANDS = (
        ("a", 0.0, 100.0),
        ("b", 0.0, 100.0),
        ("c", 0.0, 100.0),
    )

    def __init__(self, conn: aiosqlite.Connection, logger: FilteringBoundLogger) -> None:
        self._conn = conn
        self._log: FilteringBoundLogger = logger

    async def init(self, settings: "Settings") -> None:
        await self._conn.executescript(self.CREATE_TABLES_SQL)
        await self._conn.commit()
        await self._ensure_snapshots_schema()
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

    async def upsert_many(self, items: Dict[str, Tuple[float, float]]) -> None:
        if not items:
            return

        await self._conn.execute("BEGIN IMMEDIATE")
        try:
            for name, (low, high) in items.items():
                await self._conn.execute(
                    "INSERT INTO bands(name, low, high) VALUES(?,?,?) "
                    "ON CONFLICT(name) DO UPDATE SET low=excluded.low, high=excluded.high",
                    (name, low, high),
                )
        except Exception:
            await self._conn.rollback()
            raise
        else:
            await self._conn.commit()

    async def get_last_alert(self, band: str, side: str) -> Optional[int]:
        """Return raw timestamp; callers handle slot quantization for back-compat."""
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

    async def get_baseline(self) -> Optional[Tuple[float, float, int]]:
        async with self._conn.execute("SELECT sol, usdc, ts FROM baseline WHERE id=1") as cur:
            row = await cur.fetchone()
            if not row:
                return None
            sol, usdc, ts = row
            return float(sol), float(usdc), int(ts)

    async def set_baseline(self, sol: float, usdc: float, ts: int) -> None:
        await self._conn.execute(
            "INSERT INTO baseline(id, sol, usdc, ts) VALUES(1,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET sol=excluded.sol, usdc=excluded.usdc, ts=excluded.ts",
            (sol, usdc, ts),
        )
        await self._conn.commit()

    async def get_pref(self, key: str) -> Optional[str]:
        async with self._conn.execute("SELECT value FROM prefs WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
            return str(row[0]) if row and row[0] is not None else None

    async def set_pref(self, key: str, value: str) -> None:
        await self._conn.execute(
            "INSERT INTO prefs(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        await self._conn.commit()

    async def get_notional_usd(self) -> Optional[float]:
        raw = await self.get_pref("notional_usd")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    async def set_notional_usd(self, v: float) -> None:
        if not math.isfinite(v) or v < 0:
            raise ValueError("notional must be finite and non-negative")
        await self.set_pref("notional_usd", f"{v}")

    async def get_tilt_sol_frac(self) -> float:
        raw = await self.get_pref("tilt_sol_frac")
        if raw is None:
            return 0.5
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return 0.5
        return value

    async def set_tilt_sol_frac(self, f: float) -> None:
        if not math.isfinite(f):
            raise ValueError("tilt must be finite")
        clamped = min(max(f, 0.0), 1.0)
        await self.set_pref("tilt_sol_frac", f"{clamped}")

    async def insert_snapshot(self, ts: int, sol: float, usdc: float, price: float, drift: float) -> None:
        await self._conn.execute(
            "INSERT INTO snapshots(ts, sol, usdc, price, drift) VALUES(?,?,?,?,?)",
            (ts, sol, usdc, price, drift),
        )
        await self._conn.commit()

    async def get_latest_snapshot(self) -> Optional[Tuple[int, float, float, float, float]]:
        async with self._conn.execute(
            "SELECT ts, sol, usdc, price, drift FROM snapshots ORDER BY ts DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            ts, sol, usdc, price, drift = row
            return int(ts), float(sol), float(usdc), float(price), float(drift)

    async def _seed_bands_if_missing(self) -> None:
        for name, low, high in self._DEFAULT_BANDS:
            await self._conn.execute(
                "INSERT OR IGNORE INTO bands(name, low, high) VALUES(?,?,?)",
                (name, low, high),
            )
        await self._conn.commit()

    async def _ensure_snapshots_schema(self) -> None:
        async with self._conn.execute("PRAGMA table_info(snapshots)") as cur:
            rows = await cur.fetchall()

        if not rows:
            return

        column_names: List[str] = [row[1] for row in rows]
        if "id" in column_names:
            await self._cleanup_legacy_snapshots_table()
            return

        await self._conn.execute("BEGIN")
        try:
            await self._conn.execute("ALTER TABLE snapshots RENAME TO snapshots_old")
            await self._conn.executescript(
                """
                CREATE TABLE snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    sol REAL NOT NULL,
                    usdc REAL NOT NULL,
                    price REAL NOT NULL,
                    drift REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS snapshots_ts_idx ON snapshots(ts);
                """
            )
            await self._conn.execute(
                "INSERT INTO snapshots(ts, sol, usdc, price, drift) "
                "SELECT ts, sol, usdc, price, drift FROM snapshots_old"
            )
            await self._conn.execute("DROP TABLE snapshots_old")
        except Exception:
            await self._conn.rollback()
            raise
        else:
            await self._conn.commit()
        await self._cleanup_legacy_snapshots_table()

    async def _cleanup_legacy_snapshots_table(self) -> None:
        async with self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots_old'"
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return

        async with self._conn.execute("SELECT COUNT(1) FROM snapshots") as cur:
            count_row = await cur.fetchone()
        current = int(count_row[0]) if count_row and count_row[0] is not None else 0

        await self._conn.execute("BEGIN")
        try:
            if current == 0:
                await self._conn.execute(
                    "INSERT INTO snapshots(ts, sol, usdc, price, drift) "
                    "SELECT ts, sol, usdc, price, drift FROM snapshots_old"
                )
            await self._conn.execute("DROP TABLE snapshots_old")
        except Exception:
            await self._conn.rollback()
            raise
        else:
            await self._conn.commit()

    async def _upsert_bands_from_env(self, settings: "Settings") -> None:
        defaults = {name: (float(low), float(high)) for name, low, high in self._DEFAULT_BANDS}
        for name, spec in (("a", settings.BAND_A), ("b", settings.BAND_B), ("c", settings.BAND_C)):
            if not spec:
                continue
            parsed = self._parse_band_spec(spec)
            if parsed is None:
                self._log.warning("band_env_invalid", band=name, spec=spec)
                continue
            lo, hi = parsed
            async with self._conn.execute(
                "SELECT low, high FROM bands WHERE name=?",
                (name,),
            ) as cur:
                row = await cur.fetchone()
            current = (float(row[0]), float(row[1])) if row else None

            if current is None:
                await self.upsert_band(name, lo, hi)
                self._log.info("band_env_seeded", band=name, low=lo, high=hi, reason="missing")
                continue

            if self._ranges_close(current, (lo, hi)):
                self._log.debug("band_env_already_set", band=name, low=lo, high=hi)
                continue

            default_range = defaults.get(name)
            if default_range and self._ranges_close(current, default_range):
                await self.upsert_band(name, lo, hi)
                self._log.info("band_env_seeded", band=name, low=lo, high=hi, reason="default")
                continue

            self._log.info(
                "band_env_skipped_existing",
                band=name,
                existing_low=current[0],
                existing_high=current[1],
            )

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
    def _ranges_close(
        current: Tuple[float, float],
        target: Tuple[float, float],
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
    ) -> bool:
        return (
            math.isclose(current[0], target[0], rel_tol=rel_tol, abs_tol=abs_tol)
            and math.isclose(current[1], target[1], rel_tol=rel_tol, abs_tol=abs_tol)
        )

dbrepo = DBRepo
