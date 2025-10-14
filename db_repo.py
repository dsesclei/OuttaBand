from __future__ import annotations

import math
import re
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

import aiosqlite
from structlog.typing import FilteringBoundLogger

from shared_types import BAND_ORDER, BandMap, BandName, BandRange, Baseline, Side, Snapshot

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from config import Settings


# Constants
BANDS_UPSERT_SQL: Final = (
    "INSERT INTO bands(name, low, high) VALUES(?,?,?) "
    "ON CONFLICT(name) DO UPDATE SET low=excluded.low, high=excluded.high"
)

_BAND_SPEC_RE: Final = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-:]\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


class DBRepo:
    CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS bands (
        name TEXT PRIMARY KEY,
        low  REAL NOT NULL,
        high REAL NOT NULL,
        CHECK (low < high)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS alerts (
        band TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('low','high')),
        last_sent_ts INTEGER,
        PRIMARY KEY (band, side)
    );

    CREATE TABLE IF NOT EXISTS baseline (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        sol REAL NOT NULL,
        usdc REAL NOT NULL,
        ts INTEGER NOT NULL CHECK (ts >= 0)
    );

    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        sol REAL NOT NULL,
        usdc REAL NOT NULL,
        price REAL NOT NULL CHECK (price > 0),
        drift REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS snapshots_ts_idx ON snapshots(ts);

    CREATE TABLE IF NOT EXISTS prefs (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """

    _DEFAULT_BANDS: Final[tuple[tuple[BandName, float, float], ...]] = tuple(
        (band, 0.0, 100.0) for band in BAND_ORDER
    )

    def __init__(self, conn: aiosqlite.Connection, logger: FilteringBoundLogger) -> None:
        self._conn = conn
        self._conn.row_factory = aiosqlite.Row  # Named access + native types
        self._log: FilteringBoundLogger = logger

    async def init(self, settings: Settings) -> None:
        # Pragmatic defaults for reliability/perf
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=3000")

        await self._conn.executescript(self.CREATE_TABLES_SQL)
        await self._conn.commit()
        await self._ensure_snapshots_schema()
        await self._seed_bands_if_missing()
        await self._upsert_bands_from_env(settings)

    # ---------- Small helpers ----------

    @asynccontextmanager
    async def _tx(self, *, immediate: bool = False):
        await self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield
        except Exception:
            await self._conn.rollback()
            raise
        else:
            await self._conn.commit()

    async def _scalar(self, sql: str, params: tuple[object, ...] = ()) -> object | None:
        async with self._conn.execute(sql, params) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    # ---------- Bands ----------

    async def get_bands(self) -> BandMap:
        out: BandMap = {}
        async with self._conn.execute("SELECT name, low, high FROM bands ORDER BY name") as cur:
            async for row in cur:
                out[row["name"]] = (row["low"], row["high"])
        return out

    async def upsert_band(self, name: str, low: float, high: float) -> None:
        lo, hi = self._normalize_band_range(name, low, high)
        async with self._tx():
            await self._conn.execute(BANDS_UPSERT_SQL, (name, lo, hi))

    async def upsert_many(self, items: BandMap) -> None:
        if not items:
            return

        normalized: list[tuple[str, float, float]] = []
        errors: dict[str, str] = {}
        for name, (low, high) in items.items():
            try:
                lo, hi = self._normalize_band_range(name, low, high)
            except ValueError as exc:
                reason = str(exc)
                errors[name] = reason
                self._log.warning("band_upsert_invalid", band=name, reason=reason)
            else:
                normalized.append((name, lo, hi))

        if errors:
            problems = "; ".join(f"{k}: {v}" for k, v in errors.items())
            raise ValueError(f"invalid band ranges → {problems}")

        if not normalized:
            return

        async with self._tx(immediate=True):
            await self._conn.executemany(BANDS_UPSERT_SQL, normalized)

    # ---------- Alerts ----------

    async def get_last_alert(self, band: str, side: Side) -> int | None:
        ts = await self._scalar(
            "SELECT last_sent_ts FROM alerts WHERE band=? AND side=?",
            (band, side),
        )
        return int(ts) if ts is not None else None

    async def set_last_alert(self, band: str, side: Side, ts: int) -> None:
        async with self._tx():
            await self._conn.execute(
                "INSERT INTO alerts(band, side, last_sent_ts) VALUES(?,?,?) "
                "ON CONFLICT(band, side) DO UPDATE SET last_sent_ts=excluded.last_sent_ts",
                (band, side, ts),
            )

    # ---------- Baseline ----------

    async def get_baseline(self) -> Baseline | None:
        async with self._conn.execute(
            "SELECT sol, usdc, ts FROM baseline WHERE id=1 LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return Baseline(row["sol"], row["usdc"], row["ts"])

    async def set_baseline(self, sol: float, usdc: float, ts: int) -> None:
        async with self._tx():
            await self._conn.execute(
                "INSERT INTO baseline(id, sol, usdc, ts) VALUES(1,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET sol=excluded.sol, usdc=excluded.usdc, ts=excluded.ts",
                (sol, usdc, ts),
            )

    # ---------- Prefs ----------

    async def get_pref(self, key: str) -> str | None:
        v = await self._scalar("SELECT value FROM prefs WHERE key=?", (key,))
        return None if v is None else str(v)

    async def set_pref(self, key: str, value: str) -> None:
        async with self._tx():
            await self._conn.execute(
                "INSERT INTO prefs(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    async def get_notional_usd(self) -> float | None:
        raw = await self.get_pref("notional_usd")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if math.isfinite(value) and value >= 0 else None

    async def set_notional_usd(self, v: float) -> None:
        if not math.isfinite(v) or v < 0:
            raise ValueError("notional must be finite and non-negative")
        await self.set_pref("notional_usd", str(v))

    async def get_tilt_sol_frac(self) -> float:
        raw = await self.get_pref("tilt_sol_frac")
        if raw is None:
            return 0.5
        try:
            value = float(raw)
        except ValueError:
            return 0.5
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return 0.5
        return value

    async def set_tilt_sol_frac(self, f: float) -> None:
        if not math.isfinite(f):
            raise ValueError("tilt must be finite")
        clamped = min(max(f, 0.0), 1.0)
        await self.set_pref("tilt_sol_frac", str(clamped))

    # ---------- Snapshots ----------

    async def insert_snapshot(self, ts: int, sol: float, usdc: float, price: float, drift: float) -> None:
        async with self._tx():
            await self._conn.execute(
                "INSERT INTO snapshots(ts, sol, usdc, price, drift) VALUES(?,?,?,?,?)",
                (ts, sol, usdc, price, drift),
            )

    async def get_latest_snapshot(self) -> Snapshot | None:
        async with self._conn.execute(
            "SELECT ts, sol, usdc, price, drift FROM snapshots ORDER BY ts DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return Snapshot(row["ts"], row["sol"], row["usdc"], row["price"], row["drift"])

    # ---------- Schema / Migration ----------

    async def _seed_bands_if_missing(self) -> None:
        async with self._tx():
            for name, low, high in self._DEFAULT_BANDS:
                await self._conn.execute(
                    "INSERT OR IGNORE INTO bands(name, low, high) VALUES(?,?,?)",
                    (name, low, high),
                )

    async def _ensure_snapshots_schema(self) -> None:
        schema_version = await self._get_schema_version()
        if schema_version is not None and schema_version >= 1:
            return

        async with self._conn.execute("PRAGMA table_info(snapshots)") as cur:
            rows = await cur.fetchall()
        column_names = [row["name"] for row in rows] if rows else []

        if not column_names:
            if schema_version is None:
                await self.set_pref("schema_version", "1")
            return

        if "id" in column_names:
            await self._cleanup_legacy_snapshots_table()
            if schema_version is None:
                await self.set_pref("schema_version", "1")
            return

        async with self._tx():
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

        await self._cleanup_legacy_snapshots_table()
        await self.set_pref("schema_version", "1")

    async def _cleanup_legacy_snapshots_table(self) -> None:
        has_old = await self._scalar(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots_old'"
        )
        if has_old is None:
            return

        current = await self._scalar("SELECT COUNT(1) FROM snapshots")
        count = int(current or 0)

        async with self._tx():
            if count == 0:
                await self._conn.execute(
                    "INSERT INTO snapshots(ts, sol, usdc, price, drift) "
                    "SELECT ts, sol, usdc, price, drift FROM snapshots_old"
                )
            await self._conn.execute("DROP TABLE snapshots_old")

    # ---------- ENV-driven bands ----------

    async def _upsert_bands_from_env(self, settings: Settings) -> None:
        defaults: BandMap = {
            name: (float(low), float(high)) for name, low, high in self._DEFAULT_BANDS
        }

        # Single transaction, avoids nested tx in per-row upserts.
        async with self._tx():
            band_specs = (settings.BAND_A, settings.BAND_B, settings.BAND_C)
            for name, spec in zip(BAND_ORDER, band_specs, strict=False):
                if not spec:
                    continue

                parsed = self._parse_band_spec(spec)
                if parsed is None:
                    self._log.warning("band_env_invalid", band=name, spec=spec)
                    continue

                lo, hi = parsed
                async with self._conn.execute(
                    "SELECT low, high FROM bands WHERE name=? LIMIT 1",
                    (name,),
                ) as cur:
                    row = await cur.fetchone()

                current = (row["low"], row["high"]) if row else None

                if current is None:
                    await self._conn.execute(BANDS_UPSERT_SQL, (name, lo, hi))
                    self._log.info("band_env_seeded", band=name, low=lo, high=hi, reason="missing")
                    continue

                if self._ranges_close(current, (lo, hi)):
                    self._log.debug("band_env_already_set", band=name, low=lo, high=hi)
                    continue

                default_range = defaults.get(name)
                if default_range and self._ranges_close(current, default_range):
                    await self._conn.execute(BANDS_UPSERT_SQL, (name, lo, hi))
                    self._log.info("band_env_seeded", band=name, low=lo, high=hi, reason="default")
                    continue

                self._log.info(
                    "band_env_skipped_existing",
                    band=name,
                    existing_low=current[0],
                    existing_high=current[1],
                )

    # ---------- Locks (singleton guards) ----------

    async def acquire_lock(self, name: str, ttl_s: int = 120) -> bool:
        """coarse advisory lock. returns True if acquired, False if held by someone else."""
        now = int(time.time())
        until = now + max(1, int(ttl_s))
        async with self._tx(immediate=True):
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS locks (
                    name TEXT PRIMARY KEY,
                    until_ts INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            async with self._conn.execute("SELECT until_ts FROM locks WHERE name=?", (name,)) as cur:
                row = await cur.fetchone()
            if row is not None:
                try:
                    current_until = int(row["until_ts"])
                except Exception:
                    current_until = 0
                if current_until > now:
                    return False
            await self._conn.execute(
                "INSERT INTO locks(name, until_ts) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET until_ts=excluded.until_ts",
                (name, until),
            )
            return True

    # ---------- Utilities ----------

    @staticmethod
    def _parse_band_spec(spec: str) -> BandRange | None:
        m = _BAND_SPEC_RE.match(spec)
        if not m:
            return None
        lo, hi = float(m.group(1)), float(m.group(2))
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        if lo >= hi:
            return None
        return (lo, hi)

    @staticmethod
    def _normalize_band_range(name: str, low: float, high: float) -> BandRange:
        try:
            lo = float(low)
            hi = float(high)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Band '{name}': low/high must be numeric") from exc
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Band '{name}': low/high must be finite")
        if lo >= hi:
            raise ValueError(f"Band '{name}': low must be < high")
        return (lo, hi)

    async def _get_schema_version(self) -> int | None:
        raw = await self.get_pref("schema_version")
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _ranges_close(
        current: BandRange,
        target: BandRange,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
    ) -> bool:
        return (
            math.isclose(current[0], target[0], rel_tol=rel_tol, abs_tol=abs_tol)
            and math.isclose(current[1], target[1], rel_tol=rel_tol, abs_tol=abs_tol)
        )
