from __future__ import annotations

from typing import Literal, NamedTuple, TypedDict

# Core literals / atoms
BandName = Literal["a", "b", "c"]
BAND_ORDER: tuple[BandName, ...] = ("a", "b", "c")
Bucket = Literal["low", "mid", "high"]
Side = Literal["low", "high"]
UnixTs = int
BucketSplit = tuple[int, int, int]

# Band ranges / maps
BandRange = tuple[float, float]
BandMap = dict[BandName, BandRange]
PolicyWidthMap = dict[BandName, float]

# Per-band allocation amounts (SOL, USDC)
AmountsMap = dict[BandName, tuple[float, float]]


class AdvisoryPayloadBase(TypedDict):
    price: float
    sigma_pct: float | None
    bucket: Bucket
    split: BucketSplit
    ranges: BandMap


class AdvisoryPayload(AdvisoryPayloadBase, total=False):
    stale: bool


class Baseline(NamedTuple):
    sol: float
    usdc: float
    ts: UnixTs


class Snapshot(NamedTuple):
    ts: UnixTs
    sol: float
    usdc: float
    price: float
    drift: float


PendingKind = Literal["alert", "adv"]


class AlertPayload(NamedTuple):
    band: BandName
    suggested_range: BandRange


AdvPayload = BandMap
PendingPayload = AlertPayload | AdvPayload

__all__ = [
    "BAND_ORDER",
    "AdvPayload",
    "AdvisoryPayload",
    "AlertPayload",
    "AmountsMap",
    "BandMap",
    "BandName",
    "BandRange",
    "Baseline",
    "Bucket",
    "BucketSplit",
    "PendingKind",
    "PendingPayload",
    "PolicyWidthMap",
    "Side",
    "Snapshot",
    "UnixTs",
]
