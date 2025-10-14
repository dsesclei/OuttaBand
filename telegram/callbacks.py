from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from shared_types import BAND_ORDER, BandName

# Callback kinds
Kind = Literal["adv", "alert", "bands"]

# Typed payloads


@dataclass(frozen=True)
class AdvAction:
    a: Literal["apply", "ignore", "set"]
    t: str  # token (pending id)
    k: Literal["adv"] = "adv"


@dataclass(frozen=True)
class AlertAction:
    a: Literal["accept", "ignore", "set"]
    band: BandName
    t: str
    k: Literal["alert"] = "alert"


@dataclass(frozen=True)
class BandsAction:
    a: Literal["edit", "back"]
    band: BandName | None = None
    k: Literal["bands"] = "bands"


# Backward/forward compatible envelope (versioned)
class _Envelope(TypedDict, total=False):
    v: int
    k: Kind
    a: str
    t: str
    band: BandName


VERSION = 1


def encode(obj: AdvAction | AlertAction | BandsAction) -> str:
    payload: _Envelope = {"v": VERSION, "k": obj.k, "a": obj.a}
    if isinstance(obj, (AdvAction, AlertAction)):
        payload["t"] = obj.t
    if isinstance(obj, AlertAction):
        payload["band"] = obj.band
    if isinstance(obj, BandsAction) and obj.band is not None:
        payload["band"] = obj.band
    return json.dumps(payload, separators=(",", ":"))


def decode(s: str) -> AdvAction | AlertAction | BandsAction | None:
    try:
        data = json.loads(s)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    k = data.get("k")
    a = data.get("a")
    if k == "adv" and a in {"apply", "ignore", "set"} and isinstance(data.get("t"), str):
        return AdvAction(a=a, t=data["t"])
    if k == "alert" and a in {"accept", "ignore", "set"} and isinstance(data.get("t"), str):
        band = data.get("band")
        if isinstance(band, str) and band in BAND_ORDER:
            return AlertAction(a=a, band=cast(BandName, band), t=data["t"])
    if k == "bands" and a in {"edit", "back"}:
        band_val = data.get("band")
        band: BandName | None = None
        if isinstance(band_val, str) and band_val in BAND_ORDER:
            band = cast(BandName, band_val)
        return BandsAction(a=a, band=band)
    return None
