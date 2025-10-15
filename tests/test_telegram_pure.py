from __future__ import annotations

import importlib
import sys
import types

import pytest

from policy.volatility import VolReading
from shared_types import BAND_ORDER, BandMap, Baseline, PendingKind, Snapshot
from tests.conftest import Capture, now_time
from tests.test_jobs import FakeRepo

# Provide a lightweight telegram (third-party) stub so pure helpers can be imported
if "telegram" not in sys.modules or not getattr(sys.modules["telegram"], "_outtaband_stub", False):
    telegram_stub = types.ModuleType("telegram")

    class InlineKeyboardButton:
        def __init__(self, text: str, callback_data: str | None = None) -> None:
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:
        def __init__(self, keyboard: list[list[InlineKeyboardButton]]) -> None:
            self.inline_keyboard = keyboard

    telegram_stub.InlineKeyboardButton = InlineKeyboardButton
    telegram_stub.InlineKeyboardMarkup = InlineKeyboardMarkup
    telegram_stub._outtaband_stub = True
    sys.modules["telegram"] = telegram_stub

callbacks = importlib.import_module("tgbot.callbacks")
pending = importlib.import_module("tgbot.pending")
render = importlib.import_module("tgbot.render")
handlers_mod = importlib.import_module("tgbot.handlers")

BotCtx = handlers_mod.BotCtx
Handlers = handlers_mod.Handlers
Providers = handlers_mod.Providers


# --- Callbacks ----------------------------------------------------------------


def test_encode_decode_roundtrip_adv_alert_bands() -> None:
    adv_actions = [
        callbacks.AdvAction(a="apply", t="tok1"),
        callbacks.AdvAction(a="ignore", t="tok2"),
        callbacks.AdvAction(a="set", t="tok3"),
    ]
    alert_actions = [
        callbacks.AlertAction(a="accept", band="a", t="tokA"),
        callbacks.AlertAction(a="ignore", band="b", t="tokB"),
        callbacks.AlertAction(a="set", band="c", t="tokC"),
    ]
    bands_actions = [
        callbacks.BandsAction(a="edit", band="a"),
        callbacks.BandsAction(a="back"),
    ]

    for obj in adv_actions + alert_actions + bands_actions:
        encoded = callbacks.encode(obj)
        decoded = callbacks.decode(encoded)
        assert decoded == obj


def test_decode_rejects_malformed() -> None:
    cases = [
        "not json",
        "[]",
        '{"k": "adv"}',
        '{"k": "adv", "a": "apply", "t": 123}',
        '{"k": "adv", "a": "noop", "t": "tok"}',
        '{"k": "alert", "a": "accept", "t": "tok"}',
        '{"k": "alert", "a": "accept", "t": "tok", "band": "z"}',
        '{"k": "bands"}',
        '{"k": "bands", "a": "noop"}',
    ]
    for raw in cases:
        assert callbacks.decode(raw) is None


# --- PendingStore -------------------------------------------------------------


def test_put_pop_by_kind() -> None:
    store = pending.PendingStore()
    kind_adv: PendingKind = "adv"
    kind_alert: PendingKind = "alert"

    first_payload: BandMap = {"a": (1.0, 2.0)}
    token = store.put(kind_adv, first_payload)
    assert store.pop(kind_alert, token) is None

    second_payload: BandMap = {"b": (2.0, 3.0)}
    token2 = store.put(kind_adv, second_payload)
    popped = store.pop(kind_adv, token2)
    assert popped is not None
    assert popped.payload == second_payload
    assert store.pop(kind_adv, token2) is None


def test_ttl_eviction_on_access(monkeypatch: pytest.MonkeyPatch) -> None:
    store = pending.PendingStore(ttl_s=10)
    now_time(monkeypatch, 100.0)
    token = store.put("adv", {"a": (1.0, 2.0)})

    now_time(monkeypatch, 111.0)
    assert store.pop("adv", token) is None
    assert store.pop("adv", token) is None


def test_capacity_evicts_oldest() -> None:
    store = pending.PendingStore(cap=2, ttl_s=1000)
    kind_adv: PendingKind = "adv"

    payloads: list[BandMap] = [
        {"a": (1.0, 2.0)},
        {"b": (2.0, 3.0)},
        {"c": (3.0, 4.0)},
    ]
    t1 = store.put(kind_adv, payloads[0])
    t2 = store.put(kind_adv, payloads[1])
    t3 = store.put(kind_adv, payloads[2])

    assert store.pop(kind_adv, t1) is None
    assert store.pop(kind_adv, t2) is not None
    assert store.pop(kind_adv, t3) is not None


# --- Render -------------------------------------------------------------------


def test_adv_kb_layout() -> None:
    markup = render.adv_kb("token")
    assert len(markup.inline_keyboard) == 1
    labels = [button.text for button in markup.inline_keyboard[0]]
    assert labels == ["Apply All", "Set Exact", "Ignore"]


def test_alert_kb_layout() -> None:
    markup = render.alert_kb("a", "token")
    assert len(markup.inline_keyboard) == 1
    labels = [button.text for button in markup.inline_keyboard[0]]
    assert labels == ["Apply", "Ignore", "Set Exact"]


def test_sigma_summary_strings_and_stale_tag() -> None:
    reading = VolReading(sigma_pct=1.2345, bucket="high", stale=True)
    line, bucket, pct = render.sigma_summary(reading)
    assert line == "<b>Volatility</b>: 1.23% (High) [<i>Stale</i>]"
    assert bucket == "high"
    assert pct == pytest.approx(1.2345)

    line_none, bucket_none, pct_none = render.sigma_summary(None)
    assert line_none == "<b>Volatility</b>: – (Mid)"
    assert bucket_none == "mid"
    assert pct_none is None


def test_bands_menu_text_and_kb_layout() -> None:
    bands: BandMap = {"a": (90.0, 110.0), "b": (95.0, 115.0), "c": (100.0, 120.0)}
    text = render.bands_menu_text(bands)
    for name in BAND_ORDER:
        assert f"{name.upper()}:" in text
    assert text.splitlines()[0] == "<b>Configured Bands</b>:"

    markup = render.bands_menu_kb()
    assert len(markup.inline_keyboard) == len(BAND_ORDER) + 1
    for idx, name in enumerate(BAND_ORDER):
        button = markup.inline_keyboard[idx][0]
        assert button.text == f"Edit {name.upper()}"
    assert markup.inline_keyboard[-1][0].text == "Back"


# --- Handlers -----------------------------------------------------------------


def make_bot_ctx(capture: Capture) -> BotCtx:
    async def send(text: str, markup: object | None) -> None:
        capture.send(text, markup)

    async def edit_or_send(_: object, text: str, markup: object | None) -> None:
        capture.send(text, markup)

    async def edit_by_id(_: int, text: str) -> bool:
        capture.send(text, None)
        return True

    return BotCtx(send, edit_or_send, edit_by_id)


def make_handlers(
    repo: FakeRepo,
    *,
    price: float | None = None,
    sigma: VolReading | None = None,
) -> Handlers:
    async def price_provider() -> float | None:
        return price

    async def sigma_provider() -> VolReading | None:
        return sigma

    return Handlers(
        repo=repo,
        providers=Providers(
            price_provider=price_provider,
            sigma_provider=sigma_provider,
        ),
    )


def test_extract_numbers_ignores_labels_and_nonfinite() -> None:
    text = "/cmd sol 1 usdc 2 NaN 3e2 inf 4"
    numbers = Handlers._extract_numbers(text, 3, ignore_labels={"sol", "usdc"})
    assert numbers == [1.0, 2.0, 300.0]


def test_parse_tilt_accepts_ratio_fraction_and_clamps() -> None:
    repo = FakeRepo()
    handlers = make_handlers(repo)
    ratio = handlers._parse_tilt("60:40")
    assert ratio == pytest.approx((0.6, 0.4))

    single = handlers._parse_tilt("0.6")
    assert single == pytest.approx((0.6, 0.4))

    clamped = handlers._parse_tilt("-10:20")
    assert clamped == pytest.approx((0.0, 1.0))


@pytest.mark.asyncio
async def test_cmd_status_summarizes_state() -> None:
    repo = FakeRepo()
    repo.tilt = 0.6
    repo.notional = 1500.0
    repo.baseline = Baseline(10.0, 5000.0, 100)
    repo.snapshot = Snapshot(200, 12.0, 4000.0, 105.0, -790.0)

    handlers = make_handlers(
        repo,
        price=105.0,
        sigma=VolReading(sigma_pct=0.85, bucket="mid"),
    )
    capture = Capture()
    await handlers.cmd_status(make_bot_ctx(capture))

    assert len(capture.messages) == 1
    message = capture.messages[0][0]
    assert "<b>Price</b>: 105.00" in message
    assert "<b>Volatility</b>: 0.85% (Mid)" in message
    for name in BAND_ORDER:
        assert f"{name.upper()}:" in message
        assert f"{name.upper()} amount:" in message
    assert "<b>Advisory Split</b>: 50/30/20 (Mid)" in message
    assert "notional: $1500.00 | tilt sol/usdc: 60/40" in message
    assert "<b>Balances</b>: 12 SOL, 4000 USDC" in message
    assert "<b>Drift</b>:" in message


@pytest.mark.asyncio
async def test_cmd_setbaseline_usage_and_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    handlers = make_handlers(repo)
    capture = Capture()
    bot = make_bot_ctx(capture)

    await handlers.cmd_setbaseline(bot, "junk input")
    assert capture.messages[-1][0] == handlers_mod._USAGE_SETBASELINE
    assert repo.baseline is None

    now_time(monkeypatch, 1234.0)
    await handlers.cmd_setbaseline(bot, "/setbaseline 3 sol 200 usdc")
    assert capture.messages[-1][0] == "[<i>Applied</i>] Baseline → 3 SOL, 200 USDC"
    assert repo.baseline is not None
    assert repo.baseline.sol == 3
    assert repo.baseline.usdc == 200
    assert repo.baseline.ts == 1234


@pytest.mark.asyncio
async def test_cmd_setnotional_usage_and_validation() -> None:
    repo = FakeRepo()
    handlers = make_handlers(repo)
    capture = Capture()
    bot = make_bot_ctx(capture)

    await handlers.cmd_setnotional(bot, "/setnotional -10")
    assert capture.messages[-1][0] == handlers_mod._USAGE_SETNOTIONAL

    await handlers.cmd_setnotional(bot, "/setnotional 2500")
    assert capture.messages[-1][0] == "[<i>Applied</i>] Notional → $2500.00"
    assert repo.notional == pytest.approx(2500.0)


@pytest.mark.asyncio
async def test_cmd_settilt_usage_and_clamping() -> None:
    repo = FakeRepo()
    handlers = make_handlers(repo)
    capture = Capture()
    bot = make_bot_ctx(capture)

    await handlers.cmd_settilt(bot, "invalid")
    assert capture.messages[-1][0] == handlers_mod._USAGE_SETTILT

    await handlers.cmd_settilt(bot, "60:40")
    assert capture.messages[-1][0] == "[<i>Applied</i>] Tilt → SOL/USDC = 60/40"
    assert repo.tilt == pytest.approx(0.6)

    await handlers.cmd_settilt(bot, "0.6")
    assert capture.messages[-1][0] == "[<i>Applied</i>] Tilt → SOL/USDC = 60/40"
    assert repo.tilt == pytest.approx(0.6)

    await handlers.cmd_settilt(bot, "-10:20")
    assert capture.messages[-1][0] == "[<i>Applied</i>] Tilt → SOL/USDC = 0/100"
    assert repo.tilt == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cmd_updatebalances_requires_price_and_baseline_then_records_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepo()
    capture = Capture()
    bot = make_bot_ctx(capture)

    handlers_price_missing = make_handlers(repo, price=None)
    await handlers_price_missing.cmd_updatebalances(bot, "/updatebalances 1 sol 2 usdc")
    assert capture.messages[-1][0] == "[<i>Error</i>] Price unavailable. Try again later."
    assert repo.snapshot is None

    handlers_baseline_missing = make_handlers(repo, price=100.0)
    await handlers_baseline_missing.cmd_updatebalances(bot, "/updatebalances 1 sol 2 usdc")
    assert (
        capture.messages[-1][0]
        == "[<i>Error</i>] Baseline not set. Run <code>/setbaseline</code> first."
    )
    assert repo.snapshot is None

    repo.baseline = Baseline(1.0, 2.0, 10)
    now_time(monkeypatch, 2000.0)
    await handlers_baseline_missing.cmd_updatebalances(bot, "/updatebalances 1 sol 2 usdc")
    message = capture.messages[-1][0]
    assert message.startswith("[<i>Applied</i>] <b>Drift</b>:")
    assert "@ Price <b>100.00</b>" in message
    assert repo.snapshot is not None
    assert repo.snapshot.sol == pytest.approx(1.0)
    assert repo.snapshot.usdc == pytest.approx(2.0)
    assert repo.snapshot.price == pytest.approx(100.0)
    assert repo.snapshot.ts == 2000
