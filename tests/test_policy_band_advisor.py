import math
from typing import cast

import pytest

from policy import band_advisor
from shared_types import BAND_ORDER, Bucket, BucketSplit


def test_widths_for_bucket_happy_paths() -> None:
    skip_flags: dict[str, bool] = {}
    for bucket_name in ("low", "mid", "high"):
        bucket = cast(Bucket, bucket_name)
        widths, skip_a = band_advisor.widths_for_bucket(bucket)
        assert set(widths.keys()) == set(BAND_ORDER)
        assert all(width > 0 and math.isfinite(width) for width in widths.values())
        skip_flags[bucket] = skip_a

    assert skip_flags["high"] is True
    assert skip_flags["low"] is False
    assert skip_flags["mid"] is False


def test_ranges_for_price_respects_skip_a_on_high_default() -> None:
    price = 100.0
    default_high = band_advisor.ranges_for_price(price, "high")
    assert "a" not in default_high
    assert {"b", "c"} <= set(default_high.keys())

    explicit_high = band_advisor.ranges_for_price(price, "high", include_a_on_high=True)
    assert set(explicit_high.keys()) == set(BAND_ORDER)


@pytest.mark.parametrize("bucket_name", ["low", "mid", "high"])
def test_ranges_for_price_math(bucket_name: str) -> None:
    bucket = cast(Bucket, bucket_name)
    price = 137.42
    ranges = band_advisor.ranges_for_price(price, bucket, include_a_on_high=True)
    widths, _ = band_advisor.widths_for_bucket(bucket)

    for band, (lo, hi) in ranges.items():
        width = widths[band]
        expected_delta = price * width
        assert math.isclose(lo, price - expected_delta, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(hi, price + expected_delta, rel_tol=0, abs_tol=1e-9)
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo < hi


@pytest.mark.parametrize(
    "sigma_pct,expected",
    [
        (None, (60, 20, 20)),
        (0.0, (60, 20, 20)),
        (0.59, (60, 20, 20)),
        (0.6, (50, 30, 20)),
        (1.2, (50, 30, 20)),
    ],
)
def test_split_for_sigma_thresholds(sigma_pct: float | None, expected: BucketSplit) -> None:
    assert band_advisor.split_for_sigma(sigma_pct) == expected


def test_split_for_bucket_table() -> None:
    assert band_advisor.split_for_bucket("low") == (60, 20, 20)
    assert band_advisor.split_for_bucket("mid") == (50, 30, 20)
    assert band_advisor.split_for_bucket("high") == (50, 30, 20)


def test_compute_amounts_nominal() -> None:
    price = 100.0
    notional = 1000.0
    tilt = 0.6
    split: BucketSplit = (60, 20, 20)
    ranges = {"b": (90.0, 110.0), "c": (80.0, 120.0)}

    amounts, unallocated = band_advisor.compute_amounts(price, split, ranges, notional, tilt)
    assert amounts == {
        "b": (1.2, 80.0),
        "c": (1.2, 80.0),
    }
    assert unallocated == pytest.approx(600.0, abs=1e-6)

    redistributed, redistributed_unalloc = band_advisor.compute_amounts(
        price,
        split,
        ranges,
        notional,
        tilt,
        redistribute_skipped=True,
    )
    assert redistributed == {
        "b": (3.0, 200.0),
        "c": (3.0, 200.0),
    }
    assert redistributed_unalloc == pytest.approx(0.0, abs=1e-9)


def test_compute_amounts_monotone_in_notional() -> None:
    price = 150.0
    split: BucketSplit = (60, 20, 20)
    ranges = band_advisor.ranges_for_price(price, cast(Bucket, "mid"), include_a_on_high=True)

    amounts1, unalloc1 = band_advisor.compute_amounts(price, split, ranges, 1000.0, 0.55)
    amounts2, unalloc2 = band_advisor.compute_amounts(price, split, ranges, 2000.0, 0.55)

    assert unalloc1 == pytest.approx(0.0, abs=1e-9)
    assert unalloc2 == pytest.approx(0.0, abs=1e-9)

    for band in BAND_ORDER:
        sol1, usdc1 = amounts1[band]
        sol2, usdc2 = amounts2[band]
        assert math.isclose(sol2, sol1 * 2, abs_tol=1e-6)
        assert math.isclose(usdc2, usdc1 * 2, abs_tol=0.01)


@pytest.mark.parametrize("tilt", [0.0, 1.0])
def test_compute_amounts_tilt_boundaries(tilt: float) -> None:
    price = 200.0
    split: BucketSplit = (60, 20, 20)
    ranges = band_advisor.ranges_for_price(price, cast(Bucket, "low"), include_a_on_high=True)
    notional = 300.0

    amounts, _ = band_advisor.compute_amounts(price, split, ranges, notional, tilt)
    for band in BAND_ORDER:
        sol_amt, usdc_amt = amounts[band]
        if tilt == 0.0:
            assert sol_amt == pytest.approx(0.0, abs=1e-9)
            assert usdc_amt > 0.0
        else:
            assert sol_amt > 0.0
            assert usdc_amt == pytest.approx(0.0, abs=0.01)
