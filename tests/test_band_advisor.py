import band_advisor
from shared_types import BAND_ORDER


def test_ranges_for_price_skips_a_in_high_bucket_by_default() -> None:
    price = 100.0
    ranges = band_advisor.ranges_for_price(price, "high")
    assert BAND_ORDER[0] not in ranges
    assert BAND_ORDER[1] in ranges
    assert BAND_ORDER[2] in ranges


def test_ranges_for_price_includes_a_when_requested() -> None:
    price = 100.0
    ranges = band_advisor.ranges_for_price(price, "high", include_a_on_high=True)
    assert set(BAND_ORDER).issubset(ranges.keys())
