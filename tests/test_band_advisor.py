import band_advisor


def test_ranges_for_price_skips_a_in_high_bucket_by_default() -> None:
    price = 100.0
    ranges = band_advisor.ranges_for_price(price, "high")
    assert "a" not in ranges
    assert "b" in ranges
    assert "c" in ranges


def test_ranges_for_price_includes_a_when_requested() -> None:
    price = 100.0
    ranges = band_advisor.ranges_for_price(price, "high", include_a_on_high=True)
    assert {"a", "b", "c"}.issubset(ranges.keys())
