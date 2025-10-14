from policy import band_advisor, engine
from shared_types import BandMap, Bucket


def test_compute_breaches_none_when_inside() -> None:
    price = 100.0
    bucket: Bucket = "mid"
    ranges = band_advisor.ranges_for_price(price, bucket, include_a_on_high=True)

    breaches = engine.compute_breaches(price, ranges, bucket, include_a_on_high=True)

    assert breaches == []


def test_compute_breaches_low_and_high() -> None:
    price = 100.0
    bucket: Bucket = "mid"
    policy_ranges = band_advisor.ranges_for_price(price, bucket, include_a_on_high=True)
    widths, _ = band_advisor.widths_for_bucket(bucket)

    bands: BandMap = {
        "a": (policy_ranges["a"][0] + 5.0, policy_ranges["a"][1] + 5.0),  # price below low
        "b": (policy_ranges["b"][0] - 15.0, policy_ranges["b"][1] - 15.0),  # price above high
        "c": policy_ranges["c"],
    }

    breaches = engine.compute_breaches(price, bands, bucket, include_a_on_high=True)

    assert len(breaches) == 2

    first = breaches[0]
    assert first.band == "a"
    assert first.side == "low"
    assert first.current == bands["a"]
    assert first.suggested == policy_ranges["a"]
    assert first.policy_meta == (bucket, widths["a"])

    second = breaches[1]
    assert second.band == "b"
    assert second.side == "high"
    assert second.current == bands["b"]
    assert second.suggested == policy_ranges["b"]
    assert second.policy_meta == (bucket, widths["b"])


def test_suggest_ranges_passthrough() -> None:
    price = 250.0
    bucket: Bucket = "high"

    suggested = engine.suggest_ranges(price, bucket, include_a_on_high=True)
    direct = band_advisor.ranges_for_price(price, bucket, include_a_on_high=True)
    assert suggested == direct
