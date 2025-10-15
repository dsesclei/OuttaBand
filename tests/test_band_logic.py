from outtaband.band_logic import broken_bands, fmt_range, format_advisory_card
from outtaband.policy import band_advisor


def test_fmt_range_uses_en_dash() -> None:
    result = fmt_range(1.0, 2.0)
    assert "–" in result
    assert result == "1.00–2.00"


def test_broken_bands_detection() -> None:
    bands = {
        "a": (90.0, 110.0),
        "b": (80.0, 120.0),
        "c": (95.0, 130.0),
    }

    assert broken_bands(100.0, bands) == set()
    assert broken_bands(115.0, bands) == {"a"}
    assert broken_bands(125.0, bands) == {"a", "b"}
    assert broken_bands(70.0, bands) == {"a", "b", "c"}


def test_format_advisory_card_fields() -> None:
    price = 123.456
    sigma = 0.789
    bucket = "mid"
    split = band_advisor.split_for_bucket(bucket)
    ranges = band_advisor.ranges_for_price(price, bucket, include_a_on_high=True)
    amounts = {
        band: (round(0.123456 * (idx + 1), 6), round(10.0 * (idx + 1), 2))
        for idx, band in enumerate(ranges)
    }

    message = format_advisory_card(
        price,
        sigma,
        bucket,
        ranges,
        split,
        stale=True,
        amounts=amounts,
        unallocated_usd=12.34,
    )

    header_line = message.splitlines()[0]
    assert "P=<b>123.46</b>" in header_line
    assert "σ=<b>0.79%</b>" in header_line
    assert "(Mid)" in header_line
    assert "Split <b>50/30/20</b>" in header_line
    assert "[<i>Stale</i>]" in header_line

    widths, _ = band_advisor.widths_for_bucket(bucket)
    for band, (lo, hi) in ranges.items():
        pct = widths[band] * 100
        band_line = next(
            line for line in message.splitlines() if line.startswith(f"<b>{band.upper()}</b>")
        )
        assert f"±{pct:.2f}%" in band_line
        assert fmt_range(lo, hi) in band_line
        sol_amt, usdc_amt = amounts[band]
        assert f"{sol_amt:.6f} SOL" in band_line
        assert f"${usdc_amt:.2f} USDC" in band_line

    assert "<i>Unallocated</i>: $12.34" in message

    # When amounts/unallocated are omitted, formatting should drop those fragments.
    bare = format_advisory_card(price, None, bucket, ranges, split)
    assert "σ=<b>–</b>" in bare
    assert "SOL / $" not in bare
    assert "Unallocated" not in bare
