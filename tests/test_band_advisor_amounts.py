import math

import band_advisor


def test_compute_amounts_without_redistribute_keeps_unallocated_share() -> None:
    price = 100.0
    notional = 1000.0
    split = band_advisor.split_for_bucket("high")
    ranges = band_advisor.ranges_for_price(price, "high", include_a_on_high=False)
    amounts, unallocated = band_advisor.compute_amounts(
        price,
        split,
        ranges,
        notional,
        tilt_sol_frac=0.5,
        redistribute_skipped=False,
    )

    assert "a" not in amounts
    assert unallocated > 0.0

    sol_usd = sum(sol * price for sol, _ in amounts.values())
    usdc = sum(usdc_amt for _, usdc_amt in amounts.values())
    total_placed = sol_usd + usdc
    assert total_placed < notional
    assert math.isclose(total_placed + unallocated, notional, rel_tol=0.02)


def test_compute_amounts_with_redistribute_fills_missing_share() -> None:
    price = 100.0
    notional = 1000.0
    split = band_advisor.split_for_bucket("high")
    ranges = band_advisor.ranges_for_price(price, "high", include_a_on_high=False)
    amounts, unallocated = band_advisor.compute_amounts(
        price,
        split,
        ranges,
        notional,
        tilt_sol_frac=0.6,
        redistribute_skipped=True,
    )

    assert "a" not in amounts
    assert math.isclose(unallocated, 0.0, abs_tol=1e-6)

    total_usd = sum(sol_amt * price + usdc_amt for sol_amt, usdc_amt in amounts.values())
    assert math.isclose(total_usd, notional, rel_tol=0.02)
