import math
import pytest
from hypothesis import given, strategies as st

import band_advisor as ba


@given(
    price=st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
    notional=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    tilt=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_redistribute_conserves_notional(price, notional, tilt):
    split = ba.split_for_bucket("high")
    ranges = ba.ranges_for_price(price, "high", include_a_on_high=False)
    amounts, unalloc = ba.compute_amounts(
        price, split, ranges, notional, tilt_sol_frac=tilt, redistribute_skipped=True
    )
    total = sum(sol * price + usd for sol, usd in amounts.values())
    assert math.isfinite(total)
    assert math.isclose(total, notional, rel_tol=0.03)
    assert math.isclose(unalloc, 0.0, abs_tol=1e-6)
