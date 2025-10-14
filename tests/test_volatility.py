import math
import random

import pytest

from policy.volatility import compute_sigma_from_closes


def _generate_gbm_prices(count: int, *, seed: int = 42, drift: float = 0.0, vol: float = 0.002) -> list[float]:
    random.seed(seed)
    price = 20.0
    closes: list[float] = []
    for _ in range(count):
        closes.append(price)
        ret = random.gauss(drift, vol)
        price *= math.exp(ret)
    return closes


def test_compute_sigma_from_closes_flat_prices() -> None:
    closes = [10.0] * 61
    result = compute_sigma_from_closes(closes)
    assert result is not None
    sigma_pct, sample_count = result
    assert sample_count == 60
    assert sigma_pct == pytest.approx(0.0, abs=1e-9)


def test_compute_sigma_from_closes_random_series() -> None:
    closes = _generate_gbm_prices(61)
    result = compute_sigma_from_closes(closes)
    assert result is not None
    sigma_pct, sample_count = result
    assert sample_count == 60
    assert math.isfinite(sigma_pct)
    assert sigma_pct > 0.0


@pytest.mark.parametrize(
    "closes",
    [
        [10.0] * 60,  # insufficient length
        [0.0] + [10.0] * 60,  # zero price
        [10.0] * 30 + [float("nan")] + [10.0] * 30,  # non-finite price
    ],
)
def test_compute_sigma_from_closes_invalid(closes: list[float]) -> None:
    assert compute_sigma_from_closes(closes) is None
