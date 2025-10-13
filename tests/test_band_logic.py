from band_logic import broken_bands
from shared_types import BAND_ORDER


def test_broken_bands_basic() -> None:
    bands = {
        BAND_ORDER[0]: (24.0, 26.0),
        BAND_ORDER[1]: (22.0, 28.0),
        BAND_ORDER[2]: (20.0, 30.0),
    }

    assert broken_bands(25.0, bands) == set()
    assert broken_bands(23.5, bands) == {BAND_ORDER[0]}

    below = broken_bands(21.5, bands)
    assert below == {BAND_ORDER[0], BAND_ORDER[1]}

    above = broken_bands(26.5, bands)
    assert above == {BAND_ORDER[0]}
