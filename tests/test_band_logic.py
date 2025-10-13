from band_logic import broken_bands


def test_broken_bands_basic() -> None:
    bands = {
        "a": (24.0, 26.0),
        "b": (22.0, 28.0),
        "c": (20.0, 30.0),
    }

    assert broken_bands(25.0, bands) == set()
    assert broken_bands(23.5, bands) == {"a"}

    below = broken_bands(21.5, bands)
    assert below == {"a", "b"}

    above = broken_bands(26.5, bands)
    assert above == {"a"}
