from __future__ import annotations

import numpy as np
import pandas as pd

from factor.fractional import FractionalDiffLogCloseFactor


def _minute_frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 1_000.0),
        },
        index=index,
    )


def test_fractional_diff_is_price_level_invariant_and_causal() -> None:
    factor = FractionalDiffLogCloseFactor(order=0.4, threshold=1e-3)
    index = pd.date_range("2026-01-05 09:30:00", periods=60, freq="min")
    close = np.full(60, 100.0)
    baseline = factor.calculate(_minute_frame(index, close))

    assert factor.name == "fractional_diff_log_close_d04"
    assert factor.window == 50
    assert baseline.iloc[: factor.window - 1].isna().all()
    assert np.allclose(baseline.iloc[factor.window - 1 :], 0.0)

    changed_close = close.copy()
    changed_close[factor.window] = 120.0
    changed = factor.calculate(_minute_frame(index, changed_close))
    assert np.isclose(changed.iloc[factor.window - 1], baseline.iloc[factor.window - 1])


def test_fractional_diff_warmup_is_restarted_per_day() -> None:
    from scripts.generate_etf_minute_factors import calculate_factor_frame

    first_day = pd.date_range("2026-01-05 09:30:00", periods=60, freq="min")
    second_day = pd.date_range("2026-01-06 09:30:00", periods=60, freq="min")
    index = first_day.append(second_day)
    result = calculate_factor_frame(_minute_frame(index, np.full(len(index), 100.0)))
    values = result["fractional_diff_log_close_d04"]

    assert values.iloc[:49].isna().all()
    assert values.iloc[49] == 0.0
    assert values.iloc[60:109].isna().all()
    assert values.iloc[109] == 0.0
