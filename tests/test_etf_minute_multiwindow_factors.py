from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.microstructure import MarketImpactFactor
from factor.price import (
    RollingOBVFactor,
    RollingVolumePriceTrendFactor,
    VolumePriceConfirmRateFactor,
)
from scripts.generate_etf_minute_factors import (
    DEFAULT_MULTIWINDOW_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    build_factor_specs,
    calculate_factor_frame,
    resolve_output_root,
)


def _minute_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    steps = np.arange(len(index), dtype=float)
    close = 100.0 + steps * 0.02 + np.sin(steps / 3.0) * 0.05
    volume = 1_000.0 + (steps % 17) * 25.0
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.03,
            "low": close - 0.03,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_factor_profiles_have_expected_unique_columns() -> None:
    base_names = [spec.output_name for spec in build_factor_specs("base")]
    multi_names = [spec.output_name for spec in build_factor_specs("multi")]

    assert len(base_names) == len(set(base_names)) == 52
    assert len(multi_names) == len(set(multi_names)) == 156
    assert multi_names[:52] == base_names
    assert {
        "returns_w3",
        "macd_f6_s13_sig5",
        "obv_delta_w20",
        "market_impact_w10",
        "liquidity_migration_w80",
    } <= set(multi_names)


def test_profile_default_output_roots_are_separate() -> None:
    assert resolve_output_root(None, "base") == DEFAULT_OUTPUT_ROOT
    assert (
        resolve_output_root(None, "multi")
        == DEFAULT_MULTIWINDOW_OUTPUT_ROOT
    )


def test_rolling_price_volume_factors_use_complete_causal_windows() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=8, freq="min")
    close = pd.Series(np.arange(100.0, 108.0), index=index)
    volume = pd.Series(np.arange(1.0, 9.0), index=index)
    frame = pd.DataFrame({"close": close, "volume": volume})

    obv = RollingOBVFactor(window=3).calculate(frame)
    vpt = RollingVolumePriceTrendFactor(window=3).calculate(frame)
    confirm = VolumePriceConfirmRateFactor(window=3).calculate(frame)

    assert obv.iloc[:3].isna().all()
    assert obv.iloc[3] == 2.0 + 3.0 + 4.0
    expected_vpt = sum(
        volume.iloc[i] * (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
        for i in range(1, 4)
    )
    assert np.isclose(vpt.iloc[3], expected_vpt)
    assert confirm.iloc[:3].isna().all()
    assert confirm.iloc[3] == 1.0


def test_market_impact_default_is_legacy_compatible() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=60, freq="min")
    frame = _minute_frame(index)

    actual = MarketImpactFactor(window=50).calculate(frame)
    order_flow = frame["volume"] * np.sign(frame["close"].diff())
    flow_sum = order_flow.rolling(5, min_periods=5).sum()
    price_std = frame["close"].rolling(10, min_periods=10).std(ddof=0)
    expected = flow_sum / (price_std * 100 + 1e-8)
    expected = expected.where(price_std > 0, 0.0)
    valid = frame["close"].notna().rolling(50, min_periods=50).sum().eq(50)
    expected = expected.where(valid)

    pd.testing.assert_series_equal(actual, expected)


def test_short_market_impact_variant_does_not_wait_for_fifty_bars() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=30, freq="min")
    frame = _minute_frame(index)

    short = MarketImpactFactor(
        window=10, flow_window=5, volatility_window=10
    ).calculate(frame)
    legacy = MarketImpactFactor(window=50).calculate(frame)

    assert short.iloc[:9].isna().all()
    assert short.iloc[9:].notna().all()
    assert legacy.isna().all()


def test_multi_profile_smoke_preserves_base_columns_and_resets_daily() -> None:
    first_day = pd.date_range("2026-01-05 09:30:00", periods=90, freq="min")
    second_day = pd.date_range("2026-01-06 09:30:00", periods=90, freq="min")
    frame = _minute_frame(first_day.append(second_day))

    base = calculate_factor_frame(frame, "base")
    multi = calculate_factor_frame(frame, "multi")
    base_factor_names = [spec.output_name for spec in build_factor_specs("base")]

    assert len(base.columns) == len(frame.columns) + 52
    assert len(multi.columns) == len(frame.columns) + 156
    pd.testing.assert_frame_equal(
        multi[base_factor_names], base[base_factor_names]
    )
    assert multi.loc[first_day[:20], "liquidity_ratio_w30"].isna().all()
    assert multi.loc[second_day[:20], "liquidity_ratio_w30"].isna().all()
    assert multi.loc[first_day[29], "liquidity_ratio_w30"] == multi.loc[
        first_day[29], "liquidity_ratio_w30"
    ]
    assert multi.loc[second_day[29], "liquidity_ratio_w30"] == multi.loc[
        second_day[29], "liquidity_ratio_w30"
    ]
