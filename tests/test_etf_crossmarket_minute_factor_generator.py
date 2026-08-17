from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_etf_crossmarket_minute_factors import (
    FACTOR_COLUMNS,
    calculate_benchmark_factor_frame,
    merge_factor_columns,
    process_reference_group,
)


def _frames(days: tuple[str, ...], periods: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{day} 09:30:00", periods=periods, freq="min")
                .to_numpy()
                for day in days
            ]
        ),
        name="trade_time",
    )
    steps = np.arange(len(index), dtype=float)
    etf_close = 1.0 * np.exp(
        0.0003 * steps + 0.001 * np.sin(steps / 5.0)
    )
    reference_close = 3_000.0 * np.exp(
        0.0002 * steps + 0.0013 * np.cos(steps / 7.0)
    )
    return (
        pd.DataFrame({"close": etf_close}, index=index),
        pd.DataFrame({"close": reference_close}, index=index),
    )


def _raw_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index()
    result["open"] = result["close"]
    result["high"] = result["close"]
    result["low"] = result["close"]
    result["vol"] = 1_000.0
    result["amount"] = 10_000.0
    result["trade_date"] = result["trade_time"].dt.normalize()
    return result


def test_benchmark_factors_use_lagged_beta_and_expected_windows() -> None:
    etf, reference = _frames(("2026-01-05",))
    factors = calculate_benchmark_factor_frame(etf, reference)
    point = factors.index[60]
    etf_returns = np.log(etf["close"]).diff()
    reference_returns = np.log(reference["close"]).diff()
    expected_beta = etf_returns.shift(1).rolling(60, min_periods=30).cov(
        reference_returns.shift(1)
    ).loc[point] / reference_returns.shift(1).rolling(
        60, min_periods=30
    ).var().loc[point]
    residual = etf_returns - factors["rolling_market_beta_60m"] * reference_returns
    expected_momentum = residual.rolling(20, min_periods=20).sum().loc[point]
    expected_zscore = (
        residual
        - residual.rolling(60, min_periods=30).mean()
    ) / residual.rolling(60, min_periods=30).std()
    expected_corr = etf_returns.rolling(60, min_periods=30).corr(
        reference_returns
    )

    assert tuple(factors.columns) == FACTOR_COLUMNS
    assert factors["rolling_market_beta_60m"].iloc[:31].isna().all()
    assert np.isclose(factors.loc[point, "rolling_market_beta_60m"], expected_beta)
    assert np.isclose(
        factors.loc[point, "beta_residual_momentum_20m"], expected_momentum
    )
    assert np.isclose(
        factors.loc[point, "beta_residual_zscore_60m"], expected_zscore.loc[point]
    )
    assert np.isclose(
        factors.loc[point, "benchmark_correlation_60m"], expected_corr.loc[point]
    )


def test_benchmark_factors_do_not_use_future_rows_and_reset_daily() -> None:
    etf, reference = _frames(("2026-01-05", "2026-01-06"))
    changed_reference = reference.copy()
    changed_reference.loc[changed_reference.index[65]:, "close"] *= 1.3

    original = calculate_benchmark_factor_frame(etf, reference)
    changed = calculate_benchmark_factor_frame(etf, changed_reference)

    pd.testing.assert_frame_equal(original.iloc[:65], changed.iloc[:65])
    second_day = original.loc["2026-01-06"]
    assert second_day["rolling_market_beta_60m"].iloc[:31].isna().all()
    assert second_day["beta_residual_momentum_20m"].iloc[:50].isna().all()
    assert second_day["beta_residual_zscore_60m"].iloc[:60].isna().all()
    assert second_day["benchmark_correlation_60m"].iloc[:30].isna().all()


def test_benchmark_factors_require_common_minutes_and_nonconstant_benchmark() -> None:
    etf, reference = _frames(("2026-01-05",))
    missing_time = reference.index[35]
    factors = calculate_benchmark_factor_frame(etf, reference.drop(missing_time))

    assert missing_time not in factors.index

    constant_reference = reference.copy()
    constant_reference["close"] = 3_000.0
    constant = calculate_benchmark_factor_frame(etf, constant_reference)
    assert constant.isna().all().all()


def test_merge_preserves_existing_columns_and_obeys_overwrite() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=4, freq="min")
    existing = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2, 1.3],
            "existing_crossmarket": [10.0, 11.0, 12.0, 13.0],
            "rolling_market_beta_60m": [99.0, np.nan, np.nan, np.nan],
        },
        index=index,
    )
    incoming = pd.DataFrame(1.5, index=index[1:3], columns=FACTOR_COLUMNS)

    appended = merge_factor_columns(existing, incoming, overwrite=False)
    assert appended["existing_crossmarket"].equals(
        existing["existing_crossmarket"]
    )
    assert appended["rolling_market_beta_60m"].iloc[0] == 99.0
    assert appended["rolling_market_beta_60m"].iloc[1] == 1.5
    assert appended["rolling_market_beta_60m"].iloc[3] != appended[
        "rolling_market_beta_60m"
    ].iloc[3]

    replaced = merge_factor_columns(existing, incoming, overwrite=True)
    assert replaced["rolling_market_beta_60m"].iloc[0] == 99.0
    assert replaced["rolling_market_beta_60m"].iloc[1] == 1.5

    cleared = merge_factor_columns(
        existing, incoming, overwrite=True, update_index=index
    )
    assert pd.isna(cleared["rolling_market_beta_60m"].iloc[0])
    assert pd.isna(cleared["rolling_market_beta_60m"].iloc[3])


def test_process_group_merges_only_existing_crossmarket_outputs(
    tmp_path: Path,
) -> None:
    etf, reference = _frames(("2026-01-05",))
    etf_root = tmp_path / "etf"
    output_root = tmp_path / "crossmarket"
    etf_root.mkdir()
    output_root.mkdir()
    etf_path = etf_root / "159001.SZ.parquet"
    reference_path = tmp_path / "000300.SH.parquet"
    output_path = output_root / "159001.SZ.parquet"
    _raw_minute_frame(etf).to_parquet(etf_path)
    _raw_minute_frame(reference).to_parquet(reference_path)
    pd.DataFrame(
        {"close": etf["close"], "existing_crossmarket": 7.0}, index=etf.index
    ).to_parquet(output_path)

    results = process_reference_group(
        "000300.SH",
        ("159001", "159002"),
        {"159001": etf_path, "159002": etf_path},
        {"159001": output_path},
        reference_path,
        None,
        None,
        False,
    )

    assert {result["status"] for result in results} == {
        "written",
        "missing_crossmarket_output",
    }
    output = pd.read_parquet(output_path)
    assert output["existing_crossmarket"].eq(7.0).all()
    assert output["rolling_market_beta_60m"].notna().any()
