from __future__ import annotations

import numpy as np
import pandas as pd

from factor.regime import (
    build_daily_market_feature_panel,
    calculate_causal_daily_market_regime_features,
)
from scripts.generate_etf_regime_daily_factors import (
    OUTPUT_COLUMNS,
    OUTPUT_FILENAME,
    build_market_regime_daily_frame,
    load_reference_close_panel,
    write_output,
)


def _market_result(periods: int = 140) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(11)
    closes = pd.DataFrame(
        {
            f"reference_{position}": 100.0
            * np.exp(np.cumsum(rng.normal(0.0, 0.008, periods)))
            for position in range(4)
        },
        index=dates,
    )
    features = build_daily_market_feature_panel(closes)
    return dates, calculate_causal_daily_market_regime_features(
        features, training_days=80, min_training_days=50, refit_days=10
    )


def test_generator_writes_one_shared_regime_row_per_market_date() -> None:
    dates, regime = _market_result()
    result = build_market_regime_daily_frame(regime, dates)

    assert result.columns.tolist() == list(OUTPUT_COLUMNS)
    assert len(result) == len(dates) - 1
    assert result["trade_date"].is_unique
    assert (result["source_trade_date"] < result["trade_date"]).all()
    assert (
        result["available_time"]
        == result["trade_date"] + pd.Timedelta(hours=9, minutes=15)
    ).all()


def test_generator_date_filter_and_empty_intersection_are_schema_stable() -> None:
    dates, regime = _market_result()
    result = build_market_regime_daily_frame(
        regime, dates, date_from="2024-03-01", date_to="2024-03-15"
    )
    assert result["trade_date"].min() >= pd.Timestamp("2024-03-01")
    assert result["trade_date"].max() <= pd.Timestamp("2024-03-15")

    empty = build_market_regime_daily_frame(
        regime, dates, date_from="2030-01-01", date_to="2030-01-02"
    )
    assert empty.empty
    assert empty.columns.tolist() == list(OUTPUT_COLUMNS)


def test_reference_loader_rejects_missing_configured_code(tmp_path) -> None:
    with np.testing.assert_raises(FileNotFoundError):
        load_reference_close_panel(tmp_path, ("000985.CSI",))


def test_write_output_supports_resume_and_overwrite(tmp_path) -> None:
    dates, regime = _market_result()
    result = build_market_regime_daily_frame(regime, dates)[60:65]
    write_output(result.iloc[:3], tmp_path, overwrite=False)
    write_output(result.iloc[1:], tmp_path, overwrite=False)
    stored = pd.read_parquet(tmp_path / OUTPUT_FILENAME)
    assert len(stored) == 5

    changed = result.iloc[[0]].copy()
    changed["regime_confidence"] = 0.123
    write_output(changed, tmp_path, overwrite=True)
    stored = pd.read_parquet(tmp_path / OUTPUT_FILENAME)
    value = stored.loc[
        stored["trade_date"] == changed.iloc[0]["trade_date"],
        "regime_confidence",
    ].iloc[0]
    assert float(value) == 0.123
