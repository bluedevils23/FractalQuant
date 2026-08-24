from __future__ import annotations

import numpy as np
import pandas as pd

from factor.regime import build_daily_market_feature_panel, calculate_causal_daily_market_regime_features
from scripts.generate_etf_regime_daily_factors import (
    OUTPUT_COLUMNS,
    build_etf_regime_daily_frame,
    load_reference_close_panel,
    write_outputs,
)


def _market_result(periods: int = 140) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_generator_broadcasts_shared_regime_and_maps_previous_date() -> None:
    dates, regime = _market_result()
    target_dates = dates[60:90]
    etf = pd.DataFrame(
        {
            "trade_date": list(target_dates) * 2,
            "ts_code": ["159001.SZ"] * len(target_dates) + ["510300.SH"] * len(target_dates),
        }
    )
    result = build_etf_regime_daily_frame(etf, regime, dates)

    assert result.columns.tolist() == list(OUTPUT_COLUMNS)
    assert (result["source_trade_date"] < result["trade_date"]).all()
    assert (result["available_time"] == result["trade_date"] + pd.Timedelta(hours=9, minutes=15)).all()
    compare = result.pivot(index="trade_date", columns="ts_code", values="regime_prob_high").dropna()
    assert np.allclose(compare.iloc[:, 0], compare.iloc[:, 1])


def test_generator_empty_intersection_is_schema_stable() -> None:
    dates, regime = _market_result()
    etf = pd.DataFrame({"trade_date": [dates[0]], "ts_code": ["159001.SZ"]})
    result = build_etf_regime_daily_frame(etf, regime, dates)
    assert result.empty
    assert result.columns.tolist() == list(OUTPUT_COLUMNS)


def test_reference_loader_rejects_missing_configured_code(tmp_path) -> None:
    with np.testing.assert_raises(FileNotFoundError):
        load_reference_close_panel(tmp_path, ("000985.CSI",))


def test_write_outputs_supports_resume_and_overwrite(tmp_path) -> None:
    dates, regime = _market_result()
    etf = pd.DataFrame({"trade_date": dates[60:65], "ts_code": "159001.SZ"})
    result = build_etf_regime_daily_frame(etf, regime, dates)
    write_outputs(result.iloc[:3], tmp_path, overwrite=False)
    write_outputs(result.iloc[1:], tmp_path, overwrite=False)
    stored = pd.read_parquet(tmp_path / "159001.SZ.parquet")
    assert len(stored) == 5

    changed = result.iloc[[0]].copy()
    changed["regime_confidence"] = 0.123
    write_outputs(changed, tmp_path, overwrite=True)
    stored = pd.read_parquet(tmp_path / "159001.SZ.parquet")
    assert float(stored.loc[stored["trade_date"] == changed.iloc[0]["trade_date"], "regime_confidence"].iloc[0]) == 0.123
