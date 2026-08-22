from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.etf_report78 import (  # noqa: E402
    REPORT78_FACTOR_COLUMNS,
    build_report78_daily_panel,
    calculate_daily_report78_technical_factors,
)
from scripts.generate_etf_report78_factors import normalize_minute_frame  # noqa: E402


def _frame(days: int = 23, bars_per_day: int = 4) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-01-05", periods=days, freq="B"):
        for minute in range(bars_per_day):
            close = 100.0 + len(rows) * 0.02 + (0.03 if minute % 2 else -0.01)
            rows.append(
                {
                    "trade_time": day + pd.Timedelta(hours=9, minutes=30 + minute),
                    "open": close - 0.01,
                    "high": close + 0.02,
                    "low": close - 0.02,
                    "close": close,
                    "vol": 1000.0 + minute * 100.0,
                    "amount": (1000.0 + minute * 100.0) * close,
                }
            )
    return pd.DataFrame(rows)


def test_normalize_minute_frame_supports_local_etf_schema() -> None:
    normalized = normalize_minute_frame(_frame(1))
    assert normalized.index.name == "trade_time"
    assert "volume" in normalized
    assert normalized.index.is_monotonic_increasing


def test_daily_factors_are_finite_when_amount_is_present() -> None:
    frame = normalize_minute_frame(_frame(1))
    values = calculate_daily_report78_technical_factors(frame)
    assert set(values) == set(REPORT78_FACTOR_COLUMNS[:-1])
    assert all(np.isfinite(value) for value in values.values())
    assert 0.0 <= values["technical_qua"] <= 1.0


def test_daily_panel_is_causal_and_long_momentum_uses_completed_days() -> None:
    panel = build_report78_daily_panel(normalize_minute_frame(_frame()))
    assert list(panel.columns) == list(REPORT78_FACTOR_COLUMNS)
    assert panel["technical_long_mom"].iloc[:20].isna().all()
    assert panel["technical_long_mom"].iloc[20:].notna().all()
    assert panel.index.is_monotonic_increasing


def test_amount_dependent_fields_remain_null_when_amount_is_unavailable() -> None:
    frame = normalize_minute_frame(_frame(1)).drop(columns="amount")
    values = calculate_daily_report78_technical_factors(frame)
    assert np.isnan(values["technical_cvilliq"])
    assert np.isnan(values["technical_qua"])

