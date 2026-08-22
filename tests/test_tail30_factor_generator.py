from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_tail30_factors import (  # noqa: E402
    FACTOR_COLUMNS,
    OUTPUT_COLUMNS,
    calculate_tail30_factors,
    merge_symbol_output,
    process_symbol,
)


def _day_frame(day: str, missing_minute: int | None = None) -> pd.DataFrame:
    index = pd.date_range(f"{day} 14:31", periods=30, freq="min")
    if missing_minute is not None:
        index = index.delete(missing_minute)
    close = np.linspace(10.0, 10.29, len(index))
    frame = pd.DataFrame(
        {
            "open": close - 0.01,
            "close": close,
            "volume": np.arange(1, len(index) + 1, dtype=float),
            "amount": np.arange(1, len(index) + 1, dtype=float) * 100.0,
        },
        index=index,
    )
    return frame


def test_tail30_uses_1431_to_1500_and_report_weighting() -> None:
    frame = _day_frame("2026-03-31")
    row = calculate_tail30_factors(
        frame,
        "510300.SH",
        {"2026-03-31": "2026-04-01"},
    )[0]

    prices = np.concatenate(([9.99], frame["close"].to_numpy()))
    returns = np.diff(np.log(prices))
    expected_vwap = (frame["close"] * frame["volume"]).sum() / frame["volume"].sum()
    expected_bias = (frame["close"].iloc[-1] - expected_vwap) / expected_vwap
    expected_return = frame["close"].iloc[-1] / frame["open"].iloc[0] - 1.0

    assert row["tail30_bar_count"] == 30
    assert row["tail30_complete"] is True
    assert row["tail30_start_time"] == pd.Timestamp("2026-03-31 14:31")
    assert row["tail30_end_time"] == pd.Timestamp("2026-03-31 15:00")
    assert row["available_time"] == pd.Timestamp("2026-04-01 09:15")
    assert np.isclose(row["tail30_return"], expected_return)
    assert np.isclose(row["tail30_dastd"], returns.std(ddof=1))
    assert np.isclose(row["tail30_bias_proxy"], expected_bias)
    assert np.isclose(
        row["tail30_daily_reverse"],
        0.5 * row["tail30_return"] + 0.5 * row["tail30_bias_proxy"],
    )
    assert np.isclose(
        row["tail30_daily_volatility"],
        0.5 * row["tail30_dastd"] + 0.5 * row["tail30_residual_volatility_proxy"],
    )
    assert tuple(column for column in OUTPUT_COLUMNS if column in row) == tuple(OUTPUT_COLUMNS)


def test_process_symbol_skips_only_incomplete_date(tmp_path: Path) -> None:
    input_path = tmp_path / "510300.SH.parquet"
    frame = pd.concat(
        [_day_frame("2026-03-31"), _day_frame("2026-04-01", missing_minute=7)]
    )
    frame["ts_code"] = "510300.SH"
    frame.to_parquet(input_path)

    output_path, row_count, skipped_dates = process_symbol(
        "510300.SH",
        input_path,
        tmp_path / "output",
        None,
        None,
        False,
        {},
    )

    result = pd.read_parquet(output_path)
    assert row_count == 1
    assert skipped_dates == 1
    assert result["trade_date"].tolist() == ["2026-03-31"]


def test_merge_does_not_duplicate_existing_dates(tmp_path: Path) -> None:
    output_path = tmp_path / "510300.SH.parquet"
    existing = pd.DataFrame(
        {
            column: [pd.NaT if column in {"available_time", "tail30_start_time", "tail30_end_time"} else np.nan]
            for column in OUTPUT_COLUMNS
        }
    )
    existing["trade_date"] = ["2026-03-31"]
    existing["ts_code"] = ["510300.SH"]
    existing.to_parquet(output_path, index=False)
    requested = pd.DataFrame(
        [{column: ("2026-03-31" if column == "trade_date" else "510300.SH" if column == "ts_code" else 0.25 if column == "tail30_daily_milliq" else np.nan) for column in OUTPUT_COLUMNS}]
    )

    merged = merge_symbol_output(output_path, requested, overwrite=False)

    assert len(merged) == 1
    assert pd.isna(merged.loc[0, "tail30_daily_milliq"])
    assert set(FACTOR_COLUMNS).issubset(merged.columns)
