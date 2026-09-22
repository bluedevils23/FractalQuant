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
    _residual_std,
    calculate_tail30_factors,
    merge_symbol_output,
    process_symbol,
)


def _day_frame(day: str, missing_minute: int | None = None) -> pd.DataFrame:
    index = pd.date_range(f"{day} 14:30", periods=31, freq="min")
    if missing_minute is not None:
        index = index.delete(missing_minute + 1)
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
    frame.loc[frame.index[1:], "amount"] = np.array(
        [100.0] * 29 + [1_000.0]
    )
    row = calculate_tail30_factors(
        frame,
        "510300.SH",
        {"2026-03-31": "2026-04-01"},
    )[0]

    tail = frame.loc["2026-03-31 14:31":"2026-03-31 15:00"]
    returns = np.diff(np.log(frame["close"].to_numpy()))
    expected_vwap = tail["amount"].sum() / tail["volume"].sum()
    expected_bias = (frame["close"].iloc[-1] - expected_vwap) / expected_vwap
    expected_return = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0

    assert row["tail30_bar_count"] == 30
    assert row["tail30_complete"] is True
    assert row["tail30_start_time"] == pd.Timestamp("2026-03-31 14:31")
    assert row["tail30_end_time"] == pd.Timestamp("2026-03-31 15:00")
    assert row["available_time"] == pd.Timestamp("2026-04-01 09:15")
    assert row["tail30_return_start_close"] == frame["close"].iloc[0]
    assert np.isclose(row["tail30_vwap"], expected_vwap)
    assert np.isclose(row["tail30_return"], expected_return)
    assert np.isclose(row["tail30_dastd"], returns.std(ddof=1))
    assert np.isclose(row["tail30_bias_proxy"], expected_bias)
    assert np.isclose(
        row["tail30_daily_reverse"],
        0.5 * row["tail30_return"] + 0.5 * row["tail30_bias_proxy"],
    )
    assert np.isnan(row["tail30_residual_volatility_proxy"])
    assert np.isnan(row["tail30_daily_volatility"])
    assert tuple(column for column in OUTPUT_COLUMNS if column in row) == tuple(OUTPUT_COLUMNS)


def test_milliq_includes_signed_and_peak_amount_windows() -> None:
    frame = _day_frame("2026-03-31")
    frame.loc[:, "amount"] = 100.0
    frame.iloc[11:14, frame.columns.get_loc("amount")] = 1_000.0
    frame.iloc[10:14, frame.columns.get_loc("close")] = [10.30, 10.20, 10.10, 10.00]
    row = calculate_tail30_factors(frame, "510300.SH", {"2026-03-31": "2026-04-01"})[0]

    expected_signed = (np.log(10.00) - np.log(10.30)) / 3_000.0
    assert row["tail30_maxilliq_signed_proxy"] < 0
    assert np.isclose(row["tail30_maxilliq_proxy"], abs(row["tail30_maxilliq_signed_proxy"]))
    assert np.isclose(row["tail30_maxilliq_w3_signed_proxy"], expected_signed)
    assert np.isclose(
        row["tail30_maxilliq_w3_proxy"],
        abs(row["tail30_maxilliq_w3_signed_proxy"]),
    )
    assert np.isfinite(row["tail30_maxilliq_w5_proxy"])


def test_residual_volatility_uses_only_prior_days() -> None:
    current = np.array([0.005, -0.003, 0.004])
    benchmark = {f"2026-03-{day:02d}": np.array([0.001, -0.002, 0.003]) for day in range(1, 22)}
    asset = {
        date: 0.0005 + 1.3 * values + np.array([0.0002, -0.0001, 0.0003])
        for date, values in benchmark.items()
    }
    current_date = "2026-03-21"
    benchmark[current_date] = current
    asset[current_date] = 0.0005 + 1.3 * current + np.array([0.001, -0.002, 0.001])
    baseline = _residual_std(asset[current_date], benchmark, asset, current_date)

    benchmark["2026-03-22"] = np.array([0.10, -0.10, 0.10])
    asset["2026-03-22"] = np.array([-0.10, 0.10, -0.10])
    with_future = _residual_std(asset[current_date], benchmark, asset, current_date)

    assert np.isfinite(baseline)
    assert np.isclose(with_future, baseline)


def test_residual_volatility_is_missing_for_benchmark_itself() -> None:
    returns = np.array([0.001, -0.002, 0.003])
    benchmark = {f"2026-03-{day:02d}": returns for day in range(1, 22)}
    assert np.isnan(_residual_std(returns, benchmark, benchmark, "2026-03-21"))


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


def test_schema_upgrade_retries_dates_that_failed_during_upgrade(tmp_path: Path) -> None:
    input_path = tmp_path / "510300.SH.parquet"
    output_root = tmp_path / "output"
    frame = pd.concat(
        [_day_frame("2026-03-31"), _day_frame("2026-04-01")]
    )
    frame = frame.drop(pd.Timestamp("2026-04-01 14:30"))
    frame["ts_code"] = "510300.SH"
    frame.to_parquet(input_path)

    legacy = pd.DataFrame(
        {
            "trade_date": ["2026-03-31", "2026-04-01"],
            "available_time": [pd.Timestamp("2026-04-01 09:15"), pd.Timestamp("2026-04-02 09:15")],
            "ts_code": ["510300.SH", "510300.SH"],
            "tail30_daily_milliq": [0.1, 0.2],
        }
    )
    output_root.mkdir()
    legacy.to_parquet(output_root / "510300.SH.parquet", index=False)

    _, rows, skipped = process_symbol("510300.SH", input_path, output_root, None, None, False, {})
    after_failed_upgrade = pd.read_parquet(output_root / "510300.SH.parquet")
    assert (rows, skipped) == (1, 1)
    assert after_failed_upgrade["trade_date"].tolist() == ["2026-03-31"]

    repaired = pd.concat([_day_frame("2026-03-31"), _day_frame("2026-04-01")])
    repaired["ts_code"] = "510300.SH"
    repaired.to_parquet(input_path)
    _, rows, skipped = process_symbol("510300.SH", input_path, output_root, None, None, False, {})
    result = pd.read_parquet(output_root / "510300.SH.parquet")
    assert (rows, skipped) == (1, 0)
    assert result["trade_date"].tolist() == ["2026-03-31", "2026-04-01"]


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


def test_schema_upgrade_does_not_keep_old_factor_values(tmp_path: Path) -> None:
    output_path = tmp_path / "510300.SH.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2026-03-31"],
            "available_time": [pd.Timestamp("2026-04-01 09:15")],
            "ts_code": ["510300.SH"],
            "tail30_maxilliq_proxy": [0.1],
            "tail30_daily_milliq": [0.1],
        }
    ).to_parquet(output_path, index=False)
    requested = pd.DataFrame(
        [
            {
                column: (
                    "2026-03-31" if column == "trade_date" else
                    "510300.SH" if column == "ts_code" else
                    0.2 if column == "tail30_maxilliq_w3_proxy" else np.nan
                )
                for column in OUTPUT_COLUMNS
            }
        ]
    )

    merged = merge_symbol_output(output_path, requested, overwrite=False)

    assert len(merged) == 1
    assert pd.isna(merged.loc[0, "tail30_maxilliq_proxy"])
    assert merged.loc[0, "tail30_maxilliq_w3_proxy"] == 0.2
