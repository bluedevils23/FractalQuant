from __future__ import annotations

import pandas as pd

from scripts.generate_stock_orderbook_factors import (
    BASE_OUTPUT_COLUMNS,
    FACTOR_COLUMNS,
    OUTPUT_COLUMNS,
    build_output_frame,
    merge_symbol_output,
)


def _minute_frame(trade_date: str) -> pd.DataFrame:
    times = [f"{trade_date} 09:30:00", f"{trade_date} 09:31:00"]
    return pd.DataFrame(
        {
            "trade_date": [trade_date] * 2,
            "trade_time": times,
            "ts_code": ["000001.SZ"] * 2,
            "open": [10.0, 10.1],
            "high": [10.1, 10.2],
            "low": [9.9, 10.0],
            "close": [10.05, 10.15],
            "vol": [1000.0, 1200.0],
            "amount": [10000.0, 12180.0],
            "adj_factor": [1.0, 1.0],
        }
    )[BASE_OUTPUT_COLUMNS]


def test_output_matches_advanced_parquet_layout() -> None:
    minute = _minute_frame("2026-01-05")
    factor_index = pd.DatetimeIndex(
        ["2026-01-05 09:29:59", "2026-01-05 09:30:59"],
        name="trade_time",
    )
    factors = pd.DataFrame(0.0, index=factor_index, columns=FACTOR_COLUMNS)
    factors["mid_price"] = [10.0, 10.1]

    result = build_output_frame(minute, factors)

    assert isinstance(result.index, pd.RangeIndex)
    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result["trade_date"].dtype == minute["trade_date"].dtype
    assert result["trade_time"].dtype == minute["trade_time"].dtype
    assert result["mid_price"].tolist() == [10.0, 10.1]


def test_incremental_merge_replaces_only_requested_trade_date(tmp_path) -> None:
    output_path = tmp_path / "000001.SZ.parquet"
    first = build_output_frame(
        _minute_frame("2026-01-05"),
        pd.DataFrame(
            1.0,
            index=pd.DatetimeIndex(
                ["2026-01-05 09:30:00", "2026-01-05 09:31:00"],
                name="trade_time",
            ),
            columns=FACTOR_COLUMNS,
        ),
    )
    first.to_parquet(output_path)
    second = build_output_frame(
        _minute_frame("2026-01-06"),
        pd.DataFrame(
            2.0,
            index=pd.DatetimeIndex(
                ["2026-01-06 09:30:00", "2026-01-06 09:31:00"],
                name="trade_time",
            ),
            columns=FACTOR_COLUMNS,
        ),
    )

    combined = merge_symbol_output(output_path, second, overwrite=True)

    assert combined["trade_date"].tolist() == [
        "2026-01-05",
        "2026-01-05",
        "2026-01-06",
        "2026-01-06",
    ]
    assert isinstance(combined.index, pd.RangeIndex)
