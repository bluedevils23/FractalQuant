from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_etf_minute_factors import process_file  # noqa: E402
from scripts.session_path_factors import (  # noqa: E402
    SESSION_PATH_OUTPUT_COLUMNS,
    build_session_path_factor_frame_from_frame,
)


def _minute_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("2026-01-05", "2026-01-05 09:30:00"),
            ("2026-01-05", "2026-01-05 09:31:00"),
            ("2026-01-06", "2026-01-06 09:30:00"),
        ],
        names=["trade_date", "trade_time"],
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


def test_session_path_preserves_causal_time_and_cross_day_previous_close() -> None:
    result = build_session_path_factor_frame_from_frame(
        _minute_frame(), "510300.SH"
    )

    assert list(result.columns) == SESSION_PATH_OUTPUT_COLUMNS
    assert result["bar_time"].tolist() == [
        pd.Timestamp("2026-01-05 09:30:00"),
        pd.Timestamp("2026-01-05 09:31:00"),
        pd.Timestamp("2026-01-06 09:30:00"),
    ]
    assert result["available_time"].tolist() == [
        pd.Timestamp("2026-01-05 09:31:00"),
        pd.Timestamp("2026-01-05 09:32:00"),
        pd.Timestamp("2026-01-06 09:31:00"),
    ]
    assert np.isnan(result.loc[0, "intraday_return_from_prev_close"])
    assert np.isclose(result.loc[2, "intraday_return_from_prev_close"], 102.5 / 102 - 1)
    assert np.isfinite(
        result[
            [
                "intraday_drawdown_from_session_high",
                "intraday_rebound_from_session_low",
            ]
        ]
    ).all().all()


def test_minute_factor_process_reuses_loaded_frame_for_session_path(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "510300.SH.parquet"
    output_root = tmp_path / "minute_factors"
    session_root = tmp_path / "session_path"
    _minute_frame().to_parquet(input_path)
    output_root.mkdir()
    pd.DataFrame({"placeholder": [1]}).to_parquet(
        output_root / input_path.name, index=False
    )

    def fail_factor_calculation(*args, **kwargs):
        raise AssertionError("existing minute factors should not be recalculated")

    monkeypatch.setattr(
        "scripts.generate_etf_minute_factors.calculate_factor_frame",
        fail_factor_calculation,
    )

    status, output_path, row_count, column_count = process_file(
        input_path,
        output_root,
        overwrite=False,
        session_path_output_root=session_root,
    )

    assert status == "skipped"
    assert output_path == output_root / input_path.name
    assert row_count is None
    assert column_count is None
    session_output = pd.read_parquet(session_root / input_path.name)
    assert len(session_output) == 3
    assert session_output["trade_date"].nunique() == 2


def test_minute_factor_process_skips_minute_read_when_requested_session_dates_exist(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "510300.SH.parquet"
    output_root = tmp_path / "minute_factors"
    session_root = tmp_path / "session_path"
    output_root.mkdir()
    session_root.mkdir()
    pd.DataFrame({"placeholder": [1]}).to_parquet(
        output_root / input_path.name, index=False
    )
    pd.DataFrame(
        {
            "trade_date": ["2026-01-05"],
            "bar_time": [pd.Timestamp("2026-01-05 09:30")],
        }
    ).to_parquet(session_root / input_path.name, index=False)

    def fail_read(*args, **kwargs):
        raise AssertionError("existing requested session dates must skip minute read")

    monkeypatch.setattr(
        "scripts.generate_etf_minute_factors.existing_trade_dates",
        lambda path: {"2026-01-05"},
    )
    monkeypatch.setattr("scripts.generate_etf_minute_factors.pd.read_parquet", fail_read)

    status, output_path, row_count, column_count = process_file(
        input_path,
        output_root,
        overwrite=False,
        session_path_output_root=session_root,
        date_from="20260105",
        date_to="20260105",
    )

    assert status == "skipped"
    assert output_path == output_root / input_path.name
    assert row_count is None
    assert column_count is None
