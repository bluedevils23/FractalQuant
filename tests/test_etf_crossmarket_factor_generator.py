from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.crossmarket import CrossMarketCorrelationFactor
from scripts.generate_etf_crossmarket_factors import (
    ALL_FACTOR_COLUMNS,
    ETF_INDEX_RELATIVE_VALUE_COLUMNS,
    FACTOR_COLUMNS,
    RELATED_ETF_FACTOR_COLUMNS,
    build_index_file_lookup,
    calculate_related_etf_factor_frames,
    calculate_crossmarket_factor_frame,
    load_mapping_records,
    process_mapping_group,
    process_mapping_record,
    resolve_reference_path,
)


def _minute_frame(
    trade_days: tuple[str, ...],
    *,
    reference: bool = False,
) -> pd.DataFrame:
    frames = []
    for day_number, trade_day in enumerate(trade_days):
        index = pd.date_range(
            f"{trade_day} 09:30:00", periods=70, freq="min"
        )
        steps = np.arange(len(index), dtype=float)
        if reference:
            close = (
                3_000.0
                + day_number * 20.0
                + steps * 1.5
                + np.sin(steps / 5.0) * 2.0
            )
            ts_code = "399975.SZ"
        else:
            close = (
                1.0
                + day_number * 0.01
                + steps * 0.0008
                + np.sin(steps / 4.0) * 0.002
            )
            ts_code = "159008.SZ"
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": ts_code,
                    "open": close - 0.001,
                    "high": close + 0.002,
                    "low": close - 0.002,
                    "close": close,
                    "volume": 1_000.0 + steps,
                    "amount": close * (1_000.0 + steps),
                },
                index=index,
            )
        )
    result = pd.concat(frames)
    result.index.name = "trade_time"
    return result


def _raw_parquet_frame(
    trade_day: str,
    *,
    reference: bool,
) -> pd.DataFrame:
    frame = _minute_frame((trade_day,), reference=reference).rename(
        columns={"volume": "vol"}
    )
    trade_dates = frame.index.strftime("%Y-%m-%d")
    return frame.assign(
        trade_date=trade_dates,
        trade_time=frame.index,
    ).set_index(["trade_date", "trade_time"])


def test_mapping_and_reference_resolution_support_code_stem_alias(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "fund_code,reference_index_code\n"
        "159008,399975.SZ\n"
        "510300,000300.CSI\n",
        encoding="utf-8",
    )
    records = load_mapping_records(mapping_path)
    assert records["159008"].reference_index_code == "399975.SZ"

    index_root = tmp_path / "indices"
    index_root.mkdir()
    (index_root / "399975.SZ.parquet").touch()
    (index_root / "000300.SH.parquet").touch()
    (index_root / "AU99.99.SH.parquet").touch()
    by_full, by_stem = build_index_file_lookup(index_root)

    exact, exact_type = resolve_reference_path(
        "399975.SZ", by_full, by_stem
    )
    alias, alias_type = resolve_reference_path(
        "000300.CSI", by_full, by_stem
    )
    missing, missing_type = resolve_reference_path(
        "931946.CSI", by_full, by_stem
    )
    multi_dot_alias, multi_dot_type = resolve_reference_path(
        "Au99.99.SGE", by_full, by_stem
    )

    assert exact == index_root / "399975.SZ.parquet"
    assert exact_type == "exact"
    assert alias == index_root / "000300.SH.parquet"
    assert alias_type == "stem"
    assert missing is None
    assert missing_type == "missing"
    assert multi_dot_alias == index_root / "AU99.99.SH.parquet"
    assert multi_dot_type == "stem"


def test_crossmarket_reference_windows_do_not_use_future_rows() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=80, freq="min")
    steps = np.arange(len(index), dtype=float)
    etf = pd.DataFrame(
        {"close": 1.0 + steps * 0.001 + np.sin(steps / 4.0) * 0.002},
        index=index,
    )
    reference = pd.DataFrame(
        {"close": 3_000.0 + steps * 2.0 + np.sin(steps / 5.0) * 3.0},
        index=index,
    )
    changed_future = reference.copy()
    changed_future.loc[index[60]:, "close"] = (
        reference.loc[index[60]:, "close"].to_numpy()[::-1]
    )

    factor = CrossMarketCorrelationFactor()
    original = factor.calculate(etf, reference)
    changed = factor.calculate(etf, changed_future)

    pd.testing.assert_series_equal(
        original.loc[: index[59]],
        changed.loc[: index[59]],
    )
    assert not np.isclose(original.iloc[-1], changed.iloc[-1])


def test_all_crossmarket_factors_reset_each_trade_day() -> None:
    etf = _minute_frame(("2026-01-05", "2026-01-06"))
    reference = _minute_frame(
        ("2026-01-05", "2026-01-06"), reference=True
    )

    factors = calculate_crossmarket_factor_frame(etf, reference)

    assert tuple(factors.columns) == ALL_FACTOR_COLUMNS
    for _, day in factors.groupby(factors.index.normalize()):
        assert day.loc[:, FACTOR_COLUMNS].iloc[:49].isna().all().all()
        assert day.loc[:, FACTOR_COLUMNS].iloc[49:].notna().all().all()
        assert day["etf_fair_value_premium"].notna().all()
        assert pd.isna(day["etf_index_return_gap_1m"].iloc[0])
        assert day["etf_index_return_gap_1m"].iloc[1:].notna().all()
        assert day.loc[:, RELATED_ETF_FACTOR_COLUMNS].isna().all().all()
    assert (factors.loc[:, FACTOR_COLUMNS].iloc[49:].notna().sum() > 0).all()
    assert not (factors.loc[:, "cointegration"].fillna(0) == 0).all()
    assert factors["cross_market_coherence"].dropna().abs().gt(0).any()
    assert factors["cross_market_info_flow"].dropna().nunique() > 1


def test_relative_value_factors_use_same_minute_data_without_future_rows() -> None:
    etf = _minute_frame(("2026-01-05",))
    reference = _minute_frame(("2026-01-05",), reference=True)
    factors = calculate_crossmarket_factor_frame(etf, reference)
    point = factors.index[40]
    previous = factors.index[39]

    expected_gap = np.log(etf.loc[point, "close"] / etf.loc[previous, "close"])
    expected_gap -= np.log(
        reference.loc[point, "close"] / reference.loc[previous, "close"]
    )
    expected_fair_value = etf["open"].iloc[0] * (
        reference.loc[point, "close"] / reference["open"].iloc[0]
    )
    assert np.isclose(factors.loc[point, "etf_index_return_gap_1m"], expected_gap)
    assert np.isclose(
        factors.loc[point, "etf_fair_value_premium"],
        etf.loc[point, "close"] / expected_fair_value - 1.0,
    )
    assert factors.loc[point, "etf_index_tracking_error"] >= 0
    assert factors.loc[point, "etf_index_realized_vol_ratio"] > 0

    missing_same_minute = reference.drop(point)
    missing_factors = calculate_crossmarket_factor_frame(etf, missing_same_minute)
    assert pd.isna(missing_factors.loc[point, "etf_index_return_gap_1m"])
    assert pd.isna(missing_factors.loc[point, "etf_fair_value_premium"])

    changed_future = reference.copy()
    changed_future.loc[reference.index[60]:, "close"] *= 1.2
    changed_future.loc[reference.index[60]:, "amount"] *= 5.0
    changed_factors = calculate_crossmarket_factor_frame(etf, changed_future)
    pd.testing.assert_frame_equal(
        factors.loc[: reference.index[59], list(ETF_INDEX_RELATIVE_VALUE_COLUMNS)],
        changed_factors.loc[
            : reference.index[59], list(ETF_INDEX_RELATIVE_VALUE_COLUMNS)
        ],
    )


def test_related_etf_factors_are_leave_one_out_and_lag_weighted() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=70, freq="min")
    steps = np.arange(len(index), dtype=float)
    frames: dict[str, pd.DataFrame] = {}
    for multiplier, code in enumerate(("159001", "159002", "159003"), start=1):
        frame = _minute_frame(("2026-01-05",))
        frame["close"] = frame["open"] * (1.0 + multiplier * steps * 0.001)
        frame["amount"] = 100.0 * multiplier + steps
        frames[code] = frame

    related = calculate_related_etf_factor_frames(frames)
    point = index[10]
    previous = index[9]
    peer_returns = np.array(
        [
            frames["159002"].loc[point, "close"]
            / frames["159002"]["open"].iloc[0]
            - 1.0,
            frames["159003"].loc[point, "close"]
            / frames["159003"]["open"].iloc[0]
            - 1.0,
        ]
    )
    peer_weights = np.array(
        [
            frames["159002"].loc[previous, "amount"],
            frames["159003"].loc[previous, "amount"],
        ]
    )
    expected_spread = (
        frames["159001"].loc[point, "close"] / frames["159001"]["open"].iloc[0]
        - 1.0
        - np.average(peer_returns, weights=peer_weights)
    )

    assert np.isclose(
        related["159001"].loc[point, "related_etf_price_spread"],
        expected_spread,
    )
    assert related["159001"]["related_etf_liquidity_gap"].iloc[:29].isna().all()
    assert related["159001"]["related_etf_liquidity_gap"].iloc[29:].notna().all()
    assert calculate_related_etf_factor_frames(
        {"159001": frames["159001"]}
    )["159001"].isna().all().all()


def test_process_mapping_record_writes_reference_metadata(
    tmp_path: Path,
) -> None:
    etf_path = tmp_path / "159008.SZ.parquet"
    reference_path = tmp_path / "399975.SZ.parquet"
    output_root = tmp_path / "output"
    _raw_parquet_frame("2026-01-05", reference=False).to_parquet(etf_path)
    _raw_parquet_frame("2026-01-05", reference=True).to_parquet(
        reference_path
    )

    result = process_mapping_record(
        etf_path,
        reference_path,
        "399975.SZ",
        output_root,
        "20260105",
        "20260105",
        False,
    )
    output = pd.read_parquet(output_root / etf_path.name)

    assert result["status"] == "written"
    assert result["factor_non_null"] > 0
    assert set(ALL_FACTOR_COLUMNS) <= set(output.columns)
    assert output["reference_index_code"].eq("399975.SZ").all()
    assert output["reference_ts_code"].eq("399975.SZ").all()
    assert output["reference_close"].notna().all()


def test_process_mapping_group_writes_related_etf_factors(tmp_path: Path) -> None:
    reference_path = tmp_path / "399975.SZ.parquet"
    output_root = tmp_path / "output"
    _raw_parquet_frame("2026-01-05", reference=True).to_parquet(reference_path)
    member_paths: dict[str, Path] = {}
    for multiplier, code in enumerate(("159001", "159002", "159003"), start=1):
        path = tmp_path / f"{code}.SZ.parquet"
        frame = _raw_parquet_frame("2026-01-05", reference=False)
        frame["close"] = frame["open"] * (
            1.0 + multiplier * np.arange(len(frame), dtype=float) * 0.001
        )
        frame["amount"] = 100.0 * multiplier + np.arange(len(frame), dtype=float)
        frame.to_parquet(path)
        member_paths[code] = path

    results = process_mapping_group(
        member_paths,
        ("159001", "159002", "159003"),
        reference_path,
        "399975.SZ",
        output_root,
        "20260105",
        "20260105",
        False,
    )

    assert {result["status"] for result in results} == {"written"}
    output = pd.read_parquet(output_root / "159001.SZ.parquet")
    assert output["related_etf_price_spread"].iloc[1:].notna().all()
    assert output["related_etf_liquidity_gap"].iloc[29:].notna().all()
