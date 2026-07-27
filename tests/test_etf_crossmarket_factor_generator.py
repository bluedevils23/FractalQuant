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
    DAILY_STATE_COLUMNS,
    ETF_INDEX_RELATIVE_VALUE_COLUMNS,
    FACTOR_COLUMNS,
    RELATED_ETF_FACTOR_COLUMNS,
    align_previous_daily_state,
    build_index_file_lookup,
    calculate_related_etf_factor_frames,
    calculate_crossmarket_factor_frame,
    load_etf_daily_histories,
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
    return _raw_parquet_days((trade_day,), reference=reference)


def _raw_parquet_days(
    trade_days: tuple[str, ...],
    *,
    reference: bool,
) -> pd.DataFrame:
    frame = _minute_frame(trade_days, reference=reference).rename(
        columns={"volume": "vol"}
    )
    trade_dates = frame.index.strftime("%Y-%m-%d")
    return frame.assign(
        trade_date=trade_dates,
        trade_time=frame.index,
    ).set_index(["trade_date", "trade_time"])


def _daily_state_frame(
    index: pd.DatetimeIndex,
    *,
    nav: float,
    total_size: float,
    total_share: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prev_nav": nav,
            "prev_total_size": total_size,
            "prev_total_share": total_share,
        },
        index=index,
    )


def _write_daily_file(
    path: Path,
    codes: tuple[str, ...],
    *,
    previous_nav: float = 1.0,
) -> None:
    rows = []
    for number, code in enumerate(codes, start=1):
        rows.extend(
            [
                {
                    "trade_date": pd.Timestamp("2026-01-02"),
                    "ts_code": f"{code}.SZ",
                    "nav": previous_nav * number,
                    "total_size": 1_000.0 * number,
                    "total_share": 900.0 * number,
                },
                {
                    "trade_date": pd.Timestamp("2026-01-05"),
                    "ts_code": f"{code}.SZ",
                    "nav": 100.0 * number,
                    "total_size": 100_000.0 * number,
                    "total_share": 90_000.0 * number,
                },
            ]
        )
    pd.DataFrame(rows).set_index(["trade_date", "ts_code"]).to_parquet(path)


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

    daily_state = _daily_state_frame(
        etf.index, nav=1.0, total_size=1_000.0, total_share=900.0
    )
    factors = calculate_crossmarket_factor_frame(
        etf, reference, daily_state
    )

    assert tuple(factors.columns) == ALL_FACTOR_COLUMNS
    for trade_day, day in factors.groupby(factors.index.normalize()):
        assert day.loc[:, FACTOR_COLUMNS].iloc[:49].isna().all().all()
        assert day.loc[:, FACTOR_COLUMNS].iloc[49:].notna().all().all()
        assert day["etf_index_intraday_price_gap"].notna().all()
        assert pd.isna(day["etf_index_basis_1m"].iloc[0])
        assert day["etf_index_basis_1m"].iloc[1:].notna().all()
        if trade_day == pd.Timestamp("2026-01-05"):
            assert day["etf_fair_value_gap_proxy_1m"].isna().all()
        else:
            assert day["etf_fair_value_gap_proxy_1m"].notna().all()
        assert day.loc[:, RELATED_ETF_FACTOR_COLUMNS].isna().all().all()
    assert (factors.loc[:, FACTOR_COLUMNS].iloc[49:].notna().sum() > 0).all()
    assert not (factors.loc[:, "cointegration"].fillna(0) == 0).all()
    assert factors["cross_market_coherence"].dropna().abs().gt(0).any()
    assert factors["cross_market_info_flow"].dropna().nunique() > 1


def test_relative_value_factors_use_same_minute_data_without_future_rows() -> None:
    etf = _minute_frame(("2026-01-05",))
    current_reference = _minute_frame(("2026-01-05",), reference=True)
    previous_reference = pd.DataFrame(
        {
            "open": 2_990.0,
            "high": 2_990.0,
            "low": 2_990.0,
            "close": 2_990.0,
            "volume": 1_000.0,
            "amount": 3_000_000.0,
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-01-02 15:00:00")], name="trade_time"
        ),
    )
    reference = pd.concat([previous_reference, current_reference])
    daily_state = _daily_state_frame(
        etf.index, nav=1.02, total_size=1_000.0, total_share=900.0
    )
    factors = calculate_crossmarket_factor_frame(
        etf, reference, daily_state
    )
    point = factors.index[40]
    previous = factors.index[39]

    expected_basis = etf.loc[point, "close"] / etf.loc[previous, "close"] - 1.0
    expected_basis -= (
        reference.loc[point, "close"] / reference.loc[previous, "close"] - 1.0
    )
    expected_fair_value = 1.02 * (
        reference.loc[point, "close"] / previous_reference["close"].iloc[-1]
    )
    assert np.isclose(factors.loc[point, "etf_index_basis_1m"], expected_basis)
    assert np.isclose(
        factors.loc[point, "etf_fair_value_gap_proxy_1m"],
        etf.loc[point, "close"] / expected_fair_value - 1.0,
    )
    fair_gap = factors["etf_fair_value_gap_proxy_1m"]
    assert np.isclose(
        factors.loc[point, "basis_reversion_signal_5m"],
        -(fair_gap.loc[point] - fair_gap.iloc[35]),
    )
    assert factors["etf_basis_zscore_20m"].iloc[:20].isna().all()
    assert factors["etf_basis_zscore_20m"].iloc[20:].notna().all()
    assert factors.loc[point, "etf_index_tracking_error"] >= 0
    assert factors.loc[point, "etf_index_realized_vol_ratio"] > 0

    missing_nav_state = daily_state.copy()
    missing_nav_state["prev_nav"] = np.nan
    missing_nav_factors = calculate_crossmarket_factor_frame(
        etf, reference, missing_nav_state
    )
    assert missing_nav_factors["etf_fair_value_gap_proxy_1m"].isna().all()
    assert missing_nav_factors["basis_reversion_signal_5m"].isna().all()
    assert missing_nav_factors["etf_index_basis_1m"].iloc[1:].notna().all()

    missing_same_minute = reference.drop(point)
    missing_factors = calculate_crossmarket_factor_frame(
        etf, missing_same_minute, daily_state
    )
    assert pd.isna(missing_factors.loc[point, "etf_index_basis_1m"])
    assert pd.isna(
        missing_factors.loc[point, "etf_fair_value_gap_proxy_1m"]
    )

    changed_future = reference.copy()
    changed_future.loc[etf.index[60]:, "close"] *= 1.2
    changed_future.loc[etf.index[60]:, "amount"] *= 5.0
    changed_factors = calculate_crossmarket_factor_frame(
        etf, changed_future, daily_state
    )
    pd.testing.assert_frame_equal(
        factors.loc[: etf.index[59], list(ETF_INDEX_RELATIVE_VALUE_COLUMNS)],
        changed_factors.loc[
            : etf.index[59], list(ETF_INDEX_RELATIVE_VALUE_COLUMNS)
        ],
    )


def test_related_etf_factors_use_p1_and_renamed_legacy_definitions() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=70, freq="min")
    steps = np.arange(len(index), dtype=float)
    frames: dict[str, pd.DataFrame] = {}
    for multiplier, code in enumerate(("159001", "159002", "159003"), start=1):
        frame = _minute_frame(("2026-01-05",))
        frame["close"] = frame["open"] * (1.0 + multiplier * steps * 0.001)
        frame["amount"] = 100.0 * multiplier + steps
        frames[code] = frame

    daily_states = {
        code: _daily_state_frame(
            frame.index,
            nav=float(multiplier),
            total_size=1_000.0 * multiplier,
            total_share=900.0 * multiplier,
        )
        for multiplier, (code, frame) in enumerate(frames.items(), start=1)
    }
    related = calculate_related_etf_factor_frames(frames, daily_states)
    point = index[10]
    previous = index[9]
    peer_returns = np.array(
        [
            frames["159002"].loc[point, "close"]
            / frames["159002"].loc[previous, "close"]
            - 1.0,
            frames["159003"].loc[point, "close"]
            / frames["159003"].loc[previous, "close"]
            - 1.0,
        ]
    )
    own_return = (
        frames["159001"].loc[point, "close"]
        / frames["159001"].loc[previous, "close"]
        - 1.0
    )
    expected_relative_return = own_return - peer_returns.mean()
    peer_liquidity = np.array(
        [
            frames["159002"].loc[point, "amount"] / (2_000.0 * 10_000.0),
            frames["159003"].loc[point, "amount"] / (3_000.0 * 10_000.0),
        ]
    )
    expected_relative_liquidity = (
        frames["159001"].loc[point, "amount"] / (1_000.0 * 10_000.0)
        - peer_liquidity.mean()
    )

    assert np.isclose(
        related["159001"].loc[point, "same_index_relative_return_1m"],
        expected_relative_return,
    )
    assert np.isclose(
        related["159001"].loc[point, "same_index_relative_liquidity_1m"],
        expected_relative_liquidity,
    )
    assert related["159001"][
        "same_index_intraday_cumulative_return_gap"
    ].iloc[1:].notna().all()
    assert related["159001"][
        "same_index_relative_amount_shock_60m"
    ].iloc[:29].isna().all()
    assert related["159001"][
        "same_index_relative_amount_shock_60m"
    ].iloc[29:].notna().all()
    missing_aum_states = {
        code: state.copy() for code, state in daily_states.items()
    }
    missing_aum_states["159001"]["prev_total_size"] = np.nan
    missing_aum = calculate_related_etf_factor_frames(
        frames, missing_aum_states
    )["159001"]
    assert missing_aum["same_index_relative_liquidity_1m"].isna().all()
    assert missing_aum["same_index_relative_return_1m"].iloc[1:].notna().all()
    assert calculate_related_etf_factor_frames(
        {"159001": frames["159001"]},
        {"159001": daily_states["159001"]},
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
    assert set(DAILY_STATE_COLUMNS) <= set(output.columns)


def test_daily_state_uses_only_immediately_previous_record(
    tmp_path: Path,
) -> None:
    daily_path = tmp_path / "etf_daily.parquet"
    minute_path = tmp_path / "159008.SZ.parquet"
    minute_path.touch()
    _write_daily_file(daily_path, ("159008",), previous_nav=1.03)

    history = load_etf_daily_histories(
        daily_path, {"159008": minute_path}
    )["159008"]
    minute_index = pd.date_range(
        "2026-01-05 09:30:00", periods=3, freq="min"
    )
    state = align_previous_daily_state(minute_index, history)

    assert state["prev_nav"].eq(1.03).all()
    assert state["prev_total_size"].eq(1_000.0).all()
    assert state["prev_total_share"].eq(900.0).all()

    history.loc[pd.Timestamp("2026-01-02"), "nav"] = np.nan
    missing = align_previous_daily_state(minute_index, history)
    assert missing["prev_nav"].isna().all()
    assert missing["prev_total_size"].eq(1_000.0).all()


def test_process_mapping_group_writes_related_etf_factors(tmp_path: Path) -> None:
    reference_path = tmp_path / "399975.SZ.parquet"
    daily_path = tmp_path / "etf_daily.parquet"
    output_root = tmp_path / "output"
    _raw_parquet_days(
        ("2026-01-02", "2026-01-05"), reference=True
    ).to_parquet(reference_path)
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
    _write_daily_file(
        daily_path, ("159001", "159002", "159003"), previous_nav=1.0
    )

    results = process_mapping_group(
        member_paths,
        ("159001", "159002", "159003"),
        reference_path,
        "399975.SZ",
        daily_path,
        output_root,
        "20260105",
        "20260105",
        False,
    )

    assert {result["status"] for result in results} == {"written"}
    output = pd.read_parquet(output_root / "159001.SZ.parquet")
    assert output["same_index_relative_return_1m"].iloc[1:].notna().all()
    assert output["same_index_relative_liquidity_1m"].notna().all()
    assert output[
        "same_index_intraday_cumulative_return_gap"
    ].iloc[1:].notna().all()
    assert output[
        "same_index_relative_amount_shock_60m"
    ].iloc[29:].notna().all()
    assert output["prev_nav"].eq(1.0).all()
    assert output["prev_total_size"].eq(1_000.0).all()
    replaced_columns = {
        "etf_index_return_gap_1m",
        "etf_index_return_gap_5m",
        "etf_fair_value_premium",
        "etf_fair_value_premium_zscore",
        "premium_mean_reversion_speed",
        "premium_change_1m",
        "premium_change_5m",
        "related_etf_price_spread",
        "related_etf_liquidity_gap",
    }
    assert replaced_columns.isdisjoint(output.columns)
