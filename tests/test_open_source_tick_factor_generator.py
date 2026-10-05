from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor.open_source_tick import (
    _adjusted_close_returns,
    AUDIT_COLUMNS,
    TICK_FACTOR_COLUMNS,
    add_availability_columns,
    build_daily_flow_raw_features,
    build_daily_tick_raw_features,
    build_daily_order_raw_features,
    build_open_source_tick_factor_panel,
    normalize_tick_transactions,
    normalize_tick_orders,
    FLOW_BUCKET_SCHEMA,
    select_output_columns,
)
from scripts import generate_open_source_tick_factors as generator


def _transactions(day: str, *, amount_scale: float = 1.0) -> pd.DataFrame:
    morning = pd.date_range(f"{day} 09:30", periods=120, freq="min")
    afternoon = pd.date_range(f"{day} 13:00", periods=120, freq="min")
    times = morning.append(afternoon)
    prices = 10.0 + np.arange(len(times), dtype=float) * 0.001
    quantities = (100 + np.arange(len(times)) % 17) * amount_scale
    return pd.DataFrame(
        {
            "trade_time": times,
            "price": prices,
            "quantity": quantities,
            "ts_code": "000001.SZ",
            "trade_code": "0",
        }
    )


def test_tick_session_filter_and_daily_metrics() -> None:
    raw = _transactions("2025-01-02")
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                {
                    "trade_time": pd.to_datetime(["2025-01-02 09:29", "2025-01-02 11:30", "2025-01-02 12:00"]),
                    "price": [10.0, 10.0, 10.0],
                    "quantity": [100, 100, 100],
                    "ts_code": "000001.SZ",
                }
            ),
        ],
        ignore_index=True,
    )
    normalized = normalize_tick_transactions(raw)
    assert len(normalized) == 240
    daily = build_daily_tick_raw_features(raw, "000001.SZ")
    assert daily.loc[0, "valid_trade_minutes"] == 240
    assert daily.loc[0, "trade_count"] == 240
    assert daily.loc[0, "daily_mts"] == 1.0


def test_qua_drops_ten_largest_minute_observations() -> None:
    raw = _transactions("2025-01-02")
    daily = build_daily_tick_raw_features(raw, "000001.SZ")
    amounts = raw.assign(amount=raw["price"] * raw["quantity"]).groupby(
        raw["trade_time"].dt.floor("min")
    )["amount"].sum()
    clean = amounts.sort_values().iloc[:-10]
    expected = (clean.quantile(0.10) - clean.min()) / (clean.max() - clean.min())
    assert np.isclose(daily.loc[0, "daily_qua"], expected)


def test_panel_rolling_and_next_actual_date() -> None:
    frames = []
    dates = pd.bdate_range("2025-01-02", periods=21)
    for i, date in enumerate(dates):
        frames.append(_transactions(date.strftime("%Y-%m-%d"), amount_scale=1 + i / 100))
    raw = pd.concat(
        [build_daily_tick_raw_features(frame, "000001.SZ") for frame in frames],
        ignore_index=True,
    )
    raw["adj_factor"] = 1.0
    panel = add_availability_columns(build_open_source_tick_factor_panel(raw))
    assert panel[list(TICK_FACTOR_COLUMNS[:5])].iloc[19].notna().all()
    assert panel.loc[0, "available_date"] == dates[1]
    assert panel.loc[0, "available_time"] == dates[1] + pd.Timedelta(hours=9, minutes=30)
    assert panel.iloc[-1]["available_date"] == pd.Timestamp("2025-02-05")
    assert not np.isinf(panel[list(TICK_FACTOR_COLUMNS)].dropna().to_numpy()).any()


def test_ev_and_conditional_flow_factors_are_cross_sectional() -> None:
    dates = pd.bdate_range("2025-01-02", periods=21)
    rows = []
    for day_index, date in enumerate(dates):
        for symbol_index in range(20):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"{symbol_index:06d}.SZ",
                    "daily_close": 100 + symbol_index + day_index * 0.1,
                    "adj_factor": 1.0,
                    "daily_return": 0.001 * symbol_index,
                    "daily_amount": 10_000_000 + symbol_index * 100_000,
                    "trade_count": 240,
                    "valid_trade_minutes": 240,
                    "amount_available": True,
                    "daily_qua": 0.1,
                    "daily_q90_q10_ratio": 2.0,
                    "daily_mts": 0.1,
                    "daily_mte": 0.1,
                    "daily_sr_l020": 0.1,
                    "daily_mean_trade_notional": 100.0,
                    "daily_mean_trade_volume": 10.0,
                    "daily_flow_extra_buy": 1_000_000 + symbol_index * 10_000,
                    "daily_flow_extra_sell": 500_000,
                    "daily_flow_large_buy": 2_000_000 + symbol_index * 10_000,
                    "daily_flow_large_sell": 1_000_000,
                    "daily_flow_medium_buy": 3_000_000,
                    "daily_flow_medium_sell": 2_000_000,
                    "daily_flow_small_buy": 2_000_000,
                    "daily_flow_small_sell": 1_000_000,
                    "daily_flow_ge20k_buy": 5_000_000,
                    "daily_flow_ge20k_sell": 2_000_000,
                    "daily_flow_power_buy": 6_000_000,
                    "daily_flow_power_sell": 3_000_000,
                    "daily_flow_total_buy": 8_000_000,
                    "daily_flow_total_sell": 4_000_000,
                    "daily_flow_large_high_amp_buy": 1_000_000 + symbol_index * 10_000,
                    "daily_flow_large_high_amp_sell": 500_000,
                    "daily_flow_small_high_amp_buy": 500_000,
                    "daily_flow_small_high_amp_sell": 1_000_000,
                    "daily_flow_large_open30_buy": 1_000_000 + symbol_index * 10_000,
                    "daily_flow_large_open30_sell": 500_000,
                    "daily_flow_small_open30_buy": 500_000,
                    "daily_flow_small_open30_sell": 1_000_000,
                    "flow_classification_valid": True,
                }
            )
    panel = build_open_source_tick_factor_panel(pd.DataFrame(rows))
    latest = panel.loc[panel["trade_date"].eq(panel["trade_date"].max())]
    assert latest["kaiyuan_evl_m20"].notna().all()
    assert latest["kaiyuan_evm_m20"].notna().all()
    assert latest["kaiyuan_evs_m20"].notna().all()
    assert latest["kaiyuan_flow_large_high_amp_resid_m20"].notna().all()
    assert latest["kaiyuan_flow_small_open30_resid_m20"].notna().all()


def test_tick_ideal_reversal_siblings_use_daily_trade_intensity() -> None:
    frames = []
    dates = pd.bdate_range("2025-01-02", periods=22)
    for i, date in enumerate(dates):
        frame = _transactions(date.strftime("%Y-%m-%d"), amount_scale=1 + i / 50)
        frame.loc[frame.index[-1], "price"] *= 1 + (0.002 if i % 2 else -0.002)
        frames.append(build_daily_tick_raw_features(frame, "000001.SZ"))
    raw = pd.concat(frames, ignore_index=True)
    raw["adj_factor"] = 1.0
    panel = add_availability_columns(build_open_source_tick_factor_panel(raw))
    assert pd.notna(panel["kaiyuan_ideal_reversal_tick_notional_m20"].iloc[-1])
    assert pd.notna(panel["kaiyuan_ideal_reversal_tick_volume_m20"].iloc[-1])
    assert pd.notna(panel["kaiyuan_trade_notional_return_corr_m20"].iloc[-1])


def test_order_memory_factors_use_order_events_and_keep_audit_fields() -> None:
    times = pd.date_range("2025-01-02 09:30", periods=240, freq="min")
    orders = pd.DataFrame(
        {
            "trade_time": times,
            "side": np.where((np.arange(len(times)) // 3) % 2, "S", "B"),
            "order_type": "0",
            "order_id": np.arange(len(times)) + 1,
            "price": 10.0,
            "quantity": 100,
            "ts_code": "000001.SZ",
        }
    )
    raw = build_daily_order_raw_features(orders, "000001.SZ")
    assert raw.loc[0, "valid_order_events"] == 150
    assert raw.loc[0, "order_id_available"]
    assert raw.loc[0, "order_side_coverage"] == 1.0
    assert pd.notna(raw.loc[0, "daily_order_lms"])
    assert pd.notna(raw.loc[0, "daily_order_island_mean"])


def test_trade_code_cancellation_and_scaled_csv_values() -> None:
    frame = pd.DataFrame(
        {
            "自然日": [20250102, 20250102],
            "时间": [93000000, 93001000],
            "成交代码": ["0", "C"],
            "成交价格": [100000, 100000],
            "成交数量": [100, 100],
        }
    )
    normalized = normalize_tick_transactions(frame, "000001.SZ")
    assert len(normalized) == 1
    assert normalized.iloc[0]["price"] == 10.0
    assert normalized.iloc[0]["amount"] == 1000.0


def test_cli_single_symbol_smoke(tmp_path: Path) -> None:
    input_root = tmp_path / "ticks"
    output_root = tmp_path / "out"
    daily_path = tmp_path / "stock_daily.parquet"
    day_dir = input_root / "2025" / "202501" / "20250102" / "000001.SZ"
    day_dir.mkdir(parents=True)
    _transactions("2025-01-02").to_csv(day_dir / "transactions.csv", index=False)
    # The generator intentionally discovers the archive's canonical filenames.
    _transactions("2025-01-02").to_csv(day_dir / "逐笔成交.csv", index=False)
    pd.DataFrame(
        {
            "trade_time": pd.date_range("2025-01-02 09:30", periods=30, freq="min"),
            "side": np.where(np.arange(30) % 2, "S", "B"),
            "order_type": "0",
            "order_id": np.arange(30) + 1,
            "price": 10.0,
            "quantity": 100,
            "ts_code": "000001.SZ",
        }
    ).to_parquet(day_dir / "orders.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")],
            "ts_code": ["000001.SZ"],
            "adj_factor": [1.0],
        }
    ).set_index(["trade_date", "ts_code"]).to_parquet(daily_path)
    script = Path(__file__).parents[1] / "scripts" / "generate_open_source_tick_factors.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--daily-path",
            str(daily_path),
            "--symbols",
            "000001.SZ",
            "--workers",
            "1",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = pd.read_parquet(output_root / "000001.SZ.parquet")
    assert list(result.columns) == [*AUDIT_COLUMNS, *TICK_FACTOR_COLUMNS]
    assert result.loc[0, "source_level"] == "tick_transaction+order"
    assert result.loc[0, "valid_order_events"] == 30
    assert result.loc[0, "order_id_available"]


def _matched_flow_day(
    day: str,
    symbol: str,
    day_index: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range(f"{day} 09:30", periods=4, freq="min")
    trade_ids = np.arange(1, 5) + day_index * 1000
    sides = np.array(["B", "S", "B", "S"])
    transactions = pd.DataFrame(
        {
            "trade_time": times,
            "price": [10.0, 10.0, 10.0, 10.0],
            "quantity": [10, 10, 10, 10],
            "ts_code": symbol,
            "bs_flag": sides,
            "buy_order_id": trade_ids,
            "sell_order_id": trade_ids + 500,
        }
    )
    order_ids = np.concatenate([trade_ids, trade_ids + 500])
    order_sides = np.concatenate([sides, np.where(sides == "B", "S", "B")])
    order_quantities = [30_000, 100, 5_000, 100_000] * 2
    orders = pd.DataFrame(
        {
            "trade_time": times.tolist() * 2,
            "side": order_sides,
            "order_type": "0",
            "order_id": order_ids,
            "price": 10.0,
            "quantity": order_quantities,
            "ts_code": symbol,
        }
    )
    return transactions, orders


def test_transaction_aliases_keep_buy_and_sell_order_ids() -> None:
    raw = pd.DataFrame(
        {
            "自然日": [20250102],
            "时间": [93000000],
            "成交代码": ["0"],
            "成交价格": [100000],
            "成交数量": [100],
            "BS标志": ["B"],
            "叫卖序号": [22],
            "叫买序号": [11],
        }
    )
    normalized = normalize_tick_transactions(raw, "000001.SZ")
    assert normalized.loc[0, "buy_order_id"] == 11
    assert normalized.loc[0, "sell_order_id"] == 22
    assert normalized.loc[0, "price"] == 10.0


def test_order_notional_buckets_use_fill_amount_for_flow() -> None:
    transactions, orders = _matched_flow_day("2025-01-02", "000001.SZ")
    result = build_daily_flow_raw_features(transactions, orders, "000001.SZ")
    row = result.iloc[0]
    assert row["flow_classification_valid"]
    assert row["active_order_match_amount_coverage"] == 1.0
    # The first fill is 10 * 10 yuan, while its original order is 300,000 yuan.
    assert row["daily_flow_large_buy"] == 100.0
    assert row["daily_flow_total_buy"] == 200.0
    assert row["daily_flow_total_sell"] == 200.0


def test_sh_immediate_execution_taker_orders_inferred_for_flow() -> None:
    # Simulates Shanghai Level-2 where resting maker orders are in the order stream,
    # but immediate-execution taker orders are not in the order stream.
    times = pd.date_range("2025-01-02 09:30", periods=3, freq="min")
    transactions = pd.DataFrame(
        {
            "trade_time": times,
            "price": [10.0, 10.0, 10.0],
            "quantity": [5_000, 5_000, 20_000],  # 50,000, 50,000, 200,000 yuan
            "ts_code": "510050.SH",
            "bs_flag": ["B", "B", "S"],
            # Taker order 101 executes two fills: 50,000 + 50,000 = 100,000 yuan (medium)
            # Taker order 102 executes one fill: 200,000 yuan (large)
            "buy_order_id": [101, 101, 201],
            "sell_order_id": [301, 302, 102],
        }
    )
    # Orders stream only has passive maker orders (301, 302, 201), NOT taker orders (101, 102)
    orders = pd.DataFrame(
        {
            "trade_time": times.tolist(),
            "side": ["S", "S", "B"],
            "order_type": "0",
            "order_id": [301, 302, 201],
            "price": [10.0, 10.0, 10.0],
            "quantity": [5_000, 5_000, 20_000],
            "ts_code": "510050.SH",
        }
    )
    result = build_daily_flow_raw_features(transactions, orders, "510050.SH")
    row = result.iloc[0]
    assert row["flow_classification_valid"]
    assert row["active_order_match_amount_coverage"] == 1.0
    # Order 101 has total notional 100,000 yuan -> medium order bucket [40k, 200k)
    # Both fills (50,000 + 50,000 = 100,000) go to daily_flow_medium_buy
    assert row["daily_flow_medium_buy"] == 100_000.0
    assert row["daily_flow_small_buy"] == 0.0
    # Order 102 has total notional 200,000 yuan -> large order bucket [200k, 1M)
    assert row["daily_flow_large_sell"] == 200_000.0
    assert row["daily_flow_total_buy"] == 100_000.0
    assert row["daily_flow_total_sell"] == 200_000.0


def test_flow_factors_require_cross_section_and_keep_unmatched_days_null() -> None:
    raw_frames = []
    for symbol_index in range(20):
        symbol = f"{symbol_index:06d}.SZ"
        daily_transactions = []
        daily_orders = []
        for day_index, day in enumerate(pd.bdate_range("2025-01-02", periods=21)):
            transactions, orders = _matched_flow_day(
                day.strftime("%Y-%m-%d"), symbol, day_index
            )
            transactions["price"] *= 1.0 + symbol_index * 0.001 + day_index * 0.0001
            orders["price"] *= 1.0 + symbol_index * 0.001 + day_index * 0.0001
            daily_transactions.append(transactions)
            daily_orders.append(orders)
        transactions = pd.concat(daily_transactions, ignore_index=True)
        orders = pd.concat(daily_orders, ignore_index=True)
        raw_frames.append(
            build_daily_tick_raw_features(transactions, symbol).merge(
                build_daily_flow_raw_features(transactions, orders, symbol),
                on=["trade_date", "ts_code"],
                how="left",
            )
        )
    panel = build_open_source_tick_factor_panel(
        pd.concat(raw_frames, ignore_index=True).assign(adj_factor=1.0)
    )
    last_day = panel.loc[panel["trade_date"] == pd.Timestamp("2025-01-30")]
    assert last_day["flow_cross_section_count"].eq(20).all()
    assert last_day["kaiyuan_nir_mod_ge20k_cs_m20"].notna().all()
    assert last_day["kaiyuan_cnir_cs_m20"].notna().all()
    assert last_day["kaiyuan_act_pos_highret_m20_l010"].notna().all()
    assert last_day["kaiyuan_act_neg_lowret_m20_l010"].notna().all()

    incomplete = raw_frames[0].copy()
    incomplete.loc[:, "active_order_match_amount_coverage"] = 0.1
    incomplete.loc[:, "flow_classification_valid"] = False
    incomplete.loc[:, "daily_flow_large_buy"] = np.nan
    assert build_open_source_tick_factor_panel(incomplete.assign(adj_factor=1.0))[
        "kaiyuan_large_flow_s3_m20"
    ].isna().all()


def test_cross_day_tick_returns_are_invariant_to_split_adjustment() -> None:
    dates = pd.bdate_range("2025-01-02", periods=21)
    raw = pd.concat(
        [
            build_daily_tick_raw_features(
                _transactions(date.strftime("%Y-%m-%d"), amount_scale=1 + index / 100),
                "000001.SZ",
            )
            for index, date in enumerate(dates)
        ],
        ignore_index=True,
    )
    raw["adj_factor"] = 1.0
    baseline_return, baseline_ret20 = _adjusted_close_returns(raw, 20)
    split = raw.copy()
    split.loc[20, "daily_close"] /= 2.0
    split.loc[20, "adj_factor"] = 2.0
    adjusted_return, adjusted_ret20 = _adjusted_close_returns(split, 20)

    assert np.isclose(adjusted_return.loc[20], baseline_return.loc[20])
    assert np.isclose(adjusted_ret20.loc[20], baseline_ret20.loc[20])


@pytest.mark.parametrize("kind", ["transactions", "orders"])
def test_price_units_follow_schema_not_price_magnitude(kind: str) -> None:
    normalize = normalize_tick_transactions if kind == "transactions" else normalize_tick_orders
    canonical = pd.DataFrame({"trade_time": pd.to_datetime(["2026-09-18 09:31"]),
                              "price": [1500.0], "quantity": [100.0], "side": ["B"]})
    assert normalize(canonical, "600519.SH")["price"].iloc[0] == 1500.0
    assert normalize(canonical, "600519.SH", price_divisor=10000)["price"].iloc[0] == 0.15
    archive = pd.DataFrame({"raw_trade_date": [20260918] * 5,
                            "raw_time": [93100000] * 5,
                            "price": [0, 0, 0, 0, 8800], "quantity": [100] * 5,
                            "side": ["B"] * 5, "trade_code": ["C"] * 4 + ["0"]})
    normalized = normalize(archive, "159007.SZ")
    assert normalized.loc[normalized["price"].gt(0), "price"].tolist() == [0.88]
    if kind == "transactions":
        assert normalized["amount"].tolist() == [88.0]


def test_close_auction_is_preserved_for_close_amount_and_returns() -> None:
    trades = pd.DataFrame({"trade_time": pd.to_datetime([
        "2026-09-17 09:30", "2026-09-17 14:56", "2026-09-17 15:00", "2026-09-17 15:01",
        "2026-09-18 09:30", "2026-09-18 14:56", "2026-09-18 15:00"]),
        "price": [10., 10., 11., 99., 10., 10., 12.], "quantity": [100.] * 7})
    raw = build_daily_tick_raw_features(trades, "000001.SZ")
    assert raw["daily_close"].tolist() == [11., 12.]
    assert raw["daily_amount"].tolist() == [3100., 3200.]
    returns, _ = _adjusted_close_returns(raw.assign(adj_factor=1.), 1)
    assert returns.iloc[-1] == pytest.approx(12 / 11 - 1)


def test_sr_includes_opening_minute_price_move() -> None:
    trades = _transactions("2026-09-18")
    trades["price"] = 110.
    trades.loc[0, "trade_time"] += pd.Timedelta(seconds=30)
    first = trades.iloc[[0]].copy()
    first["trade_time"] -= pd.Timedelta(seconds=30)
    first["price"] = 100.
    first["quantity"] = 100000.
    raw = build_daily_tick_raw_features(pd.concat([first, trades]), "000001.SZ")
    assert raw.loc[0, "daily_sr_l020"] == pytest.approx(0.1)


def _raw_ev_panel(symbol_count: int = 25, days: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(15)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2026-09-14", periods=days)):
        for symbol_index in range(symbol_count):
            row = {"trade_date": date, "ts_code": f"{symbol_index:06d}.SZ",
                   "daily_close": 100. + day_index + symbol_index, "adj_factor": 1.,
                   "daily_return": 0.01, "daily_amount": 1_000_000., "trade_count": 240,
                   "valid_trade_minutes": 240, "amount_available": True,
                   "daily_mean_trade_notional": 1000., "daily_mean_trade_volume": 100.,
                   "flow_bucket_schema": FLOW_BUCKET_SCHEMA, "flow_classification_valid": True}
            for column in ("daily_qua", "daily_q90_q10_ratio", "daily_mts", "daily_mte", "daily_sr_l020"):
                row[column] = .1
            for category, share in zip(("extra", "large", "medium", "small"), rng.dirichlet([1.] * 4)):
                row[f"daily_flow_{category}_buy"] = 1_000_000. * share * .6
                row[f"daily_flow_{category}_sell"] = 1_000_000. * share * .4
            rows.append(row)
    return pd.DataFrame(rows)


def test_ev_uses_three_separate_regressions_on_extra_share() -> None:
    raw = _raw_ev_panel()
    panel = build_open_source_tick_factor_panel(raw, window=4)
    latest = panel.loc[panel["trade_date"].eq(panel["trade_date"].max())].set_index("ts_code")
    averages = raw.groupby("ts_code").mean(numeric_only=True)
    y = (averages["daily_flow_extra_buy"] + averages["daily_flow_extra_sell"]) / averages["daily_amount"]
    for category, label in (("large", "evl"), ("medium", "evm"), ("small", "evs")):
        x = (averages[f"daily_flow_{category}_buy"] + averages[f"daily_flow_{category}_sell"]) / averages["daily_amount"]
        centered_x, centered_y = x - x.mean(), y - y.mean()
        slope = (centered_x * centered_y).sum() / centered_x.pow(2).sum()
        expected = centered_y - slope * centered_x
        actual = latest[f"kaiyuan_{label}_m20"]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert actual.std() > .001
        assert latest[f"flow_{label}_r2"].lt(.999).all()


def test_volume_reversal_is_split_invariant_and_requires_adjustment() -> None:
    raw = _raw_ev_panel(symbol_count=1)
    raw["daily_return"] = [.01, -.01, .02, -.02]
    raw["daily_mean_trade_volume"] = [100., 200., 150., 180.]
    baseline = build_open_source_tick_factor_panel(raw, window=4)
    split = raw.copy()
    split.loc[2:, "daily_close"] /= 2
    split.loc[2:, "adj_factor"] = 2.
    split.loc[2:, "daily_mean_trade_volume"] *= 2
    adjusted = build_open_source_tick_factor_panel(split, window=4)
    column = "kaiyuan_ideal_reversal_tick_volume_m20"
    assert baseline[column].iloc[-1] == pytest.approx(-.03)
    np.testing.assert_allclose(baseline[column], adjusted[column], equal_nan=True)
    split["adj_factor"] = np.nan
    assert build_open_source_tick_factor_panel(split, window=4)[column].isna().all()


def test_conditional_buckets_are_disjoint_and_follow_daily_buckets() -> None:
    amounts = [19999., 20000., 30000., 39999., 40000., 199999., 200000., 999999., 1000000.]
    times = pd.date_range("2026-09-18 09:30", periods=len(amounts), freq="min")
    trades = pd.DataFrame({"trade_time": times, "price": 10., "quantity": 100.,
                           "bs_flag": "B", "buy_order_id": np.arange(1, len(amounts) + 1)})
    orders = pd.DataFrame({"trade_time": times, "price": 10., "quantity": np.array(amounts) / 10,
                           "side": "B", "order_type": "0", "order_id": np.arange(1, len(amounts) + 1)})
    row = build_daily_flow_raw_features(trades, orders, "000001.SZ").iloc[0]
    assert row["daily_flow_large_open30_buy"] == row["daily_flow_large_buy"] == 2000.
    assert row["daily_flow_small_open30_buy"] == row["daily_flow_small_buy"] == 4000.
    assert row["daily_flow_ge20k_buy"] == 8000.
    assert row["daily_flow_total_buy"] == 9000.


def test_availability_uses_calendar_even_at_input_end() -> None:
    raw = pd.DataFrame({"trade_date": pd.to_datetime(["2026-09-21", "2026-09-24", "2026-09-30"]),
                        "ts_code": ["000001.SZ"] * 3})
    result = add_availability_columns(raw)
    assert result["available_date"].tolist() == list(pd.to_datetime(["2026-09-22", "2026-09-28", "2026-10-08"]))
    assert result["available_time"].dt.hour.eq(9).all()
    assert result["available_time"].dt.minute.eq(30).all()


def test_bucket_schema_survives_panel_and_parquet_roundtrip(tmp_path: Path) -> None:
    result = select_output_columns(add_availability_columns(build_open_source_tick_factor_panel(_raw_ev_panel(1))))
    path = tmp_path / "factor.parquet"
    result.to_parquet(path, index=False)
    assert pd.read_parquet(path)["flow_bucket_schema"].eq(FLOW_BUCKET_SCHEMA).all()


def test_missing_bucket_schema_gets_explicit_contract_value() -> None:
    raw = _raw_ev_panel(1).drop(columns="flow_bucket_schema")
    panel = build_open_source_tick_factor_panel(raw)
    assert panel["flow_bucket_schema"].eq(FLOW_BUCKET_SCHEMA).all()


def test_order_only_cli_emits_order_factors_without_daily_prices(tmp_path: Path, monkeypatch) -> None:
    input_root, output_root = tmp_path / "ticks", tmp_path / "out"
    for date in pd.bdate_range("2026-08-03", periods=20):
        day = date.strftime("%Y%m%d")
        directory = input_root / day[:4] / day[:6] / day / "000001.SZ"
        directory.mkdir(parents=True)
        orders = pd.DataFrame({"trade_time": pd.date_range(date + pd.Timedelta(hours=9, minutes=30), periods=100, freq="s"),
                               "side": np.where(np.arange(100) % 6 < 3, "B", "S"),
                               "order_type": "0", "order_id": np.arange(100) + 1,
                               "price": 10., "quantity": 100.})
        orders.to_parquet(directory / "orders.parquet", index=False)
    monkeypatch.setattr(sys, "argv", ["generate_open_source_tick_factors.py", "--input-root", str(input_root),
                                     "--output-root", str(output_root), "--daily-path", str(tmp_path / "missing.parquet"), "--workers", "1"])
    assert generator.main() == 0
    result = pd.read_parquet(output_root / "000001.SZ.parquet")
    assert result["source_level"].eq("tick_order").all()
    assert result["kaiyuan_order_lms_m20"].notna().sum() == 1
    assert result["kaiyuan_order_island_mean_m20"].iloc[-1] > 1
    assert result[list(TICK_FACTOR_COLUMNS[:8])].isna().all().all()
    assert result["available_time"].notna().all()


def test_cli_subset_and_limit_keep_full_ev_cross_section(tmp_path: Path, monkeypatch) -> None:
    import json
    raw = _raw_ev_panel(symbol_count=25, days=20)
    symbols = sorted(raw["ts_code"].unique())
    files = {symbol: {"transactions": [], "orders": []} for symbol in symbols}
    def discover(input_root, selected, limit, date_from, date_to):
        assert selected is None and limit is None and date_from is None
        return files
    monkeypatch.setattr(generator, "discover_symbol_files", discover)
    monkeypatch.setattr(generator, "build_raw_panel", lambda files, workers: (raw.drop(columns="adj_factor"), []))
    monkeypatch.setattr(generator, "load_daily_adjustment_factors", lambda *args: raw[["trade_date", "ts_code", "adj_factor"]])
    for folder, extra in (("full", []), ("subset", ["--symbols", symbols[0]]), ("limited", ["--limit", "1"])):
        monkeypatch.setattr(sys, "argv", ["generate_open_source_tick_factors.py", "--input-root", str(tmp_path / "ticks"),
                                         "--output-root", str(tmp_path / folder), "--workers", "1"] + extra)
        assert generator.main() == 0
    full = pd.read_parquet(tmp_path / "full" / f"{symbols[0]}.parquet")
    assert full["kaiyuan_evl_m20"].notna().any()
    for folder in ("subset", "limited"):
        pd.testing.assert_frame_equal(full, pd.read_parquet(tmp_path / folder / f"{symbols[0]}.parquet"))
        manifest = json.loads((tmp_path / folder / "manifest.json").read_text())
        assert manifest["calculation_universe"] == symbols
        assert len(list((tmp_path / folder).glob("*.parquet"))) == 1
    files["999999.SZ"] = {"transactions": [], "orders": []}
    monkeypatch.setattr(sys, "argv", ["generate_open_source_tick_factors.py", "--input-root", str(tmp_path / "ticks"),
                                     "--output-root", str(tmp_path / "full"), "--overwrite"])
    with pytest.raises(ValueError, match="contract changed"):
        generator.main()


@pytest.mark.parametrize("workers", [1, 2])
def test_extraction_failure_aborts_all_output(tmp_path: Path, monkeypatch, workers: int) -> None:
    directory = tmp_path / "input" / "2026" / "202609" / "20260918"
    good, bad = directory / "000001.SZ", directory / "000002.SZ"
    good.mkdir(parents=True)
    bad.mkdir()
    _transactions("2026-09-18").to_parquet(good / "transactions.parquet", index=False)
    (bad / "transactions.parquet").write_bytes(b"broken parquet")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["generate_open_source_tick_factors.py", "--input-root", str(tmp_path / "input"),
                                     "--output-root", str(output), "--symbols", "000001.SZ", "--workers", str(workers)])
    assert generator.main() == 1
    assert not output.exists()


def test_cli_rejects_legacy_outputs_before_extracting(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "000001.SZ.parquet"
    sentinel.write_bytes(b"legacy")
    monkeypatch.setattr(generator, "discover_symbol_files", lambda *args: {"000001.SZ": {"transactions": [], "orders": []}})
    monkeypatch.setattr(sys, "argv", ["generate_open_source_tick_factors.py", "--output-root", str(output), "--overwrite"])
    with pytest.raises(ValueError, match="no calculation manifest"):
        generator.main()
    assert sentinel.read_bytes() == b"legacy"
