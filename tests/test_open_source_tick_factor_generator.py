from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factor.open_source_tick import (
    AUDIT_COLUMNS,
    TICK_FACTOR_COLUMNS,
    add_availability_columns,
    build_daily_flow_raw_features,
    build_daily_tick_raw_features,
    build_daily_order_raw_features,
    build_open_source_tick_factor_panel,
    normalize_tick_transactions,
)


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
    panel = add_availability_columns(build_open_source_tick_factor_panel(raw))
    assert panel[list(TICK_FACTOR_COLUMNS[:5])].iloc[19].notna().all()
    assert panel.loc[0, "available_date"] == dates[1]
    assert panel.loc[0, "available_time"] == dates[1] + pd.Timedelta(hours=9, minutes=30)
    assert pd.isna(panel.iloc[-1]["available_date"])
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
    script = Path(__file__).parents[1] / "scripts" / "generate_open_source_tick_factors.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
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
        pd.concat(raw_frames, ignore_index=True)
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
    assert build_open_source_tick_factor_panel(incomplete)[
        "kaiyuan_large_flow_s3_m20"
    ].isna().all()
