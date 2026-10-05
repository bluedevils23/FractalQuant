from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.auction_tick_cache import AuctionTickCache
from scripts.generate_close_auction_factors import (
    _cancel_notional_by_side,
    OUTPUT_COLUMNS,
    calculate_daily_close_auction_factors,
    merge_symbol_output,
    process_symbol_series,
)


@pytest.mark.parametrize("ids,quantities", [([1, 999], [20, 50]), ([1, 1], [80, 80])])
def test_invalid_cancel_coverage_and_cumulative_quantity(ids, quantities):
    adds = pd.DataFrame({"order_id": [1], "side": ["B"], "price": [10.], "quantity": [100.]})
    cancels = pd.DataFrame({"order_id": ids, "quantity": quantities})
    assert not _cancel_notional_by_side(adds, cancels)[2]


def test_empty_transactions_are_unknown_not_zero():
    orders = _orders(("2026-03-31 14:58:00", 1, "A", "B", 10., 100.))
    row = calculate_daily_close_auction_factors(orders, _transactions(), "600000.SH", "20260401")
    assert not row["close_auction_event_reconstruction_ok"]
    assert pd.isna(row["close_auction_matched_notional"])
    assert pd.isna(row["close_auction_order_imbalance"])


def test_failed_refresh_preserves_old_record(tmp_path):
    path = tmp_path / "old.parquet"
    old = pd.DataFrame([{c: np.nan for c in OUTPUT_COLUMNS}])
    old["trade_date"] = "2026-03-31"
    old.to_parquet(path)
    result = merge_symbol_output(path, pd.DataFrame(columns=OUTPUT_COLUMNS), False, {"2026-03-31"})
    assert result.trade_date.tolist() == ["2026-03-31"]


def test_millisecond_close_match_is_included():
    orders = _orders(("2026-03-31 14:58:00", 1, "A", "B", 10., 100.),
                     ("2026-03-31 14:58:00", 2, "A", "S", 10., 100.))
    trades = _transactions(("2026-03-31 15:00:00.500", "0", 10., 50., 2, 1))
    row = calculate_daily_close_auction_factors(orders, trades, "600000.SH", "20260401")
    assert row["close_auction_matched_notional"] == 500.


def _orders(*rows: tuple[str, int, str, str, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["trade_time", "order_id", "order_type", "side", "price", "quantity"],
    ).assign(trade_time=lambda frame: pd.to_datetime(frame["trade_time"]))


@pytest.mark.parametrize("day,asset_type,applicable", [
    ("2026-07-03", "etf", False), ("2026-07-06", "etf", True),
    ("2026-07-03", "stock", True),
])
def test_sh_fund_session_switch(day, asset_type, applicable):
    orders = _orders((day + " 14:58", 1, "A", "B", 10., 100.),
                     (day + " 14:58", 2, "A", "S", 10., 100.))
    trades = _transactions((day + " 15:00:01.350", "", 10., 50., 2, 1))
    row = calculate_daily_close_auction_factors(orders, trades, "510300.SH", None, asset_type=asset_type)
    assert row["auction_applicable"] == applicable
    if not applicable:
        assert row["auction_status"] == "legacy_continuous"
        assert not row["close_auction_event_reconstruction_ok"]
        assert not pd.isna(row["close_auction_order_imbalance"])
    else:
        assert row["close_auction_event_reconstruction_ok"]


def test_delayed_match_old_order_linkage_and_postclose_exclusion():
    orders = _orders(("2026-07-27 10:00", 1, "A", "B", 10., 100.),
                     ("2026-07-27 14:58", 2, "A", "S", 10., 100.),
                     ("2026-07-27 14:58", 3, "A", "B", 10., 100.),
                     ("2026-07-27 14:59", 1, "D", "B", 10., 10.))
    trades = _transactions(("2026-07-27 15:00:01.350", "", 10., 50., 2, 1),
                           ("2026-07-27 15:05:00.000", "", 10., 20., 2, 3))
    row = calculate_daily_close_auction_factors(orders, trades, "510300.SH", None, asset_type="etf")
    assert row["close_auction_event_reconstruction_ok"]
    assert row["close_auction_matched_notional"] == 500.
    assert row["close_auction_match_count"] == 1
    assert row["close_auction_order_count"] == 2
    assert row["close_auction_cancel_buy_notional"] == 0.
    assert row["close_auction_attributed_buy_match_notional"] == 0.
    assert row["close_auction_attributed_sell_match_notional"] == 500.
    assert row["source_available_time"] == pd.Timestamp("2026-07-27 15:00:01.350")
    trades.loc[0, "bid_order_id"] = 999
    assert not calculate_daily_close_auction_factors(orders, trades, "510300.SH", None, asset_type="etf")["close_auction_event_reconstruction_ok"]


def _transactions(*rows: tuple[str, str, float, float, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["trade_time", "trade_code", "price", "quantity", "ask_order_id", "bid_order_id"],
    ).assign(
        trade_time=lambda frame: pd.to_datetime(frame["trade_time"]),
        bs_flag="",
    )


def test_close_auction_factors_link_matches_and_defer_availability() -> None:
    orders = _orders(
        ("2026-03-31 14:57:00", 1, "A", "B", 10.0, 100.0),
        ("2026-03-31 14:58:00", 2, "A", "S", 11.0, 200.0),
        ("2026-03-31 14:59:00", 2, "D", "S", 11.0, 50.0),
    )
    transactions = _transactions(
        ("2026-03-31 14:59:59", "", 10.5, 40.0, 2, 1),
        ("2026-03-31 15:00:00", "", 10.5, 100.0, 2, 1),
    )

    row = calculate_daily_close_auction_factors(
        orders, transactions, "600000.SH", "20260401"
    )

    assert row["trade_date"] == "2026-03-31"
    assert row["available_time"] == pd.Timestamp("2026-04-01 09:15:00")
    assert row["source_available_time"] == pd.Timestamp("2026-03-31 15:00:00")
    assert row["close_auction_order_count"] == 2
    assert row["close_auction_cancel_count"] == 1
    assert row["close_auction_match_count"] == 1
    assert row["close_auction_submitted_buy_notional"] == 1000.0
    assert row["close_auction_submitted_sell_notional"] == 2200.0
    assert row["close_auction_cancel_sell_notional"] == 550.0
    assert row["close_auction_matched_notional"] == 1050.0
    assert row["close_auction_buy_match_to_submitted_notional_proxy"] == 1.05
    assert row["close_auction_sell_match_to_submitted_notional_proxy"] == 1050.0 / 2200.0
    assert row["close_auction_match_to_submitted_notional_proxy"] == 2100.0 / 3200.0


def test_sz_cancellation_uses_transaction_order_identifier() -> None:
    orders = _orders(
        ("2026-03-31 14:57:00", 1, "0", "B", 10.0, 100.0),
        ("2026-03-31 14:58:00", 2, "0", "S", 11.0, 200.0),
    )
    transactions = _transactions(
        ("2026-03-31 14:59:00", "C", 0.0, 50.0, 2, 0),
        ("2026-03-31 15:00:00", "0", 10.5, 100.0, 2, 1),
    )

    row = calculate_daily_close_auction_factors(
        orders, transactions, "000001.SZ", "20260401"
    )

    assert row["close_auction_event_reconstruction_ok"]
    assert row["close_auction_cancel_sell_notional"] == 550.0
    assert row["close_auction_cancel_notional_ratio"] == 550.0 / 3200.0


def test_close_tick_cache_uses_expected_boundaries(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "2026" / "202603" / "20260331" / "600000.SH"
    symbol_dir.mkdir(parents=True)
    orders = pd.DataFrame(
        {
            "自然日": [20260331] * 4,
            "时间": [145659999, 145700000, 145959999, 150000000],
            "交易所委托号": [1, 2, 3, 4],
            "委托类型": ["A"] * 4,
            "委托代码": ["B"] * 4,
            "委托价格": [100000] * 4,
            "委托数量": [100] * 4,
        }
    )
    transactions = pd.DataFrame(
        {
            "自然日": [20260331] * 6,
            "时间": [145659999, 145700000, 145959999, 150000000, 150001350, 150500000],
            "成交代码": ["0"] * 6,
            "BS标志": ["B"] * 6,
            "成交价格": [100000] * 6,
            "成交数量": [100] * 6,
            "叫卖序号": [1] * 6,
            "叫买序号": [2] * 6,
        }
    )
    orders.to_csv(symbol_dir / "逐笔委托.csv", index=False, encoding="gbk")
    transactions.to_csv(symbol_dir / "逐笔成交.csv", index=False, encoding="gbk")

    cache = AuctionTickCache(None)
    close_orders = cache.load_close_orders(symbol_dir)
    close_transactions = cache.load_close_transactions(symbol_dir)
    history = cache.load_close_order_history(symbol_dir)
    assert history["order_id"].tolist() == [1, 2, 3]

    assert close_orders["trade_time"].tolist() == [
        pd.Timestamp("2026-03-31 14:57:00"),
        pd.Timestamp("2026-03-31 14:59:59.999"),
    ]
    assert close_transactions["trade_time"].tolist() == [
        pd.Timestamp("2026-03-31 14:57:00"),
        pd.Timestamp("2026-03-31 14:59:59.999"),
        pd.Timestamp("2026-03-31 15:00:00"),
        pd.Timestamp("2026-03-31 15:00:01.350"),
    ]


def test_merge_replaces_requested_date_without_duplicates(tmp_path: Path) -> None:
    output_path = tmp_path / "600000.SH.parquet"
    existing = pd.DataFrame(
        {
            column: pd.Series([np.nan, np.nan], dtype="object")
            for column in OUTPUT_COLUMNS
        }
    )
    existing.loc[:, "trade_date"] = ["2026-03-30", "2026-03-31"]
    existing.loc[:, "ts_code"] = "600000.SH"
    existing.to_parquet(output_path, index=False)
    requested = existing.iloc[[1]].copy()
    requested.loc[:, "close_auction_order_imbalance"] = 0.25

    merged = merge_symbol_output(output_path, requested, overwrite=True)

    assert len(merged) == 2
    assert not merged["trade_date"].duplicated().any()
    assert merged.loc[merged["trade_date"].eq("2026-03-31"), "close_auction_order_imbalance"].iloc[0] == 0.25


def test_existing_close_auction_date_skips_tick_load(tmp_path: Path, monkeypatch) -> None:
    symbol_dir = tmp_path / "2026" / "202603" / "20260331" / "600000.SH"
    symbol_dir.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    existing = pd.DataFrame(
        {column: pd.Series([np.nan], dtype="object") for column in OUTPUT_COLUMNS}
    )
    existing.loc[:, "trade_date"] = "2026-03-31"
    existing.loc[:, "ts_code"] = "600000.SH"
    existing.to_parquet(output_root / "600000.SH.parquet", index=False)

    def fail_loader(*args, **kwargs):
        raise AssertionError("existing output must skip tick loading")

    monkeypatch.setattr(AuctionTickCache, "load_close_order_history", fail_loader)
    result = process_symbol_series(
        "stock",
        "600000.SH",
        [symbol_dir],
        output_root,
        "20260331",
        "20260331",
        False,
        {},
        None,
        False,
    )

    assert result[2] == 0


def test_close_auction_symbol_continues_after_single_invalid_date(
    tmp_path: Path, monkeypatch
) -> None:
    valid_dir = tmp_path / "2026" / "202603" / "20260331" / "600000.SH"
    invalid_dir = tmp_path / "2026" / "202604" / "20260401" / "600000.SH"
    valid_dir.mkdir(parents=True)
    invalid_dir.mkdir(parents=True)
    orders = _orders(("2026-03-31 14:57:00", 1, "A", "B", 10.0, 100.0),
                     ("2026-03-31 14:57:00", 2, "A", "S", 10.0, 100.0))
    transactions = _transactions(
        ("2026-03-31 15:00:00", "", 10.0, 100.0, 2, 1)
    )

    def load_orders(_cache: AuctionTickCache, path: Path) -> pd.DataFrame:
        return orders if path == valid_dir else orders.iloc[0:0]

    monkeypatch.setattr(AuctionTickCache, "load_close_order_history", load_orders)
    monkeypatch.setattr(AuctionTickCache, "load_close_transactions", lambda *args: transactions)

    with pytest.raises(ValueError, match="dates failed"):
        process_symbol_series(
        "stock",
        "600000.SH",
        [valid_dir, invalid_dir],
        tmp_path / "output",
        None,
        None,
        False,
        {"20260331": "20260401", "20260401": "20260402"},
        None,
        False,
    )

    result = pd.read_parquet(tmp_path / "output" / "600000.SH.parquet")
    assert result["trade_date"].tolist() == ["2026-03-31"]
    assert result["close_auction_event_reconstruction_ok"].all()


def test_legacy_close_auction_schema_backfills_existing_date(
    tmp_path: Path, monkeypatch
) -> None:
    symbol_dir = tmp_path / "2026" / "202603" / "20260331" / "600000.SH"
    symbol_dir.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    pd.DataFrame({"trade_date": ["2026-03-31"]}).to_parquet(
        output_root / "600000.SH.parquet", index=False
    )
    orders = _orders(("2026-03-31 14:57:00", 1, "A", "B", 10.0, 100.0),
                     ("2026-03-31 14:57:00", 2, "A", "S", 10.0, 100.0))
    transactions = _transactions(
        ("2026-03-31 15:00:00", "", 10.0, 100.0, 2, 1)
    )
    monkeypatch.setattr(AuctionTickCache, "load_close_order_history", lambda *args: orders)
    monkeypatch.setattr(
        AuctionTickCache, "load_close_transactions", lambda *args: transactions
    )

    _, output_path, row_count = process_symbol_series(
        "stock",
        "600000.SH",
        [symbol_dir],
        output_root,
        "20260331",
        "20260331",
        False,
        {"20260331": "20260401"},
        None,
        False,
    )

    result = pd.read_parquet(output_path)
    assert row_count == 1
    assert set(OUTPUT_COLUMNS).issubset(result.columns)
    assert result["trade_date"].tolist() == ["2026-03-31"]
