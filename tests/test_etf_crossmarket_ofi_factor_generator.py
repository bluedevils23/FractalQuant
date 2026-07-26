from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_etf_crossmarket_ofi_factors import (
    FACTOR_COLUMNS,
    calculate_group_ofi_factors,
    merge_factor_columns,
    normalize_orderbook_frame,
    process_reference_group,
)


def _orderbook_frame(index: pd.DatetimeIndex, multiplier: float) -> pd.DataFrame:
    steps = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "trade_date": index.strftime("%Y-%m-%d"),
            "trade_time": index.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 100.0 * multiplier + steps,
            "normalized_ofi_l1_60s": multiplier * steps + steps**2 / 70.0,
            "normalized_mlofi_l5_60s": (
                (multiplier + 1.0) * steps + steps**2 / 60.0
            ),
        }
    )


def _group_frames(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    return {
        code: normalize_orderbook_frame(_orderbook_frame(index, multiplier))
        for code, multiplier in (("159001", 1.0), ("159002", 2.0), ("159003", 3.0))
    }


def test_group_factors_use_leave_one_out_lagged_amount_weights() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="min")
    frames = _group_frames(index)
    result = calculate_group_ofi_factors(frames)["159001"]
    point = 35
    previous = point - 1
    peer_ofi = (
        frames["159002"].iloc[point]["normalized_ofi_l1_60s"],
        frames["159003"].iloc[point]["normalized_ofi_l1_60s"],
    )
    peer_weights = (
        frames["159002"].iloc[previous]["amount"],
        frames["159003"].iloc[previous]["amount"],
    )
    expected_market = np.average(peer_ofi, weights=peer_weights)
    expected_idiosyncratic = (
        frames["159001"].iloc[point]["normalized_ofi_l1_60s"]
        - result.iloc[point]["market_ofi_beta_l1_60s"] * expected_market
    )

    assert tuple(result.columns) == FACTOR_COLUMNS
    assert result["market_ofi_beta_l1_60s"].iloc[:31].isna().all()
    assert np.isclose(
        result.iloc[point]["idiosyncratic_ofi_l1_60s"],
        expected_idiosyncratic,
    )
    assert np.isclose(
        result.iloc[point]["lead_market_ofi_l1_60s"],
        np.average(
            (
                frames["159002"].iloc[previous]["normalized_ofi_l1_60s"],
                frames["159003"].iloc[previous]["normalized_ofi_l1_60s"],
            ),
            weights=(
                frames["159002"].iloc[previous - 1]["amount"],
                frames["159003"].iloc[previous - 1]["amount"],
            ),
        ),
    )
    assert result["sector_ofi_dispersion_l1_60s"].iloc[point] > 0
    assert result["market_ofi_beta_mlofi_l5_60s"].iloc[point] != result[
        "market_ofi_beta_l1_60s"
    ].iloc[point]


def test_group_factors_do_not_use_future_rows_and_reset_daily() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="min").append(
        pd.date_range("2026-01-06 09:30:00", periods=40, freq="min")
    )
    original = _group_frames(index)
    changed = {code: frame.copy() for code, frame in original.items()}
    changed["159002"].loc[index[35]:, "normalized_ofi_l1_60s"] *= -5.0
    changed["159003"].loc[index[35]:, "normalized_mlofi_l5_60s"] *= -3.0

    original_result = calculate_group_ofi_factors(original)["159001"]
    changed_result = calculate_group_ofi_factors(changed)["159001"]

    pd.testing.assert_frame_equal(
        original_result.iloc[:35], changed_result.iloc[:35]
    )
    second_day = original_result.loc["2026-01-06"]
    assert second_day["market_ofi_beta_l1_60s"].iloc[:31].isna().all()
    assert second_day["market_ofi_beta_l1_60s"].iloc[31:].notna().all()


def test_insufficient_members_and_missing_orderbook_columns() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="min")
    one_member = {"159001": _group_frames(index)["159001"]}
    result = calculate_group_ofi_factors(one_member)["159001"]

    assert result.isna().all().all()
    with pytest.raises(ValueError, match="normalized_mlofi_l5_60s"):
        normalize_orderbook_frame(pd.DataFrame({"amount": [1.0]}, index=index[:1]))


def test_no_valid_weights_and_missing_peer_timestamps() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="min")
    zero_amount_frames = _group_frames(index)
    for frame in zero_amount_frames.values():
        frame["amount"] = 0.0
    zero_amount_result = calculate_group_ofi_factors(zero_amount_frames)["159001"]
    assert zero_amount_result.isna().all().all()

    missing_timestamp_frames = _group_frames(index)
    missing_timestamp_frames["159003"] = missing_timestamp_frames["159003"].drop(
        index[20]
    )
    result = calculate_group_ofi_factors(missing_timestamp_frames)["159001"]
    assert result.index.equals(index)
    assert result.loc[index[20], "sector_ofi_dispersion_l1_60s"] > 0


def test_merge_preserves_existing_columns_and_obeys_overwrite() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=3, freq="min")
    existing = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "reference_index_code": "000300.SH",
            "legacy_factor": [10.0, 11.0, 12.0],
            "market_ofi_beta_l1_60s": [99.0, np.nan, np.nan],
        },
        index=index,
    )
    factor_frame = pd.DataFrame(1.5, index=index[1:], columns=FACTOR_COLUMNS)

    appended = merge_factor_columns(existing, factor_frame, overwrite=False)
    assert appended["legacy_factor"].equals(existing["legacy_factor"])
    assert appended["market_ofi_beta_l1_60s"].iloc[0] == 99.0
    assert appended["market_ofi_beta_l1_60s"].iloc[1] == 1.5
    assert set(FACTOR_COLUMNS) <= set(appended.columns)

    replaced = merge_factor_columns(existing, factor_frame, overwrite=True)
    assert replaced["market_ofi_beta_l1_60s"].iloc[0] == 99.0
    assert replaced["market_ofi_beta_l1_60s"].iloc[1] == 1.5


def test_process_group_skips_missing_crossmarket_output(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="min")
    orderbook_root = tmp_path / "orderbook"
    crossmarket_root = tmp_path / "crossmarket"
    orderbook_root.mkdir()
    crossmarket_root.mkdir()
    codes = ("159001", "159002", "159003")
    for multiplier, code in enumerate(codes, start=1):
        _orderbook_frame(index, float(multiplier)).to_parquet(
            orderbook_root / f"{code}.SZ.parquet"
        )
    output_path = crossmarket_root / "159001.SZ.parquet"
    pd.DataFrame({"close": 1.0, "legacy_factor": 3.0}, index=index).to_parquet(
        output_path
    )

    results = process_reference_group(
        "000300.SH",
        codes,
        ("159001", "159003"),
        {
            code: orderbook_root / f"{code}.SZ.parquet"
            for code in codes
        },
        {"159001": output_path},
        None,
        None,
        False,
    )

    assert {result["status"] for result in results} == {
        "written",
        "missing_crossmarket_output",
    }
    output = pd.read_parquet(output_path)
    assert output["legacy_factor"].eq(3.0).all()
    assert output.loc[index[31]:, "market_ofi_beta_l1_60s"].notna().all()
