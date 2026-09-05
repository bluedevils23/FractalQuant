"""Test script for auction turnover rate factors."""
import pandas as pd
import numpy as np


def _apply_turnover_rate_factors(result: pd.DataFrame, index: int) -> None:
    """Calculate auction turnover rate factors.

    auction_turnover_rate = auction_matched_volume / float_share * 100
    auction_turnover_rate_free = auction_matched_volume / free_share * 100
    """
    auction_volume = result.at[index, "auction_matched_volume"]
    float_share = result.at[index, "previous_day_float_share"]
    free_share = result.at[index, "previous_day_free_share"]

    # Initialize columns
    result.at[index, "auction_turnover_rate"] = np.nan
    result.at[index, "auction_turnover_rate_free"] = np.nan

    if not np.isfinite(auction_volume) or auction_volume <= 0:
        return

    # Calculate turnover rate using float_share
    if np.isfinite(float_share) and float_share > 0:
        result.at[index, "auction_turnover_rate"] = float(
            auction_volume / float_share * 100.0
        )

    # Calculate turnover rate using free_share
    if np.isfinite(free_share) and free_share > 0:
        result.at[index, "auction_turnover_rate_free"] = float(
            auction_volume / free_share * 100.0
        )


def test_turnover_rate_calculation():
    """Test basic turnover rate calculation logic."""
    print("Testing turnover rate calculation...")
    print("=" * 60)

    # Create test data
    test_data = pd.DataFrame({
        "trade_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
        "ts_code": ["000001.SZ"] * 3,
        "auction_matched_volume": [1000000, 2000000, 0],  # shares
        "previous_day_float_share": [100000000, 100000000, 100000000],  # shares
        "previous_day_free_share": [80000000, 80000000, 80000000],  # shares
    })

    # Apply turnover rate factors
    for index, row in test_data.iterrows():
        _apply_turnover_rate_factors(test_data, index)

    # Expected results
    # Day 1: 1000000 / 100000000 * 100 = 1.0%
    # Day 1 (free): 1000000 / 80000000 * 100 = 1.25%
    # Day 2: 2000000 / 100000000 * 100 = 2.0%
    # Day 2 (free): 2000000 / 80000000 * 100 = 2.5%
    # Day 3: volume=0, should be NaN

    print("\nTest Results:")
    print("-" * 60)
    for index, row in test_data.iterrows():
        print(f"\nDate: {row['trade_date']}")
        print(f"  Auction Volume: {row['auction_matched_volume']:,.0f}")
        print(f"  Float Share: {row['previous_day_float_share']:,.0f}")
        print(f"  Free Share: {row['previous_day_free_share']:,.0f}")
        print(f"  Turnover Rate (float): {row.get('auction_turnover_rate', np.nan):.4f}%")
        print(f"  Turnover Rate (free): {row.get('auction_turnover_rate_free', np.nan):.4f}%")

    # Validate day 1
    assert abs(test_data.at[0, "auction_turnover_rate"] - 1.0) < 0.0001, \
        f"Day 1 float turnover rate should be 1.0%, got {test_data.at[0, 'auction_turnover_rate']}"
    assert abs(test_data.at[0, "auction_turnover_rate_free"] - 1.25) < 0.0001, \
        f"Day 1 free turnover rate should be 1.25%, got {test_data.at[0, 'auction_turnover_rate_free']}"

    # Validate day 2
    assert abs(test_data.at[1, "auction_turnover_rate"] - 2.0) < 0.0001, \
        f"Day 2 float turnover rate should be 2.0%, got {test_data.at[1, 'auction_turnover_rate']}"
    assert abs(test_data.at[1, "auction_turnover_rate_free"] - 2.5) < 0.0001, \
        f"Day 2 free turnover rate should be 2.5%, got {test_data.at[1, 'auction_turnover_rate_free']}"

    # Validate day 3 (zero volume)
    assert np.isnan(test_data.at[2, "auction_turnover_rate"]), \
        "Day 3 should have NaN turnover rate (zero volume)"
    assert np.isnan(test_data.at[2, "auction_turnover_rate_free"]), \
        "Day 3 should have NaN free turnover rate (zero volume)"

    print("\n" + "=" * 60)
    print("[OK] All turnover rate calculation tests passed!")
    print("=" * 60)


def test_edge_cases():
    """Test edge cases: missing data, zero shares."""
    print("\nTesting edge cases...")
    print("=" * 60)

    test_data = pd.DataFrame({
        "trade_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
        "ts_code": ["000001.SZ"] * 3,
        "auction_matched_volume": [1000000, 1000000, 1000000],
        "previous_day_float_share": [np.nan, 0, 100000000],  # missing, zero, valid
        "previous_day_free_share": [80000000, 80000000, np.nan],  # valid, valid, missing
    })

    for index, row in test_data.iterrows():
        _apply_turnover_rate_factors(test_data, index)

    print("\nEdge Case Results:")
    print("-" * 60)

    # Case 1: missing float_share
    print("\nCase 1: Missing float_share")
    print(f"  Float turnover rate: {test_data.at[0, 'auction_turnover_rate']}")
    print(f"  Free turnover rate: {test_data.at[0, 'auction_turnover_rate_free']:.4f}%")
    assert np.isnan(test_data.at[0, "auction_turnover_rate"]), \
        "Should be NaN when float_share is missing"
    assert np.isfinite(test_data.at[0, "auction_turnover_rate_free"]), \
        "Should have valid free turnover rate"

    # Case 2: zero float_share
    print("\nCase 2: Zero float_share")
    print(f"  Float turnover rate: {test_data.at[1, 'auction_turnover_rate']}")
    assert np.isnan(test_data.at[1, "auction_turnover_rate"]), \
        "Should be NaN when float_share is zero"

    # Case 3: missing free_share
    print("\nCase 3: Missing free_share")
    print(f"  Float turnover rate: {test_data.at[2, 'auction_turnover_rate']:.4f}%")
    print(f"  Free turnover rate: {test_data.at[2, 'auction_turnover_rate_free']}")
    assert np.isfinite(test_data.at[2, "auction_turnover_rate"]), \
        "Should have valid float turnover rate"
    assert np.isnan(test_data.at[2, "auction_turnover_rate_free"]), \
        "Should be NaN when free_share is missing"

    print("\n" + "=" * 60)
    print("[OK] All edge case tests passed!")
    print("=" * 60)


def test_reasonable_ranges():
    """Test that turnover rates are within reasonable ranges."""
    print("\nTesting reasonable value ranges...")
    print("=" * 60)

    # Typical auction: 0.5% - 3% turnover rate
    test_data = pd.DataFrame({
        "trade_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
        "ts_code": ["000001.SZ"] * 3,
        "auction_matched_volume": [500000, 2000000, 5000000],
        "previous_day_float_share": [100000000] * 3,
        "previous_day_free_share": [80000000] * 3,
    })

    for index, row in test_data.iterrows():
        _apply_turnover_rate_factors(test_data, index)

    print("\nTurnover Rate Ranges:")
    print("-" * 60)
    for index, row in test_data.iterrows():
        float_rate = row["auction_turnover_rate"]
        free_rate = row["auction_turnover_rate_free"]
        print(f"\nVolume: {row['auction_matched_volume']:,}")
        print(f"  Float turnover rate: {float_rate:.4f}%")
        print(f"  Free turnover rate: {free_rate:.4f}%")

        # Verify reasonable ranges (0% - 10%)
        assert 0 <= float_rate <= 10, \
            f"Float turnover rate {float_rate}% outside expected range [0%, 10%]"
        assert 0 <= free_rate <= 10, \
            f"Free turnover rate {free_rate}% outside expected range [0%, 10%]"

        # Free share turnover should be higher than float share turnover
        # (since free_share <= float_share)
        assert free_rate >= float_rate, \
            "Free turnover rate should be >= float turnover rate"

    print("\n" + "=" * 60)
    print("[OK] All value range tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_turnover_rate_calculation()
        test_edge_cases()
        test_reasonable_ranges()

        print("\n" + "=" * 60)
        print("[SUCCESS] All turnover rate factor tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAILED] Test assertion failed: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
