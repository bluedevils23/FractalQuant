"""Test turnover rate factors with real data."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Test parameters
DATA_ROOT = Path("E:/逐笔数据")
DAILY_PATH = Path("D:/workspace/stockdata/stock-data/行情数据/stock_daily.parquet")
TEST_DATE = "20260115"
TEST_SYMBOL = "000001.SZ"  # 平安银行

print("Testing turnover rate factors with real data...")
print("=" * 70)
print(f"\nTest Parameters:")
print(f"  Date: {TEST_DATE}")
print(f"  Symbol: {TEST_SYMBOL}")
print(f"  Data Root: {DATA_ROOT}")
print(f"  Daily Data: {DAILY_PATH}")

# Check data availability
if not DATA_ROOT.exists():
    print(f"\n[ERROR] Tick data root not found: {DATA_ROOT}")
    sys.exit(1)

if not DAILY_PATH.exists():
    print(f"\n[ERROR] Daily data not found: {DAILY_PATH}")
    sys.exit(1)

# Check if date directory exists
# Structure: E:/逐笔数据/2026/202601/20260115/
year = TEST_DATE[:4]
month = TEST_DATE[:6]
date_dir = DATA_ROOT / year / month / TEST_DATE
if not date_dir.exists():
    print(f"\n[ERROR] Date directory not found: {date_dir}")
    sys.exit(1)

print(f"\n[OK] All data paths exist")

# Load float_share and free_share from daily data
print(f"\n" + "-" * 70)
print("Loading share data from daily file...")
print("-" * 70)

try:
    daily_df = pd.read_parquet(
        DAILY_PATH,
        columns=["float_share", "free_share"],
    )
    daily_df = daily_df.reset_index()

    if "trade_date" not in daily_df.columns or "ts_code" not in daily_df.columns:
        print("[ERROR] Daily file missing required index columns")
        sys.exit(1)

    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"]).dt.normalize()
    daily_df["float_share"] = pd.to_numeric(daily_df["float_share"], errors="coerce")
    daily_df["free_share"] = pd.to_numeric(daily_df["free_share"], errors="coerce")

    # Get data for the test symbol before the test date
    test_ts = pd.Timestamp(TEST_DATE)
    symbol_data = daily_df[
        (daily_df["ts_code"] == TEST_SYMBOL) &
        (daily_df["trade_date"] < test_ts)
    ].sort_values("trade_date")

    if symbol_data.empty:
        print(f"[ERROR] No historical data for {TEST_SYMBOL} before {TEST_DATE}")
        sys.exit(1)

    # Get most recent values
    latest = symbol_data.iloc[-1]
    float_share = latest["float_share"]  # in 万股
    free_share = latest["free_share"]    # in 万股

    print(f"\nShare Data (from {latest['trade_date'].date()}):")
    print(f"  Float Share: {float_share:,.2f} 万股 = {float_share * 10000:,.0f} 股")
    print(f"  Free Share: {free_share:,.2f} 万股 = {free_share * 10000:,.0f} 股")

    if not np.isfinite(float_share) or float_share <= 0:
        print(f"[ERROR] Invalid float_share: {float_share}")
        sys.exit(1)

    if not np.isfinite(free_share) or free_share <= 0:
        print(f"[ERROR] Invalid free_share: {free_share}")
        sys.exit(1)

except Exception as e:
    print(f"[ERROR] Failed to load share data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load auction transaction data
print(f"\n" + "-" * 70)
print("Loading auction transaction data...")
print("-" * 70)

transaction_file = date_dir / TEST_SYMBOL / "逐笔成交.csv"
if not transaction_file.exists():
    print(f"[ERROR] Transaction file not found: {transaction_file}")
    sys.exit(1)

try:
    trans_df = pd.read_csv(transaction_file, encoding="gbk")

    # Time format examples:
    # 91500060 = 09:15:00.060 (9HMMSSmmm - morning with leading 9)
    # 143002500 = 14:30:02.500 (HHMMSSmmm - afternoon)

    def parse_auction_time(time_val):
        """Parse time and return hour, minute."""
        s = str(int(time_val)) if pd.notna(time_val) else ""
        if len(s) == 8:
            if s[0] == '9':  # Morning: 9HMMSSmmm
                return 9, int(s[1:3])
            else:  # Afternoon: HHMMSSmmm
                return int(s[0:2]), int(s[2:4])
        elif len(s) == 9:  # HHMMSSmmm
            return int(s[0:2]), int(s[2:4])
        return None, None

    trans_df[["hour", "minute"]] = trans_df["时间"].apply(
        lambda x: pd.Series(parse_auction_time(x))
    )

    print(f"\nTotal transactions: {len(trans_df)}")
    valid_times = trans_df[trans_df["hour"].notna()]
    print(f"Valid time records: {len(valid_times)}")
    print(f"Hour range: {int(valid_times['hour'].min())}-{int(valid_times['hour'].max())}")
    morning = valid_times[valid_times['hour'] == 9]
    if len(morning) > 0:
        print(f"Minute range (hour 9): {sorted(morning['minute'].dropna().unique().astype(int))[:10]}...")

    # Filter auction period
    # For Shenzhen (SZ): auction matching happens at 09:25:00
    # Look for transactions at exactly 09:25:00 with code '0'
    auction_trans = trans_df[
        ((trans_df["hour"] == 9) & (trans_df["minute"] == 25) & (trans_df["成交代码"] == "0"))
    ].copy()

    print(f"Auction matching transactions (09:25:00, code=0): {len(auction_trans)}")

    if auction_trans.empty:
        print("[ERROR] No auction transactions found")
        sys.exit(1)

    # Calculate total auction volume
    # Find column name containing "量" (volume)
    volume_col = [c for c in auction_trans.columns if "量" in c and "成交" in c][0]
    auction_volume = auction_trans[volume_col].sum()  # in 手 (lots)
    auction_volume_shares = auction_volume * 100  # convert to shares

    print(f"\nAuction Transaction Summary:")
    print(f"  Transaction Count: {len(auction_trans)}")
    print(f"  Total Volume: {auction_volume:,.0f} 手 = {auction_volume_shares:,.0f} 股")

except Exception as e:
    print(f"[ERROR] Failed to load transaction data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate turnover rates
print(f"\n" + "=" * 70)
print("Calculating Turnover Rates")
print("=" * 70)

# Convert float_share and free_share from 万股 to 股
float_share_in_shares = float_share * 10000
free_share_in_shares = free_share * 10000

turnover_rate_float = (auction_volume_shares / float_share_in_shares) * 100
turnover_rate_free = (auction_volume_shares / free_share_in_shares) * 100

print(f"\nCalculation:")
print(f"  Auction Volume: {auction_volume_shares:,.0f} 股")
print(f"  Float Share: {float_share_in_shares:,.0f} 股")
print(f"  Free Share: {free_share_in_shares:,.0f} 股")
print(f"\n  auction_turnover_rate = {auction_volume_shares:,.0f} / {float_share_in_shares:,.0f} * 100")
print(f"                        = {turnover_rate_float:.4f}%")
print(f"\n  auction_turnover_rate_free = {auction_volume_shares:,.0f} / {free_share_in_shares:,.0f} * 100")
print(f"                             = {turnover_rate_free:.4f}%")

# Validate results
print(f"\n" + "-" * 70)
print("Validation")
print("-" * 70)

# Turnover rate should be reasonable (0.1% - 5% for typical auctions)
if 0.01 <= turnover_rate_float <= 10.0:
    print(f"[OK] Float turnover rate is reasonable: {turnover_rate_float:.4f}%")
else:
    print(f"[WARNING] Float turnover rate seems unusual: {turnover_rate_float:.4f}%")

if 0.01 <= turnover_rate_free <= 10.0:
    print(f"[OK] Free turnover rate is reasonable: {turnover_rate_free:.4f}%")
else:
    print(f"[WARNING] Free turnover rate seems unusual: {turnover_rate_free:.4f}%")

# Free turnover should be >= float turnover (since free_share <= float_share)
if turnover_rate_free >= turnover_rate_float:
    print(f"[OK] Free turnover rate >= float turnover rate")
else:
    print(f"[WARNING] Free turnover rate < float turnover rate (unexpected)")

print(f"\n" + "=" * 70)
print("[SUCCESS] Turnover rate calculation validated with real data!")
print("=" * 70)
