"""Replay the expanded raw sample audit without modifying production factors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.auction_tick_cache import AuctionTickCache
from scripts.generate_close_auction_factors import calculate_daily_close_auction_factors


def main():
    output = Path("D:/workspace/stock-backtest/analysis_outputs/close_auction_raw_audit_20260927")
    samples = pd.read_csv(output / "samples.csv", dtype={"day": str})
    cache = AuctionTickCache(None)
    results = []
    for sample in samples.loc[samples.status.eq("ok")].itertuples():
        day, code = sample.day, sample.asset
        path = Path("E:/逐笔数据") / day[:4] / day[:6] / day / code
        orders = cache.load_close_order_history(path)
        trades = cache.load_close_transactions(path)
        row = calculate_daily_close_auction_factors(orders, trades, code, None, asset_type="etf")
        raw = pd.read_csv(path / "逐笔成交.csv", encoding="gbk", dtype=str)
        number = lambda col: pd.to_numeric(raw[col], errors="coerce")
        mask = number("时间").ge(150000000) & number("时间").lt(150500000)
        mask &= ~raw["成交代码"].eq("C") & number("成交价格").gt(0) & number("成交数量").gt(0)
        expected = float((number("成交价格")[mask] / 10000 * number("成交数量")[mask]).sum())
        applicable = not (code.endswith("SH") and day < "20260706")
        passed = row["auction_applicable"] == applicable
        if applicable:
            passed &= bool(row["close_auction_event_reconstruction_ok"])
            passed &= row["close_auction_match_count"] == int(mask.sum())
            passed &= bool(np.isclose(row["close_auction_matched_notional"], expected, rtol=1e-12, atol=1e-6))
        else:
            passed &= row["auction_status"] == "not_applicable" and pd.isna(row["close_auction_matched_notional"])
        results.append(dict(day=day, asset=code, status=row["auction_status"], passed=bool(passed),
                            raw_matches=int(mask.sum()), raw_notional=expected,
                            generated_notional=row["close_auction_matched_notional"]))
    frame = pd.DataFrame(results)
    frame.to_csv(output / "fixed_generator_validation.csv", index=False)
    saved = pd.read_csv(output / "fixed_generator_validation.csv")
    summary = dict(samples=len(saved), passed=int(saved.passed.sum()),
                   status_counts=saved.status.value_counts().to_dict(), missing_source_samples=int(samples.status.ne("ok").sum()))
    (output / "fixed_generator_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    assert saved.passed.all(), saved.loc[~saved.passed].to_string(index=False)


if __name__ == "__main__":
    main()
