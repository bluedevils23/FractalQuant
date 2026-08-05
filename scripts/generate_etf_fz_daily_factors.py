"""Generate daily FangZheng ETF factor exposures from 1-minute market data.

The implementation lives in the legacy module so the former minute entrypoint can
remain a compatible alias while no longer emitting non-causal minute backfills.
"""

try:
    from scripts.generate_etf_fz_minute_factors import main
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from generate_etf_fz_minute_factors import main


if __name__ == "__main__":
    raise SystemExit(main())
