"""ETF-configured entrypoint for the generic daily FangZheng generator."""

from __future__ import annotations

try:
    from scripts.generate_fz_daily_factors import *  # noqa: F403
    from scripts.generate_fz_daily_factors import main as _daily_main
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from generate_fz_daily_factors import *  # noqa: F403
    from generate_fz_daily_factors import main as _daily_main


def main(argv: list[str] | None = None) -> int:
    return _daily_main(argv, default_asset_type="etf")


if __name__ == "__main__":
    raise SystemExit(main())
