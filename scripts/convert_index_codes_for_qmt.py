"""Map ETF reference indexes to symbols exposed by MiniQMT."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING_PATH = Path(
    r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping_qmt.csv"
)
DEFAULT_UNIVERSE_PATH = Path(
    r"D:\workspace\stockdata\etf-factors\etf_universe\qmt_reference_index_universe.txt"
)
DEFAULT_SECTOR = "沪深指数"


def _clean_code(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _code_base(value: str) -> str:
    return value.rsplit(".", 1)[0] if "." in value else value


def map_code_to_qmt(value: object, qmt_symbols: list[str]) -> tuple[str, str]:
    """Map one provider code using only symbols returned by QMT.

    Matching is deliberately conservative: an ambiguous base-code match is
    not guessed and is returned as ``ambiguous``.
    """
    source = _clean_code(value)
    if not source:
        return "", "missing_source"
    normalized = {_clean_code(symbol) for symbol in qmt_symbols if _clean_code(symbol)}
    if source in normalized:
        return source, "exact"

    base = _code_base(source)
    matches = sorted(symbol for symbol in normalized if _code_base(symbol) == base)
    if len(matches) == 1:
        return matches[0], "base_unique"
    if len(matches) > 1:
        return "", "ambiguous_base"
    return "", "not_in_qmt_sector"


def convert_mapping(
    mapping: pd.DataFrame,
    qmt_symbols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add QMT reference/tracking columns and return a unique QMT universe."""
    result = mapping.copy()
    for source_column, prefix in (
        ("reference_index_code", "reference"),
        ("tracking_index_code", "tracking"),
    ):
        if source_column not in result.columns:
            continue
        mapped = result[source_column].map(
            lambda value: map_code_to_qmt(value, qmt_symbols)
        )
        result[f"qmt_{prefix}_index_code"] = mapped.map(lambda item: item[0])
        result[f"qmt_{prefix}_index_match_status"] = mapped.map(lambda item: item[1])

    if "qmt_reference_index_code" in result:
        universe = sorted(
            code
            for code in result["qmt_reference_index_code"].dropna().astype(str)
            if code
        )
    else:
        universe = []
    return result, sorted(set(universe))


def fetch_qmt_index_symbols(sector_name: str = DEFAULT_SECTOR) -> list[str]:
    """Fetch symbols from a running MiniQMT client."""
    try:
        from xtquant import xtdata
    except ImportError as exc:  # pragma: no cover - depends on QMT runtime
        raise RuntimeError(
            "xtquant is unavailable; run this script in the MiniQMT Python environment"
        ) from exc

    sectors = xtdata.get_sector_list()
    if sector_name not in sectors:
        raise RuntimeError(
            f"QMT sector {sector_name!r} was not found; available sectors include "
            f"{list(sectors)[:10]!r}"
        )
    symbols = xtdata.get_stock_list_in_sector(sector_name)
    if not symbols:
        raise RuntimeError(f"QMT sector {sector_name!r} returned no symbols")
    return sorted({_clean_code(symbol) for symbol in symbols if _clean_code(symbol)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map ETF reference indexes to symbols listed by MiniQMT."
    )
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--sector-name", default=DEFAULT_SECTOR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = pd.read_csv(args.mapping_path)
    qmt_symbols = fetch_qmt_index_symbols(args.sector_name)
    converted, universe = convert_mapping(mapping, qmt_symbols)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    args.universe_path.parent.mkdir(parents=True, exist_ok=True)
    args.universe_path.write_text("\n".join(universe) + "\n", encoding="utf-8")

    for column in converted.columns:
        if column.endswith("_match_status"):
            print(column, converted[column].value_counts(dropna=False).to_dict())
    print(f"qmt_sector_symbols={len(qmt_symbols)}")
    print(f"mapped_reference_indexes={len(universe)}")
    print(f"mapping_output={args.output_path}")
    print(f"universe_output={args.universe_path}")


if __name__ == "__main__":
    main()
