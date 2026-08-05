from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.import_index_minute_excel import (
    build_output_code_lookup,
    import_source_file,
    normalize_source_frame,
)


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["931573", "港股通科技", "2026-05-04 09:31:00", 10, 11, 9, 10.5, 0, 0, 100, 1_000],
            ["931573", "港股通科技", "2026-05-04 09:32:00", 10.5, 12, 10, 11.5, 1, 1, 120, 1_400],
            ["931573", "港股通科技", "bad-time", 10, 11, 9, 10.5, 0, 0, 100, 1_000],
        ]
    )


def test_output_code_lookup_canonicalizes_cny_alias(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "fund_code,reference_index_code\n"
        "159001,931573.CSI\n"
        "159269,931573CNY00.CSI\n"
        "159002,HSBIO.HI\n",
        encoding="utf-8",
    )

    assert build_output_code_lookup(mapping_path) == {
        "931573": "931573.CSI",
        "HSBIO": "HSBIO.HI",
    }


def test_normalize_source_frame_matches_index_store_schema() -> None:
    result = normalize_source_frame(_source_frame(), "931573.CSI")

    assert result.index.names == ["trade_date", "trade_time"]
    assert result.columns.tolist() == [
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    assert len(result) == 2
    assert result["ts_code"].eq("931573.CSI").all()
    assert np.isclose(result.iloc[1]["amount"], 1_400.0)


def test_import_source_file_respects_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "K线导出_931573_1分钟线数据.xlsx"
    source_path.touch()
    output_path = tmp_path / "931573.CSI.parquet"
    monkeypatch.setattr(
        "scripts.import_index_minute_excel.pd.read_excel",
        lambda _: _source_frame(),
    )

    written = import_source_file(
        source_path, output_path, "931573.CSI", overwrite=False
    )
    skipped = import_source_file(
        source_path, output_path, "931573.CSI", overwrite=False
    )

    assert written["status"] == "written"
    assert skipped["status"] == "skipped"
    output = pd.read_parquet(output_path)
    assert output.index.names == ["trade_date", "trade_time"]
    assert output["ts_code"].eq("931573.CSI").all()
