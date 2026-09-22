from __future__ import annotations

import pandas as pd

from scripts.convert_index_codes_for_qmt import convert_mapping, map_code_to_qmt


def test_map_code_prefers_exact_then_unique_base_match() -> None:
    qmt_symbols = ["000300.SH", "931946.SH", "HSCAUPG.HK"]

    assert map_code_to_qmt("000300.SH", qmt_symbols) == ("000300.SH", "exact")
    assert map_code_to_qmt("931946.CSI", qmt_symbols) == (
        "931946.SH",
        "base_unique",
    )
    assert map_code_to_qmt("HSCAUPG.HI", qmt_symbols) == (
        "HSCAUPG.HK",
        "base_unique",
    )


def test_map_code_does_not_guess_missing_or_ambiguous_symbols() -> None:
    assert map_code_to_qmt("XIN9.OTH", ["XIN9.SH", "XIN9.SZ"]) == (
        "",
        "ambiguous_base",
    )
    assert map_code_to_qmt("399001.SZ", ["000300.SH"]) == (
        "",
        "not_in_qmt_sector",
    )
    assert map_code_to_qmt(None, ["000300.SH"]) == ("", "missing_source")


def test_convert_mapping_adds_qmt_columns_and_unique_universe() -> None:
    mapping = pd.DataFrame(
        {
            "fund_code": ["159001", "159002", "159003"],
            "reference_index_code": ["931946.CSI", "931946.CSI", "XIN9.OTH"],
            "tracking_index_code": ["931946", "931946", None],
        }
    )

    converted, universe = convert_mapping(mapping, ["931946.SH"])

    assert converted.loc[0, "qmt_reference_index_code"] == "931946.SH"
    assert converted.loc[2, "qmt_reference_index_match_status"] == "not_in_qmt_sector"
    assert converted.loc[2, "qmt_tracking_index_match_status"] == "missing_source"
    assert universe == ["931946.SH"]
