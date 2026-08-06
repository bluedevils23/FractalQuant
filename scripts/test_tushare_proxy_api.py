"""Test daily, stock-minute, and index-minute access through a Tushare proxy.

Credentials are supplied through ``TUSHARE_PROXY_API_KEY`` or ``--api-key`` and
are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


DEFAULT_BASE_URL = "https://ai-tool.indevs.in/tushare/pro"
DEFAULT_STOCK_CODE = "000001.SZ"
DEFAULT_INDEX_CODE = "000001.SH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TUSHARE_PROXY_API_KEY"),
        help="Proxy API key; defaults to TUSHARE_PROXY_API_KEY.",
    )
    parser.add_argument("--stock-code", default=DEFAULT_STOCK_CODE)
    parser.add_argument("--index-code", default=DEFAULT_INDEX_CODE)
    parser.add_argument("--daily-start", default="20260101")
    parser.add_argument("--daily-end", default="20260110")
    parser.add_argument("--minute-start", default="2023-08-25 09:00:00")
    parser.add_argument("--minute-end", default="2023-08-25 19:00:00")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def call_api(
    session: requests.Session,
    base_url: str,
    api_name: str,
    params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = session.post(
        base_url,
        json={"api_name": api_name, "params": params},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{api_name} returned {type(payload).__name__}, not JSON object")
    return payload


def print_result(api_name: str, payload: dict[str, Any]) -> bool:
    print(f"\n=== {api_name} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload.get("code") == 0


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print(
            "Set TUSHARE_PROXY_API_KEY or pass --api-key before running.",
            file=sys.stderr,
        )
        return 2
    session = requests.Session()
    session.headers.update({"X-API-Key": args.api_key})
    session.trust_env = False
    session.proxies = {"http": "", "https": ""}

    calls = (
        (
            "daily",
            {
                "ts_code": args.stock_code,
                "start_date": args.daily_start,
                "end_date": args.daily_end,
            },
        ),
        (
            "stk_mins",
            {
                "ts_code": args.stock_code,
                "freq": "1min",
                "start_date": args.minute_start,
                "end_date": args.minute_end,
            },
        ),
        (
            "idx_mins",
            {
                "ts_code": args.index_code,
                "freq": "1min",
                "start_date": args.minute_start,
                "end_date": args.minute_end,
            },
        ),
    )

    succeeded = True
    for api_name, params in calls:
        try:
            succeeded = print_result(
                api_name,
                call_api(session, args.base_url, api_name, params, args.timeout),
            ) and succeeded
        except (requests.RequestException, ValueError) as exc:
            succeeded = False
            print(f"\n=== {api_name} ===\nERROR: {exc}", file=sys.stderr)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
