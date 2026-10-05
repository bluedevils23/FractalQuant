import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_etf_cicc_daily_factors import FACTOR_FUNCTION_NAMES

ROOT = Path(r'D:\workspace\stockdata\etf-data\etf_cicc_daily_factors_1267_20261001')
SOURCE = Path(r'D:\workspace\stockdata\etf-data\etf_1min')
UNIVERSE = Path(r'D:\workspace\stockdata\etf-factors\etf_universe\non_day_turnover_stock_index_universe.txt')
REPORT = Path(__file__).resolve().parent / 'cicc_regeneration_20261001'
REPORT.mkdir(exist_ok=True)
symbols = list(dict.fromkeys(line.split('#', 1)[0].strip() for line in UNIVERSE.read_text(encoding='utf-8-sig').splitlines() if line.split('#', 1)[0].strip()))
expected = ['ts_code'] + [name.removeprefix('cal_') for name in FACTOR_FUNCTION_NAMES]
actual = {p.stem for p in ROOT.glob('*.parquet')}
issues = []
records = []
null_counts = pd.Series(0, index=expected[1:], dtype='int64')
for symbol in symbols:
    path = ROOT / f'{symbol}.parquet'
    if not path.exists():
        issues.append([symbol, 'missing_file'])
        continue
    frame = pd.read_parquet(path)
    dates = pd.DatetimeIndex(frame.index)
    source_dates = pq.read_table(SOURCE / path.name, columns=['trade_date'], filters=[('trade_date', '>=', pd.Timestamp('2022-01-01')), ('trade_date', '<=', pd.Timestamp('2026-09-18'))]).column('trade_date').unique().to_pandas()
    source_dates = pd.DatetimeIndex(source_dates).sort_values()
    if list(frame.columns) != expected or frame.index.name != 'trade_date':
        issues.append([symbol, 'schema'])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        issues.append([symbol, 'duplicate_or_unsorted_dates'])
    if not dates.equals(source_dates):
        issues.append([symbol, 'source_dates_mismatch'])
    if frame.empty or not frame.ts_code.eq(symbol).all():
        issues.append([symbol, 'empty_or_wrong_symbol'])
    values = frame[expected[1:]]
    if np.isinf(values.to_numpy(dtype=float)).any():
        issues.append([symbol, 'infinity'])
    if values.isna().all(axis=1).any():
        issues.append([symbol, 'all_null_row'])
    null_counts += values.isna().sum()
    records.append({'symbol': symbol, 'rows': len(frame), 'date_min': dates.min().isoformat(), 'date_max': dates.max().isoformat(), 'source_days': len(source_dates), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
if len(symbols) != 1267 or actual != set(symbols):
    issues.append(['universe', {'missing': sorted(set(symbols)-actual), 'extra': sorted(actual-set(symbols)), 'expected_count': len(symbols)}])
if list(ROOT.glob('*.tmp')):
    issues.append(['output', 'temporary_files'])
pd.DataFrame(records).to_csv(REPORT / 'files.csv', index=False)
summary = {'status': 'validated' if not issues else 'failed', 'universe_file': str(UNIVERSE), 'source_root': str(SOURCE), 'staging_root': str(ROOT), 'expected_files': len(symbols), 'actual_files': len(actual), 'rows': sum(r['rows'] for r in records), 'factor_columns': len(expected)-1, 'date_min': min(r['date_min'] for r in records), 'date_max': max(r['date_max'] for r in records), 'issues': issues, 'null_counts': null_counts.to_dict()}
(REPORT / 'audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'null_counts'}, ensure_ascii=False))
raise SystemExit(bool(issues))
