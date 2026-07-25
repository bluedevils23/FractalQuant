# ETF minute factor quality audit

- Valid factor files audited: 1397
- Invalid/corrupt factor files: 1
- Distinct factor fields: 156
- Fields requiring review: 15
- Source-file linkage issues: 891

## Interpretation

- Missing values in the first 80 bars of each session are treated separately as expected rolling-window warm-up. `late_missing_rate` measures missing values after that opening region.
- `zero_rate_among_finite` is exact. `mean_batch_sample_mode_rate` is a weighted batch-level sampled concentration diagnostic; unlike a maximum, it is not triggered by one isolated flat batch.
- A factor's zero/constant result is not automatically a source-data fault. The detailed source report and the factor implementation must be read together before removing a field.

## Highest-priority factor fields

| Factor | Assessment | Missing | Late missing | Zero | Mean sample-mode |
|---|---:|---:|---:|---:|---:|
| volume_price_confirm_rate_w20 | review_late_missing | 85.04% | 84.90% | 2.39% | 17.52% |
| volume_price_confirm_rate_w10 | review_late_missing | 81.07% | 81.72% | 9.87% | 25.72% |
| volume_price_confirm_rate_w5 | review_late_missing | 76.20% | 77.27% | 28.80% | 37.44% |
| volatility_regime_s3_l10 | review_late_missing | 50.24% | 49.42% | 37.16% | 37.12% |
| volume_momentum | review_late_missing | 47.00% | 47.01% | 0.12% | 37.13% |
| volume_momentum_w10 | review_late_missing | 42.42% | 40.79% | 0.03% | 48.83% |
| adx_w7 | review_late_missing | 41.32% | 37.51% | 0.08% | 0.54% |
| adx_w28 | review_late_missing | 41.32% | 37.51% | 0.08% | 0.34% |
| volatility_regime | review_late_missing | 40.58% | 36.10% | 32.85% | 32.84% |
| adx | review_late_missing | 38.36% | 34.71% | 0.08% | 0.43% |
| volume_momentum_w20 | review_late_missing | 36.91% | 31.85% | 0.01% | 55.93% |
| volatility_regime_s10_l40 | review_late_missing | 40.20% | 28.81% | 28.86% | 28.87% |
| price_zscore_w10 | review_late_missing | 33.15% | 28.15% | 28.41% | 28.41% |
| price_zscore | review_late_missing | 33.03% | 26.55% | 12.47% | 12.49% |
| price_zscore_w40 | review_late_missing | 33.85% | 20.23% | 9.93% | 9.94% |

## Invalid factor files

| File | Bytes | Error |
|---|---:|---|
| 513190.SH.parquet | 6421669 | ArrowInvalid: Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file. |

Detailed machine-readable results are in the CSV files beside this report.
