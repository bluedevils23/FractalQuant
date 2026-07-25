# ETF minute factor quality audit

- Valid factor files audited: 1397
- Invalid/corrupt factor files: 1
- Distinct factor fields: 156
- Fields requiring review: 156
- Source-file linkage issues: 891

## Interpretation

- Missing values in the first 80 bars of each session are treated separately as expected rolling-window warm-up. `late_missing_rate` measures missing values after that opening region.
- `zero_rate_among_finite` is exact. `max_sample_mode_rate` is a batch-level sampled concentration diagnostic; it is intended to flag nearly constant values, not to estimate global cardinality.
- A factor's zero/constant result is not automatically a source-data fault. The detailed source report and the factor implementation must be read together before removing a field.

## Highest-priority factor fields

| Factor | Assessment | Missing | Late missing | Zero | Sample-mode |
|---|---:|---:|---:|---:|---:|
| volume_price_confirm_rate_w20 | review_late_missing | 85.04% | 84.90% | 2.39% | 100.00% |
| volume_price_confirm_rate_w10 | review_late_missing | 81.07% | 81.72% | 9.87% | 100.00% |
| volume_price_confirm_rate_w5 | review_late_missing | 76.20% | 77.27% | 28.80% | 100.00% |
| volatility_regime_s3_l10 | review_late_missing | 50.24% | 49.42% | 37.16% | 100.00% |
| volume_momentum | review_late_missing | 47.00% | 47.01% | 0.12% | 100.00% |
| volume_momentum_w10 | review_late_missing | 42.42% | 40.79% | 0.03% | 100.00% |
| adx_w7 | review_late_missing | 41.32% | 37.51% | 0.08% | 100.00% |
| adx_w28 | review_late_missing | 41.32% | 37.51% | 0.08% | 100.00% |
| volatility_regime | review_late_missing | 40.58% | 36.10% | 32.85% | 100.00% |
| adx | review_late_missing | 38.36% | 34.71% | 0.08% | 100.00% |
| volume_momentum_w20 | review_late_missing | 36.91% | 31.85% | 0.01% | 100.00% |
| volatility_regime_s10_l40 | review_late_missing | 40.20% | 28.81% | 28.86% | 100.00% |
| price_zscore_w10 | review_late_missing | 33.15% | 28.15% | 28.41% | 100.00% |
| price_zscore | review_late_missing | 33.03% | 26.55% | 12.47% | 100.00% |
| price_zscore_w40 | review_late_missing | 33.85% | 20.23% | 9.93% | 100.00% |
| liquidity_migration_w80 | review_dominant_value | 32.78% | 0.00% | 24.05% | 100.00% |
| price_volume_decoupling_w80 | review_dominant_value | 32.78% | 0.00% | 20.18% | 100.00% |
| liquidity_depth_w80 | review_dominant_value | 32.78% | 0.00% | 20.13% | 100.00% |
| orderflow_significance_w80 | review_dominant_value | 32.78% | 0.00% | 20.13% | 100.00% |
| liquidity_ratio_w80 | review_dominant_value | 32.78% | 0.00% | 20.13% | 100.00% |
| volume_clustering_w80 | review_dominant_value | 32.78% | 0.00% | 19.31% | 100.00% |
| market_efficiency_w80 | review_dominant_value | 32.78% | 0.00% | 0.00% | 100.00% |
| trade_size_distribution_w80 | review_dominant_value | 32.78% | 0.00% | 0.00% | 100.00% |
| awesome_oscillator_s10_l68 | review_dominant_value | 27.80% | 0.00% | 21.87% | 100.00% |
| volatility_skew_w60 | review_dominant_value | 24.90% | 0.00% | 23.32% | 100.00% |
| volatility_kurtosis_w60 | review_dominant_value | 24.90% | 0.00% | 0.00% | 100.00% |
| obv_delta_w50 | review_dominant_value | 20.75% | 0.00% | 25.88% | 100.00% |
| volume_price_trend_delta_w50 | review_dominant_value | 20.75% | 0.00% | 25.57% | 100.00% |
| price_momentum_w40 | review_dominant_value | 16.60% | 0.00% | 36.45% | 100.00% |
| volatility_skew_w40 | review_dominant_value | 16.60% | 0.00% | 28.41% | 100.00% |

## Invalid factor files

| File | Bytes | Error |
|---|---:|---|
| 513190.SH.parquet | 6421669 | ArrowInvalid: Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file. |

Detailed machine-readable results are in the CSV files beside this report.
