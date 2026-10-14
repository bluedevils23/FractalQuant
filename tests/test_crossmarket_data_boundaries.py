from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import generate_etf_crossmarket_factors as price
from scripts.generate_etf_crossmarket_minute_factors import calculate_benchmark_factor_frame
from scripts.generate_etf_crossmarket_ofi_factors import _calculate_single_ofi_panel
from scripts.generate_etf_daily_states import generate_daily_states


def bars(index, scale=1.0):
    steps = np.arange(len(index))
    close = scale * np.exp(steps * .001 + .003 * np.sin(steps / 4))
    return pd.DataFrame(dict(open=close, high=close, low=close, close=close,
                             vol=100., amount=1000.), index=index)


def save_bars(frame, path):
    frame.rename_axis('trade_time').reset_index().to_parquet(path, index=False)


def test_gap_does_not_become_one_minute_return():
    ix = pd.date_range('2026-01-05 09:31', periods=90, freq='min')
    etf, ref = bars(ix), bars(ix, 1000)
    ref = ref.drop(ix[40])
    result = price.calculate_crossmarket_factor_frame(etf, ref)
    assert pd.isna(result.loc[ix[41], 'etf_index_basis_1m'])
    assert pd.isna(result.loc[ix[41], 'cross_market_correlation'])
    sparse = calculate_benchmark_factor_frame(etf, ref)
    dense = calculate_benchmark_factor_frame(etf, ref.reindex(ix))
    pd.testing.assert_frame_equal(sparse, dense.loc[sparse.index])
    assert pd.isna(sparse.loc[ix[41], 'beta_residual_momentum_20m'])
    peers = price.calculate_related_etf_factor_frames({'a': etf.drop(ix[40]), 'b': bars(ix)})
    assert pd.isna(peers['a'].loc[ix[41], 'same_index_relative_return_1m'])


def test_late_open_cannot_change_earlier_rows():
    ix = pd.date_range('2026-01-05 09:31', periods=70, freq='min')
    etf, ref = bars(ix), bars(ix, 1000)
    etf.loc[ix[:3], 'open'] = np.nan
    before = price.calculate_crossmarket_factor_frame(etf, ref)
    etf.loc[ix[3], 'open'] *= 2
    after = price.calculate_crossmarket_factor_frame(etf, ref)
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
    assert after['etf_index_intraday_price_gap'].iloc[:3].isna().all()
    assert price._day_cumulative_return(etf).iloc[:3].isna().all()


def test_daily_states_index_codes_missing_day_and_missing_current_day(tmp_path):
    minute_root = tmp_path / 'minutes'
    minute_root.mkdir()
    calendar = pd.to_datetime(['2026-01-02 09:31', '2026-01-05 09:31',
                               '2026-01-06 09:31', '2026-01-07 09:31'])
    minute_path = minute_root / '159040.SZ.parquet'
    save_bars(bars(calendar), minute_path)
    daily = pd.DataFrame({'ts_code': ['159040.SZ'] * 2,
                          'trade_date': pd.to_datetime(['2026-01-02', '2026-01-06']),
                          'nav': [1., 2.], 'total_size': [100., 200.],
                          'total_share': [100., 100.]})
    daily_path, state_path = tmp_path / 'daily.parquet', tmp_path / 'states.parquet'
    daily.set_index(['trade_date', 'ts_code']).to_parquet(daily_path)
    states = generate_daily_states(daily_path, state_path, etf_root=minute_root)
    loaded = price.load_etf_daily_states(state_path, {'159040': minute_path})['159040']
    assert loaded.loc['2026-01-05', 'prev_nav'] == 1.
    assert loaded.loc['2026-01-06'].isna().all()
    assert loaded.loc['2026-01-07', 'prev_nav'] == 2.
    # Also accept full exchange codes, and reject a stale source date.
    states['ts_code'] = '159040.SZ'
    states.loc[states.trade_date.eq('2026-01-06'), ['prev_nav', 'source_trade_date']] = [1., pd.Timestamp('2026-01-02')]
    states.to_parquet(state_path, index=False)
    loaded = price.load_etf_daily_states(state_path, {'159040': minute_path})['159040']
    assert loaded.loc['2026-01-05', 'prev_nav'] == 1.
    assert loaded.loc['2026-01-06'].isna().all()
    states.drop(columns='source_trade_date').to_parquet(state_path, index=False)
    with pytest.raises(ValueError, match='rebuild'):
        price.load_etf_daily_states(state_path, {'159040': minute_path})


def test_lunch_break_is_not_counted_as_missing_trading_minutes():
    ix = pd.to_datetime(['2026-01-05 11:29', '2026-01-05 11:30',
                         '2026-01-05 13:01', '2026-01-05 13:03'])
    expected = pd.to_datetime(['2026-01-05 11:29', '2026-01-05 11:30',
                               '2026-01-05 13:01', '2026-01-05 13:02',
                               '2026-01-05 13:03'])
    pd.testing.assert_index_equal(price.trading_minute_grid(ix), expected)


def test_ofi_own_zero_amount_does_not_invalidate_peers():
    ix = pd.date_range('2026-01-05 09:31', periods=40, freq='min')
    ofi = pd.DataFrame({'a': np.arange(40.), 'b': np.arange(40.) * 2}, index=ix)
    amount = pd.DataFrame({'a': 0., 'b': 100.}, index=ix)
    result = _calculate_single_ofi_panel(ofi, amount, 'l1_60s')['a']
    assert result['lead_market_ofi_l1_60s'].iloc[2] == 2.
    assert np.isclose(result['market_ofi_beta_l1_60s'].iloc[-1], .5)


def test_partial_rebuild_preserves_history_and_supplements(tmp_path):
    ix = pd.date_range('2026-01-05 09:31', periods=70, freq='min').append(
        pd.date_range('2026-01-06 09:31', periods=70, freq='min'))
    etf_path, ref_path = tmp_path / '159040.SZ.parquet', tmp_path / 'reference.parquet'
    save_bars(bars(ix), etf_path)
    ref = bars(ix, 1000).assign(trade_date=ix.normalize(), trade_time=ix)
    ref = ref.set_index(['trade_date', 'trade_time'])
    ref.to_parquet(ref_path)
    out = tmp_path / 'output'
    price.process_mapping_record(etf_path, ref_path, '931637.CSI', out, None, None, False)
    original = pd.read_parquet(out / etf_path.name)
    original['rolling_market_beta_60m'] = 17.
    original['market_ofi_beta_l1_60s'] = 18.
    original.to_parquet(out / etf_path.name)
    price.process_mapping_record(etf_path, ref_path, '931637.CSI', out, '20260106', '20260106', True)
    updated = pd.read_parquet(out / etf_path.name)
    pd.testing.assert_frame_equal(original.loc['2026-01-05'], updated.loc['2026-01-05'])
    assert updated['rolling_market_beta_60m'].eq(17.).all()
    assert updated['market_ofi_beta_l1_60s'].eq(18.).all()
    # Rebuilding a day without an index must not delete other days.
    ref.loc[ref.index.get_level_values('trade_date') == pd.Timestamp('2026-01-05')].to_parquet(ref_path)
    price.process_mapping_record(etf_path, ref_path, '931637.CSI', out, '20260106', '20260106', True)
    remaining = pd.read_parquet(out / etf_path.name)
    pd.testing.assert_frame_equal(remaining, original.loc['2026-01-05'])
