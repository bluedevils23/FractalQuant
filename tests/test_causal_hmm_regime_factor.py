from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from factor.regime import (
    HMM_FEATURE_COLUMNS,
    HMM_REGIME_STATES,
    CausalHMMRegimeConfidenceFactor,
    CausalHMMRegimeEntropyFactor,
    CausalHMMRegimeProbabilityFactor,
    CausalHMMRegimeTransitionFactor,
    _state_order_by_variance,
    calculate_causal_hmm_regime_features,
    calculate_causal_hmm_regime_probabilities,
)

from scripts.generate_etf_minute_factors import (
    build_factor_specs,
    calculate_factor_frame,
)

HMM_COLUMNS = [f"hmm_regime_prob_{state}_vol" for state in HMM_REGIME_STATES]


def _minute_frame(days: int = 3, bars_per_day: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for day_offset in range(days):
        index = pd.date_range(
            pd.Timestamp("2026-01-05") + pd.Timedelta(days=day_offset, hours=9, minutes=30),
            periods=bars_per_day,
            freq="min",
        )
        scales = np.resize(np.array([0.5, 2.0, 6.0]), bars_per_day)
        returns_bps = rng.normal(0.0, scales)
        close = 100.0 * np.exp(np.cumsum(returns_bps / 10_000.0))
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.0002,
                    "low": close * 0.9998,
                    "close": close,
                    "volume": np.full(bars_per_day, 1_000.0),
                },
                index=index,
            )
        )
    return pd.concat(frames)


def test_hmm_state_order_is_defined_by_conditional_variance() -> None:
    model = SimpleNamespace(
        n_components=3,
        covars_=np.array([[[4.0]], [[1.0]], [[9.0]]]),
    )

    assert _state_order_by_variance(model).tolist() == [1, 0, 2]


def test_hmm_probabilities_use_only_preceding_days_and_sum_to_one() -> None:
    frame = _minute_frame()
    probabilities = calculate_causal_hmm_regime_probabilities(frame)
    days = frame.index.normalize().unique()

    assert probabilities.loc[frame.index.normalize() == days[0]].isna().all().all()
    for day in days[1:]:
        day_probabilities = probabilities.loc[frame.index.normalize() == day]
        assert day_probabilities.iloc[0].isna().all()
        finite = day_probabilities.dropna()
        assert not finite.empty
        assert np.allclose(finite.sum(axis=1), 1.0)
        assert ((finite >= 0.0) & (finite <= 1.0)).all().all()


def test_hmm_summary_features_are_bounded_and_strategy_ready() -> None:
    frame = _minute_frame()
    features = calculate_causal_hmm_regime_features(frame)

    assert features.columns.tolist() == list(HMM_FEATURE_COLUMNS)
    finite = features.dropna()
    assert not finite.empty
    assert np.allclose(
        finite[[f"hmm_regime_prob_{state}_vol" for state in HMM_REGIME_STATES]]
        .sum(axis=1),
        1.0,
    )
    assert ((finite["hmm_regime_confidence"] >= 1 / 3) &
            (finite["hmm_regime_confidence"] <= 1.0)).all()
    assert ((finite["hmm_regime_entropy"] >= 0.0) &
            (finite["hmm_regime_entropy"] <= 1.0)).all()
    assert ((finite["hmm_regime_transition_score"] >= 0.0) &
            (finite["hmm_regime_transition_score"] <= 1.0)).all()


def test_hmm_filter_is_causal_within_current_day() -> None:
    frame = _minute_frame()
    changed = frame.copy()
    last_day = frame.index.normalize().unique()[-1]
    last_day_positions = np.flatnonzero(frame.index.normalize() == last_day)
    change_start = last_day_positions[-20]
    changed.iloc[change_start:, changed.columns.get_loc("close")] *= 1.05

    baseline = calculate_causal_hmm_regime_probabilities(frame)
    modified = calculate_causal_hmm_regime_probabilities(changed)

    pd.testing.assert_frame_equal(
        baseline.iloc[:change_start], modified.iloc[:change_start]
    )


def test_hmm_returns_missing_for_insufficient_or_constant_history() -> None:
    short = _minute_frame(days=2, bars_per_day=60)
    constant = _minute_frame(days=2)
    constant.loc[:, "close"] = 100.0

    assert calculate_causal_hmm_regime_probabilities(short).isna().all().all()
    assert calculate_causal_hmm_regime_probabilities(constant).isna().all().all()


def test_hmm_factor_cache_and_generator_registration() -> None:
    frame = _minute_frame(days=2)
    factors = [
        CausalHMMRegimeProbabilityFactor(state) for state in HMM_REGIME_STATES
    ] + [
        CausalHMMRegimeConfidenceFactor(),
        CausalHMMRegimeEntropyFactor(),
        CausalHMMRegimeTransitionFactor(),
    ]
    result = pd.concat([factor.calculate(frame) for factor in factors], axis=1)
    specs = build_factor_specs("base")
    hmm_specs = [spec for spec in specs if spec.output_name in HMM_FEATURE_COLUMNS]

    assert result.columns.tolist() == list(HMM_FEATURE_COLUMNS)
    assert [spec.output_name for spec in hmm_specs] == list(HMM_FEATURE_COLUMNS)
    assert all(not spec.reset_daily for spec in hmm_specs)


def test_etf_minute_generator_emits_cross_day_hmm_probabilities() -> None:
    frame = _minute_frame(days=2)
    result = calculate_factor_frame(frame, "base")
    second_day = frame.index.normalize().unique()[1]
    probabilities = result.loc[
        frame.index.normalize() == second_day, HMM_FEATURE_COLUMNS
    ]

    assert probabilities.iloc[0].isna().all()
    finite = probabilities.dropna()
    assert not finite.empty
    assert np.allclose(
        finite[[f"hmm_regime_prob_{state}_vol" for state in HMM_REGIME_STATES]]
        .sum(axis=1),
        1.0,
    )
