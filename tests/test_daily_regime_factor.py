from __future__ import annotations

import numpy as np
import pandas as pd

from factor.regime import (
    DAILY_REGIME_OUTPUT_COLUMNS,
    build_daily_market_feature_panel,
    calculate_causal_daily_market_regime_features,
)


def _close_panel(periods: int = 320) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            f"reference_{position}": 100.0
            * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
            for position in range(4)
        },
        index=index,
    )


def _features(periods: int = 320) -> pd.DataFrame:
    return build_daily_market_feature_panel(_close_panel(periods))


def test_daily_regime_outputs_ordered_probabilities_and_summaries() -> None:
    result = calculate_causal_daily_market_regime_features(
        _features(), training_days=100, min_training_days=60, refit_days=10
    )

    assert result.columns.tolist() == list(DAILY_REGIME_OUTPUT_COLUMNS)
    finite = result.dropna(subset=["regime_prob_low"])
    assert not finite.empty
    probabilities = finite[
        ["regime_prob_low", "regime_prob_mid", "regime_prob_high"]
    ]
    next_probabilities = finite[
        ["regime_next_prob_low", "regime_next_prob_mid", "regime_next_prob_high"]
    ]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(next_probabilities.sum(axis=1), 1.0)
    assert ((finite["regime_confidence"] >= 1 / 3) & (finite["regime_confidence"] <= 1)).all()
    assert ((finite["regime_entropy"] >= 0) & (finite["regime_entropy"] <= 1)).all()
    assert ((finite["regime_transition_score"].dropna() >= 0) &
            (finite["regime_transition_score"].dropna() <= 1)).all()


def test_daily_regime_is_causal() -> None:
    features = _features()
    changed = features.copy()
    cutoff = 220
    changed.iloc[cutoff:, changed.columns.get_loc("market_return_1d")] += 5.0

    baseline = calculate_causal_daily_market_regime_features(
        features, training_days=100, min_training_days=60, refit_days=10
    )
    modified = calculate_causal_daily_market_regime_features(
        changed, training_days=100, min_training_days=60, refit_days=10
    )
    pd.testing.assert_frame_equal(baseline.iloc[:cutoff], modified.iloc[:cutoff])


def test_daily_regime_requires_history_and_feature_columns() -> None:
    short = _features(50)
    result = calculate_causal_daily_market_regime_features(
        short, training_days=100, min_training_days=60, refit_days=10
    )
    assert result.isna().all().all()

    with np.testing.assert_raises(ValueError):
        calculate_causal_daily_market_regime_features(short.drop(columns=["market_vol_5d"]))


def test_close_panel_features_are_finite_only_after_required_windows() -> None:
    features = build_daily_market_feature_panel(_close_panel(30))
    assert features["market_vol_5d"].iloc[:4].isna().all()
    assert features["market_vol_20d"].iloc[:19].isna().all()
    assert features["market_drawdown_20d"].iloc[:19].isna().all()
