from __future__ import annotations

import logging
import warnings
import weakref
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.exceptions import ConvergenceWarning

from .base import BaseFactor

HMM_REGIME_STATES = ("low", "mid", "high")
HMM_TRAINING_DAYS = 5
HMM_MIN_TRAINING_OBSERVATIONS = 120
HMM_RETURN_SCALE = 10_000.0
HMM_RANDOM_STATE = 42
HMM_INITIAL_ITERATIONS = 100
HMM_WARM_START_ITERATIONS = 20

DAILY_REGIME_STATES = ("low", "mid", "high")
DAILY_REGIME_FEATURE_COLUMNS = (
    "market_return_1d",
    "market_breadth_1d",
    "market_dispersion_1d",
    "market_vol_5d",
    "market_vol_20d",
    "market_drawdown_20d",
)
DAILY_REGIME_OUTPUT_COLUMNS = (
    "regime_state",
    "regime_next_state",
    "regime_prob_low",
    "regime_prob_mid",
    "regime_prob_high",
    "regime_next_prob_low",
    "regime_next_prob_mid",
    "regime_next_prob_high",
    "regime_expected_vol",
    "regime_confidence",
    "regime_entropy",
    "regime_transition_score",
)
DAILY_REGIME_RANDOM_STATE = 42

_HMM_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[pd.DataFrame], dict[tuple[int, int], pd.DataFrame]],
] = {}


def _regime_cache(df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    key = id(df)
    entry = _HMM_CACHE.get(key)
    if entry is None or entry[0]() is not df:
        ref = weakref.ref(
            df,
            lambda _ref, cache_key=key: _HMM_CACHE.pop(cache_key, None),
        )
        entry = (ref, {})
        _HMM_CACHE[key] = entry
    return entry[1]


def _state_order_by_variance(model: GaussianHMM) -> np.ndarray:
    variances = np.asarray(model.covars_, dtype=float).reshape(
        model.n_components, -1
    )[:, 0]
    return np.argsort(variances, kind="stable")


def build_daily_market_feature_panel(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Build causal shared-market features from aligned reference closes."""
    if not isinstance(close_panel.index, pd.DatetimeIndex):
        raise TypeError("close_panel requires a DatetimeIndex")
    if close_panel.index.has_duplicates:
        raise ValueError("close_panel index contains duplicate dates")
    if close_panel.empty or close_panel.shape[1] == 0:
        return pd.DataFrame(index=close_panel.index, columns=DAILY_REGIME_FEATURE_COLUMNS)

    closes = close_panel.apply(pd.to_numeric, errors="coerce").sort_index()
    closes = closes.where(closes > 0)
    returns = np.log(closes).diff()
    market_return = returns.mean(axis=1, skipna=True)
    breadth = returns.gt(0).where(returns.notna()).mean(axis=1, skipna=True)
    dispersion = returns.std(axis=1, skipna=True, ddof=0)
    market_vol_5d = market_return.rolling(5, min_periods=5).std(ddof=0)
    market_vol_20d = market_return.rolling(20, min_periods=20).std(ddof=0)
    market_level = np.exp(market_return.fillna(0).cumsum())
    market_drawdown = market_level / market_level.rolling(20, min_periods=20).max() - 1.0
    return pd.DataFrame(
        {
            "market_return_1d": market_return,
            "market_breadth_1d": breadth,
            "market_dispersion_1d": dispersion,
            "market_vol_5d": market_vol_5d,
            "market_vol_20d": market_vol_20d,
            "market_drawdown_20d": market_drawdown,
        },
        index=closes.index,
    )


def _daily_hmm_fit(
    values: np.ndarray,
    random_state: int = DAILY_REGIME_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a standardized three-state daily HMM and order states by variance."""
    if len(values) < 3 or not np.isfinite(values).all():
        return None
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    standardized = (values - center) / scale
    if np.unique(standardized, axis=0).shape[0] < 3:
        return None
    model = GaussianHMM(
        n_components=len(DAILY_REGIME_STATES),
        covariance_type="diag",
        min_covar=1e-4,
        n_iter=100,
        tol=1e-3,
        random_state=random_state,
        implementation="scaling",
    )
    hmm_logger = logging.getLogger("hmmlearn.base")
    previous_level = hmm_logger.level
    try:
        hmm_logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(standardized)
    except (FloatingPointError, ValueError):
        return None
    finally:
        hmm_logger.setLevel(previous_level)
    raw_covars = np.asarray(model.covars_, dtype=float)
    covars_diag = (
        np.diagonal(raw_covars, axis1=1, axis2=2)
        if raw_covars.ndim == 3
        else raw_covars
    )
    arrays = (
        np.asarray(model.startprob_, dtype=float),
        np.asarray(model.transmat_, dtype=float),
        np.asarray(model.means_, dtype=float),
        covars_diag,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        return None
    order = np.argsort(covars_diag.sum(axis=1), kind="stable")
    start, transition, means, covars = arrays
    transition = transition[np.ix_(order, order)]
    return (
        start[order] / max(start[order].sum(), 1e-12),
        transition / np.maximum(transition.sum(axis=1, keepdims=True), 1e-12),
        means[order],
        covars[order],
    )


def calculate_causal_daily_market_regime_features(
    reference_panel: pd.DataFrame,
    training_days: int = 756,
    min_training_days: int = 252,
    refit_days: int = 21,
) -> pd.DataFrame:
    """Return one causal three-state regime result per source trading day.

    The model for day ``d`` is fitted only on rows strictly before ``d`` and
    filters the observation on ``d``.  ``regime_next_prob_*`` is the one-step
    projection for the next trading day.
    """
    if not isinstance(reference_panel.index, pd.DatetimeIndex):
        raise TypeError("reference_panel requires a DatetimeIndex")
    if reference_panel.index.has_duplicates:
        raise ValueError("reference_panel index contains duplicate dates")
    if training_days <= 0 or min_training_days <= 0 or refit_days <= 0:
        raise ValueError("training_days, min_training_days and refit_days must be positive")
    missing = [column for column in DAILY_REGIME_FEATURE_COLUMNS if column not in reference_panel]
    if missing:
        raise ValueError(f"reference_panel is missing feature columns: {missing}")

    ordered = reference_panel.sort_index().loc[:, DAILY_REGIME_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    rows: list[dict[str, object]] = []
    fit_failure_dates: list[str] = []
    insufficient_history_dates: list[str] = []
    if ordered.empty:
        empty = pd.DataFrame(index=ordered.index, columns=DAILY_REGIME_OUTPUT_COLUMNS)
        empty.attrs["model_fit_failure_dates"] = fit_failure_dates
        empty.attrs["insufficient_history_dates"] = insufficient_history_dates
        return empty

    values = ordered.to_numpy(dtype=float)
    previous_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    filtered: np.ndarray | None = None
    previous_probability: np.ndarray | None = None
    last_fit_position = -refit_days

    for position in range(len(ordered)):
        start = max(0, position - training_days)
        training = values[start:position]
        training = training[np.isfinite(training).all(axis=1)]
        if len(training) < min_training_days:
            insufficient_history_dates.append(ordered.index[position].strftime("%Y-%m-%d"))
            previous_model = None
            filtered = None
            previous_probability = None
            continue

        if previous_model is None or position - last_fit_position >= refit_days:
            center = training.mean(axis=0)
            scale = training.std(axis=0, ddof=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
            fitted = _daily_hmm_fit(training)
            if fitted is None:
                fit_failure_dates.append(ordered.index[position].strftime("%Y-%m-%d"))
                previous_model = None
                filtered = None
                previous_probability = None
                continue
            startprob, transition, means, covars = fitted
            previous_model = (startprob, transition, means, covars, center, scale)
            filtered = startprob.copy()
            previous_probability = None
            last_fit_position = position

        if previous_model is None or filtered is None:
            continue
        startprob, transition, means, covars, center, scale = previous_model
        observation = values[position]
        if not np.isfinite(observation).all():
            previous_probability = None
            continue
        observation = (observation - center) / scale
        predicted = filtered @ transition
        variances = np.maximum(covars, 1e-8)
        log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * variances).sum(axis=1)
            + (np.square(observation - means) / variances).sum(axis=1)
        )
        log_posterior = np.log(np.maximum(predicted, 1e-300)) + log_likelihood
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        normalizer = posterior.sum()
        if not np.isfinite(normalizer) or normalizer <= 0:
            previous_probability = None
            continue
        posterior /= normalizer
        next_probability = posterior @ transition
        state_vol = np.sqrt(np.maximum(covars[:, 0], 1e-12)) * scale[0]
        valid_previous = previous_probability is not None
        entropy = -np.sum(np.clip(posterior, 1e-12, 1.0) * np.log(np.clip(posterior, 1e-12, 1.0))) / np.log(3.0)
        rows.append(
            {
                "_position": position,
                "regime_state": DAILY_REGIME_STATES[int(np.argmax(posterior))],
                "regime_next_state": DAILY_REGIME_STATES[int(np.argmax(next_probability))],
                "regime_prob_low": posterior[0],
                "regime_prob_mid": posterior[1],
                "regime_prob_high": posterior[2],
                "regime_next_prob_low": next_probability[0],
                "regime_next_prob_mid": next_probability[1],
                "regime_next_prob_high": next_probability[2],
                "regime_expected_vol": float(
                    np.sqrt(np.sum(next_probability * np.square(state_vol)))
                ),
                "regime_confidence": float(np.max(posterior)),
                "regime_entropy": float(entropy),
                "regime_transition_score": (
                    float(0.5 * np.abs(posterior - previous_probability).sum())
                    if valid_previous
                    else np.nan
                ),
            }
        )
        filtered = posterior
        previous_probability = posterior
    if not rows:
        empty = pd.DataFrame(index=ordered.index, columns=DAILY_REGIME_OUTPUT_COLUMNS)
        empty.attrs["model_fit_failure_dates"] = fit_failure_dates
        empty.attrs["insufficient_history_dates"] = insufficient_history_dates
        return empty
    result = pd.DataFrame(rows).set_index("_position")
    result = result.reindex(range(len(ordered)))
    result.index = ordered.index
    result = result.reindex(columns=DAILY_REGIME_OUTPUT_COLUMNS)
    result.attrs["model_fit_failure_dates"] = fit_failure_dates
    result.attrs["insufficient_history_dates"] = insufficient_history_dates
    return result


def _fit_hmm(
    training_returns: np.ndarray,
    sequence_lengths: list[int],
    min_training_observations: int,
    initial_model: GaussianHMM | None = None,
) -> GaussianHMM | None:
    if (
        len(training_returns) < min_training_observations
        or np.unique(training_returns).size < len(HMM_REGIME_STATES)
        or np.std(training_returns) < 1e-8
    ):
        return None

    warm_start = initial_model is not None
    model = GaussianHMM(
        n_components=len(HMM_REGIME_STATES),
        covariance_type="diag",
        min_covar=1e-3,
        n_iter=(HMM_WARM_START_ITERATIONS if warm_start else HMM_INITIAL_ITERATIONS),
        tol=1e-3,
        random_state=HMM_RANDOM_STATE,
        implementation="scaling",
        init_params="" if warm_start else "stmc",
    )
    if initial_model is not None:
        model.startprob_ = initial_model.startprob_.copy()
        model.transmat_ = initial_model.transmat_.copy()
        model.means_ = initial_model.means_.copy()
        model._covars_ = initial_model._covars_.copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(training_returns.reshape(-1, 1), lengths=sequence_lengths)
    except (FloatingPointError, ValueError):
        return None

    if not (
        np.isfinite(model.startprob_).all()
        and np.isfinite(model.transmat_).all()
        and np.isfinite(model.means_).all()
        and np.isfinite(model.covars_).all()
    ):
        return None
    return model


def _filter_session_probabilities(
    model: GaussianHMM,
    returns: np.ndarray,
) -> np.ndarray:
    probabilities = np.full(
        (len(returns), len(HMM_REGIME_STATES)), np.nan, dtype=float
    )
    state_order = _state_order_by_variance(model)
    means = np.asarray(model.means_, dtype=float).reshape(model.n_components, -1)[
        :, 0
    ]
    variances = np.asarray(model.covars_, dtype=float).reshape(
        model.n_components, -1
    )[:, 0]
    variances = np.maximum(variances, 1e-8)
    filtered = np.asarray(model.startprob_, dtype=float).copy()
    filtered /= filtered.sum()
    has_observation = False

    for position, value in enumerate(returns):
        if has_observation:
            filtered = filtered @ model.transmat_
        if not np.isfinite(value):
            continue

        log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * variances) + np.square(value - means) / variances
        )
        log_posterior = np.log(np.maximum(filtered, 1e-300)) + log_likelihood
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        normalizer = posterior.sum()
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            continue
        filtered = posterior / normalizer
        has_observation = True
        probabilities[position] = filtered[state_order]

    return probabilities


def calculate_causal_hmm_regime_probabilities(
    df: pd.DataFrame,
    training_days: int = HMM_TRAINING_DAYS,
    min_training_observations: int = HMM_MIN_TRAINING_OBSERVATIONS,
) -> pd.DataFrame:
    """Fit on preceding days and filter each current session without lookahead."""
    columns = [f"hmm_regime_prob_{state}_vol" for state in HMM_REGIME_STATES]
    result = pd.DataFrame(np.nan, index=df.index, columns=columns, dtype=float)
    if df.empty:
        return result
    if training_days <= 0 or min_training_observations <= 0:
        raise ValueError("training_days and min_training_observations must be positive")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("HMM regime factors require a DatetimeIndex")

    order = np.argsort(df.index.asi8, kind="stable")
    ordered_index = df.index[order]
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)[order]
    day_labels = ordered_index.normalize()
    unique_days = pd.unique(day_labels)
    returns = np.full(len(close), np.nan, dtype=float)

    day_positions: list[np.ndarray] = []
    for day in unique_days:
        positions = np.flatnonzero(day_labels == day)
        day_positions.append(positions)
        day_close = close[positions]
        valid_pair = (
            np.isfinite(day_close[1:])
            & np.isfinite(day_close[:-1])
            & (day_close[1:] > 0)
            & (day_close[:-1] > 0)
        )
        day_returns = np.full(len(positions) - 1, np.nan, dtype=float)
        day_returns[valid_pair] = (
            np.log(day_close[1:][valid_pair] / day_close[:-1][valid_pair])
            * HMM_RETURN_SCALE
        )
        returns[positions[1:]] = day_returns

    ordered_result = np.full_like(result.to_numpy(dtype=float), np.nan)
    previous_model: GaussianHMM | None = None
    for day_offset in range(1, len(day_positions)):
        training_start = max(0, day_offset - training_days)
        training_sequences: list[np.ndarray] = []
        for positions in day_positions[training_start:day_offset]:
            day_returns = returns[positions]
            day_returns = day_returns[np.isfinite(day_returns)]
            if len(day_returns):
                training_sequences.append(day_returns)
        if not training_sequences:
            continue
        training_returns = np.concatenate(training_sequences)
        if len(training_returns) < min_training_observations:
            continue

        model = _fit_hmm(
            training_returns,
            [len(sequence) for sequence in training_sequences],
            min_training_observations,
            initial_model=previous_model,
        )
        if model is None:
            continue
        previous_model = model
        positions = day_positions[day_offset]
        ordered_result[positions] = _filter_session_probabilities(
            model, returns[positions]
        )

    result.iloc[order] = ordered_result
    return result


class CausalHMMRegimeProbabilityFactor(BaseFactor):
    """One variance-ordered state probability from a causal daily HMM."""

    def __init__(
        self,
        state: Literal["low", "mid", "high"],
        training_days: int = HMM_TRAINING_DAYS,
        min_training_observations: int = HMM_MIN_TRAINING_OBSERVATIONS,
    ) -> None:
        if state not in HMM_REGIME_STATES:
            raise ValueError(f"Unsupported HMM regime state: {state}")
        super().__init__(f"hmm_regime_prob_{state}_vol", min_training_observations)
        self.state = state
        self.training_days = training_days
        self.min_training_observations = min_training_observations

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        cache = _regime_cache(df)
        key = (self.training_days, self.min_training_observations)
        if key not in cache:
            cache[key] = calculate_causal_hmm_regime_probabilities(
                df,
                training_days=self.training_days,
                min_training_observations=self.min_training_observations,
            )
        return cache[key][self.name]
