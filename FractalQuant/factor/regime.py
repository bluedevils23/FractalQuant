from __future__ import annotations

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
