"""Online (streaming) regime detection.

Provides an incremental Hidden Markov Model that updates regime
estimates with each new observation, suitable for real-time
applications. Wraps the batch HMM from :mod:`quantlite.regimes.hmm`
and makes it incremental via windowed re-inference.

Example::

    import quantlite as ql

    detector = ql.OnlineRegimeDetector(n_regimes=3)

    # Pre-train on historical data
    detector.fit(historical_returns)

    # Update incrementally
    for new_return in live_returns:
        result = detector.update(new_return)
        print(f"Regime: {result.regime}, Confidence: {result.confidence:.2f}")
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "OnlineRegimeDetector",
    "RegimeUpdate",
]


@dataclass
class RegimeUpdate:
    """Result of an online regime detection update.

    Attributes:
        regime: The current most likely regime index.
        confidence: Posterior probability of the current regime
            (between 0 and 1).
        probabilities: Full posterior distribution over regimes.
        regime_changed: Whether the regime changed from the
            previous observation.
        observation_count: Total number of observations processed.
    """

    regime: int
    confidence: float
    probabilities: np.ndarray
    regime_changed: bool
    observation_count: int


class OnlineRegimeDetector:
    """Streaming regime detector using windowed HMM inference.

    Maintains a sliding window of recent observations and performs
    regime inference on each update. Can be pre-trained on historical
    data or fitted on the fly once enough observations accumulate.

    Args:
        n_regimes: Number of hidden regimes. Defaults to 2.
        window_size: Size of the sliding window for inference.
            Defaults to 252 (one trading year).
        min_observations: Minimum observations required before
            fitting the model. Defaults to 50.
        refit_interval: Re-fit the HMM every N observations to
            adapt to structural changes. Defaults to 100.
            Set to 0 to disable automatic refitting.
        rng_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        window_size: int = 252,
        min_observations: int = 50,
        refit_interval: int = 100,
        rng_seed: int | None = None,
    ) -> None:
        self._n_regimes = n_regimes
        self._window_size = window_size
        self._min_observations = min_observations
        self._refit_interval = refit_interval
        self._rng_seed = rng_seed

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._observation_count = 0
        self._last_regime: int | None = None
        self._model: object | None = None  # RegimeModel
        self._updates_since_fit = 0
        self._history: list[RegimeUpdate] = []

    @property
    def n_regimes(self) -> int:
        """Number of hidden regimes."""
        return self._n_regimes

    @property
    def observation_count(self) -> int:
        """Total number of observations processed."""
        return self._observation_count

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has a fitted model."""
        return self._model is not None

    @property
    def history(self) -> list[RegimeUpdate]:
        """List of all regime updates."""
        return list(self._history)

    @property
    def current_regime(self) -> int | None:
        """The current regime, or None if not yet fitted."""
        return self._last_regime

    def fit(
        self,
        returns: np.ndarray | Sequence[float],
    ) -> None:
        """Pre-train the detector on historical return data.

        Args:
            returns: Historical return series. The last
                ``window_size`` observations are kept in the buffer.
        """
        from quantlite.regimes.hmm import fit_regime_model

        arr = np.asarray(returns, dtype=float)
        self._model = fit_regime_model(
            arr, n_regimes=self._n_regimes, rng_seed=self._rng_seed
        )

        # Fill the buffer with the tail of the training data
        tail = arr[-self._window_size:]
        self._buffer.clear()
        for val in tail:
            self._buffer.append(float(val))

        self._observation_count = len(arr)
        self._updates_since_fit = 0

        # Set initial regime from the last observation
        probs = self._infer_current_probabilities()
        if probs is not None:
            self._last_regime = int(np.argmax(probs))

    def update(self, observation: float) -> RegimeUpdate:
        """Process a new observation and return the regime estimate.

        If the model has not been fitted and enough observations have
        accumulated, the model is fitted automatically.

        Args:
            observation: A new return observation.

        Returns:
            A ``RegimeUpdate`` with the current regime estimate.

        Raises:
            RuntimeError: If not enough observations have been
                collected and no pre-trained model exists.
        """
        self._buffer.append(float(observation))
        self._observation_count += 1
        self._updates_since_fit += 1

        # Auto-fit if we have enough data but no model yet
        if self._model is None:
            if len(self._buffer) >= self._min_observations:
                self._fit_from_buffer()
            else:
                # Not enough data yet; return a uniform estimate
                probs = np.ones(self._n_regimes) / self._n_regimes
                update = RegimeUpdate(
                    regime=0,
                    confidence=1.0 / self._n_regimes,
                    probabilities=probs,
                    regime_changed=False,
                    observation_count=self._observation_count,
                )
                self._history.append(update)
                return update

        # Periodic refit
        if (
            self._refit_interval > 0
            and self._updates_since_fit >= self._refit_interval
            and len(self._buffer) >= self._min_observations
        ):
            self._fit_from_buffer()

        probs = self._infer_current_probabilities()
        if probs is None:
            probs = np.ones(self._n_regimes) / self._n_regimes

        regime = int(np.argmax(probs))
        confidence = float(probs[regime])
        changed = self._last_regime is not None and regime != self._last_regime
        self._last_regime = regime

        update = RegimeUpdate(
            regime=regime,
            confidence=confidence,
            probabilities=probs,
            regime_changed=changed,
            observation_count=self._observation_count,
        )
        self._history.append(update)
        return update

    def batch_update(
        self,
        observations: np.ndarray | Sequence[float],
    ) -> list[RegimeUpdate]:
        """Process multiple observations at once.

        Args:
            observations: Sequence of return observations.

        Returns:
            List of ``RegimeUpdate`` results, one per observation.
        """
        arr = np.asarray(observations, dtype=float).ravel()
        return [self.update(float(x)) for x in arr]

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._buffer.clear()
        self._observation_count = 0
        self._last_regime = None
        self._model = None
        self._updates_since_fit = 0
        self._history.clear()

    def _fit_from_buffer(self) -> None:
        """Fit the HMM on the current buffer contents."""
        from quantlite.regimes.hmm import fit_regime_model

        arr = np.array(list(self._buffer), dtype=float)
        try:
            self._model = fit_regime_model(
                arr, n_regimes=self._n_regimes, rng_seed=self._rng_seed
            )
            self._updates_since_fit = 0
        except Exception:
            # If fitting fails, keep the old model
            pass

    def _infer_current_probabilities(self) -> np.ndarray | None:
        """Infer regime probabilities for the current window.

        Returns:
            Posterior probabilities for the last observation, or
            None if inference fails.
        """
        if self._model is None:
            return None

        from quantlite.regimes.hmm import regime_probabilities

        arr = np.array(list(self._buffer), dtype=float)
        try:
            probs = regime_probabilities(self._model, arr)
            return probs[-1]  # Last observation's probabilities
        except Exception:
            return None
