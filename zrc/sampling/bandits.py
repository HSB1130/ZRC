"""Multi-armed bandit algorithms for curriculum sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..config import BanditConfig


class BanditStrategy(ABC):
    """Abstract base class for curriculum bandits."""

    def __init__(self, config: BanditConfig) -> None:
        self.config = config

    @abstractmethod
    def select(self) -> int:
        """Return the index of the cluster to sample next."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update the bandit posterior with new feedback."""

    @abstractmethod
    def ensure_arm(self, arm: int) -> None:
        """Ensure state exists for the arm."""


@dataclass
class _BetaStats:
    alpha: float
    beta: float


class ThompsonSamplingBandit(BanditStrategy):
    """Thompson Sampling with Beta posteriors."""

    def __init__(self, config: BanditConfig) -> None:
        super().__init__(config)
        self._posterior: Dict[int, _BetaStats] = {}

    def ensure_arm(self, arm: int) -> None:
        self._posterior.setdefault(
            arm,
            _BetaStats(alpha=self.config.prior_alpha, beta=self.config.prior_beta),
        )

    def select(self) -> int:
        if not self._posterior:
            raise RuntimeError("No arms registered; call ensure_arm() first.")
        samples = {
            arm: np.random.beta(stats.alpha, stats.beta)
            for arm, stats in self._posterior.items()
        }
        return int(max(samples, key=samples.get))

    def update(self, arm: int, reward: float) -> None:
        self.ensure_arm(arm)
        stats = self._posterior[arm]
        stats.alpha += reward
        stats.beta += 1.0 - reward


class UCB1Bandit(BanditStrategy):
    """Deterministic Upper Confidence Bound."""

    def __init__(self, config: BanditConfig) -> None:
        super().__init__(config)
        self._counts: Dict[int, int] = {}
        self._values: Dict[int, float] = {}
        self._total: int = 0

    def ensure_arm(self, arm: int) -> None:
        self._counts.setdefault(arm, 0)
        self._values.setdefault(arm, 0.0)

    def select(self) -> int:
        if not self._counts:
            raise RuntimeError("No arms registered; call ensure_arm() first.")
        self._total += 1
        scores: Dict[int, float] = {}
        for arm, count in self._counts.items():
            if count == 0:
                scores[arm] = float("inf")
                continue
            exploration = np.sqrt(
                self.config.ucb_confidence * np.log(self._total) / count
            )
            scores[arm] = self._values[arm] + exploration
        return int(max(scores, key=scores.get))

    def update(self, arm: int, reward: float) -> None:
        self.ensure_arm(arm)
        self._counts[arm] += 1
        count = self._counts[arm]
        value = self._values[arm]
        self._values[arm] = value + (reward - value) / count

