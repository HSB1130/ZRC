"""Intrinsic difficulty estimator combining pass-rate and self-verification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np

from ..config import DifficultyConfig


@dataclass
class DifficultySummary:
    """Rolling statistics for a single task."""

    successes: int = 0
    attempts: int = 0
    verification_failures: int = 0
    window_successes: Deque[int] = None  # type: ignore
    window_verifications: Deque[int] = None  # type: ignore

    def __post_init__(self) -> None:
        self.window_successes = deque(maxlen=0)
        self.window_verifications = deque(maxlen=0)


class DifficultyEstimator:
    """Maintains moving difficulty scores for a task corpus.

    Each task accumulates:
    * Empirical pass rate smoothed over a rolling window.
    * Frequency of self-verification failures (model disagrees with itself).
    """

    def __init__(self, config: DifficultyConfig, window_size: int = 200) -> None:
        self.config = config
        self.window_size = max(window_size, config.min_window)
        self._stats: Dict[str, DifficultySummary] = {}

    def update(
        self,
        task_ids: Iterable[str],
        successes: Iterable[bool],
        verification_failures: Optional[Iterable[bool]] = None,
    ) -> None:
        """Update statistics for a batch of tasks."""
        if verification_failures is None:
            verification_failures = (False for _ in task_ids)

        for task_id, success, verification_failure in zip(
            task_ids, successes, verification_failures
        ):
            summary = self._stats.setdefault(
                task_id,
                DifficultySummary(
                    window_successes=deque(maxlen=self.window_size),
                    window_verifications=deque(maxlen=self.window_size),
                ),
            )

            summary.attempts += 1
            summary.successes += int(success)
            summary.verification_failures += int(verification_failure)
            summary.window_successes.append(int(success))
            summary.window_verifications.append(int(verification_failure))

    def score(self, task_ids: Iterable[str]) -> np.ndarray:
        """Return the intrinsic difficulty score for each task."""
        scores: List[float] = []
        for task_id in task_ids:
            summary = self._stats.get(task_id)
            if summary is None or summary.attempts == 0:
                # Unseen tasks default to medium difficulty.
                scores.append(0.5)
                continue

            mean_success = np.mean(summary.window_successes) if summary.window_successes else summary.successes / summary.attempts
            mean_verification = (
                np.mean(summary.window_verifications)
                if summary.window_verifications
                else summary.verification_failures / summary.attempts
            )

            smoothed_fail_rate = (1 - self.config.pass_rate_smoothing) * (1 - mean_success) + self.config.pass_rate_smoothing * (summary.verification_failures + 1) / (
                summary.attempts + 2
            )
            combined = smoothed_fail_rate + self.config.verification_weight * mean_verification
            scores.append(float(np.clip(combined, 0.0, 1.0)))
        return np.asarray(scores, dtype=np.float32)

    def has_history(self, task_id: str) -> bool:
        summary = self._stats.get(task_id)
        return summary is not None and summary.attempts > 0

