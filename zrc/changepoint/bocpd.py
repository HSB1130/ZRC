"""Bayesian Online Change Point Detection (Adams & MacKay, 2007)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np

from ..config import BOCPDConfig


@dataclass
class ClusterHistory:
    """Tracks success-rate history required for BOCPD."""

    run_length_prob: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    last_recluster_step: int = 0
    successes: int = 0
    trials: int = 0
    beta_params: Tuple[float, float] = (1.0, 1.0)


class BOCPD:
    """Maintains change-point probabilities for cluster success rates."""

    def __init__(self, config: BOCPDConfig) -> None:
        self.config = config
        self._clusters: Dict[int, ClusterHistory] = {}
        self._step = 0

    def reset_cluster(self, cluster_id: int) -> None:
        self._clusters[cluster_id] = ClusterHistory(last_recluster_step=self._step)

    def update(
        self, cluster_id: int, successes: int, trials: int
    ) -> float:
        """Update BOCPD state for a cluster and return change-point probability."""
        if trials == 0:
            return 0.0

        self._step += 1
        history = self._clusters.setdefault(
            cluster_id, ClusterHistory(last_recluster_step=self._step)
        )
        hazard = self.config.hazard
        run_length_prob = history.run_length_prob

        log_likelihoods = []
        for run_length, prob in enumerate(run_length_prob):
            alpha = history.beta_params[0] + history.successes
            beta = history.beta_params[1] + history.trials - history.successes
            mean = alpha / (alpha + beta)
            ll = successes * np.log(mean + 1e-8) + (trials - successes) * np.log(
                1 - mean + 1e-8
            )
            log_likelihoods.append(ll + np.log(prob + 1e-12))

        log_likelihoods = np.asarray(log_likelihoods)
        growth_probs = (1 - hazard) * np.exp(log_likelihoods - log_likelihoods.max())
        cp_prob = hazard * np.sum(np.exp(log_likelihoods - log_likelihoods.max()))
        normalizer = cp_prob + growth_probs.sum()
        if normalizer <= 0:
            cp_prob = 0.0
        else:
            cp_prob /= normalizer

        history.successes += successes
        history.trials += trials
        history.run_length_prob = np.concatenate(
            [growth_probs / max(normalizer, 1e-12), np.array([cp_prob])]
        )

        return float(cp_prob)

    def should_recluster(self, cluster_id: int) -> bool:
        """Return True if enough evidence and time elapsed for reclustering."""
        history = self._clusters.get(cluster_id)
        if history is None:
            return False
        if self._step - history.last_recluster_step < self.config.min_steps_between_recluster:
            return False
        cp_prob = history.run_length_prob[-1] if history.run_length_prob.size else 0.0
        return cp_prob > self.config.threshold

    def mark_reclustered(self, cluster_id: int) -> None:
        if cluster_id in self._clusters:
            self._clusters[cluster_id].last_recluster_step = self._step
            self._clusters[cluster_id].run_length_prob = np.array([1.0])
            self._clusters[cluster_id].successes = 0
            self._clusters[cluster_id].trials = 0

