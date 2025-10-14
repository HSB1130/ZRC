"""End-to-end Zero-Resource Curriculum manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..changepoint import BOCPD
from ..clustering import DPMeans
from ..config import ZRCConfig
from ..difficulty import DifficultyEstimator
from ..sampling import BanditStrategy, ThompsonSamplingBandit, UCB1Bandit


@dataclass
class ClusterStats:
    successes: int = 0
    trials: int = 0

    @property
    def success_rate(self) -> float:
        if self.trials == 0:
            return 0.0
        return self.successes / self.trials


@dataclass
class CurriculumState:
    assignments: Dict[str, int] = field(default_factory=dict)
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    cluster_stats: Dict[int, ClusterStats] = field(default_factory=dict)
    stabilization_remaining: int = 0
    step: int = 0


class ZeroResourceCurriculum:
    """Coordinates difficulty estimation, clustering, change detection, and sampling."""

    def __init__(
        self,
        config: ZRCConfig,
        embedding_dim: int,
        semantic_encoder: Optional[callable] = None,
    ) -> None:
        self.config = config
        self.embedding_dim = embedding_dim
        self.semantic_encoder = semantic_encoder

        self.difficulty = DifficultyEstimator(config.difficulty)
        self.dpmeans = DPMeans(config.clustering)
        self.bocpd = BOCPD(config.changepoint)
        self.bandit: BanditStrategy = (
            ThompsonSamplingBandit(config.bandit)
            if config.bandit.strategy == "thompson"
            else UCB1Bandit(config.bandit)
        )

        self.state = CurriculumState()
        self._semantic_embeddings: Dict[str, np.ndarray] = {}
        self._old_distribution: Optional[np.ndarray] = None
        self._new_distribution: Optional[np.ndarray] = None
        self._cluster_order: List[int] = []
        self._initial_difficulty: Dict[str, float] = {}

    def register_tasks(
        self,
        task_ids: Sequence[str],
        semantic_embeddings: Optional[np.ndarray] = None,
        initial_difficulty: Optional[Dict[str, float]] = None,
    ) -> None:
        """Register tasks and run initial clustering."""
        task_ids = list(task_ids)
        if len(task_ids) == 0:
            raise ValueError("register_tasks requires at least one task.")

        if semantic_embeddings is None:
            if self.semantic_encoder is None:
                raise ValueError(
                    "semantic_embeddings must be provided when no encoder is set."
                )
            semantic_embeddings = self.semantic_encoder(task_ids)

        semantic_embeddings = np.asarray(semantic_embeddings, dtype=np.float32)
        if semantic_embeddings.shape != (len(task_ids), self.embedding_dim):
            raise ValueError(
                f"semantic_embeddings must be of shape ({len(task_ids)}, {self.embedding_dim})"
            )

        for task_id, embedding in zip(task_ids, semantic_embeddings):
            self._semantic_embeddings[task_id] = embedding

        self._initial_difficulty = dict(initial_difficulty or {})
        self._initialise_curriculum(task_ids)

    def _initialise_curriculum(self, task_ids: Sequence[str]) -> None:
        embeddings, ids = self._task_embeddings(task_ids)
        assignment = self.dpmeans.fit(ids, embeddings)
        self.state.assignments = assignment.assignments
        self.state.centroids = assignment.centroids
        self.state.cluster_stats = {
            cluster_id: ClusterStats() for cluster_id in set(assignment.assignments.values())
        }

        for cluster_id in self.state.cluster_stats:
            self.bandit.ensure_arm(cluster_id)
            self.bocpd.reset_cluster(cluster_id)

        self._cluster_order = sorted(self.state.cluster_stats.keys())
        self.state.stabilization_remaining = 0
        self.state.step = 0
        self._new_distribution = self._compute_uniform_distribution()
        self._old_distribution = None

    def _task_embeddings(
        self, task_ids: Sequence[str]
    ) -> Tuple[np.ndarray, List[str]]:
        ids = list(task_ids)
        semantic = np.stack([self._semantic_embeddings[task_id] for task_id in ids])
        difficulty_scores = self.difficulty.score(ids)
        if self._initial_difficulty:
            for idx, task_id in enumerate(ids):
                if not self.difficulty.has_history(task_id) and task_id in self._initial_difficulty:
                    difficulty_scores[idx] = self._initial_difficulty[task_id]
        difficulty_scores = difficulty_scores[:, None]
        combined = np.concatenate([semantic, difficulty_scores], axis=1)
        return combined, ids

    def select_cluster(self) -> int:
        """Choose the next cluster to sample from."""
        if not self.state.assignments:
            raise RuntimeError("Curriculum has no assignments; call register_tasks().")

        if self.state.stabilization_remaining > 0:
            assert self._old_distribution is not None and self._new_distribution is not None
            alpha = self.state.stabilization_remaining / max(
                1, self.config.stabilization.window
            )
            mixture = alpha * self._old_distribution + (1 - alpha) * self._new_distribution
            mixture /= mixture.sum()
            idx = int(np.random.choice(len(mixture), p=mixture))
            return self._cluster_order[idx]

        return self.bandit.select()

    def sample_tasks_from_cluster(self, cluster_id: int, num_tasks: int) -> List[str]:
        """Return task ids belonging to a cluster."""
        candidates = [
            task_id
            for task_id, assignment in self.state.assignments.items()
            if assignment == cluster_id
        ]
        if len(candidates) == 0:
            return []
        choice = np.random.choice(candidates, size=min(num_tasks, len(candidates)), replace=False)
        return choice.tolist()

    def record_outcomes(
        self,
        task_ids: Iterable[str],
        successes: Iterable[bool],
        verification_failures: Optional[Iterable[bool]] = None,
    ) -> None:
        """Log training outcomes to update difficulty, bandit rewards, and BOCPD."""
        task_ids = list(task_ids)
        successes = list(successes)
        if verification_failures is not None:
            verification_failures = list(verification_failures)

        self.state.step += 1
        self.difficulty.update(task_ids, successes, verification_failures)

        cluster_rewards: Dict[int, List[int]] = {}
        for idx, task_id in enumerate(task_ids):
            cluster_id = self.state.assignments.get(task_id)
            if cluster_id is None:
                continue
            cluster_rewards.setdefault(cluster_id, []).append(int(successes[idx]))

        for cluster_id, outcomes in cluster_rewards.items():
            stats = self.state.cluster_stats.setdefault(cluster_id, ClusterStats())
            trials = len(outcomes)
            successes_cluster = sum(outcomes)
            stats.trials += trials
            stats.successes += successes_cluster

            reward = successes_cluster / max(trials, 1)
            self.bandit.ensure_arm(cluster_id)
            self.bandit.update(cluster_id, reward)

            cp_prob = self.bocpd.update(cluster_id, successes_cluster, trials)
            if cp_prob > self.config.changepoint.threshold:
                self._trigger_recluster()
                break

        if self.state.stabilization_remaining > 0:
            self.state.stabilization_remaining -= 1

    def _trigger_recluster(self) -> None:
        self._old_distribution = self._compute_distribution()
        embeddings, ids = self._task_embeddings(self.state.assignments.keys())
        assignment = self.dpmeans.fit(ids, embeddings)
        self.state.assignments = assignment.assignments
        self.state.centroids = assignment.centroids
        self.state.cluster_stats = {
            cluster_id: ClusterStats()
            for cluster_id in set(assignment.assignments.values())
        }

        for cluster_id in self.state.cluster_stats:
            self.bandit.ensure_arm(cluster_id)
            self.bocpd.mark_reclustered(cluster_id)

        self._cluster_order = sorted(self.state.cluster_stats.keys())
        self._new_distribution = self._compute_uniform_distribution()
        self.state.stabilization_remaining = self.config.stabilization.window

    def _compute_distribution(self) -> np.ndarray:
        ordered_items = sorted(self.state.cluster_stats.items())
        self._cluster_order = [cluster_id for cluster_id, _ in ordered_items]
        counts = [stats.trials for _, stats in ordered_items]
        counts = np.asarray(counts, dtype=np.float32)
        if counts.sum() == 0:
            return self._compute_uniform_distribution()
        return counts / counts.sum()

    def _compute_uniform_distribution(self) -> np.ndarray:
        self._cluster_order = sorted({cluster for cluster in self.state.assignments.values()})
        num_clusters = len(self._cluster_order)
        if num_clusters == 0:
            return np.array([], dtype=np.float32)
        return np.ones(num_clusters, dtype=np.float32) / num_clusters

    def curriculum_state(self) -> CurriculumState:
        """Return a snapshot of the current curriculum state."""
        return CurriculumState(
            assignments=dict(self.state.assignments),
            centroids=np.copy(self.state.centroids),
            cluster_stats={k: ClusterStats(v.successes, v.trials) for k, v in self.state.cluster_stats.items()},
            stabilization_remaining=self.state.stabilization_remaining,
            step=self.state.step,
        )
