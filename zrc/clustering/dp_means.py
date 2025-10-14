"""DP-means clustering as described in the ZRC paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..config import DPMeansConfig


@dataclass
class ClusterAssignment:
    """Associates task ids with cluster indices."""

    assignments: Dict[str, int]
    centroids: np.ndarray


class DPMeans:
    """Non-parametric clustering with hard cluster creation."""

    def __init__(self, config: DPMeansConfig) -> None:
        self.config = config
        self._assignments: Dict[str, int] = {}
        self._centroids: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def assignments(self) -> Dict[str, int]:
        return dict(self._assignments)

    @property
    def centroids(self) -> np.ndarray:
        return np.copy(self._centroids)

    def fit(
        self, task_ids: Iterable[str], embeddings: np.ndarray
    ) -> ClusterAssignment:
        """Run DP-means and return cluster assignments."""
        task_ids = list(task_ids)
        if len(task_ids) == 0:
            raise ValueError("No task ids provided for clustering.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(task_ids):
            raise ValueError("Embeddings must be [N, D] aligned with task_ids.")

        # Initialise with the first point.
        centroids = [embeddings[0]]
        assignments = np.zeros(embeddings.shape[0], dtype=np.int32)

        for idx in range(1, embeddings.shape[0]):
            distances = np.linalg.norm(embeddings[idx] - np.stack(centroids), axis=1)
            min_distance = np.min(distances)
            if min_distance > self.config.lambda_param:
                centroids.append(embeddings[idx])
                assignments[idx] = len(centroids) - 1
            else:
                assignments[idx] = int(np.argmin(distances))

        centroids = np.stack(centroids)

        for _ in range(self.config.max_iterations):
            old_assignments = assignments.copy()
            # Update centroids.
            for cluster_idx in range(centroids.shape[0]):
                mask = assignments == cluster_idx
                if not np.any(mask):
                    continue
                centroids[cluster_idx] = embeddings[mask].mean(axis=0)

            # Re-assign points.
            distances = np.linalg.norm(
                embeddings[:, None, :] - centroids[None, :, :], axis=2
            )
            min_distances = distances.min(axis=1)
            new_assignments = distances.argmin(axis=1)

            for idx, distance in enumerate(min_distances):
                if distance > self.config.lambda_param:
                    centroids = np.vstack([centroids, embeddings[idx]])
                    new_assignments[idx] = centroids.shape[0] - 1

            assignments = new_assignments
            if np.all(assignments == old_assignments):
                break

        self._assignments = {task_id: int(cluster) for task_id, cluster in zip(task_ids, assignments)}
        self._centroids = centroids.astype(np.float32)
        return ClusterAssignment(assignments=self.assignments, centroids=self.centroids)

    def soft_reassign(self, task_ids: Iterable[str], embeddings: np.ndarray) -> Dict[str, int]:
        """Reassign tasks to existing centroids without moving them."""
        if self._centroids.size == 0:
            raise RuntimeError("DPMeans has no centroids; call fit() first.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        distances = np.linalg.norm(
            embeddings[:, None, :] - self._centroids[None, :, :], axis=2
        )
        assignments = distances.argmin(axis=1)
        result = {task_id: int(cluster) for task_id, cluster in zip(task_ids, assignments)}
        self._assignments.update(result)
        return result

