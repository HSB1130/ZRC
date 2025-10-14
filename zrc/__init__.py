"""Zero-Resource Curriculum (ZRC) package.

This package implements the core components described in the paper
“Zero-Resource Curriculum Reinforcement Learning for LLMs”. The modules
under ``zrc`` provide composable building blocks for:

* Estimating intrinsic difficulty from model behaviour.
* Clustering tasks with a non-parametric DP-means variant.
* Detecting distribution shifts via Bayesian Online Change Point Detection.
* Stabilising the curriculum after reclustering events.
* Sampling tasks with multi-armed bandit strategies.
* Integrating the curriculum with reinforcement learning policy updates.
"""

from .config import ZRCConfig
from .curriculum.manager import ZeroResourceCurriculum

__all__ = ["ZRCConfig", "ZeroResourceCurriculum"]
