"""Configuration dataclasses for the ZRC framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class DifficultyConfig:
    """Hyperparameters controlling intrinsic difficulty estimation."""

    pass_rate_smoothing: float = 0.05
    verification_weight: float = 1.0
    min_window: int = 32


@dataclass
class DPMeansConfig:
    """Hyperparameters for the DP-means clustering procedure."""

    lambda_param: float = 0.5
    max_iterations: int = 50
    tolerance: float = 1e-4


@dataclass
class BOCPDConfig:
    """Hyperparameters for Bayesian Online Change Point Detection."""

    hazard: float = 0.01
    threshold: float = 0.6
    soft_threshold: float = 0.75
    hard_threshold: float = 0.9
    min_steps_between_recluster: int = 100


@dataclass
class StabilizationConfig:
    """Hyperparameters for the stabilization window after reclustering."""

    window: int = 50
    kl_scale: float = 2.0


@dataclass
class BanditConfig:
    """Hyperparameters for the bandit sampler."""

    strategy: str = "thompson"
    ucb_confidence: float = 2.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


@dataclass
class TransformerConfig:
    """Configuration for loading Hugging Face transformers models."""

    model_name: str = "sshleifer/tiny-gpt2"
    tokenizer_name: Optional[str] = None
    device_map: Optional[str] = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class TRLConfig:
    """Configuration for TRL GRPO training."""

    policy_model_name: str = "Qwen/Qwen2.5-Math-7B"
    tokenizer_name: Optional[str] = None
    learning_rate: float = 5e-7
    gradient_checkpointing: bool = True
    num_generations: int = 4
    max_completion_length: int = 256
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_enable_sleep_mode: bool = False
    beta: float = 0.0
    steps_per_generation: Optional[int] = 1
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.7, "top_p": 1.0}
    )


@dataclass
class VLLMConfig:
    """Configuration for vLLM inference."""

    model: str = "sshleifer/tiny-gpt2"
    tensor_parallel_size: int = 1
    dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class RLConfig:
    """Settings for the reinforcement learning updater."""

    update_fn: Optional[Callable] = None
    update_batch_size: int = 32
    max_kl: float = 0.01
    gamma: float = 0.99
    trl: TRLConfig = field(default_factory=TRLConfig)


@dataclass
class ZRCConfig:
    """Container for all configuration sections."""

    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    clustering: DPMeansConfig = field(default_factory=DPMeansConfig)
    changepoint: BOCPDConfig = field(default_factory=BOCPDConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    transformers: TransformerConfig = field(default_factory=TransformerConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
