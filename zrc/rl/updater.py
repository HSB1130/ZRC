"""Interfaces for connecting ZRC to policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch

from ..config import RLConfig


Batch = Dict[str, torch.Tensor]


@dataclass
class RLUpdater:
    """Wraps an arbitrary policy update function."""

    config: RLConfig

    def update(self, policy: Any, batch: Batch) -> Dict[str, Any]:
        if self.config.update_fn is None:
            raise RuntimeError("No update_fn provided in RLConfig.")
        return self.config.update_fn(policy, batch, self.config)


def update_policy_with_ppo_like_step(
    policy: Any, batch: Batch, config: RLConfig
) -> Dict[str, Any]:
    """Reference implementation for a PPO-style clipped update.

    This is intentionally lightweight: it assumes ``policy`` implements
    ``policy.evaluate_actions(observations, actions)`` returning log-probs
    and value estimates, and ``policy.optimizer`` exposes ``zero_grad`` and
    ``step`` methods after calling ``loss.backward()``.
    """
    observations = batch["observations"]
    actions = batch["actions"]
    advantages = batch["advantages"]
    old_log_probs = batch["log_probs"]
    returns = batch["returns"]

    eval_result = policy.evaluate_actions(observations, actions)
    log_probs = eval_result["log_probs"]
    values = eval_result["values"]

    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.max_kl, 1.0 + config.max_kl)
    policy_loss = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))

    value_loss = torch.mean((returns - values) ** 2)
    entropy_tensor = eval_result.get("entropy")
    entropy_loss = -torch.mean(entropy_tensor) if entropy_tensor is not None else torch.tensor(0.0, device=policy_loss.device)
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

    policy.optimizer.zero_grad()
    total_loss.backward()
    policy.optimizer.step()

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float((-entropy_loss).detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
