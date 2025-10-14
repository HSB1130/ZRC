"""Wrapper around vLLM for high-throughput inference."""

from __future__ import annotations

from typing import Iterable, List, Optional

from vllm import LLM, SamplingParams

from ..config import VLLMConfig


class VLLMEngine:
    """Instantiates a vLLM engine and exposes a simple generate API."""

    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._engine = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
        )

    def generate(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs = self._engine.generate(list(prompts), sampling)
        return [output.outputs[0].text for output in outputs]

