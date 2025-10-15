"""Utilities to load transformer-based causal language models."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import TransformerConfig

logger = logging.getLogger(__name__)


def load_causal_lm(config: TransformerConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and tokenizer according to the configuration."""
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding_side="left",
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, config.torch_dtype, torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    original_vocab = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab = len(tokenizer)
    if tokenizer_vocab > original_vocab:
        model.resize_token_embeddings(tokenizer_vocab)
    elif tokenizer_vocab < original_vocab:
        logger.warning(
            "Tokenizer vocab size (%s) smaller than model (%s); skipping resize to keep original weights.",
            tokenizer_vocab,
            original_vocab,
        )
    model.eval()
    return model, tokenizer
