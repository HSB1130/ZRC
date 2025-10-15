#!/usr/bin/env python3
"""ZRC training script using TRL's GRPOTrainer with vLLM rollouts."""

from __future__ import annotations

import argparse
import json
import random
import re
import os
from setproctitle import setproctitle, getproctitle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zrc import ZRCConfig, ZeroResourceCurriculum
from zrc.config import TransformerConfig
from zrc.models import load_causal_lm

# Process 이름 설정
setproctitle("sungbinhan9039")
print("변경된 프로세스 이름:", getproctitle())

# GPU 0번과 1번만 보이도록 제한
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print("사용 가능한 GPU 개수:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))

SYSTEM_PROMPT = (
    "You are a helpful assistant. Please reason step by step, and put your final numeric "
    "answer within \\boxed{...}. Do not include any extra text after the boxed answer."
)

STOP_SEQUENCES = ["</s>", "\\end{", "\n\n\n"]


def format_chat_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Match the chat template used during initial difficulty measurements."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_model_output(model_output_str: str) -> Optional[float]:
    """Extract numeric value from model output, prioritizing \\boxed{...} content."""
    m = re.search(r"\\boxed\{([^{}]+)\}", model_output_str)
    cand = m.group(1).strip() if m else None

    if cand is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", model_output_str)
        cand = nums[-1] if nums else None

    if cand is None:
        return None

    try:
        cand = cand.replace(",", "")
        if "/" in cand and cand.count("/") == 1:
            parts = cand.split("/")
            return float(parts[0].strip()) / float(parts[1].strip())
        return float(cand)
    except ValueError:
        return None


def parse_ground_truth_from_deepscaler(answer_str: str, solution_str: str) -> Optional[float]:
    """Mirror initial difficulty parsing for DeepScaleR answers."""
    s = answer_str.strip()
    m = re.fullmatch(r"\s*[-+]?\d+(?:/\d+|\.\d+)?\s*", s)
    if m:
        try:
            if "/" in s:
                parts = s.split("/")
                return float(parts[0]) / float(parts[1])
            return float(s)
        except Exception:
            pass

    v = parse_model_output(solution_str)
    if v is not None:
        return v

    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
    return float(nums[-1]) if nums else None


@dataclass
class PrecomputedDifficulty:
    difficulty: float
    ground_truth: str
    solve_rate: Optional[float] = None


@dataclass
class TaskRecord:
    task_id: str
    prompt: str
    answer: str
    difficulty: Optional[float]


class CurriculumIterableDataset(IterableDataset):
    """Infinite dataset that streams prompts according to the ZRC scheduler."""

    def __init__(self, curriculum: ZeroResourceCurriculum, task_lookup: Dict[str, TaskRecord]) -> None:
        self.curriculum = curriculum
        self.task_lookup = task_lookup

    def __iter__(self) -> Iterator[Dict[str, str]]:
        while True:
            cluster_id = self.curriculum.select_cluster()
            task_ids = self.curriculum.sample_tasks_from_cluster(cluster_id, num_tasks=1)
            if not task_ids:
                continue
            task_id = task_ids[0]
            record = self.task_lookup[task_id]
            yield {
                "prompt": record.prompt,
                "task_id": task_id,
                "answer": record.answer,
            }


def build_prompt(problem: str) -> str:
    return format_chat_prompt(problem)


def sanitize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\\boxed\\s*\\{([^}]*)\\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = re.sub(r"\\frac\\{([^{}]+)\\}\\{([^{}]+)\\}", r"(\1)/(\2)", text)
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\\text\\{([^}]*)\\}", r"\1", text)
    text = re.sub(r"\\sqrt\\{([^}]*)\\}", r"sqrt(\1)", text)
    text = re.sub(r"\\__", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    text = text.strip(".")
    return text


def canonicalize_answer(text: str) -> str:
    text = sanitize_answer(text)
    return re.sub(r"[()]", "", text)


def evaluate_response(response: str, expected: str, eps: float = 1e-6) -> Tuple[bool, float]:
    pred_value = parse_model_output(response)
    expected_value = parse_model_output(expected)
    success = False

    if pred_value is not None and expected_value is not None:
        success = abs(pred_value - expected_value) < eps
    else:
        normalized_prediction = canonicalize_answer(response)
        normalized_expected = canonicalize_answer(expected)
        if normalized_expected:
            success = normalized_prediction == normalized_expected

    reward = 1.0 if success else -1.0
    return success, reward


def deterministic_embeddings(
    texts: Iterable[str],
    tokenizer,
    model,
    embedding_dim: int,
    max_length: int = 128,
) -> np.ndarray:
    embed_layer = model.get_input_embeddings()
    embeddings: List[np.ndarray] = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(embed_layer.weight.device)
        with torch.no_grad():
            token_vecs = embed_layer(input_ids)
            pooled = (
                token_vecs.mean(dim=1)
                .squeeze(0)
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        if pooled.shape[0] != embedding_dim:
            raise ValueError("Embedding dimension mismatch.")
        embeddings.append(pooled.astype(np.float32))
    return np.stack(embeddings, axis=0)


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_precomputed_difficulty(path: Path) -> Dict[str, PrecomputedDifficulty]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    lookup: Dict[str, PrecomputedDifficulty] = {}
    for entry in data:
        question = normalize_problem(entry["question"])
        difficulty_score = float(entry.get("difficulty_score", 50.0))
        difficulty_norm = float(np.clip(difficulty_score / 100.0, 0.0, 1.0))
        ground_truth = entry.get("ground_truth_answer", "")
        solve_rate = entry.get("solve_rate")
        lookup[question] = PrecomputedDifficulty(
            difficulty=difficulty_norm,
            ground_truth=str(ground_truth),
            solve_rate=float(solve_rate) if solve_rate is not None else None,
        )
    return lookup


def load_task_records(
    split: str,
    limit: int,
    precomputed: Dict[str, PrecomputedDifficulty],
) -> List[TaskRecord]:
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split=split)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    records: List[TaskRecord] = []
    for idx, row in enumerate(dataset):
        problem = row["problem"]
        key = normalize_problem(problem)
        stats = precomputed.get(key)
        prompt = build_prompt(problem)
        difficulty = stats.difficulty if stats else None
        if stats and stats.ground_truth:
            answer = stats.ground_truth
        else:
            raw_answer = row.get("answer", "")
            parsed = parse_ground_truth_from_deepscaler(raw_answer or "", row.get("solution", ""))
            answer = str(parsed) if parsed is not None else (raw_answer or "")
        records.append(
            TaskRecord(
                task_id=f"dsr_{idx:05d}",
                prompt=prompt,
                answer=answer,
                difficulty=difficulty,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZRC with GRPO and vLLM on DeepScaleR preview data.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split specification, e.g., 'train' or 'train[:500]'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of tasks to load (0 for all).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of optimizer steps for GRPOTrainer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size supplied to GRPOTrainer.",
    )
    parser.add_argument(
        "--precomputed-path",
        type=str,
        default="final_results.json",
        help="Path to JSON file containing precomputed difficulty stats.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo",
        help="Directory to store GRPO checkpoints and logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    precomputed_map = load_precomputed_difficulty(Path(args.precomputed_path))

    config = ZRCConfig()
    config.transformers = TransformerConfig(
        model_name="Qwen/Qwen2.5-Math-7B",
        tokenizer_name="Qwen/Qwen2.5-Math-7B",
        device_map="auto",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    config.rl.trl.policy_model_name = config.transformers.model_name
    config.rl.trl.tokenizer_name = config.transformers.tokenizer_name
    config.rl.trl.generation_kwargs.setdefault("temperature", 0.7)
    config.rl.trl.generation_kwargs.setdefault("top_p", 0.95)
    config.rl.trl.generation_kwargs.setdefault("stop", STOP_SEQUENCES)

    policy_model, tokenizer = load_causal_lm(config.transformers)
    records = load_task_records(args.split, args.limit, precomputed_map)
    if not records:
        raise RuntimeError("No tasks loaded from dataset.")

    embedding_dim = policy_model.config.hidden_size
    semantic_embeddings = deterministic_embeddings(
        [record.prompt for record in records],
        tokenizer,
        policy_model,
        embedding_dim=embedding_dim,
        max_length=256,
    )

    curriculum = ZeroResourceCurriculum(config=config, embedding_dim=embedding_dim)
    initial_difficulty = {
        record.task_id: record.difficulty
        for record in records
        if record.difficulty is not None
    }
    curriculum.register_tasks(
        [record.task_id for record in records],
        semantic_embeddings,
        initial_difficulty=initial_difficulty,
    )

    task_lookup: Dict[str, TaskRecord] = {record.task_id: record for record in records}
    dataset = CurriculumIterableDataset(curriculum, task_lookup)
    trainer_holder: Dict[str, Optional[GRPOTrainer]] = {"trainer": None}
    step_counter = {"calls": 0}

    def _apply_curriculum_kl_penalty(trainer: Optional[GRPOTrainer]) -> None:
        if trainer is None:
            return
        beta_value = curriculum.current_kl_penalty()
        if hasattr(trainer, "args"):
            trainer.args.beta = beta_value
        if hasattr(trainer, "beta"):
            trainer.beta = beta_value  # type: ignore[attr-defined]
        scheduler = getattr(trainer, "beta_scheduler", None)
        if scheduler is not None:
            if hasattr(scheduler, "value"):
                scheduler.value = beta_value
            if hasattr(scheduler, "set_value"):
                try:
                    scheduler.set_value(beta_value)  # type: ignore[call-arg]
                except TypeError:
                    pass
        controller = getattr(trainer, "kl_controller", None)
        if controller is not None:
            if hasattr(controller, "value"):
                controller.value = beta_value
            if hasattr(controller, "target_kl"):
                controller.target_kl = beta_value

    def reward_function(
        *,
        prompts: List[str],
        completions: List[str],
        completion_ids,
        task_id: List[str],
        answer: List[str],
        trainer_state,
        **kwargs,
    ) -> List[float]:
        del completion_ids, trainer_state, kwargs
        trainer = trainer_holder.get("trainer")
        if step_counter["calls"] > 0:
            curriculum.advance_stabilization()
        _apply_curriculum_kl_penalty(trainer)
        # Expand metadata if vLLM generated multiple completions per prompt.
        task_ids = list(task_id)
        answers = list(answer)
        if len(task_ids) != len(completions) and len(task_ids) > 0:
            repeats = len(completions) // len(task_ids)
            task_ids = [tid for tid in task_ids for _ in range(max(1, repeats))]
            answers = [ans for ans in answers for _ in range(max(1, repeats))]
        rewards: List[float] = []
        successes: List[bool] = []
        for completion, tid, ans in zip(completions, task_ids, answers):
            success, reward = evaluate_response(completion, ans)
            rewards.append(reward)
            successes.append(success)
        verification_failures = [not flag for flag in successes]
        if task_ids:
            curriculum.record_outcomes(task_ids, successes, verification_failures)
            _apply_curriculum_kl_penalty(trainer)
        step_counter["calls"] += 1
        return rewards

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.rl.trl.learning_rate,
        max_steps=args.steps,
        bf16=True,
        remove_unused_columns=False,
        use_vllm=config.rl.trl.use_vllm,
        vllm_mode=config.rl.trl.vllm_mode,
        vllm_tensor_parallel_size=config.rl.trl.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=config.rl.trl.vllm_enable_sleep_mode,
        generation_kwargs=config.rl.trl.generation_kwargs,
        num_generations=config.rl.trl.num_generations,
        max_completion_length=config.rl.trl.max_completion_length,
        steps_per_generation=config.rl.trl.steps_per_generation,
        beta=config.rl.trl.beta,
        gradient_checkpointing=config.rl.trl.gradient_checkpointing,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        accelerator_config={"dispatch_batches": False},
    )

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=reward_function,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer_holder["trainer"] = trainer
    _apply_curriculum_kl_penalty(trainer)

    trainer.train()

    final_state = curriculum.curriculum_state()
    print("Final cluster success rates:")
    for cluster_id, stats in final_state.cluster_stats.items():
        rate = stats.success_rate * 100.0
        print(f"  Cluster {cluster_id}: {rate:.1f}% over {stats.trials} trials")


if __name__ == "__main__":
    main()
