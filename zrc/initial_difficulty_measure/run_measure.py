import os
import json
import re
import csv
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset
from vllm import LLM, SamplingParams
import numpy as np
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

def setup_distributed():
    """분산 학습 환경 설정"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def parse_ground_truth_from_deepscaler(answer_str: str, solution_str: str) -> float:
    """DeepScaleR 데이터셋의 answer와 solution 필드에서 정답 추출"""
    # 1) answer 필드에서 우선 시도 (순수 숫자인 경우 많음)
    s = answer_str.strip()
    # 분수, 소수, 부호 허용
    m = re.fullmatch(r"\s*[-+]?\d+(?:/\d+|\.\d+)?\s*", s)
    if m:
        try:
            # "3/4" 같은 분수 처리
            if '/' in s:
                parts = s.split('/')
                return float(parts[0]) / float(parts[1])
            else:
                return float(s)
        except Exception:
            pass
    
    # 2) solution의 \boxed{...}에서 추출
    v = parse_model_output(solution_str)
    if v is not None:
        return v
    
    # 3) answer 필드에서 마지막 숫자 fallback
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
    return float(nums[-1]) if nums else None

def parse_model_output(model_output_str: str) -> float:
    """모델 출력에서 \\boxed{...}의 값을 우선 파싱, 실패 시 마지막 숫자를 사용"""
    # 1) boxed 우선
    m = re.search(r"\\boxed\{([^{}]+)\}", model_output_str)
    cand = m.group(1).strip() if m else None
    
    # 2) fallback: 텍스트 내 마지막 숫자
    if cand is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", model_output_str)
        cand = nums[-1] if nums else None
    
    if cand is None:
        return None
    
    # 분수 처리 및 float 변환
    try:
        # 쉼표 제거
        cand = cand.replace(",", "")
        
        # 분수 형태 처리
        if '/' in cand and cand.count('/') == 1:
            parts = cand.split('/')
            return float(parts[0].strip()) / float(parts[1].strip())
        else:
            return float(cand)
    except ValueError:
        return None

def is_correct(model_answer: float, ground_truth_answer: float, eps: float = 1e-6) -> bool:
    """두 float 값을 epsilon 오차 내에서 비교"""
    if model_answer is None or ground_truth_answer is None:
        return False
    return abs(model_answer - ground_truth_answer) < eps

class DeepScaleREvaluator:
    def __init__(self, rank: int, world_size: int, local_rank: int, num_samples: int = 32, use_tensor_parallel: bool = False):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_samples = num_samples
        
        # VLLM 설정 - GPU 병렬화
        if use_tensor_parallel:
            # Tensor parallel 모드 (단일 프로세스, 여러 GPU)
            self.model = LLM(
                model="Qwen/Qwen2.5-Math-7B-Instruct",
                tensor_parallel_size=4,  # 4개 GPU 사용
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                max_model_len=4096,  # 모델의 실제 최대 길이로 수정
                dtype="bfloat16"  # A100에서 bfloat16 사용
            )
        else:
            # Data parallel 모드 (여러 프로세스, 각각 1 GPU)
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            self.model = LLM(
                model="Qwen/Qwen2.5-Math-7B-Instruct",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                max_model_len=4096,  # 모델의 실제 최대 길이로 수정
                dtype="bfloat16"
            )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,  # 다양성을 위한 temperature
            top_p=0.95,
            max_tokens=2048,
            stop=["</s>", "\\end{", "\n\n\n"]
        )
        
        # System prompt
        self.system_prompt = "You are a helpful assistant. Please reason step by step, and put your final numeric answer within \\boxed{...}. Do not include any extra text after the boxed answer."
        
    def load_dataset(self):
        """데이터셋 로드 및 프로세스별 분할"""
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        
        # Tensor parallel 모드이거나 단일 프로세스일 경우 전체 데이터셋 사용
        if self.world_size == 1:
            self.dataset = dataset
            print(f"Processing all {len(dataset)} samples")
        else:
            # Data parallel 모드: 프로세스별로 데이터 분할
            total_samples = len(dataset)
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank
            
            if self.rank == self.world_size - 1:
                # 마지막 프로세스는 남은 샘플 모두 처리
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_rank
            
            self.dataset = dataset.select(range(start_idx, end_idx))
            print(f"Rank {self.rank}: Processing samples {start_idx} to {end_idx}")
        
    def create_prompt(self, question: str) -> str:
        """프롬프트 생성"""
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def evaluate_single_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 문제를 num_samples번 평가"""
        question = item['problem']
        answer_field = item.get('answer', '')
        solution_field = item.get('solution', '')
        
        # Ground truth 추출 (개선된 파싱 로직 사용)
        gt_answer = parse_ground_truth_from_deepscaler(answer_field, solution_field)
        
        if gt_answer is None:
            print(f"Warning: Could not parse ground truth for question: {question[:50]}...")
            gt_answer = 0.0  # 또는 skip 처리
        
        prompt = self.create_prompt(question)
        
        # num_samples번 반복 생성
        prompts = [prompt] * self.num_samples
        outputs = self.model.generate(prompts, self.sampling_params)
        
        correct_count = 0
        predictions = []
        
        for output in outputs:
            generated_text = output.outputs[0].text
            predicted_answer = parse_model_output(generated_text)
            
            is_correct_answer = is_correct(predicted_answer, gt_answer)
            if is_correct_answer:
                correct_count += 1
            
            predictions.append({
                'generated': generated_text[:500] + '...' if len(generated_text) > 500 else generated_text,  # 저장 공간 절약
                'extracted_answer': str(predicted_answer) if predicted_answer is not None else None,
                'is_correct': is_correct_answer
            })
        
        accuracy = correct_count / self.num_samples
        
        return {
            'question': question,
            'ground_truth_answer': str(gt_answer),
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_samples': self.num_samples,
            'solve_rate': accuracy,  # 해결률
            'difficulty_score': 100.0 * (1.0 - accuracy),  # 난이도 점수
            'predictions': predictions
        }
    
    def evaluate(self, save_dir: str = "./results"):
        """전체 데이터셋 평가"""
        self.load_dataset()
        
        results = []
        total_accuracy = 0
        total_difficulty = 0
        
        # 진행 상황 표시
        pbar = tqdm(self.dataset, desc=f"Rank {self.rank} Evaluation")

        for idx, item in enumerate(pbar):
            result = self.evaluate_single_question(item)
            results.append(result)

            total_accuracy += result['accuracy']
            total_difficulty += result['difficulty_score']
            avg_accuracy = total_accuracy / (idx + 1)
            avg_difficulty = total_difficulty / (idx + 1)

            pbar.set_postfix({
                'avg_acc': f"{avg_accuracy:.4f}",
                'avg_diff': f"{avg_difficulty:.2f}"
            })
            
            # 주기적으로 결과 저장
            if (idx + 1) % 10 == 0:
                self.save_results(results, save_dir, intermediate=True)
            
            # 디버그 출력
            if (idx + 1) % 5 == 0:
                print(f"\n[Question {idx+1}] Accuracy: {result['accuracy']:.2%}, "
                      f"Difficulty: {result['difficulty_score']:.2f}")

        # 최종 결과 저장
        self.save_results(results, save_dir, intermediate=False)
        
        return results
    
    def save_results(self, results: List[Dict], save_dir: str, intermediate: bool = False):
        """결과 저장 (JSON 및 CSV)"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_intermediate" if intermediate else "_final"
        
        # 상세 결과 저장 (JSON)
        output_file = f"{save_dir}/rank_{self.rank}_results{suffix}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV 저장 (요약 정보)
        csv_file = f"{save_dir}/rank_{self.rank}_results{suffix}_{timestamp}.csv"
        if results:
            import csv
            fieldnames = ["id", "question", "successful_attempts", "total_attempts", 
                         "solve_rate", "difficulty_score", "ground_truth_answer"]
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx, r in enumerate(results):
                    writer.writerow({
                        "id": idx,
                        "question": r['question'][:100] + '...' if len(r['question']) > 100 else r['question'],
                        "successful_attempts": r['correct_count'],
                        "total_attempts": r['total_samples'],
                        "solve_rate": r['solve_rate'],
                        "difficulty_score": r['difficulty_score'],
                        "ground_truth_answer": r['ground_truth_answer']
                    })
        
        # 요약 통계 저장
        if results:
            total_accuracy = sum(r['accuracy'] for r in results) / len(results)
            avg_difficulty = sum(r['difficulty_score'] for r in results) / len(results)
            
            summary = {
                'rank': self.rank,
                'total_questions': len(results),
                'average_accuracy': total_accuracy,
                'average_solve_rate': total_accuracy,
                'average_difficulty_score': avg_difficulty,
                'num_samples_per_question': self.num_samples,
                'timestamp': timestamp
            }
            
            summary_file = f"{save_dir}/rank_{self.rank}_summary{suffix}_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"Rank {self.rank}: Average accuracy = {total_accuracy:.4f}")
            print(f"Rank {self.rank}: Average difficulty = {avg_difficulty:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5-Math on DeepScaleR dataset')
    parser.add_argument('--num_samples', type=int, default=32, 
                       help='Number of samples per question')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--use_tensor_parallel', action='store_true',
                       help='Use tensor parallelism instead of data parallelism')
    args = parser.parse_args()
    
    if args.use_tensor_parallel:
        # Tensor parallel 모드: 단일 프로세스로 실행
        print("Running with tensor parallelism on 4 GPUs...")
        evaluator = DeepScaleREvaluator(
            rank=0, 
            world_size=1, 
            local_rank=0,
            num_samples=args.num_samples,
            use_tensor_parallel=True
        )
        results = evaluator.evaluate(save_dir=args.save_dir)
    else:
        # Data parallel 모드
        # rank, world_size, local_rank = setup_distributed()
        
        evaluator = DeepScaleREvaluator(
            rank=int(os.environ.get("RANK")), 
            world_size=int(os.environ.get("WORLD_SIZE")), 
            local_rank=int(os.environ.get("LOCAL_RANK")),
            num_samples=args.num_samples,
            use_tensor_parallel=False
        ) 

        results = evaluator.evaluate(save_dir=args.save_dir)
        
        # 모든 프로세스가 완료될 때까지 대기
        if os.environ.get("WORLD_SIZE") > 1:
            dist.barrier()
        
        # Rank 0에서 모든 결과 통합
        if os.environ.get("RANK") == 0 and os.environ.get("WORLD_SIZE") > 1:
            aggregate_results(args.save_dir, os.environ.get("WORLD_SIZE"))
        
        # 정리
        if os.environ.get("WORLD_SIZE") > 1:
            dist.destroy_process_group()

def aggregate_results(save_dir: str, world_size: int):
    """모든 rank의 결과를 통합"""
    all_results = []
    
    # 각 rank의 최종 결과 파일 찾기
    for rank in range(world_size):
        rank_files = list(Path(save_dir).glob(f"rank_{rank}_results_final_*.json"))
        if rank_files:
            latest_file = max(rank_files, key=os.path.getctime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
    
    if all_results:
        # 전체 통계 계산
        total_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
        avg_difficulty = sum(r['difficulty_score'] for r in all_results) / len(all_results)
        
        # 통합 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과 (JSON)
        with open(f"{save_dir}/all_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # CSV 저장
        import csv
        csv_file = f"{save_dir}/all_results_{timestamp}.csv"
        fieldnames = ["id", "question", "successful_attempts", "total_attempts", 
                     "solve_rate", "difficulty_score", "ground_truth_answer"]
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, r in enumerate(all_results):
                writer.writerow({
                    "id": idx,
                    "question": r['question'][:100] + '...' if len(r['question']) > 100 else r['question'],
                    "successful_attempts": r['correct_count'],
                    "total_attempts": r['total_samples'],
                    "solve_rate": r['solve_rate'],
                    "difficulty_score": r['difficulty_score'],
                    "ground_truth_answer": r['ground_truth_answer']
                })
        
        # 요약
        summary = {
            'total_questions': len(all_results),
            'average_accuracy': total_accuracy,
            'average_solve_rate': total_accuracy,
            'average_difficulty_score': avg_difficulty,
            'world_size': world_size,
            'timestamp': timestamp
        }
        
        with open(f"{save_dir}/final_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Final Results:")
        print(f"Total questions evaluated: {len(all_results)}")
        print(f"Average accuracy (over {all_results[0]['total_samples']} samples): {total_accuracy:.4f}")
        print(f"Average difficulty score: {avg_difficulty:.2f}")
        print(f"Results saved to:")
        print(f"  - JSON: {save_dir}/all_results_{timestamp}.json")
        print(f"  - CSV: {csv_file}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
