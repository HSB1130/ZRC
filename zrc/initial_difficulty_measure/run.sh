export WORLD_SIZE=4

RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 python run_measure.py --num_samples 32 &
RANK=1 LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=1 python run_measure.py --num_samples 32 &
RANK=2 LOCAL_RANK=2 CUDA_VISIBLE_DEVICES=2 python run_measure.py --num_samples 32 &
RANK=3 LOCAL_RANK=3 CUDA_VISIBLE_DEVICES=3 python run_measure.py --num_samples 32 &
wait