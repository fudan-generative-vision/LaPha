export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=840
export NCCL_TIMEOUT=840
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

ACCELERATE_FORCE_IP=127.0.0.1 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
accelerate launch --main_process_port 29501 \
  --num_processes 6 \
  --config_file deepspeed_zero3.yaml \
  run_dapo.py --config lapha.yaml