set -e
PORT=8000
HOST_IP=$(hostname -I | awk '{print $1}')

echo "${HOST_IP}:${PORT}" > ./vllmServer_addr.txt


CUDA_VISIBLE_DEVICES=0,1 \
trl vllm-serve \
  --model Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 12288 \