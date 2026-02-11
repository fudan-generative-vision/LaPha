#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/inspire/ssd/project/sais-bio/public/hanchen"
SRC_CKPT="${ROOT}/save/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/checkpoint-80"
OUT_POLICY="${ROOT}/save/split/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/policy_model_ckpt80"
OUT_VHEAD="${ROOT}/save/split/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/value_head_ckpt80.pt"

BASE_MODEL="${ROOT}/qwen2.5-math-1.5b-instruct"

echo "[INFO] using src checkpoint: ${SRC_CKPT}"
echo "[INFO] out policy dir:      ${OUT_POLICY}"
echo "[INFO] out value-head file: ${OUT_VHEAD}"

python "${ROOT}/LaPha/split_valuehead.py" \
  --src "${SRC_CKPT}" \
  --out-policy "${OUT_POLICY}" \
  --out-vhead "${OUT_VHEAD}" \
  --copy-tokenizer \
  --trust-remote-code

if [[ -d "${BASE_MODEL}" ]]; then
  echo "[INFO] copying *.json from base model: ${BASE_MODEL}"
  cp "${BASE_MODEL}"/*json "${OUT_POLICY}/" || true
else
  echo "[WARN] base model dir not found: ${BASE_MODEL} (skip extra json copy)"
fi

echo "[OK] Done. Policy: ${OUT_POLICY}"
echo "[OK] Value head: ${OUT_VHEAD}"
