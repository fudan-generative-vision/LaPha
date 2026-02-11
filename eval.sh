#!/usr/bin/env bash
# eval.sh — Orchestrate rollout and scoring.
# Usage:
#   ./eval.sh            # all datasets
#   ./eval.sh aime25     # single dataset
# Env knobs (policy):
#   BASE_URL, TOKENIZER_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY, MIN_P
# Env knobs (ReAct):
#   REACT_DEPTH=16 (breadth is hard-capped to 1 for ReActAgent)
# Env knobs (MCTS/value):
#   MODE=value
#   REACT_BREADTH=4              # used as MCTS breadth (candidates per expansion)
#   VALUE_MODEL or (VALUE_BASE + VALUE_HEAD)
#   VALUE_DEVICE=cuda:1, VALUE_DTYPE=auto
#   MAX_MODEL_LEN=32768
#   MCTS_NUM_SIM=128, MCTS_C_PUCT=1.0, MCTS_V_PRIOR=0.5, MCTS_VALUE_TRUST=0.5
#   MCTS_PRUNE_PER=128, MCTS_MAX_EXPANDS=2
#   MCTS_NUM_POS_SIM=4, MCTS_PASSK_THRESHOLD=1.0, MCTS_EVAL_ONLY=0/1

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Directories
mkdir -p eval/rollouts eval/results eval/logs

# Policy
export ENGINE="${ENGINE:-vllm}"
export BASE_URL="${BASE_URL:-http://localhost:8000}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/inspire/ssd/project/sais-bio/public/hanchen/save/split/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/policy_model_ckpt80}"

export USE_LLM_JUDGE="${USE_LLM_JUDGE:-1}"
export JUDGE_TOKENIZER_PATH="${JUDGE_TOKENIZER_PATH:-$TOKENIZER_PATH}"
export JUDGE_ENGINE="${JUDGE_ENGINE:-$ENGINE}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$BASE_URL}"

# Decoding
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TEMPERATURE="${TEMPERATURE:-0.3}"
export TOP_P="${TOP_P:-0.8}"
export TOP_K="${TOP_K:-20}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
export MIN_P="${MIN_P:-0.0}"

export REACT_DEPTH="${REACT_DEPTH:-6}"
export REACT_BREADTH="${REACT_BREADTH:-6}"
export PASSATK_K="${PASSATK_K:-16}"

export MODE="${MODE:-value}"

# Value / MCTS
export VALUE_MODEL="${VALUE_MODEL:-}"
export VALUE_BASE="${VALUE_BASE:-/inspire/ssd/project/sais-bio/public/hanchen/save/split/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/policy_model_ckpt80}"
export VALUE_HEAD="${VALUE_HEAD:-/inspire/ssd/project/sais-bio/public/hanchen/save/split/laphaStar_0104_qwen2.5-math-1.5b_mix_grpo_rooth0_nokl/value_head_ckpt80.pt}"
# export VALUE_BASE="${VALUE_BASE:-}"
# export VALUE_HEAD="${VALUE_HEAD:-}"
export VALUE_DEVICE="${VALUE_DEVICE:-cuda:1}"
export VALUE_DTYPE="${VALUE_DTYPE:-auto}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

export MCTS_NUM_SIM="${MCTS_NUM_SIM:-128}"
export MCTS_C_PUCT="${MCTS_C_PUCT:-1.0}"
export MCTS_V_PRIOR="${MCTS_V_PRIOR:-0.0}"
export MCTS_VALUE_TRUST="${MCTS_VALUE_TRUST:-1.0}"
export MCTS_PRUNE_PER="${MCTS_PRUNE_PER:-129}"
export MCTS_MAX_EXPANDS="${MCTS_MAX_EXPANDS:-decay}"
export MCTS_NUM_POS_SIM="${MCTS_NUM_POS_SIM:-1}"
export MCTS_PASSK_THRESHOLD="${MCTS_PASSK_THRESHOLD:-1.0}"

# Standardized data registry (override via env if needed)
DATA_DIR_AIME24="${DATA_DIR_AIME24:-data/aime-24.jsonl}"
DATA_DIR_AIME25="${DATA_DIR_AIME25:-data/aime-25.jsonl}"
DATA_DIR_MATH="${DATA_DIR_MATH:-data/math-500.jsonl}"
DATA_DIR_GAOKAO2023="${DATA_DIR_GAOKAO2023:-data/gaokao-23.jsonl}"
DATA_DIR_OLYMPIAD="${DATA_DIR_OLYMPIAD:-data/olympiad.jsonl}"

TARGET="${1:-all}"
DATASETS=("aime24" "aime25" "math" "gaokao2023" "olympiadbench")
if [[ "$TARGET" != "all" ]]; then
  DATASETS=("$TARGET")
fi

python_bin="${PYTHON:-python}"

rollout_one() {
  local ds="$1"
  local data_path=""
  case "$ds" in
    aime24)        data_path="$DATA_DIR_AIME24" ;;
    aime25)        data_path="$DATA_DIR_AIME25" ;;
    math)          data_path="$DATA_DIR_MATH" ;;
    gaokao2023)    data_path="$DATA_DIR_GAOKAO2023" ;;
    olympiadbench) data_path="$DATA_DIR_OLYMPIAD" ;;
    *) echo "Unknown dataset: $ds" >&2; exit 1 ;;
  esac

  local out_path="eval/rollouts/${ds}.pred.jsonl"
  local log="eval/logs/${ds}.rollout.log"

  echo "[rollout] $ds -> $out_path"
  set +e
  mode_to_use="${MODE:-value}"

  if [[ "$mode_to_use" == "value" ]]; then
    "$python_bin" -m eval.rollout_jsonl \
      --data "$data_path" \
      --out "$out_path" \
      --dataset-name "$ds" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --engine "$ENGINE" \
      --base-url "$BASE_URL" \
      --mode value \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --repetition-penalty "$REPETITION_PENALTY" \
      --min-p "$MIN_P" \
      --depth "$REACT_DEPTH" \
      --breadth "$REACT_BREADTH" \
      --value-base "$VALUE_BASE" \
      ${VALUE_HEAD:+--value-head "$VALUE_HEAD"} \
      ${VALUE_MODEL:+--value-model "$VALUE_MODEL"} \
      --value-device "$VALUE_DEVICE" \
      --value-dtype "$VALUE_DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --mcts-num-sim "$MCTS_NUM_SIM" \
      --mcts-c-puct "$MCTS_C_PUCT" \
      --mcts-v-prior "$MCTS_V_PRIOR" \
      --mcts-value-trust "$MCTS_VALUE_TRUST" \
      --mcts-prune-per "$MCTS_PRUNE_PER" \
      --mcts-max-expands "$MCTS_MAX_EXPANDS" \
      --mcts-num-pos-sim "$MCTS_NUM_POS_SIM" \
      --mcts-passk-threshold "$MCTS_PASSK_THRESHOLD" \
      >"$log" 2>&1

  elif [[ "$mode_to_use" == "react" ]]; then
    "$python_bin" -m eval.rollout_jsonl \
      --data "$data_path" \
      --out "$out_path" \
      --dataset-name "$ds" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --engine "$ENGINE" \
      --base-url "$BASE_URL" \
      --mode react \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --repetition-penalty "$REPETITION_PENALTY" \
      --min-p "$MIN_P" \
      --depth "$REACT_DEPTH" \
      --breadth 1 \
      >"$log" 2>&1

  else
    "$python_bin" -m eval.rollout_jsonl \
      --data "$data_path" \
      --out "$out_path" \
      --dataset-name "$ds" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --engine "$ENGINE" \
      --base-url "$BASE_URL" \
      --mode single \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --repetition-penalty "$REPETITION_PENALTY" \
      --min-p "$MIN_P" \
      >"$log" 2>&1
  fi

  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[rollout] FAILED: $ds. Tail of $log:"
    tail -n 120 "$log"
    exit $rc
  fi
}

# 0) Code server
# nohup gunicorn rpc_python_server:app \
#   --workers 4 \
#   --worker-class uvicorn.workers.UvicornWorker \
#   --bind 0.0.0.0:8001 \
#   --max-requests 1000 \
#   > "eval/code_server.log" 2>&1 &

# 1) Rollout for each dataset
for ds in "${DATASETS[@]}"; do
  rollout_one "$ds"
done

# 2) Score
"$python_bin" -u eval_math.py --dataset "$TARGET"
