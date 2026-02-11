#!/usr/bin/env bash
set -Eeuo pipefail
source condaEnvs/anaconda3/bin/activate lapha
cd LaPha

LOG_DIR=logs
CONTROL_FILE="$PWD/cmd.txt"
mkdir -p "$LOG_DIR"


pid_vllm=""
pid_test=""
pgid_vllm=""
pgid_test=""

start_jobs() {
  echo "[manager] starting vLLM + train (each in its own process group)..."

  setsid bash vllm_start.sh \
    > "$LOG_DIR/vllm.log" 2>&1 &
  pid_vllm=$!
  pgid_vllm=$pid_vllm

  setsid bash run_dapo.sh \
    > "$LOG_DIR/run_dapo.log" 2>&1 &
  pid_test=$!
  pgid_test=$pid_test

  echo "[manager] vLLM pgid=$pgid_vllm (pid=$pid_vllm), train pgid=$pgid_test (pid=$pid_test)"
}


stop_jobs() {
  echo "[manager] stopping vLLM/train jobs by process group ..."

  if [[ -n "${pgid_vllm:-}" ]]; then
    kill -- -"$pgid_vllm" 2>/dev/null || true
  fi
  if [[ -n "${pgid_test:-}" ]]; then
    kill -- -"$pgid_test" 2>/dev/null || true
  fi

  sleep 5

  if [[ -n "${pgid_vllm:-}" ]]; then
    kill -KILL -- -"$pgid_vllm" 2>/dev/null || true
  fi
  if [[ -n "${pgid_test:-}" ]]; then
    kill -KILL -- -"$pgid_test" 2>/dev/null || true
  fi

  if [[ -n "${pid_vllm:-}" ]]; then
    wait "$pid_vllm" 2>/dev/null || true
  fi
  if [[ -n "${pid_test:-}" ]]; then
    wait "$pid_test" 2>/dev/null || true
  fi

  pid_vllm=""
  pid_test=""
  pgid_vllm=""
  pgid_test=""

  echo "[manager] vLLM/train jobs (and their descendants) should be gone."
}


enter_stop_mode() {
  echo "[manager] entering STOP mode; waiting for 'restart' ..."
  echo "stop" > "$CONTROL_FILE"

  stop_jobs

  while true; do
    local cmd=""
    if [[ -f "$CONTROL_FILE" ]]; then
      read -r cmd < "$CONTROL_FILE" || cmd=""
    fi

    if [[ "$cmd" == "restart" ]]; then
      echo "[manager] detected 'restart' in $CONTROL_FILE, restarting jobs..."
      start_jobs
      echo "run" > "$CONTROL_FILE"
      return
    fi

    sleep 5
  done
}


nohup gunicorn eval.rpc_python_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --max-requests 1000 \
  > "$LOG_DIR/code_server.log" 2>&1 &

echo "[manager] started gunicorn (not monitored)."

echo "run" > "$CONTROL_FILE"
start_jobs

while true; do
  cmd="run"
  if [[ -f "$CONTROL_FILE" ]]; then
    read -r cmd < "$CONTROL_FILE" || cmd="run"
  fi

  case "$cmd" in
    stop)
      enter_stop_mode
      ;;

    restart)
      echo "[manager] got 'restart' command, restarting jobs..."
      stop_jobs
      start_jobs
      echo "run" > "$CONTROL_FILE"
      ;;

    run|"")
      ;;

    *)
      ;;
  esac

  if [[ "$cmd" == "run" ]]; then
    dead=0

    if [[ -n "$pid_vllm" ]] && ! kill -0 "$pid_vllm" 2>/dev/null; then
      echo "[manager] vLLM process died."
      dead=1
    fi

    if [[ -n "$pid_test" ]] && ! kill -0 "$pid_test" 2>/dev/null; then
      echo "[manager] train/test process died."
      dead=1
    fi

    if [[ "$dead" -eq 1 ]]; then
      echo "[manager] program crashed, switching to STOP mode."
      enter_stop_mode
    fi
  fi

  sleep 5
done
