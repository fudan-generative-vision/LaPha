# LaPha: Latent Poincaré Shaping for Agentic Reinforcement Learning

This repository contains the reference implementation of **LaPha**, a method for training **AlphaZero-like LLM agents**
in a **prompt-centered Poincaré latent space**, enabling dense process reward shaping and lightweight value-guided
test-time scaling.

> Paper: *Latent Poincaré Shaping for Agentic Reinforcement Learning* (https://arxiv.org/pdf/2602.09375)

<p align="center">
  <img src="assets/fig1_overview.png" width="900" alt="LaPha overview (Figure 1)"/>
</p>

---

## What is LaPha?

LaPha maps each **agent state** (a prompt + tool feedback + partial reasoning trace) into a **root-centered Poincaré ball**
and uses **hyperbolic geometry** to define a *potential function* that provides **dense process rewards** during RL.

Core ideas (matching the paper + current codebase):

- **Prompt-centered hyperbolic latent state**
  - Mean-pool the LM backbone hidden states into a state embedding.
  - Root-center w.r.t. the prompt (root state), and map into the Poincaré ball via an exponential map.
- **Potential-based dense shaping (process reward)**
  - Let \(y_i\) be the latent for node \(i\), and \(\mathcal{Y}^+\) be verified-correct terminal leaves.
  - Define distances:
    - \(d_i^{\text{goal}} = \min_{y_\omega \in \mathcal{Y}^+} d_\mathbb{D}(y_i, y_\omega)\)
    - \(d_i^{\text{root}} = d_\mathbb{D}(y_i, 0)\)
  - Potential:
    \[
      V(i) = \frac{d_i^{\text{root}}}{d_i^{\text{root}} + d_i^{\text{goal}}} \in [0, 1]
    \]
  - Process reward on edge \((i \to j)\):
    \[
      r(i,j) = V(j) - V(i)
    \]
- **Lightweight value head on the same shared latent**
  - A linear value head predicts \( \hat{V}(s) \) with sigmoid and is trained with MSE to match the geometry-derived potential.
  - At test time, the value head guides MCTS with **almost no extra overhead**.
- **Latent-space pruning**
  - Periodically cluster visited nodes in latent space and prune redundant paraphrastic branches, improving exploration efficiency.

---

## Key visualizations

### Value head aligns with geometry-derived potential (Figure 2)

<p align="center">
  <img src="assets/fig2_poincare_rollout.png" width="850" alt="Rollout tree in Poincaré disk and value prediction"/>
</p>

### During training, value-head top-1 selection improves beyond average leaf correctness (Figure 2)

<p align="center">
  <img src="assets/fig2_pass1_vs_avgacc.png" width="600" alt="Pass@1 vs Average Accuracy"/>
</p>

---

## Repository layout

- `trainer/`
  - `agent.py`: MCTS agent, latent distance shaping, pruning / clustering logic
  - `mtpo_trainer.py`: LaPha trainer (GRPO-style optimization + MCTS rollouts)
  - `mtpo_config.py`: training + MCTS hyperparameters (depth/breadth/num_sim/prune_per, distance shaping, etc.)
  - `latent_bank.py`: append-only latent store (GPU/CPU mirroring)
  - `vllm_client.py`: vLLM HTTP client for fast generation during training/eval
- `tools/`
  - `rpc_python_server.py`: FastAPI service used by the `execute_python_code` tool
  - `remote_python_code_interpreter.py`: tool wrapper calling the RPC server
- `eval/`
  - `rollout_jsonl.py`: roll out predictions on JSONL datasets (react/value/single modes)
  - `rewards.py`: rule-based graders + optional LLM-as-judge fallback
- `data/`: evaluation JSONLs (AIME’24/25, MATH-500, etc.)
- `run_dapo.py`, `run_dapo.sh`, `lapha.yaml`: training entry points/config
- `eval.sh`, `eval_math.py`: evaluation scripts (rollout + scoring)

---

## Installation
> The provided `environment.yml` is a **conda explicit environment file** (see its header).

```bash
export PYTHONNOUSERSITE=1
source /usr/share/miniconda/bin/activate
conda create -n lapha-test -y -c conda-forge -c defaults \
  python=3.11 pip git ninja cmake cxx-compiler make pkg-config \
  cairo pango poppler graphviz

conda activate lapha-test
python -m pip install -U pip

python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

awk '/^  - pip:/{p=1;next} p && /^prefix:/{p=0} p && /^      - /{sub(/^      - /,""); print}' environment.yml > requirements.txt

grep -Ev \
'^(torch|torchvision|torchaudio|triton|flash-attn|cupy-cuda12x|deepspeed|xformers|vllm)(==|$)|^nvidia-(cublas|cuda|cudnn|cufft|cufile|curand|cusolver|cusparse|cusparselt|nccl|nvjitlink|nvtx)-cu12==' \
requirements.txt > requirements.rest.txt
python -m pip install -r requirements.rest.txt

python -m pip install triton==3.4.0
python -m pip install cupy-cuda12x==13.6.0
python -m pip install xformers==0.0.32.post1
python -m pip install deepspeed==0.18.0
python -m pip install vllm==0.11.0
python -m pip install flash-attn==2.8.3 --no-build-isolation --no-deps
```

> Notes:
>
> * vLLM requires a CUDA GPU setup.
> * If you use `attn_implementation: flash_attention_2`, you may need `flash-attn` matching your CUDA/PyTorch.

---

## Quickstart: evaluation on provided JSONL benchmarks

### 0) Start the Python tool server (for `execute_python_code`)

> **Security note:** this server executes Python. Bind to localhost or trusted network only.

```bash
gunicorn tools.rpc_python_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --max-requests 1000
```

### 1) Start a vLLM server for your policy model

Example (edit model & TP according to your GPUs):

```bash
trl vllm-serve \
  --model cbyzju/LaPHA-Math-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

### 2) Run eval
```bash
ENGINE=vllm BASE_URL=http://localhost:8000 \
TOKENIZER_PATH=/path/to/policy_model \
MODE=value \
VALUE_BASE=/path/to/policy_model \
VALUE_HEAD=/path/to/value_head.pt \
REACT_DEPTH=6 REACT_BREADTH=6 \
MCTS_NUM_SIM=128 \
bash eval.sh aime24
```

Outputs:

* rollouts: `eval/rollouts/*.pred.jsonl`
* scores: `eval/results/*.csv` (and logs under `eval/logs/`)
