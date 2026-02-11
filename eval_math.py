# -*- coding: utf-8 -*-
"""
Score predictions (*.pred.jsonl) against standardized gold JSONL.

- Reads the gold file (question/answer)
- Reads rollout pred files (supports sharded outputs *.rank*.jsonl)
- Computes pass@1 using:
    (1) rule-based reward functions (max over multiple graders)
    (2) optional LLM-as-judge fallback (self-judge), engine = jet or vllm

IMPORTANT:
- We DO NOT require any extra pip installs for rule-based rewards.
- LLM judge uses the same adapter interface:
      llm.generate(prompts=[...], sampling_params=GenParams)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trainer.vllm_client import VLLMClient

from eval.adapters import GenParams, build_engine_adapter
from eval.rewards import REWARD_FUNCS, LLMJudge, with_llm_judge


# ----------------------------
# Directories
# ----------------------------
EVAL_DIR = Path("eval")
LOG_DIR = EVAL_DIR / "logs"
ROLLOUTS_DIR = EVAL_DIR / "rollouts"
RESULTS_DIR = EVAL_DIR / "results"
for d in (LOG_DIR, ROLLOUTS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Data registry
# ----------------------------
DATA_DIR: Dict[str, str] = {
    "aime24":        os.getenv("DATA_DIR_AIME24",        "data/aime-24.jsonl"),
    "aime25":        os.getenv("DATA_DIR_AIME25",        "data/aime-25.jsonl"),
    "math":          os.getenv("DATA_DIR_MATH",          "data/math-500.jsonl"),
    "gaokao2023":    os.getenv("DATA_DIR_GAOKAO2023",    "data/gaokao-23.jsonl"),
    "olympiadbench": os.getenv("DATA_DIR_OLYMPIAD",      "data/olympiad.jsonl"),
}


def _qid(question: str) -> str:
    qn = unicodedata.normalize("NFKC", question).encode("utf-8")
    return hashlib.sha1(qn).hexdigest()[:10]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().strip("$")
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class RunSummary:
    dataset: str
    time: str
    num: int
    correct_at_1: int
    pass_at_1: float
    tool: str

    k: int
    correct_at_k: int
    pass_at_k: float
    avg_at_k: float          # average accuracy among answered (per-question averaged)
    answered: int
    answered_rate: float     # answered / (num * k_eff)


def _load_gold(path: Path) -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = str(ex["question"])
            qid = str(ex.get("id") or _qid(q))
            m[qid] = {"question": q, "answer": str(ex["answer"])}
    return m


def _load_preds(paths: List[Path]) -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                qid = str(ex.get("qid") or _qid(ex.get("question", "")))
                if qid not in m:
                    m[qid] = ex
    return m


def _collect_pred_paths(dataset: str) -> List[Path]:
    """
    Collect prediction files:
      - single file: eval/rollouts/{ds}.pred.jsonl
      - or shards:   eval/rollouts/{ds}.pred.rank*.jsonl
    """
    base = Path("eval/rollouts") / f"{dataset}.pred.jsonl"
    if base.exists():
        return [base]
    pattern = base.parent / f"{dataset}.pred.rank*.jsonl"
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(f"Predictions not found: {base} or {pattern}")
    return files


def _build_judge_adapter(judge_tokenizer: "PreTrainedTokenizerBase"):
    """
    Build LLM-as-judge engine adapter.
    Supported engines: jet or vllm.

    Env:
      USE_LLM_JUDGE=1
      JUDGE_ENGINE=jet|vllm
      JUDGE_TOKENIZER_PATH=...
      JUDGE_BASE_URL=...         (only for vllm)
      JUDGE_MODEL_PATH=...       (only for jet; defaults to JUDGE_TOKENIZER_PATH)

      JET_TP_SIZE=... (for jet judge too, if desired)
    """
    engine = (os.getenv("JUDGE_ENGINE") or "vllm").strip().lower()

    if engine == "vllm":
        base_url = os.getenv("JUDGE_BASE_URL") or "http://localhost:8000"
        return build_engine_adapter(
            name="vllm",
            tokenizer=judge_tokenizer,
            vllm_client=VLLMClient(base_url=base_url),
        )

    if engine == "jet":
        tp = int(os.getenv("JET_TP_SIZE", "1"))
        enforce_eager = os.getenv("JET_ENFORCE_EAGER", "1") in ("1", "true", "True")
        model_path = os.getenv("JUDGE_MODEL_PATH") or os.getenv("JUDGE_TOKENIZER_PATH")
        if not model_path:
            raise ValueError("JUDGE_ENGINE=jet requires JUDGE_MODEL_PATH or JUDGE_TOKENIZER_PATH")
        return build_engine_adapter(
            name="jet",
            tokenizer=judge_tokenizer,
            jet_model_path=model_path,
            jet_tp_size=tp,
            jet_enforce_eager=enforce_eager,
            jet_mask_token_id=None,
            jet_block_length=int(os.getenv("SDAR_BLOCK_LENGTH", "4")),
            jet_max_model_len=int(os.getenv("JET_MAX_MODEL_LEN", "0")) or None,
            jet_max_num_seqs=int(os.getenv("JET_MAX_NUM_SEQS", "0")) or None,
            jet_max_active=int(os.getenv("JET_MAX_ACTIVE", "0")) or None,
        )

    raise ValueError(f"Unknown JUDGE_ENGINE: {engine}")


def _score_dataset(dataset: str, gold_map: Dict[str, Dict], pred_map: Dict[str, Dict], k: int) -> RunSummary:
    """
    Scoring:
      - pass@1: uses the first sample (backward compat: completion/final_answer)
      - pass@k: any of the first k samples correct
      - avg@k : per-question average accuracy among ANSWERED samples only
               (answered = final_answer non-empty)
    """
    reward_primary = REWARD_FUNCS.get(dataset, None)

    use_llm_judge = os.getenv("USE_LLM_JUDGE", "0") in ("1", "true", "True")
    judge = None
    if use_llm_judge:
        from transformers import AutoTokenizer
        JUDGE_TOKENIZER_PATH = os.getenv("JUDGE_TOKENIZER_PATH")
        judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_TOKENIZER_PATH, trust_remote_code=True, use_fast=True)
        judge_llm = _build_judge_adapter(judge_tokenizer=judge_tokenizer)
        judge = LLMJudge(judge_tokenizer, judge_llm)
    
    if reward_primary is None:
        reward_fn = None
        tool_tag = "reward:strict-em"
    else:
        if use_llm_judge and judge is not None:
            reward_fn = with_llm_judge(reward_primary, judge.score)
            tool_tag = "reward:rule-max+judge"
        else:
            reward_fn = (lambda c, a: float(reward_primary(c, a)))
            tool_tag = "reward:rule-max"

    tot = 0
    hit1 = 0
    hitk = 0
    avg_sum = 0.0
    answered_total = 0
    denom_total = 0

    k = int(k) if int(k) > 0 else 1

    for qid, g in gold_map.items():
        tot += 1
        p = pred_map.get(qid, {})

        # candidates: prefer list fields if present
        comp_list = p.get("completions", None)
        fa_list = p.get("final_answers", None)

        if isinstance(comp_list, list) and comp_list:
            completions = [str(x) for x in comp_list]
            if isinstance(fa_list, list) and len(fa_list) == len(completions):
                finals = [str(x) for x in fa_list]
            else:
                # fallback: use single final_answer (may be empty)
                finals = [str(p.get("final_answer", "")) for _ in completions]
        else:
            completions = [str(p.get("completion", ""))]
            finals = [str(p.get("final_answer", ""))]

        k_eff = min(k, len(completions))
        completions = completions[:k_eff]
        finals = finals[:k_eff]
        denom_total += k_eff

        # score each candidate
        correct_flags = []
        answered_flags = []

        for comp_i, fin_i in zip(completions, finals):
            answered_i = bool(_norm(fin_i))
            answered_flags.append(answered_i)

            if reward_fn is None:
                corr_i = int(_norm(fin_i) == _norm(g["answer"]))
            else:
                try:
                    corr_i = int(float(reward_fn(comp_i, g["answer"])) >= 1.0)
                except Exception:
                    corr_i = 0
            correct_flags.append(corr_i)

        # pass@1 = first sample correctness
        hit1 += int(correct_flags[0] == 1)

        # pass@k
        hitk += int(any(c == 1 for c in correct_flags))

        # avg@k (answered only, per-question)
        answered_cnt = sum(1 for a in answered_flags if a)
        correct_answered_cnt = sum(c for c, a in zip(correct_flags, answered_flags) if a)
        answered_total += answered_cnt

        avg_i = (correct_answered_cnt / answered_cnt) if answered_cnt > 0 else 0.0
        avg_sum += float(avg_i)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pass1 = (hit1 / tot) if tot else 0.0
    passk = (hitk / tot) if tot else 0.0
    avgk = (avg_sum / tot) if tot else 0.0
    answered_rate = (answered_total / denom_total) if denom_total else 0.0

    out = {
        "dataset": dataset,
        "timestamp": ts,
        "tool": tool_tag,
        "num": tot,
        "correct@1": hit1,
        "pass@1": pass1,
        "k": int(k),
        "correct@k": hitk,
        "pass@k": passk,
        "avg@k_answered": avgk,
        "answered": answered_total,
        "answered_rate": answered_rate,
    }
    with open(RESULTS_DIR / f"{dataset}.summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return RunSummary(dataset, ts, tot, hit1, pass1, tool_tag, int(k), hitk, passk, avgk, answered_total, answered_rate)


def main():
    ap = argparse.ArgumentParser(description="Score predictions against standardized gold JSONL.")
    ap.add_argument("--dataset", default="all",
                    choices=["all", "aime24", "aime25", "math", "gaokao2023", "olympiadbench"])
    ap.add_argument("--k", type=int, default=int(os.getenv("PASS_K", "1")),
                    help="Use first k completions per question to compute pass@k / avg@k.")
    args = ap.parse_args()

    datasets = list(DATA_DIR.keys()) if args.dataset == "all" else [args.dataset]
    rows: List[RunSummary] = []

    for ds in datasets:
        gold_path = Path(DATA_DIR[ds])
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold not found: {gold_path}")

        pred_paths = _collect_pred_paths(ds)
        gold_map = _load_gold(gold_path)
        pred_map = _load_preds(pred_paths)
        rows.append(_score_dataset(ds, gold_map, pred_map, k=int(args.k)))

    csv_path = Path("eval/results") / "summary.csv"
    need_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        wr = csv.writer(cf)
        if need_header:
            wr.writerow([
                "time", "dataset", "#_samples", "k",
                "pass@1",
                "pass@k", "avg@k"
            ])
        for r in rows:
            wr.writerow([
                r.time, r.dataset, r.num, r.k, 
                r.pass_at_1,
                r.pass_at_k, r.avg_at_k
            ])


if __name__ == "__main__":
    main()
