# -*- coding: utf-8 -*-
"""
Dataset-specific reward functions for local pass@1 evaluation.

User requirements:
  (1) Keep the user's existing rule-based rewards
  (2) Absorb additional "math reward" logic from common RL/eval pipelines (e.g. VERL / lm-eval-harness style)
      WITHOUT requiring extra pip installs.
  (3) The final reward is max(rule_rewards). If still 0, optionally fallback to self-judge.

Design:
  - RULE_REWARD_FUNCS[dataset] = list of functions: (completion:str, gt:str) -> float(0/1)
  - REWARD_FUNCS[dataset] = max over RULE_REWARD_FUNCS[dataset]
  - with_llm_judge(...) wraps a (completion,gt)->score function into (q,c,gt)->score with judge fallback.

No mandatory third-party deps here (sympy is optional; we won't require it).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple

from eval.adapters import GenParams  # engine-agnostic sampling params for judge


# ----------------- Basic utils -----------------

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _strip_dollars(s: str) -> str:
    return (s or "").strip().strip("$")

ANS_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED   = re.compile(r"\\boxed\{([^}]*)\}")
FINAL   = re.compile(r"(?i)(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*([^\n]+)")


# ----------------- (A) Minerva-style normalization (from your existing code) -----------------

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square","ways","integers","dollars","mph","inches","hours","km","units",
    "\\ldots","sue","points","feet","minutes","digits","cents","degrees","cm",
    "gm","pounds","meters","meals","edges","students","childrentickets",
    "multiples","\\text{s}","\\text{.}","\\text{\\ns}","\\text{}^2","\\text{}^3",
    "\\text{\\n}","\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;",
    r",\!", "{,}", '"', "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Minerva-like normalization: strip text wrappers, reduce to inline math,
    normalize TeX shorthands, and canonicalize digits.
    """
    final_answer = (final_answer or "").split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # keep only the last inline math $...$
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # TeX shorthands
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # digits with commas -> digits
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# ----------------- (B) VERL / lm-eval-harness style "strip_string" grader (inlined) -----------------
#
# This family of functions is used across multiple open-source math eval pipelines:
#   - strip whitespace and common latex wrappers (\left \right)
#   - normalize \frac, \sqrt
#   - remove trailing units or irrelevant tokens
#
# We keep it purely regex/string based (no sympy).
#

def _fix_frac(s: str) -> str:
    # Convert "frac12" -> "frac{1}{2}" pattern (lightweight heuristic).
    return re.sub(r"(\\frac)([0-9])([0-9])", r"\\frac{\2}{\3}", s)

def _fix_sqrt(s: str) -> str:
    # Convert "\sqrt2" -> "\sqrt{2}" pattern (lightweight heuristic).
    return re.sub(r"(\\sqrt)([0-9])", r"\\sqrt{\2}", s)

def strip_string(s: str) -> str:
    """
    Aggressive canonicalization used in many math eval scripts.
    Intended for short final answers (not full CoT).
    """
    s = _nfkc(s)
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\$", "")
    s = s.replace(" ", "")
    s = s.replace("\u00a0", "")  # non-breaking space

    # Remove common text wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Normalize sqrt/frac shorthands
    s = _fix_frac(s)
    s = _fix_sqrt(s)

    # Remove redundant outer dollars
    s = s.strip("$")

    # Normalize trailing .0 for pure floats
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]

    # Handle "0.5" vs ".5"
    if re.fullmatch(r"-?\.\d+", s):
        s = s.replace(".", "0.", 1)

    return s

def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last '\\boxed{...}' from a LaTeX snippet, or None."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not (s.startswith(left) and s.endswith("}")):
        return s
    return s[len(left):-1]

def extract_from_completion(completion: str) -> str:
    """
    Priority: <answer>...</answer> > \\boxed{...} > 'Final Answer: ...' > last numeric-like token.
    """
    if not completion:
        return ""
    m = ANS_TAG.search(completion)
    if m:
        return _strip_dollars(_nfkc(m.group(1)))
    m = BOXED.search(completion)
    if m:
        return _strip_dollars(_nfkc(m.group(1)))
    m = FINAL.search(completion)
    if m:
        seg = m.group(1).strip()
        seg = re.split(r"[\n。]", seg)[0]
        return _strip_dollars(_nfkc(seg))
    nums = re.findall(r"[-+]?\d+(?:/\d+)?|\d*\.\d+|\\sqrt\{[^}]+\}", completion)
    if nums:
        return _strip_dollars(_nfkc(nums[-1]))
    return ""


# ----------------- AIME rewards -----------------

def extract_aime_int(completion: str) -> Optional[int]:
    """Extract an AIME-style integer (0..999) from a completion."""
    cand = extract_from_completion(completion)
    m = re.search(r"(\d{1,3})\b", cand)
    if not m:
        ints = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", completion or "")
        cand = ints[-1] if ints else None
    else:
        cand = m.group(1)
    if cand is None:
        return None
    try:
        val = int(cand)
        if 0 <= val <= 999:
            return val
    except Exception:
        return None
    return None

def reward_aime_strict(completion: str, gt: str) -> float:
    """AIME: strict int equality (0..999)."""
    pred_int = extract_aime_int(completion)
    gt_clean = _nfkc(gt).strip()

    gt_int = None
    m = re.fullmatch(r"\s*0*(\d{1,3})\s*$", gt_clean)
    if m:
        gt_int = int(m.group(1))
    else:
        gt_box = last_boxed_only_string(gt_clean)
        if gt_box is not None:
            digits = re.sub(r"\D", "", remove_boxed(gt_box))
            if digits:
                gt_int = int(digits)
        else:
            g = re.findall(r"(\d{1,3})", gt_clean)
            if g:
                gt_int = int(g[-1])

    if pred_int is None or gt_int is None:
        return 0.0
    return 1.0 if pred_int == gt_int else 0.0

def reward_aime_strip_match(completion: str, gt: str) -> float:
    """
    AIME: alternative rule grader (common in some pipelines):
      - compare extracted answers after aggressive strip_string normalization.
    """
    pred = strip_string(extract_from_completion(completion))
    gold = strip_string(extract_from_completion(gt))
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0


# ----------------- MATH rewards -----------------

def reward_math_minerva(completion: str, gt: str) -> float:
    """MATH(-500): Minerva-style normalized string equality."""
    m = ANS_TAG.search(completion or "")
    pred_raw = m.group(1) if m else extract_from_completion(completion)
    pred = normalize_final_answer(pred_raw)

    gt_box = last_boxed_only_string(gt or "")
    gt_raw = remove_boxed(gt_box) if gt_box is not None else (gt or "")
    gt_norm = normalize_final_answer(gt_raw)

    return 1.0 if pred == gt_norm and pred != "" else 0.0

def reward_math_strip_string(completion: str, gt: str) -> float:
    """
    Another very common math grader:
      - extract answer, then strip_string compare.
    """
    pred = strip_string(extract_from_completion(completion))
    gt_box = last_boxed_only_string(gt or "")
    gt_raw = remove_boxed(gt_box) if gt_box is not None else (gt or "")
    gold = strip_string(gt_raw)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0

def reward_math_numeric_if_possible(completion: str, gt: str) -> float:
    """
    Numeric fallback:
      - If both pred & gt look like plain numbers, compare as rational/float tolerant.
    This is purely rule-based and has no external deps.
    """
    pred = extract_from_completion(completion)
    gt_box = last_boxed_only_string(gt or "")
    gt_raw = remove_boxed(gt_box) if gt_box is not None else (gt or "")
    gold = extract_from_completion(gt_raw) or gt_raw

    pred_s = strip_string(pred)
    gold_s = strip_string(gold)

    # Integers
    if pred_s.isdigit() and gold_s.isdigit():
        return 1.0 if int(pred_s) == int(gold_s) else 0.0

    # Simple decimals
    try:
        fp = float(pred_s)
        fg = float(gold_s)
        return 1.0 if abs(fp - fg) <= 1e-9 else 0.0
    except Exception:
        return 0.0


# ----------------- Gaokao rewards -----------------

def _extract_choice_letter(s: str) -> Optional[str]:
    """Extract a single MCQ letter A-E."""
    s = _nfkc(s).upper()
    m = ANS_TAG.search(s)
    field = m.group(1) if m else s
    m2 = (re.search(r"\b([A-E])\b", field)
          or re.search(r"[(（\[]\s*([A-E])\s*[)）\]]", field)
          or re.search(r"[：:]\s*([A-E])\b", field))
    return m2.group(1) if m2 else None

def reward_gaokao_choice_or_math(completion: str, gt: str) -> float:
    """
    Gaokao-2023:
      - If GT is a single MCQ letter A-E, compare letters.
      - Otherwise compare using math graders (max).
    """
    gt_clean = _nfkc(gt).strip().upper()
    if re.fullmatch(r"[A-E]", gt_clean):
        pred = _extract_choice_letter(completion)
        return 1.0 if pred == gt_clean else 0.0

    # If not MCQ, treat like math
    return max(
        reward_math_minerva(completion, gt),
        reward_math_strip_string(completion, gt),
        reward_math_numeric_if_possible(completion, gt),
    )


# ----------------- OlympiadBench rewards -----------------

def reward_olympiad_rule_max(completion: str, gt: str) -> float:
    """
    OlympiadBench:
      - Since no extra deps requested, we avoid mandatory sympy.
      - Use multiple string-based graders and take max.
    """
    return max(
        reward_math_minerva(completion, gt),
        reward_math_strip_string(completion, gt),
        reward_math_numeric_if_possible(completion, gt),
    )


# ----------------- Compose "max over rule rewards" -----------------

def _max_rule_reward(fns: List[Callable[[str, str], float]]) -> Callable[[str, str], float]:
    """
    Return a function that computes max(fn_i(completion, gt)).
    """
    def _r(completion: str, gt: str) -> float:
        best = 0.0
        for fn in fns:
            try:
                best = max(best, float(fn(completion, gt)))
            except Exception:
                continue
        return 1.0 if best >= 1.0 else 0.0
    return _r


RULE_REWARD_FUNCS: Dict[str, List[Callable[[str, str], float]]] = {
    "aime24": [reward_aime_strict, reward_aime_strip_match],
    "aime25": [reward_aime_strict, reward_aime_strip_match],
    "math":   [reward_math_minerva, reward_math_strip_string, reward_math_numeric_if_possible],
    "gaokao2023": [reward_gaokao_choice_or_math],
    "olympiadbench": [reward_olympiad_rule_max],
}

REWARD_FUNCS: Dict[str, Callable[[str, str], float]] = {
    k: _max_rule_reward(v) for k, v in RULE_REWARD_FUNCS.items()
}


# ----------------- LLM-as-judge helpers -----------------

def _extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANS_TAG.search(text)
    if m:
        return m.group(1).strip()
    m = BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = FINAL.search(text)
    if m:
        return re.split(r"[\n。]", m.group(1).strip())[0].strip()
    return None

def _normalize_basic(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip().strip("$")
    s = re.sub(r"\s+", " ", s)
    return s

def _make_judge_params() -> GenParams:
    """
    Deterministic and short output for judge:
      - temperature 0
      - short max_tokens to avoid verbose outputs
    """
    return GenParams(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0, max_tokens=16)

def _parse_binary_score(text: str) -> float:
    m = ANS_TAG.findall(text or "")
    if not m:
        return 0.0
    tail = m[-1].strip()
    try:
        v = float(tail)
        return 1.0 if abs(v - 1.0) < 1e-6 else 0.0
    except Exception:
        return 0.0


class LLMJudge:
    """
    Minimal binary judge:
      - First tries deterministic normalized equality on extracted final answer.
      - If that fails, asks an LLM:
            output ONLY <answer>1</answer> or <answer>0</answer>
    """
    def __init__(self, tokenizer, llm):
        self.tokenizer = tokenizer   # HF tokenizer with apply_chat_template
        self.llm = llm               # engine adapter with .generate(prompts, sampling_params)

    def score(self, model_output: str, ground_truth: str) -> float:
        extracted = _extract_final_answer(model_output)
        if extracted is None:
            return 0.0

        pred_norm = _normalize_basic(extracted)
        gold_norm = _normalize_basic(ground_truth)
        if pred_norm == gold_norm:
            return 1.0
        if pred_norm.isdigit() and gold_norm.isdigit() and int(pred_norm) == int(gold_norm):
            return 1.0

        prompt = f"""You are a grader.

Task: Decide if the model's answer matches the ground truth.
Rules:
- Output ONLY "<answer>1</answer>" if they are the same (equal numeric value or same exact text).
- Otherwise output ONLY "<answer>0</answer>".
- Do not include any explanation.

# Model Answer
{extracted}

# Ground Truth
{ground_truth}
"""
        msgs = [{"role": "user", "content": prompt}]

        # Tokenizers vary: some accept enable_thinking, some don't. Use a safe fallback.
        try:
            chat = self.tokenizer.apply_chat_template(
                conversation=msgs, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            chat = self.tokenizer.apply_chat_template(
                conversation=msgs, tokenize=False
            )

        out = self.llm.generate(
            prompts=[chat],
            sampling_params=_make_judge_params(),
            use_tqdm=False
        )
        toks = out[0].outputs[0].token_ids
        text = self.tokenizer.decode(toks, skip_special_tokens=True)
        return _parse_binary_score(text)


def with_llm_judge(
    primary_reward: Callable[[str, str], float],
    judge_callable: Callable[[str, str, str], float],
) -> Callable[[str, str, str], float]:
    """
    Compose rule reward with judge fallback.

    Returned signature:
      fn(question, completion, gold) -> float

    Policy:
      - If primary rule reward == 1.0 => return 1.0
      - Else if no <answer> tag => return 0.0 (avoid noisy judge calls)
      - Else ask judge for 0/1
    """
    def _wrapped(completion: str, gold: str) -> float:
        try:
            s = float(primary_reward(completion, gold))
        except Exception:
            s = 0.0
        if s >= 1.0:
            return 1.0
        # Escalate only when model attempted a final answer format.
        if not ANS_TAG.search(completion or ""):
            return 0.0

        try:
            return float(judge_callable(completion, gold))
        except Exception:
            return 0.0

    return _wrapped
