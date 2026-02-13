# -*- coding: utf-8 -*-
"""
Roll out predictions (pass@1) from a standardized JSONL dataset
({"question": "...", "answer": "..."} per line).

Engines supported (ONLY):
  - jet  : JetEngine local inference (`from jetengine import LLM, SamplingParams`) for SDAR.
  - vllm : vLLM HTTP server via trainer.vllm_client.VLLMClient.

Modes:
  1) react  : Balanced ReAct (multi-step + tool call), breadth is hard-capped to 1.
  2) value  : MCTS via trainer.agent.MCTSAgent (optional; keeps prior wiring).
  3) single : Single-turn (no tools).

IMPORTANT FORMAT RULE:
  - We write `completion` to JSONL as a STRING.
  - We always decode completion from token ids:
        completion = tokenizer.decode(token_ids, skip_special_tokens=True)
  - We NEVER serialize engine objects/dicts into `completion`.

Sanity check:
  - For the first example, we assert `completion` is str and does not start with "{".
    That catches the "dict-string-tokenized" bug early.
"""

from __future__ import annotations

from typing import Any, Dict, List
from typing import ClassVar, Optional, Tuple, Union
import abc
import argparse
import copy
import gc
import hashlib
import json
import os
import re
import traceback
import unicodedata

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from trainer.agent import dump_with_rich, parse_tool_calls, MCTSAgent
from trainer.agent import pick_best_leaf
from trainer.vllm_client import VLLMClient
from trainer.mtpo_trainer import LinearValueHead

import pandas as pd

from eval.adapters import GenParams, build_engine_adapter


# -----------------------------
# Minimal ReAct tools
# -----------------------------
from tools.remote_python_code_interpreter import execute_python_code, description


# -----------------------------
# Answer extraction helpers
# -----------------------------
_ANS_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:^|\n)\s*(?:Final\s*Answer|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)


def _qid(question: str) -> str:
    qn = unicodedata.normalize("NFKC", question).encode("utf-8")
    return hashlib.sha1(qn).hexdigest()[:10]

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return s.strip().strip("$")

def _extract_answer(text: str, aime_hint: bool = False) -> str:
    """
    Extract a final answer string heuristically.
    This is used for `final_answer` field in output JSONL.
    """
    if not text:
        return ""
    m = _ANS_TAG.search(text)
    if m:
        return _norm(m.group(1))
    m = _BOXED.search(text)
    if m:
        return _norm(m.group(1))
    m = _FINAL.search(text)
    if m:
        cand = m.group(1).strip()
        cand = re.split(r"[\n。]", cand)[0]
        return _norm(cand)
    if aime_hint:
        ints = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text)
        if ints:
            return (ints[-1].lstrip("0") or "0")
    m_all = re.findall(r"[-+]?\d+(?:/\d+)?|\d*\.\d+|\\sqrt\{[^}]+\}", text)
    if m_all:
        return _norm(m_all[-1])
    return ""

# -----------------------------
# Rewards registry for MCTS
# -----------------------------
def _build_reward_fns(dataset_name: str):
    """
    Returns a list of reward functions.
    Each function should be: fn(completion:str, ground_truth:str) -> float
    """
    # try:
    #     from eval.rewards import REWARD_FUNCS
    #     fn = REWARD_FUNCS.get(dataset_name, None)
    #     if fn is None:
    #         return []
    #     return [fn]
    # except Exception:
    #     return []
    from eval.rewards import REWARD_FUNCS, with_llm_judge, LLMJudge
    from eval_math import _build_judge_adapter
    use_llm_judge = os.getenv("USE_LLM_JUDGE", "0") in ("1", "true", "True")
    judge = None
    if use_llm_judge:
        JUDGE_TOKENIZER_PATH = os.getenv("JUDGE_TOKENIZER_PATH")
        judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_TOKENIZER_PATH, trust_remote_code=True, use_fast=True)
        judge_llm = _build_judge_adapter(judge_tokenizer=judge_tokenizer)
        judge = LLMJudge(judge_tokenizer, judge_llm)
    
    reward_primary = REWARD_FUNCS.get(dataset_name, None)
    if use_llm_judge and judge is not None:
        reward_fns = [with_llm_judge(reward_primary, judge.score)]
    else:
        reward_fns = [lambda c, a: float(reward_primary(c, a))]
        
    return reward_fns
            
# -----------------------------
# Agents
# -----------------------------
class ReActAgent(abc.ABC):
    SYSTEM_TEMPLATE: ClassVar[str]
    USER_TEMPLATE: ClassVar[str]
    TOOLS: ClassVar[Dict[str, callable]]
    TOOLS_DESCRIPTION: ClassVar[Optional[List[Dict[str, Any]]]]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        depth: int,
        breadth: int,
        output_dir: str,
        llm: Any,
        sampling_params: Any,
        max_model_len: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: HF tokenizer used both for chat templates and decoding.
            depth:     Max ReAct depth.
            breadth:   Beam width (usually 1 for ReAct).
            output_dir:Where to dump temporary rollouts.
            llm:       Engine adapter (jet/vllm) with .generate(prompts, sampling_params).
            sampling_params: Engine-specific sampling params (GenParams or vLLM SamplingParams).
            max_model_len: Hard cap on total prompt length in tokens. If the rendered
                           chat prompt exceeds this length, we early-stop and DO NOT
                           call the underlying engine (to avoid OOM or engine errors).
        """
        self.tokenizer = tokenizer
        self.depth = depth
        self.breadth = breadth
        self.output_dir = output_dir
        self.llm = llm
        self.sampling_params = sampling_params
        self.max_model_len = max_model_len

    def _generate_batch(self, messages: List[dict], ground_truth: str, n_variants: int) -> List[dict]:
        """
        Render messages -> prompt; optionally early-stop on max_model_len; otherwise
        call engine.generate(); decode completion tokens.

        EARLY STOP LOGIC:
        ------------------
        1) We render the full chat prompt string.
        2) We tokenize it with the HF tokenizer to get prompt_ids.
        3) If len(prompt_ids) > max_model_len (when max_model_len is not None),
           we DO NOT call the engine at all and instead return a stub completion:
                "<think>Context length exceeded max_model_len; stopping generation.</think><answer></answer>"
           This avoids sending over-long prompts to Jet/vLLM and keeps RAM/VRAM safe.
        """
        # 1) Render chat prompt
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=self.TOOLS_DESCRIPTION,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) Tokenize prompt to check length. Keep on CPU; we just need the length.
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"][0]  # shape: [seq_len]

        # 3) Early stop if context too long
        if self.max_model_len is not None and prompt_ids.numel() > self.max_model_len:
            # Stub completion that contains <answer> so ReAct recursion terminates.
            stub_completion = (
                "<think>Context length exceeded max_model_len; stopping generation."
                "</think><answer></answer>"
            )

            results: List[dict] = []
            for _ in range(n_variants):
                result = {
                    "prompt": prompt,
                    "completion": stub_completion,
                    "prompt_ids": prompt_ids.clone(),                 # store as torch tensor
                    "completion_ids": torch.tensor([], dtype=torch.long),
                    "ground_truth": ground_truth,
                    "reward": None,
                }
                dump_with_rich(result, os.path.join(self.output_dir, "tmp.txt"))
                results.append(result)
            return results

        # 4) Normal path: call engine.generate and decode completion tokens.
        outs = self.llm.generate(
            prompts=[prompt] * n_variants,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        results: List[dict] = []
        for out in outs:
            # ---- token_ids must be completion-only List[int] ----
            token_ids = out.outputs[0].token_ids
            completion = self.tokenizer.decode(token_ids, skip_special_tokens=False)  # preserve `<think>` token
            completion = completion.replace("<|im_end|>", "")
            result = {
                "prompt": prompt,
                "completion": completion,
                "prompt_ids": prompt_ids.clone(),
                "completion_ids": torch.tensor(token_ids, dtype=torch.long),
                "ground_truth": ground_truth,
                "reward": None,
            }
            dump_with_rich(result, os.path.join(self.output_dir, "tmp.txt"))
            results.append(result)
        return results

    def read_support_material(self, table_paths: Optional[List[str]]):
        if table_paths:
            support_material = dict()
            for i in range(len(table_paths)):
                try:
                    support_material[f"df{i}"] = pd.read_csv(table_paths[i])
                except Exception:
                    with open(table_paths[i]) as f:
                        support_material[f"tb{i}"] = f.read()

            support_material_str = "\n".join(
                (
                    f"Var: {k}; Type: {type(v)}\n{v}\n{v.dtypes}"
                    if isinstance(v, pd.DataFrame)
                    else f"Var: {k}; Type: {type(v)}\n{v}"
                )
                for k, v in support_material.items()
            )
        else:
            support_material, support_material_str = dict(), str()
        return support_material, support_material_str

    def react_recursive(
        self,
        question: str,
        support_material_path: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        assistant_and_tool_msg: Optional[List[dict]] = None,
        current_chain: Optional[List[dict]] = None,
        current_depth: int = 1,
        previous_variables: dict = dict(),
    ) -> List[List[dict]]:
        """
        Simple depth-limited ReAct. In --mode react, breadth is forced to 1.

        NOTE:
        -----
        Early-stop on max_model_len is implemented in _generate_batch().
        If the context grows too large at any depth, _generate_batch() will
        return a stub completion with <answer>, and this recursion will treat
        it as a terminal leaf (no further calls to the engine).
        """
        support_material, support_material_str = self.read_support_material(support_material_path)

        assistant_and_tool_msg = copy.deepcopy(assistant_and_tool_msg) if assistant_and_tool_msg else []
        current_chain = current_chain if current_chain else []

        support_material_str = f"# Given this:\n{support_material_str}" if support_material_str else ""
        system_prompt = self.SYSTEM_TEMPLATE
        user_prompt = self.USER_TEMPLATE.format(
            support_material_str=support_material_str,
            question=question,
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        responses = self._generate_batch(
            messages=msgs + assistant_and_tool_msg,
            ground_truth=ground_truth or "",
            n_variants=self.breadth,
        )

        all_chains: List[List[dict]] = []

        for resp in responses:
            resp["current_depth"] = current_depth
            local_msgs = copy.deepcopy(assistant_and_tool_msg)
            local_chain = current_chain.copy()
            resp["results"] = []

            try:
                # Early stop if <answer> is present or the model repeats.
                if "<answer>" in resp["completion"] or resp["completion"] in [m.get("content", "") for m in assistant_and_tool_msg]:
                    local_chain.append(resp)
                    all_chains.append(local_chain)
                    continue

                try:
                    assistant_msg = parse_tool_calls(resp["completion"])
                except Exception as e:
                    assistant_msg = {"role": "assistant", "content": resp["completion"]}
                    resp["results"].append({"parse_error": str(e)})
                    local_msgs.append(
                        {"role": "tool", "name": "none", "content": f"Parse error: {type(e).__name__}: {e}"}
                    )

                local_msgs.append(assistant_msg)
                tool_calls = assistant_msg.get("tool_calls", [])

                if tool_calls:
                    for call in tool_calls:
                        tool_name = call["function"]["name"]
                        tool_args = call["function"]["arguments"] or {}

                        if tool_name not in self.TOOLS:
                            raise ValueError(f"Unknown tool: {tool_name}")

                        context = {**previous_variables, **support_material}
                        try:
                            output_str, new_context = self.TOOLS[tool_name](**tool_args, context=context)
                        except Exception:
                            output_str, new_context = (
                                f"Tool execution error:\n{traceback.format_exc()}",
                                context,
                            )

                        local_msgs.append({"role": "tool", "name": tool_name, "content": output_str})
                        previous_variables = {k: v for k, v in new_context.items() if k not in support_material}

                else:
                    local_msgs.append({"role": "user", "content": "Please continue."})

            except Exception:
                resp["results"].append({"error": traceback.format_exc()})
                local_msgs.append({"role": "tool", "name": "none", "content": traceback.format_exc()})

            local_chain.append(resp)

            if current_depth + 1 <= self.depth:
                all_chains.extend(
                    self.react_recursive(
                        question=question,
                        support_material_path=support_material_path,
                        ground_truth=ground_truth,
                        assistant_and_tool_msg=local_msgs,
                        current_chain=local_chain,
                        current_depth=current_depth + 1,
                        previous_variables=previous_variables,
                    )
                )
            else:
                all_chains.append(local_chain)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_chains


class CoderAgent(ReActAgent):
    TOOLS = {"execute_python_code": execute_python_code}
    TOOLS_DESCRIPTION = description
    SYSTEM_TEMPLATE = """\
SOLVE THE PROBLEM STEP-BY-STEP. PRESENT THE ANSWER TO EXIT THE LOOP.

# Guidelines
→ Each assistant response must contain exactly one "<think>...</think>" block.
  · If the final answer is ready, use "<answer>...</answer>" block to terminate the loop.
  · No content other than whitespace may appear outside these tags.
→ Begin every response with "STEP-(\\d+):\\n<think>...", 1 step per response."""
    USER_TEMPLATE = """\
{support_material_str}
# Please answer:
{question}
👆
format the FINAL answer as `<answer>...</answer>`"""


class PoorAgent(ReActAgent):
    """
    Single-turn non-agent prompt (no tools, no STEP prefix).
    """
    TOOLS = {}
    TOOLS_DESCRIPTION = None
    SYSTEM_TEMPLATE = """\
You are NOT an agent. Answer in ONE message.
Rules:
  - Produce exactly one "<think>...</think>" block.
  - Then produce one "<answer>...</answer>" block with the final answer only.
  - Do not use tools. Do not include any "STEP-" prefixes. Stop after the final answer."""
    USER_TEMPLATE = """\
{support_material_str}
# Please answer (single-turn):
{question}
👆
format the FINAL answer as `<answer>...</answer>`"""

    def react_recursive(self, question: str, **kwargs):
        msgs = [
            {"role": "system", "content": self.SYSTEM_TEMPLATE},
            {"role": "user", "content": self.USER_TEMPLATE.format(support_material_str="", question=question)},
        ]
        resp = self._generate_batch(msgs, ground_truth=kwargs.get("ground_truth", "") or "", n_variants=1)[0]
        return [[resp]]


# class HFValueFunction(nn.Module):
#     """
#     Value function wrapper built on top of the new LinearValueHead.

#     This wrapper is designed to be used by MCTSAgent (or any search algorithm)
#     that expects:

#         value_fn(input_ids, attention_mask, **kwargs) -> (latent_vec, value)

#     where:
#       - latent_vec: a *CPU float32* tensor used for banking/pruning/radius map
#                    (we return LinearValueHead's y_state)
#       - value:      a *CPU float32* tensor in [0, 1] used as state value
#                    (we return LinearValueHead's v_pred)

#     Key design choices:
#     - We always load a base LM (HF CausalLM).
#     - We always *wrap* it with your LinearValueHead to compute:
#         last_hidden -> masked mean -> h0 -> y_state (Exp0 in Poincaré ball)
#                                       -> v_pred  (sigmoid(W h0))
#     - We support multiple checkpoint formats for loading the value head:
#         (1) HEAD-ONLY checkpoint: contains only value_head.{weight,bias} or {weight,bias}
#         (2) FULL checkpoint: contains LinearValueHead state dict (may also include base_lm.*)
#         (3) Nested checkpoints: {"state_dict": ...} or {"model": ...}
#         (4) DDP prefix: "module." prefix on keys

#     Notes:
#     - We DO NOT rely on external packages.
#     - We do not change your pool_mask logic; it is implemented inside LinearValueHead.forward.
#     """

#     def __init__(
#         self,
#         base_lm_path: str,
#         value_head_path: Optional[str] = None,
#         *,
#         device: str = "cuda:0",
#         dtype: str = "auto",
#         # ----- LinearValueHead hyper-params -----
#         curvature: float = 1.0,
#         eps: float = 1e-6,
#         eps_ball: float = 1e-4,
#         no_head_scale: float = 0.0,          # 0 -> sqrt(H)
#         value_activation: str = "sigmoid",   # "sigmoid" or "none"
#     ):
#         super().__init__()

#         # -------------------------
#         # Resolve device
#         # -------------------------
#         if device == "auto":
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = torch.device(device)

#         # -------------------------
#         # Resolve dtype
#         # -------------------------
#         if dtype == "auto":
#             torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
#         else:
#             d = str(dtype).lower()
#             dmap = {
#                 "fp32": torch.float32, "float32": torch.float32,
#                 "fp16": torch.float16, "float16": torch.float16,
#                 "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
#             }
#             torch_dtype = dmap.get(d, torch.float32)

#         # -------------------------
#         # Load base LM
#         # -------------------------
#         self.base_lm = AutoModelForCausalLM.from_pretrained(
#             base_lm_path,
#             trust_remote_code=True,
#             torch_dtype=torch_dtype,
#             device_map=None,
#         ).to(self.device)

#         self.base_lm.eval()
#         for p in self.base_lm.parameters():
#             p.requires_grad_(False)

#         # -------------------------
#         # Build LinearValueHead wrapper
#         # -------------------------
#         # Update this import path to wherever you placed LinearValueHead.
#         # e.g. from trainer.linear_value_head import LinearValueHead

#         self.model = LinearValueHead(
#             base_lm=self.base_lm,
#             curvature=curvature,
#             eps=eps,
#             eps_ball=eps_ball,
#             no_head_scale=no_head_scale,
#             value_activation=value_activation,
#         )
#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad_(False)

#         # -------------------------
#         # Optionally load value head weights / wrapper weights
#         # -------------------------
#         self.value_head_path = value_head_path
#         if value_head_path:
#             self._load_value_checkpoint(value_head_path)

#     # ------------------------------------------------------------------
#     # Robust checkpoint loading
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
#         """
#         Normalize common checkpoint formats into a flat state_dict.
#         Supported:
#           - plain dict of tensors
#           - {"state_dict": {...}}
#           - {"model": {...}}
#         """
#         if not isinstance(obj, dict):
#             return {}

#         if "state_dict" in obj and isinstance(obj["state_dict"], dict):
#             obj = obj["state_dict"]
#         elif "model" in obj and isinstance(obj["model"], dict):
#             obj = obj["model"]

#         # Keep only tensor entries.
#         sd: Dict[str, torch.Tensor] = {}
#         for k, v in obj.items():
#             if isinstance(v, torch.Tensor):
#                 sd[str(k)] = v
#         return sd

#     @staticmethod
#     def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
#         if not prefix:
#             return sd
#         out: Dict[str, torch.Tensor] = {}
#         for k, v in sd.items():
#             if k.startswith(prefix):
#                 out[k[len(prefix):]] = v
#             else:
#                 out[k] = v
#         return out

#     def _load_value_checkpoint(self, path: str) -> None:
#         """
#         Load value head / wrapper checkpoint into LinearValueHead.

#         We try multiple strategies:
#           1) If checkpoint looks like head-only: {weight,bias} -> load into model.value_head
#           2) Else try to load as full state_dict into LinearValueHead with strict=False
#         """
#         try:
#             ckpt = torch.load(path, map_location="cpu")
#         except Exception as e:
#             print(f"[HFValueFunction] WARN: torch.load({path}) failed: {type(e).__name__}: {e}")
#             return

#         sd = self._unwrap_state_dict(ckpt)
#         if not sd:
#             print(f"[HFValueFunction] WARN: empty/invalid checkpoint: {path}")
#             return

#         # Remove DDP prefix if exists.
#         sd = self._strip_prefix(sd, "module.")

#         # ---- Case 1: head-only checkpoints (common & preferred) ----
#         # Supported formats:
#         #   A) {"weight": ..., "bias": ...}
#         #   B) {"value_head.weight": ..., "value_head.bias": ...}
#         #   C) {"model.value_head.weight": ...} etc. (we'll rely on strict=False below)
#         head_only_a = ("weight" in sd and "bias" in sd)
#         head_only_b = ("value_head.weight" in sd and "value_head.bias" in sd)

#         if head_only_a:
#             try:
#                 self.model.value_head.weight.data.copy_(sd["weight"].to(self.model.value_head.weight.device))
#                 self.model.value_head.bias.data.copy_(sd["bias"].to(self.model.value_head.bias.device))
#                 print(f"[HFValueFunction] INFO: loaded head-only checkpoint (weight/bias) from {path}")
#                 return
#             except Exception as e:
#                 print(f"[HFValueFunction] WARN: head-only(weight/bias) load failed, fallback strict=False: {e}")

#         if head_only_b:
#             try:
#                 self.model.value_head.weight.data.copy_(sd["value_head.weight"].to(self.model.value_head.weight.device))
#                 self.model.value_head.bias.data.copy_(sd["value_head.bias"].to(self.model.value_head.bias.device))
#                 print(f"[HFValueFunction] INFO: loaded head-only checkpoint (value_head.*) from {path}")
#                 return
#             except Exception as e:
#                 print(f"[HFValueFunction] WARN: head-only(value_head.*) load failed, fallback strict=False: {e}")

#         # ---- Case 2: full wrapper checkpoint or mixed keys ----
#         # We load with strict=False so it can contain:
#         #   - base_lm.* (ignored because already loaded, but can match)
#         #   - value_head.* (will be applied)
#         #   - any extra keys from older versions (ignored)
#         try:
#             missing, unexpected = self.model.load_state_dict(sd, strict=False)
#             print(
#                 f"[HFValueFunction] INFO: loaded checkpoint via strict=False from {path}. "
#                 f"missing={len(missing)}, unexpected={len(unexpected)}"
#             )
#         except Exception as e:
#             print(f"[HFValueFunction] WARN: strict=False load failed for {path}: {type(e).__name__}: {e}")

#     # ------------------------------------------------------------------
#     # Call interface (used by MCTSAgent)
#     # ------------------------------------------------------------------
#     @torch.inference_mode()
#     def __call__(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         *,
#         response_mask: Optional[torch.Tensor] = None,
#         prompt_mask: Optional[torch.Tensor] = None,
#         hidden_states: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Compute (y_state, v_pred) using LinearValueHead.forward(value_output=True).

#         Returns:
#           - y_state: CPU float32 tensor [B, H]
#           - v_pred:  CPU float32 tensor [B] (or [B,] after squeeze)
#         """
#         input_ids = input_ids.to(self.device)

#         # If attention_mask is not provided, build it from pad_token_id if possible.
#         if attention_mask is None:
#             pad_id = getattr(self.base_lm.config, "pad_token_id", None)
#             if pad_id is not None:
#                 attention_mask = (input_ids != pad_id).long()
#             else:
#                 attention_mask = torch.ones_like(input_ids, dtype=torch.long)
#         else:
#             attention_mask = attention_mask.to(self.device)

#         if response_mask is not None:
#             response_mask = response_mask.to(self.device)
#         if prompt_mask is not None:
#             prompt_mask = prompt_mask.to(self.device)
#         if hidden_states is not None:
#             hidden_states = hidden_states.to(self.device)

#         # LinearValueHead handles pool_mask logic internally:
#         #   pool_mask = ((response_mask if provided else attention_mask) OR prompt_mask) AND attention_mask
#         y_state, v_pred = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             value_output=True,
#             response_mask=response_mask,
#             prompt_mask=prompt_mask,
#             hidden_states=hidden_states,
#         )

#         # Ensure output types for downstream (bank/prune expects CPU float).
#         if isinstance(y_state, torch.Tensor):
#             y_state = y_state.detach().float().cpu()
#         else:
#             y_state = torch.tensor(y_state, dtype=torch.float32)

#         if isinstance(v_pred, torch.Tensor):
#             # v_pred is expected to already be float32 in your LinearValueHead,
#             # but we still force float32 on CPU for safety.
#             v_pred = v_pred.detach().float().cpu()
#         else:
#             v_pred = torch.tensor(v_pred, dtype=torch.float32)

#         # Shape safety: ensure v_pred is 1D [B]
#         if v_pred.dim() > 1:
#             v_pred = v_pred.squeeze(-1)

#         return y_state, v_pred

class HFValueFunction(nn.Module):
    """
    Value function wrapper for evaluation/search, built on top of the new LinearValueHead
    (supports root-centered euclidean translation BEFORE exp0, and return_h0).

    Expected interface (for MCTS/search):
        value_fn(input_ids, attention_mask, **kwargs) -> (y_state_cpu, v_pred_cpu)  or (y_state_cpu, v_pred_cpu, h0_raw_cpu)

    Where:
      - y_state_cpu: CPU float32 [B, H] : Poincaré point (root-centered if root_h0 provided)
      - v_pred_cpu : CPU float32 [B]    : scalar value (sigmoid(W h0_raw)) or raw logit if activation=none
      - h0_raw_cpu : CPU float32 [B, H] : pooled euclidean vector (for caching root_h0) when return_h0=True

    New features:
      - root_h0: (H,) or (1,H) or (B,H) on CPU/GPU/list/np ok. Used for (h0_raw - root_h0) centering before exp0.
      - cached_root_h0: you can cache root_h0 once per tree/question and reuse automatically.
    """

    def __init__(
        self,
        base_lm_path: str,
        value_head_path: Optional[str] = None,
        *,
        device: str = "cuda:0",
        dtype: str = "auto",
        # ----- LinearValueHead hyper-params -----
        curvature: float = 1.0,
        eps: float = 1e-6,
        eps_ball: float = 1e-4,
        no_head_scale: float = 0.0,          # 0 -> sqrt(H)
        value_activation: str = "sigmoid",   # "sigmoid" or "none"
    ):
        super().__init__()

        # -------------------------
        # Resolve device
        # -------------------------
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # -------------------------
        # Resolve dtype
        # -------------------------
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        else:
            d = str(dtype).lower()
            dmap = {
                "fp32": torch.float32, "float32": torch.float32,
                "fp16": torch.float16, "float16": torch.float16,
                "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            }
            torch_dtype = dmap.get(d, torch.float32)

        # -------------------------
        # Load base LM
        # -------------------------
        self.base_lm = AutoModelForCausalLM.from_pretrained(
            base_lm_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(self.device)

        self.base_lm.eval()
        for p in self.base_lm.parameters():
            p.requires_grad_(False)

        # -------------------------
        # Build LinearValueHead wrapper
        # -------------------------
        self.model = LinearValueHead(
            base_lm=self.base_lm,
            curvature=curvature,
            eps=eps,
            eps_ball=eps_ball,
            no_head_scale=no_head_scale,
            value_activation=value_activation,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # -------------------------
        # Cached root_h0 (CPU float32)
        # -------------------------
        self._cached_root_h0: Optional[torch.Tensor] = None  # (H,) or (1,H) or (B,H) on CPU float32

        # -------------------------
        # Optionally load value head weights / wrapper weights
        # -------------------------
        self.value_head_path = value_head_path
        if value_head_path:
            self._load_value_checkpoint(value_head_path)

    # ------------------------------------------------------------------
    # Root-h0 cache helpers
    # ------------------------------------------------------------------
    def set_root_h0(self, root_h0: Union[torch.Tensor, Any]) -> None:
        """
        Cache root_h0 on CPU float32.
        Accepts list/np/tensor; shape can be (H,), (1,H), or (B,H).
        """
        if root_h0 is None:
            self._cached_root_h0 = None
            return
        if torch.is_tensor(root_h0):
            t = root_h0.detach().to("cpu", dtype=torch.float32)
        else:
            t = torch.as_tensor(root_h0, dtype=torch.float32).detach().to("cpu")
        self._cached_root_h0 = t

    def clear_root_h0(self) -> None:
        self._cached_root_h0 = None

    def get_root_h0(self) -> Optional[torch.Tensor]:
        return self._cached_root_h0

    # ------------------------------------------------------------------
    # Robust checkpoint loading (same as your old version)
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(obj, dict):
            return {}

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]

        sd: Dict[str, torch.Tensor] = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                sd[str(k)] = v
        return sd

    @staticmethod
    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if not prefix:
            return sd
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    def _load_value_checkpoint(self, path: str) -> None:
        """
        Load value head / wrapper checkpoint into LinearValueHead.
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[HFValueFunction] WARN: torch.load({path}) failed: {type(e).__name__}: {e}")
            return

        sd = self._unwrap_state_dict(ckpt)
        if not sd:
            print(f"[HFValueFunction] WARN: empty/invalid checkpoint: {path}")
            return

        sd = self._strip_prefix(sd, "module.")

        head_only_a = ("weight" in sd and "bias" in sd)
        head_only_b = ("value_head.weight" in sd and "value_head.bias" in sd)

        if head_only_a:
            try:
                self.model.value_head.weight.data.copy_(sd["weight"].to(self.model.value_head.weight.device))
                self.model.value_head.bias.data.copy_(sd["bias"].to(self.model.value_head.bias.device))
                print(f"[HFValueFunction] INFO: loaded head-only checkpoint (weight/bias) from {path}")
                return
            except Exception as e:
                print(f"[HFValueFunction] WARN: head-only(weight/bias) load failed, fallback strict=False: {e}")

        if head_only_b:
            try:
                self.model.value_head.weight.data.copy_(sd["value_head.weight"].to(self.model.value_head.weight.device))
                self.model.value_head.bias.data.copy_(sd["value_head.bias"].to(self.model.value_head.bias.device))
                print(f"[HFValueFunction] INFO: loaded head-only checkpoint (value_head.*) from {path}")
                return
            except Exception as e:
                print(f"[HFValueFunction] WARN: head-only(value_head.*) load failed, fallback strict=False: {e}")

        try:
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(
                f"[HFValueFunction] INFO: loaded checkpoint via strict=False from {path}. "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as e:
            print(f"[HFValueFunction] WARN: strict=False load failed for {path}: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Forward interface (used by MCTSAgent / evaluator)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        response_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,

        # NEW:
        root_h0: Optional[Union[torch.Tensor, Any]] = None,
        use_cached_root: bool = True,
        return_h0: bool = False,
        cache_root_h0: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute (y_state, v_pred) using LinearValueHead.forward(value_output=True).

        If return_h0=True, also returns h0_raw (CPU float32) for caching as root_h0.

        root_h0 precedence:
          - if root_h0 is not None: use it
          - elif use_cached_root and self._cached_root_h0 is not None: use cached one
          - else: no centering (old behavior)
        """
        # ---- move input ----
        input_ids = input_ids.to(self.device)

        # ---- attention mask ----
        if attention_mask is None:
            pad_id = getattr(self.base_lm.config, "pad_token_id", None)
            if pad_id is not None:
                attention_mask = (input_ids != int(pad_id)).long()
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(self.device)

        if response_mask is not None:
            response_mask = response_mask.to(self.device)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(self.device)
        if hidden_states is not None:
            hidden_states = hidden_states.to(self.device)

        # ---- choose root_h0 ----
        rh = root_h0
        if rh is None and use_cached_root:
            rh = self._cached_root_h0

        rh_dev = None
        if rh is not None:
            if torch.is_tensor(rh):
                rh_t = rh.detach()
            else:
                rh_t = torch.as_tensor(rh)
            # LinearValueHead expects float32 root_h0
            rh_dev = rh_t.to(self.device, dtype=torch.float32)

        # ---- call LinearValueHead ----
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            value_output=True,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            hidden_states=hidden_states,
            root_h0=rh_dev,
            return_h0=bool(return_h0),
        )

        if return_h0:
            y_state, v_pred, h0_raw = out
        else:
            y_state, v_pred = out
            h0_raw = None

        # ---- force CPU float32 outputs ----
        y_state_cpu = y_state.detach().float().cpu() if torch.is_tensor(y_state) else torch.as_tensor(y_state, dtype=torch.float32)
        v_pred_cpu  = v_pred.detach().float().cpu()  if torch.is_tensor(v_pred)  else torch.as_tensor(v_pred, dtype=torch.float32)

        if v_pred_cpu.dim() > 1:
            v_pred_cpu = v_pred_cpu.squeeze(-1)

        if return_h0:
            h0_cpu = h0_raw.detach().float().cpu() if torch.is_tensor(h0_raw) else torch.as_tensor(h0_raw, dtype=torch.float32)

            # optional: cache this h0 as root_h0
            if cache_root_h0:
                # Supports B=1 or B>1; LinearValueHead can accept (H,) / (1,H) / (B,H)
                # Instead of forcibly squeezing, we directly cache (B, H) for greater versatility.
                self._cached_root_h0 = h0_cpu

            return y_state_cpu, v_pred_cpu, h0_cpu

        return y_state_cpu, v_pred_cpu
    
    
class MCoderAgent(MCTSAgent):
    TOOLS = {"execute_python_code": execute_python_code}
    TOOLS_DESCRIPTION = description
    SYSTEM_TEMPLATE = """\
SOLVE THE PROBLEM STEP-BY-STEP. PRESENT THE ANSWER TO EXIT THE LOOP.

# Guidelines
→ Each assistant response must contain exactly one "<think>...</think>" block.
  · If the final answer is ready, use "<answer>...</answer>" block to terminate the loop.
  · No content other than whitespace may appear outside these tags.
→ Finish your REACTION within {step_limit} step(s).
→ Begin every response with "STEP-(\\d+):\\n<think>...", 1 step per response."""
    USER_TEMPLATE = """\
{support_material_str}
# Please answer:
{question}
"""


def main():
    ap = argparse.ArgumentParser(description="Roll out predictions (pass@1) with ReAct or MCTS.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", required=True)

    ap.add_argument("--mode", choices=["react", "value", "single"], default="react")

    # Engine selection (ONLY jet/vllm)
    ap.add_argument("--tokenizer-path", required=True, help="HF tokenizer/model dir (also used as model dir for jet)")
    ap.add_argument("--engine", choices=["vllm", "jet"], default="jet")
    ap.add_argument("--base-url", default="", help="Only for ENGINE=vllm")

    # Jet TP
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("JET_TP_SIZE", "1")),
                    help="JetEngine tensor parallel size (single process multi-GPU).")

    # Decoding (common)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--min-p", type=float, default=0.0)

    # Traversal
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--breadth", type=int, default=6)

    # pass@k / avg@k: run k independent chains (breadth still = 1 inside each chain)
    ap.add_argument("--k", "--rollout-k", dest="rollout_k", type=int, default=1,
                    help="Number of independent rollouts per question for react/single.")

    # Value/MCTS wiring
    ap.add_argument("--value-base", default=None, help="HF base LM path for value function (usually equals tokenizer-path)") 
    ap.add_argument("--value-head", default=None, help="Path to value head .pt (state_dict or {'weight','bias'})") 
    ap.add_argument("--value-model", default=None, help="Optional full HF dir of LM+value_head; overrides base/head")
    ap.add_argument("--value-device", default="cuda:1")
    ap.add_argument("--value-dtype", default="auto")
    
    ap.add_argument("--max-model-len", type=int, default=10240)

    # MCTS hyper-params (kept to avoid breaking your pipeline)
    ap.add_argument("--mcts-c-puct", type=float, default=1.0)
    ap.add_argument("--mcts-v-prior", type=float, default=0.5)
    ap.add_argument("--mcts-value-trust", type=float, default=0.5)
    ap.add_argument("--mcts-num-sim", type=int, default=64)
    ap.add_argument("--mcts-prune-per", type=int, default=128)
    ap.add_argument("--mcts-max-expands", default=2)
    ap.add_argument("--mcts-num-pos-sim", type=int, default=4)
    ap.add_argument("--mcts-passk-threshold", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Tokenizer (for prompt rendering + decoding completion)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build engine-agnostic sampling params
    sampling = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        # SDAR diffusion params from env
        remasking_strategy=os.getenv("SDAR_REMASKING_STRATEGY", "low_confidence_dynamic"),
        block_length=int(os.getenv("SDAR_BLOCK_LENGTH", "4")),
        denoising_steps=int(os.getenv("SDAR_DENOISING_STEPS", "4")),
        dynamic_threshold=float(os.getenv("SDAR_DYNAMIC_THRESHOLD", "0.9")),
    )

    # Construct engine adapter
    engine = args.engine.lower()
    if engine == "vllm":
        if not args.base_url:
            raise ValueError("ENGINE=vllm requires --base-url (or BASE_URL env in eval.sh).")
        llm = build_engine_adapter(
            name="vllm",
            tokenizer=tokenizer,
            vllm_client=VLLMClient(base_url=args.base_url),
        )
    else:
        # JetEngine local inference (TP is SINGLE PROCESS multi-GPU)
        tp_size = int(args.tensor_parallel_size)
        enforce_eager = os.getenv("JET_ENFORCE_EAGER", "1") in ("1", "true", "True")
        # "mask token id" is SDAR-model specific; only pass if you know it.
        mask_token_id = os.getenv("SDAR_MASK_TOKEN_ID", "").strip()
        mask_token_id = int(mask_token_id) if mask_token_id.isdigit() else None
        block_length = int(os.getenv("SDAR_BLOCK_LENGTH", "4"))

        jet_max_model_len = int(os.getenv("JET_MAX_MODEL_LEN", "0")) or None
        jet_max_num_seqs = int(os.getenv("JET_MAX_NUM_SEQS", "0")) or None
        jet_max_active = int(os.getenv("JET_MAX_ACTIVE", "0")) or None

        llm = build_engine_adapter(
            name="jet",
            tokenizer=tokenizer,
            jet_model_path=os.getenv("JET_MODEL_PATH", args.tokenizer_path),
            jet_tp_size=tp_size,
            jet_enforce_eager=enforce_eager,
            jet_mask_token_id=mask_token_id,
            jet_block_length=block_length,
            jet_max_model_len=jet_max_model_len,
            jet_max_num_seqs=jet_max_num_seqs,
            jet_max_active=jet_max_active,
        )

    # Build agent(s)
    agent = None
    mcts_agent = None

    if args.mode == "value":
        value_fn = HFValueFunction(
            base_lm_path=(args.value_base or args.tokenizer_path),
            value_head_path=args.value_head,
            device=args.value_device,
            dtype=args.value_dtype,
        )
        
        from trainer.latent_bank import LatentBank
        hid_bank = LatentBank(
            device=args.value_device,
            dtype=torch.bfloat16,
            store_cpu_copy=False,
            normalize=False,
        )        
        
        reward_fns = _build_reward_fns(args.dataset_name)
        mcts_agent = MCoderAgent(
            tokenizer=tokenizer,
            depth=max(1, args.depth),
            breadth=max(1, args.breadth),
            output_dir="./eval",
            llm=llm,
            max_model_len=int(args.max_model_len),
            sampling_params=sampling,
            value_fn=value_fn,
            reward_fns=reward_fns,
            c_puct=float(args.mcts_c_puct),
            v_prior=float(args.mcts_v_prior),
            value_trust=float(args.mcts_value_trust),
            num_sim=int(args.mcts_num_sim),
            prune_per=int(args.mcts_prune_per),
            max_expands=args.mcts_max_expands,
            num_pos_sim=int(args.mcts_num_pos_sim),
            passk_threshold=float(args.mcts_passk_threshold),
        )
        mcts_agent.hid_bank = hid_bank
                    
    elif args.mode == "single":
        agent = PoorAgent(
            tokenizer=tokenizer,
            depth=1,
            breadth=1,
            output_dir="./eval",
            llm=llm,
            sampling_params=sampling,
            max_model_len=int(args.max_model_len),
        )
    else:
        agent = CoderAgent(
            tokenizer=tokenizer,
            depth=max(1, args.depth),
            breadth=1,
            output_dir="./eval",
            llm=llm,
            sampling_params=sampling,
            max_model_len=int(args.max_model_len),
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    wrote_any = False
    with open(args.out, "w", encoding="utf-8") as fout, open(args.data, "r", encoding="utf-8") as f:
        total = 0
        for line in f:
            if args.limit is not None and total >= args.limit:
                break
            if not line.strip():
                continue

            ex = json.loads(line)
            question = str(ex["question"])
            qid = str(ex.get("id") or _qid(question))

            completions: List[str] = []
            finals: List[str] = []
            k_used = 1

            if args.mode == "value":
                chains = mcts_agent.search(
                    question=question,
                    ground_truth=ex["answer"],
                    support_material_path=None,
                    cot=None,
                )

                # # Collect unique leaves (chain[-1])
                # leaves = []
                # seen = set()
                # for ch in chains:
                #     if not ch:
                #         continue
                #     lf = ch[-1]
                #     # Dedup in case multiple chains reference same leaf dict
                #     sid = id(lf)
                #     if sid in seen:
                #         continue
                #     seen.add(sid)
                #     leaves.append(lf)

                # def _vpred(st: dict) -> float:
                #     v = st.get("v_pred", None)
                #     try:
                #         return float(v)
                #     except Exception:
                #         return float("-inf")

                # # Prefer leaves that contain <answer>...</answer>, to match "pass@1" semantics
                # answer_leaves = [st for st in leaves if _ANS_TAG.search(st.get("completion", "") or "")]
                # candidates = answer_leaves if answer_leaves else leaves
                # print({_extract_answer(c["completion"]): c["v_pred"] for c in candidates})
                
                # best_leaf = max(candidates, key=_vpred) if candidates else None
                # completion = best_leaf.get("completion", "") if best_leaf else "<think>...</think><answer></answer>"
                # final = _extract_answer(completion, aime_hint=("aime" in args.dataset_name.lower()))
                
                best_leaf = pick_best_leaf(chains, prefer_answer=True)
                completion = best_leaf["completion"]
                final = _extract_answer(completion)
                print(f"Ground Truth: {ex['answer']}")
                print(f"Answer: {final}")
                print("----------------------------------")

                completions = [completion]
                finals = [final]
                k_used = 1

            else:
                assert agent is not None
                k_used = max(1, int(getattr(args, "rollout_k", 1) or 1))

                for _ in range(k_used):
                    chains = agent.react_recursive(
                        question=question,
                        support_material_path=None,
                        ground_truth=ex["answer"],
                        assistant_and_tool_msg=None,
                        current_chain=None,
                        current_depth=1,
                        previous_variables=dict(),
                    )
                    best_leaf = None
                    for ch in chains:
                        if not ch:
                            continue
                        cand = ch[-1]
                        if _ANS_TAG.search(cand.get("completion", "")):
                            best_leaf = cand
                            break
                        if best_leaf is None:
                            best_leaf = cand

                    comp_i = best_leaf.get("completion", "") if best_leaf else "<think>...</think><answer></answer>"
                    fin_i = _extract_answer(comp_i, aime_hint=("aime" in args.dataset_name.lower()))
                    completions.append(comp_i)
                    finals.append(fin_i)

                # Keep backward-compat fields (treat first sample as pass@1)
                completion = completions[0] if completions else "<think>...</think><answer></answer>"
                final = finals[0] if finals else _extract_answer(completion, aime_hint=("aime" in args.dataset_name.lower()))

            # ---- Unit-style sanity check (first example only) ----
            if not wrote_any:
                assert isinstance(completion, str), f"completion is not str: {type(completion)}"
                assert not completion.lstrip().startswith("{"), (
                    "completion looks like a stringified dict/object. "
                    "This usually happens when adapter tokenizes str(req_out)."
                )
                wrote_any = True

            fout.write(
                json.dumps(
                    dict(
                        dataset=args.dataset_name,
                        qid=qid,
                        question=question,
                        completion=completion,          # backward compat
                        final_answer=final,             # backward compat
                        completions=completions,        # for pass@k / avg@k
                        final_answers=finals,
                        k=int(k_used),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            total += 1


if __name__ == "__main__":
    main()
