from __future__ import annotations

import random
import heapq
import math

import json
import ast
import numpy as np
import pandas as pd

import os
import re
import gc
import abc
import copy
import traceback

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, ClassVar

import torch
from transformers import AutoTokenizer
from trainer.vllm_client import _VLLMServerAdapter

from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

from uuid import uuid4
from io import StringIO
from pathlib import Path




def dump_with_rich(step: dict, logfile: str):
    """Rich dump that prefers `state_value`, is backward-compatible with `reward`,
    and optionally shows `v_pred`."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, record=False)

    # Backward-compat: prefer state_value, fall back to reward (old field).
    console.print(Panel(escape(str(step.get("reward", step.get("state_value", None)))), title="STATE VALUE"))
    console.print(Panel(escape(str(step.get("completion_ids", "").shape[-1]+step.get("prompt_ids", "").shape[-1])), title="CONTEXT LENGTH"))
    console.print(Panel(escape(str(step.get("prompt", ""))), title="PROMPT"))
    console.print(Panel(escape(str(step.get("completion", ""))), title="COMPLETION"))
    console.print(Panel(escape(str(step.get("ground_truth", ""))), title="GROUND TRUTH"))
    text = buf.getvalue()

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    Path(logfile).write_text(text, encoding="utf-8")
    return logfile


def parse_tool_calls(content: str):
    """
    Supports two tool call representations:
      1) <tool_call>{ "name": "...", "arguments": {...} }</tool_call>
      2) ```python ... ```   ← only mapped to execute_python_code

    Return：
      {"role":"assistant", "content": <text after removing all tool blocks>, "tool_calls":[...]}
    """
    import re, json, ast

    TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    PY_RE   = re.compile(r"```(?:python)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    segments = []
    tool_calls = []

    hits = []
    for m in TOOL_RE.finditer(content):
        hits.append(("tool", m.start(), m))
    for m in PY_RE.finditer(content):
        hits.append(("py", m.start(), m))
    hits.sort(key=lambda t: t[1])

    last = 0
    decoder = json.JSONDecoder(strict=False)

    for kind, start, m in hits:
        if start > last:
            head = content[last:start]
            if head.strip():
                segments.append(head)
        raw = m.group(1)

        if kind == "tool":
            try:
                func = decoder.decode(raw.strip())
            except Exception:
                func = ast.literal_eval(raw.strip())
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = decoder.decode(args)
                except Exception:
                    args = ast.literal_eval(args)
            func["arguments"] = args
            tool_calls.append({"type": "function", "function": func})

        elif kind == "py":
            code = raw if isinstance(raw, str) else str(raw)
            func = {"name": "execute_python_code", "arguments": {"code": code}}
            tool_calls.append({"type": "function", "function": func})

        last = m.end()

    if last < len(content):
        tail = content[last:]
        if tail.strip():
            segments.append(tail)

    if tool_calls:
        c = "\n".join(s.strip() for s in segments if s and s.strip())
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}

    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


def _poincare_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> float:
    """
    u, v: 1D numpy arrays (Poincaré points), returns geodesic distance
    """
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    uv_sq = float(np.maximum(0.0, uu + vv - 2.0 * np.dot(u, v)))
    denom = max(eps, (1.0 - uu) * (1.0 - vv))
    arg = 1.0 + 2.0 * uv_sq / denom
    arg = max(arg, 1.0 + 1e-7)
    return float(np.arccosh(arg))


class Node:
    """MCTS tree node."""
    def __init__(
        self,
        parent      : Optional["Node"],
        p_prior     : float,
        step_dict   : Dict[str, Any],
        messages    : List[Dict[str, Any]],
        context     : Dict[str, Any], 
        depth       : int
    ):
        self.parent   = parent
        self.depth    = depth
        self.children : List["Node"] = []
        self.P        = p_prior
        self.N        = 0
        self.W        = 0.0
        self.Q        = 0.0
        self.step     = step_dict
        self.messages = messages
        self.context  = context
        
        self.hid        = step_dict.get("hid", None)
        self.hid_idx    = step_dict.get("hid_idx", None)
        self.cluster_id = step_dict.get("cluster_id", None)
        self.disabled   = step_dict.get("disabled", False)

        self.v_pred      = step_dict.get("v_pred", None)       # model-predicted scalar (value head)
        self.state_value = step_dict.get("state_value", None)  # value used for search/backup

        self.is_terminal = False
        self.expand_calls = int(step_dict.get("expand_calls", 0))

    def u_score(self, c_puct: float, total_N: int) -> float:
        """PUCT upper-confidence bound component."""
        return c_puct * self.P * math.sqrt(total_N) / (1 + self.N)

    def best_child(self, c_puct: float) -> "Node":
        active = [ch for ch in self.children if not ch.disabled]
        if not active:
            return None
        total_N = sum(ch.N for ch in active) or 1
        best, best_sc = None, -1e18
        for ch in active:
            score = ch.Q + ch.u_score(c_puct, total_N)
            if score > best_sc:
                best_sc, best = score, ch
        return best

    def backup(self, value: float):
        """Back-propagate using the *search* value (state_value), not raw v_pred."""
        self.N += 1
        self.W += value
        self.Q  = self.W / self.N
        if self.parent:
            self.parent.backup(value)


class MCTSAgent(abc.ABC):

    SYSTEM_TEMPLATE: ClassVar[str]
    USER_TEMPLATE: ClassVar[str]
    TOOLS:             ClassVar[Dict[str, callable]]
    TOOLS_DESCRIPTION: ClassVar[List[Dict[str, Any]]]

    def __init__(
        self,
        tokenizer          : AutoTokenizer,
        depth              : int,
        breadth            : int,
        output_dir         : str,

        llm                : _VLLMServerAdapter,
        max_model_len  , 
        sampling_params,

        value_fn,
        reward_fns         : list = None,

        c_puct             : float = 1.0,
        v_prior            : float = 0.5,
        value_trust        : float = 0.5,

        num_sim              : int   = 128,
        prune_per            : int   = 129,
        max_expands          : int | str = 2,

        num_pos_sim        : int   = 4,
        passk_threshold    : float = 1.0,
    ):
        self.tokenizer       = tokenizer
        self.depth           = depth
        self.breadth         = breadth
        self.output_dir      = output_dir
        self.sampling_params = sampling_params
            
        self.llm             = llm
        self.max_model_len   = max_model_len

        self.value_fn   = value_fn
        self.reward_fns = reward_fns

        self.c_puct      = c_puct
        self.v_prior     = v_prior
        self.value_trust = value_trust

        self.num_sim   = num_sim
        self.prune_per = prune_per

        self.num_pos_sim     = num_pos_sim
        self.passk_threshold = passk_threshold

        self._all_nodes: List[Node] = []                   # All created nodes (except root)
        self._next_cluster_id: int = 0
        self._cluster_centers: Dict[int, np.ndarray] = {}  # For logging / visualization

        # Progressive widening
        self.progressive_widening = True
        # Latent state bank
        self.hid_bank = None
        self.max_expands = max_expands

        # ------------------------------------------------------------------
        # CoT‑guided "hostage" prefix (optional):
        #
        # _cot_think_prefix:
        #   - a short prefix y extracted from the <think>...</think> part of
        #     the given CoT, used as a "pre‑fill" after STEP‑k:\n<think>\n.
        # _cot_used:
        #   - True once we have injected the CoT prefix in one expansion
        #     round during this search (so we only do it once).
        # _inject_cot_this_round:
        #   - Flag used by the current expansion round to tell
        #     `_expand_and_evaluate` whether it should inject the prefix.
        # ------------------------------------------------------------------
        self._cot_think_prefix: Optional[str] = None
        self._cot_used: bool = False
        self._inject_cot_this_round: bool = False
        
    def _build_response_mask_for_last_assistant(
        self,
        full_ids_1d: torch.Tensor,
        assistant_span_ids_1d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a (1, L) response mask that marks the *last occurrence* of
        `assistant_span_ids_1d` inside `full_ids_1d`.

        This is robust to chat-template wrappers and avoids needing any special
        hostage handling: hostage text is already part of assistant_span_ids_1d.

        Notes:
        - We search for the span in token space (exact match).
        - If not found, we fall back to masking the last non-pad tokens.
        """
        full_ids = full_ids_1d.view(-1).tolist()
        span_ids = assistant_span_ids_1d.view(-1).tolist()
        L = len(full_ids)

        # Empty span -> no response tokens; mask nothing.
        if not span_ids:
            return torch.zeros((1, L), dtype=torch.long)

        # Find last match (search from the end).
        start = None
        n, m = len(full_ids), len(span_ids)
        for s in range(n - m, -1, -1):
            if full_ids[s : s + m] == span_ids:
                start = s
                break

        rm = torch.zeros((1, L), dtype=torch.long)
        if start is not None:
            rm[0, start : start + m] = 1
            return rm

        # Fallback: mark the last m non-pad tokens as response.
        # (This is a safe fallback if templates add/alter tokens unexpectedly.)
        # Caller typically pads with pad_id; here we just assume full_ids has no trailing pad.
        tail = min(m, L)
        rm[0, L - tail : L] = 1
        return rm

    def _extract_cot_think_prefix(self, cot: str) -> Optional[str]:
        """
        Given a CoT string of the form:
            <think>...</think><answer>...</answer>

        Extract the <think>...</think> content and take the first
        min(max_tokens // 2, len(think_tokens) // 2) tokens, then
        decode back to text.

        This prefix is then injected right after "STEP-k:\\n<think>\\n"
        so that the model continues from this partially given chain of
        thought.

        Returns:
            prefix_text (str) or None if extraction/tokenization fails.
        """
        if not cot:
            return None

        try:
            # 1) Extract the <think>...</think> span.
            m = re.search(r"<think>(.*?)</think>", str(cot), flags=re.S)
            if not m:
                return None
            think_text = m.group(1)
            if not think_text:
                return None

            # 2) Tokenize the think body WITHOUT special tokens.
            tok = self.tokenizer(
                think_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ids = tok["input_ids"].squeeze(0)
            if ids.numel() == 0:
                return None

            think_len = ids.size(0)
            half_think = max(1, think_len // 2)

            # 3) Limit by sampling_params.max_tokens // 2, if available.
            max_tok_cfg = getattr(self.sampling_params, "max_tokens", None)
            if max_tok_cfg is None:
                max_front = half_think
            else:
                max_front = min(max_tok_cfg // 2, half_think)

            L = int(max_front)
            if L <= 0:
                return None

            prefix_ids = ids[:L]
            # Do NOT aggressively strip; we want to preserve formatting
            # as much as possible while still skipping special tokens.
            prefix_text = self.tokenizer.decode(
                prefix_ids,
                skip_special_tokens=True,
            )
            return prefix_text
        except Exception:
            # Safe fallback: if anything goes wrong, we simply do not
            # use CoT injection for this search.
            return None
        
    def read_support_material(self, table_paths):
        if table_paths:
            support_material = dict()
            for i in range(len(table_paths)):
                try:
                    support_material[f"df{i}"] = pd.read_csv(table_paths[i])
                except Exception as e:
                    with open(table_paths[i]) as f: support_material[f"tb{i}"] = f.read()

            support_material_str = "\n".join(
                f"Var: {k}; Type: {type(v)}\n{v}" + f"\n{v.dtypes}" if isinstance(v, pd.DataFrame) else f"Var: {k}; Type: {type(v)}\n{v}" for k, v in support_material.items()
            )
        else:
            support_material, support_material_str = dict(), str()
        return support_material, support_material_str

    def _truncate_on_k_step(self, text: str, k=1) -> tuple[str, bool]:
        STEP_STOP_RE = re.compile(r'(?im)^[ \t]*STEP[ \t]*-[ \t]*\d+[ \t]*:?', re.MULTILINE)
        it = STEP_STOP_RE.finditer(text)
        m2 = None
        for i, m in enumerate(it, 1):
            if i == k:
                m2 = m
                break
        if m2:
            return text[:m2.start()].rstrip(), True
        return text, False

    def cluster_and_prune(self):
        """
        Global auto-clustering + pruning on Poincaré points y (node.hid):
        - Distance = Poincaré geodesic
        - Average-linkage agglomerative; cut by largest relative jump
        - Randomly disable ~1/3 per cluster (kept from your original logic)
        """
        nodes = [n for n in self._all_nodes if (n.hid is not None) and (not n.disabled)]
        N = len(nodes)
        if N <= 1:
            if N == 1 and nodes[0].cluster_id is None:
                nodes[0].cluster_id = self._next_cluster_id
                nodes[0].step["cluster_id"] = self._next_cluster_id
                self._cluster_centers[self._next_cluster_id] = np.asarray(nodes[0].hid, dtype="float32")
                self._next_cluster_id += 1
            return

        Z = np.stack([np.asarray(n.hid, dtype="float32") for n in nodes], axis=0)  # (N, Dp), Poincaré y
        # pairwise geodesic distance
        D = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                d = _poincare_distance(Z[i], Z[j])
                D[i, j] = D[j, i] = d

        clusters = [[i] for i in range(N)]
        snapshots = [[c[:] for c in clusters]]
        merge_dists = []
        while len(clusters) > 1:
            m = len(clusters)
            M = np.full((m, m), np.inf, dtype=np.float32)
            for i in range(m):
                idx_i = clusters[i]
                for j in range(i + 1, m):
                    idx_j = clusters[j]
                    sub = D[np.ix_(idx_i, idx_j)]
                    dist = float(sub.mean())
                    M[i, j] = dist
            k = int(np.argmin(M))
            i, j = divmod(k, M.shape[1])
            if i == j: break
            d = float(M[i, j]); merge_dists.append(d)
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)
            snapshots.append([c[:] for c in clusters])

        if len(merge_dists) == 0:
            cut_merges = 0
        elif len(merge_dists) == 1:
            cut_merges = 1
        else:
            d = np.asarray(merge_dists, dtype=np.float32)
            deltas = np.diff(d)
            ratio = deltas / (np.abs(d[:-1]) + 1e-8)
            cut_merges = int(np.argmax(ratio)) + 1
            cut_merges = min(cut_merges, len(snapshots) - 1)
        final_clusters = snapshots[cut_merges]
        if len(final_clusters) >= len(snapshots[0]) and len(snapshots) > 1:
            forced_merges = min(max(1, len(snapshots) // 4), len(snapshots) - 1)
            final_clusters = snapshots[forced_merges]

        centers = []
        for idxs in final_clusters:
            # center: Fréchet mean
            mean = Z[idxs].mean(axis=0)
            # clamp to ball
            norm = np.linalg.norm(mean) + 1e-12
            max_norm = 1.0 - 1e-4
            if norm > max_norm:
                mean = mean * (max_norm / norm)
            centers.append(mean.astype("float32"))

        cid = self._next_cluster_id
        self._cluster_centers = {}
        for c_idx, idxs in enumerate(final_clusters):
            members = [nodes[i] for i in idxs]
            for m in members:
                m.cluster_id = cid
                m.step["cluster_id"] = cid
            center = centers[c_idx]
            self._cluster_centers[cid] = center
            n = len(members)
            remove_cnt = max(0, n // 3)
            if remove_cnt >= n: remove_cnt = n - 1
            to_disable = set(random.sample(members, remove_cnt)) if remove_cnt > 0 else set()
            for m in members:
                if m in to_disable:
                    m.disabled = True; m.step["disabled"] = True
                else:
                    m.disabled = False; m.step["disabled"] = False
            cid += 1
        self._next_cluster_id = cid
    
    def _global_score(self, node: "Node", expand_total: int) -> float:
        """
        Globally comparable score for picking WHERE to expand:
          S(n) = Q_eff(n) + c * P(n) * sqrt(expand_total+1) / (1 + N(n)) - depth_pen * depth(n)

        Q_eff(n) = Q(n) if visited else state_value (usually v_pred smoothing).
        Only terminal children trigger backup -> N(n) increases only along those paths.
        """
        Qeff = float(node.Q if node.N > 0 else (node.state_value or 0.0))
        P    = float(getattr(node, "P", 0.0))
        Nn   = int(node.N)
        Nt   = int(expand_total) + 1
        c    = float(getattr(self, "c_puct", 1.0))
        return Qeff + c * P * (Nt ** 0.5) / (1.0 + Nn)

    def _can_expand(self, node: "Node") -> bool:
        if node.is_terminal or node.disabled:
            return False
        if isinstance(self.max_expands, int):
            return node.expand_calls < self.max_expands
        if isinstance(self.max_expands, str) and self.max_expands == "decay":
            return node.expand_calls < max(1, self.depth - node.depth ** 2 + 1)
        return True

    def _is_frontier(self, node: "Node") -> bool:
        """A node is expandable if it's not terminal/disabled AND below expand cap."""
        return self._can_expand(node)

    def _push_frontier(self, heap, node: "Node", expand_total: int):
        """Push (neg-score, unique-id, node) to act as a max-heap."""
        if not self._is_frontier(node):
            return
        score = self._global_score(node, expand_total)
        heapq.heappush(heap, (-score, id(node), node))
        
    def _select_leaves(self, frontier, k: int, expand_total: int) -> list:
        """
        Pick up to `k` globally best expansion targets from the frontier heap.
        Uses the same global score as `_push_frontier` (max-heap).
        Skips disabled/terminal/duplicated nodes.
        """
        batch, seen_ids = [], set()
        while frontier and len(batch) < max(1, int(k)):
            _neg, _sid, cand = heapq.heappop(frontier)
            if (id(cand) in seen_ids) or cand.disabled or cand.is_terminal:
                continue
            if not self._can_expand(cand):
                continue
            seen_ids.add(id(cand))
            batch.append(cand)
        return batch

    def search(
        self,
        question: str,
        support_material_path: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        cot: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        MCTS search:
        - Decide CoT prefix ONCE, at most one expansion round, after halfway point,
        only if no positive leaf has been found so far.
        - Apply injection to the frontier nodes selected for expansion in that round.
        """
        # ---- Build root ----
        support_material, support_material_str = self.read_support_material(support_material_path)
        support_material_str = f"# Given this:\n{support_material_str}" if support_material_str else ""

        system_prompt = self.SYSTEM_TEMPLATE.format(step_limit=self.depth)
        user_prompt = self.USER_TEMPLATE.format(
            support_material_str=support_material_str,
            question=question,
        )
        root_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation=root_msgs,
            tools=self.TOOLS_DESCRIPTION,
            tokenize=False,
            add_generation_prompt=True,
        )

        root_step = {
            "prompt": prompt,
            "prompt_ids": self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].squeeze(0),
            "ground_truth": ground_truth,
            "completion": "",
            "completion_ids": [],
            "hostaged": False,
            "state_value": None,
            "current_depth": 0,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        root = Node(
            parent    = None,
            depth     = 0,
            p_prior   = 1.0,
            step_dict = root_step,
            messages  = root_msgs,
            context   = copy.deepcopy(support_material),
        )

        # ---- Root eval (hid + v_pred) ----
        root_ids = root.step["prompt_ids"].view(1, -1).long().cpu()
        pad_id = int(self.tokenizer.pad_token_id or 0)
        root_attn = (root_ids != pad_id).long().cpu()

        # respect max_model_len by left-truncating
        max_len = int(getattr(self, "max_model_len", 0) or 0)
        if max_len > 0 and root_ids.size(1) > max_len:
            root_ids = root_ids[:, -max_len:]
            root_attn = root_attn[:, -max_len:]

        # Get v_pred + root_h0, but store y_root as 0 in bank
        h_root_raw, v_root, h0_root = self.value_fn(
            input_ids=root_ids,
            attention_mask=root_attn,
            response_mask=root_attn,
            prompt_mask=root_attn,
            root_h0=None,
            return_h0=True,
        )

        vp = float(v_root[0].item())
        root.step["v_pred"] = vp
        root.step["state_value"] = float(self.value_trust * vp + (1.0 - self.value_trust) * self.v_prior)

        # Cache root_h0 for the entire search
        root_h0 = h0_root[0].detach().to("cpu", dtype=torch.float32).view(-1)  # (H,)
        self.root_h0 = root_h0
        root.step["root_h0"] = root_h0  # The trainer side can also retrieve it; if you are concerned about serialization, you can store it using root_h0.tolist().

        # Root embedding in bank must be 0
        h_root_zero = torch.zeros_like(h_root_raw)  # (1,H)

        if getattr(self, "hid_bank", None) is not None:
            root_hid_idx = self.hid_bank.add(h_root_zero)
        else:
            root_hid_idx = None

        root.step["hid_idx"] = root_hid_idx
        root.step["hid"] = h_root_zero[0].float().cpu().numpy().astype(np.float16).tolist()
        self._root_step = root.step

        # ---- CoT prefix (prepared once; used at most once) ----
        cot_prefix: Optional[str] = self._extract_cot_think_prefix(cot) if cot else None
        cot_used_once = False

        # ---- Frontier heap ----
        expand_total = 0
        frontier: list = []
        self._push_frontier(frontier, root, expand_total)

        # ---- DDP-aware leaves_per_sim ----
        try:
            import torch.distributed as dist
            ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        except Exception:
            ws = 1

        leaves_per_sim = 1 if ws <= 1 else max(1, ws // self.breadth)
        total_rounds = max(1, self.num_sim // leaves_per_sim)
        half_round = total_rounds // 2

        self.pos_counter = 0
        for sim_i in range(total_rounds):
            # Stop early once enough positive terminal leaves collected.
            if self.pos_counter >= self.num_pos_sim:
                break

            # Decide whether to inject CoT prefix THIS round (one-off).
            cot_prefix_for_round: Optional[str] = None
            if (
                (cot_prefix is not None)
                and (not cot_used_once)
                and (sim_i >= half_round)
                and (self.pos_counter == 0)
            ):
                cot_prefix_for_round = cot_prefix
                cot_used_once = True  # ensure one-off

            # 1) Pick expansion targets from global frontier.
            frontier_nodes = self._select_leaves(frontier, leaves_per_sim, expand_total)
            if not frontier_nodes:
                break

            # 2) Expand & value-evaluate in batch (frontier nodes expanded are "leaves" here).
            creations = self._expand_and_evaluate(
                leaves=frontier_nodes,
                ground_truth=ground_truth,
                breadth=self.breadth,
                cot_prefix=cot_prefix_for_round,   # <-- explicit, no hidden flags
            )
            expand_total += len(frontier_nodes)

            # 3) Backup ONLY terminal kids, re-push frontier for parent + non-terminal kids
            for parent_node, new_children in creations:
                for ch in new_children:
                    if ch.is_terminal:
                        val = float(ch.state_value if ch.state_value is not None else 0.0)
                        ch.backup(val)

                if self._is_frontier(parent_node):
                    self._push_frontier(frontier, parent_node, expand_total)

                for ch in new_children:
                    if self._is_frontier(ch):
                        self._push_frontier(frontier, ch, expand_total)

            # Optional pruning; rebuild heap after pruning
            if self.prune_per and ((sim_i + 1) % self.prune_per == 0):
                self.cluster_and_prune()

                frontier = []
                stack = [root]
                seen = set()
                while stack:
                    cur = stack.pop()
                    if id(cur) in seen:
                        continue
                    seen.add(id(cur))
                    if self._is_frontier(cur):
                        self._push_frontier(frontier, cur, expand_total)
                    for ch in cur.children:
                        if not ch.disabled:
                            stack.append(ch)

        # Extract chains
        chains: List[List[Dict[str, Any]]] = []

        def dfs(n: Node, chain: List[Dict[str, Any]]):
            if n.parent:
                n.step["_N"] = int(n.N)
                n.step["_Q"] = float(n.Q)
                n.step["_P"] = float(n.P)
                n.step["_depth"] = int(n.depth)
                n.step["_terminal"] = bool(n.is_terminal)
                n.step["_disabled"] = bool(n.disabled)
                chain = chain + [n.step]
            if not n.children:
                chains.append(chain)
            else:
                for ch in n.children:
                    dfs(ch, chain)

        dfs(root, [])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return chains

    @torch.no_grad()
    def _expand_and_evaluate(
        self,
        leaves: List["Node"],
        ground_truth,
        breadth: int,
        *,
        cot_prefix: Optional[str] = None,
    ) -> List[Tuple["Node", List["Node"]]]:
        """
        Expand `breadth` children for each frontier node in `leaves`,
        then evaluate ALL children with one batched `value_fn` call.

        Key invariants
        --------------------------
        - We prefill injection text (CoT prefix or "Wait") into the PROMPT only.
        That means:
            * prompt_ids already include the injected prefix
            * completion_ids must be ONLY the model-generated continuation token ids
        This avoids double-counting injected tokens in full_ids.

        - Value evaluation uses:
            full_ids       = prompt_ids || completion_ids
            attention_mask = 1 for all real tokens
            response_mask  = 1 only for completion tokens (and only until first eos)
            prompt_mask    = 1 only for prompt tokens (optional; can be omitted)
        And if we left-truncate to max_model_len, we must truncate full_ids, response_mask, and prompt_mask together (same slice).
        """
        # ---------------------------------------------------------------------
        # 0) Filter expandable nodes
        # ---------------------------------------------------------------------
        frontier_nodes = [n for n in leaves if self._can_expand(n)]
        if not frontier_nodes:
            return []

        # ---------------------------------------------------------------------
        # 1) Build prompts (one per frontier node)
        # ---------------------------------------------------------------------
        prompt_texts: List[str] = []
        prompt_ids_list: List[torch.Tensor] = []
        think_headers: List[str] = []
        node_messages_list: List[List[Dict[str, Any]]] = []
        inject_text_list: List[str] = []      # actual injected string for this parent ("" | "wait" | cot_prefix)
        inject_mode_list: List[str] = []      # "cot" | "wait" | "none"

        for node in frontier_nodes:
            node.expand_calls = int(getattr(node, "expand_calls", 0)) + 1
            node.step["expand_calls"] = node.expand_calls

            depth = int(node.step.get("current_depth", 0)) + 1
            think_header = f"STEP-{depth}:\n<think>\n"
            think_headers.append(think_header)

            # mutually-exclusive injection modes
            wait_hostage = bool(node.step.get("hostaged", False))
            use_cot = bool(cot_prefix) and (not wait_hostage)
            if use_cot:
                inject_mode = "cot"
                inject_text = cot_prefix or ""
            elif wait_hostage:
                inject_mode = "wait"
                inject_text = "wait"
            else:
                inject_mode = "none"
                inject_text = ""

            inject_mode_list.append(inject_mode)
            inject_text_list.append(inject_text)

            # Base chat prompt up to the generation prompt
            base_prompt = self.tokenizer.apply_chat_template(
                conversation=node.messages,
                tools=self.TOOLS_DESCRIPTION,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Prefill the step header and injection text in the PROMPT
            prompt = base_prompt + think_header + inject_text

            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"].squeeze(0)

            prompt_texts.append(prompt)
            prompt_ids_list.append(prompt_ids)
            node_messages_list.append(node.messages)

        # ---------------------------------------------------------------------
        # 2) Generate (multi-prompt, breadth=n per prompt)
        # ---------------------------------------------------------------------
        self.sampling_params.n = int(breadth)
        response_list = self.llm.generate(
            prompts=prompt_texts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        output_list = [r.outputs for r in response_list]

        # ---------------------------------------------------------------------
        # 3) Parse children specs (flat), keep priors per parent
        # ---------------------------------------------------------------------
        child_specs_flat: List[Tuple["Node", Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], bool, int, int]] = []
        priors_groups: List[List[float]] = []

        for parent_i, output in enumerate(output_list):
            # priors = softmax(cumulative_logprob) among siblings (stable)
            cum_logps_all = [float(o.cumulative_logprob) for o in output]
            if cum_logps_all:
                m = max(cum_logps_all)
                exps = [math.exp(x - m) for x in cum_logps_all]
                Z = sum(exps)
                priors = [v / Z for v in exps] if Z > 0 else [1.0 / len(exps)] * len(exps)
            else:
                priors = []

            priors_groups.append(priors)

            parent = frontier_nodes[parent_i]
            prompt = prompt_texts[parent_i]
            prompt_ids = prompt_ids_list[parent_i]
            msgs_prefix = node_messages_list[parent_i]
            think_header = think_headers[parent_i]
            inject_text = inject_text_list[parent_i]
            inject_mode = inject_mode_list[parent_i]

            for k, o in enumerate(output):
                cum_logps = o.cumulative_logprob
                # vLLM output token ids are the *model-generated continuation* after the prompt.
                raw_ids = o.token_ids if isinstance(o.token_ids, list) else list(o.token_ids)
                gen_ids = torch.tensor(raw_ids, dtype=torch.long)
                gen_text = self.tokenizer.decode(raw_ids, skip_special_tokens=True)
                
                # For logging/reward formatting, we reconstruct the full assistant "think" body:
                #   assistant_body = inject_text + gen_text
                # But we do NOT add inject_text into completion_ids.
                assistant_body = (inject_text + gen_text) if inject_text else gen_text

                # completion string used for reward/formatting/tools is the full assistant step,
                # including STEP header + <think> prefix.
                completion = think_header + assistant_body

                completion_ids = gen_ids
                should_terminate = False

                # Terminate if answer exists
                if re.findall(r"<answer>(.*?)</answer>", completion):
                    should_terminate = True

                current_depth = int(parent.step["current_depth"]) + 1
                # Echo detection vs prompt (use body text to avoid false matches on headers)
                # NOTE: completion starts with STEP header by construction.
                _hdr = re.match(r"^STEP-\d+:\r?\n<think>\r?\n?", completion)
                # _hdr = re.match(r"<think>\r?\n?", completion)
                completion_body_nohdr = completion[_hdr.end():].strip() if _hdr else completion.strip()

                is_body_echo = bool(completion_body_nohdr) and (completion_body_nohdr in prompt)
                tool_call_blocks = re.findall(r"<tool_call>.*?</tool_call>", completion, flags=re.S)
                is_tool_echo = any((blk.strip() and (blk.strip() in prompt)) for blk in tool_call_blocks)
                is_echo_from_prompt = is_body_echo or is_tool_echo

                if (current_depth >= self.depth) or is_echo_from_prompt:
                    should_terminate = True

                # Optional hostage logic (kept for compatibility; currently unreachable if you always terminate on <answer>)
                hostaged = False
                if (not should_terminate) and re.findall(r"<answer>(.*?)</answer>", completion):
                    hostaged = True
                    completion = completion.split("<answer>")[0]
                    # Retokenize only the generated continuation part is tricky; fall back to full-text tokenize.
                    # This is rarely used; if you rely on it, consider storing a "generated_only_text" separately.
                    completion_ids = self.tokenizer(
                        # completion already includes think_header; we add an explicit end marker
                        completion + "<|im_end|>",
                        return_tensors="pt",
                        add_special_tokens=True,
                    )["input_ids"].squeeze(0).long()

                # Length cap (prompt + generated continuation)
                max_model_len = int(getattr(self, "max_model_len", 0) or 0)
                # max_gen_len = int(getattr(self.sampling_params, "max_tokens", 0) or 0)
                if (max_model_len > 0 and (int(prompt_ids.numel()) + int(completion_ids.numel()) >= max_model_len)):
                    # or (max_gen_len > 0 and int(completion_ids.numel()) >= max_gen_len):
                    should_terminate = True

                # ---- tool parsing ----
                results: List[Dict[str, Any]] = []
                new_context = parent.context.copy()
                try:
                    assistant_msg = parse_tool_calls(completion)
                except Exception:
                    assistant_msg = {"role": "assistant", "content": completion}
                    tool_response = [{
                        "role": "user",
                        "content": "Error: can not parse your <tool_call></tool_call> block.",
                    }]
                else:
                    tool_calls = assistant_msg.get("tool_calls", []) or []
                    kept = []
                    tool_response = []

                    for _call in tool_calls:
                        fn = _call.get("function") or {}
                        func_name = fn.get("name", None)
                        args = fn.get("arguments", {})

                        if not func_name:
                            tool_response.append({"role": "user", "content": f"Error: tool name missing for '<tool_call>{fn}</tool_call>'."})
                            assistant_msg["content"] = (assistant_msg.get("content") or "") + f"<tool_call>{fn}</tool_call>"
                            continue
                        func = self.TOOLS.get(func_name)
                        if func is None:
                            tool_response.append({"role": "user", "content": f"Error: no such a tool named '{func_name}'."})
                            assistant_msg["content"] = (assistant_msg.get("content") or "") + f"<tool_call>{fn}</tool_call>"
                            continue

                        if isinstance(args, str):
                            try:
                                import json
                                args = json.loads(args)
                            except Exception:
                                tool_response.append({"role": "user", "content": f"Error: tool arguments must be JSON object. Got string: {args[:200]}..."})
                                assistant_msg["content"] = (assistant_msg.get("content") or "") + f"<tool_call>{fn}</tool_call>"
                                continue

                        if not isinstance(args, dict):
                            tool_response.append({"role": "user", "content": f"Error: tool arguments must be an object/dict, got {type(args).__name__}."})
                            assistant_msg["content"] = (assistant_msg.get("content") or "") + f"<tool_call>{fn}</tool_call>"
                            continue

                        try:
                            output, new_ctx = func(context=new_context, **args)
                        except Exception as e:
                            tool_response.append({"role": "tool", "name": func_name, "content": f"Var: e; Type: {type(e).__name__}\n{e}"})
                            continue

                        new_context.update(new_ctx)
                        results.append(new_ctx)
                        tool_response.append({"role": "tool", "name": func_name, "content": output})
                        kept.append(_call)

                    assistant_msg["tool_calls"] = kept


                node_messages = msgs_prefix + [assistant_msg] + tool_response

                step_dict: Dict[str, Any] = {
                    "prompt": prompt,
                    "prompt_ids": prompt_ids,
                    "completion": completion,
                    "completion_ids": completion_ids,
                    "ground_truth": ground_truth,
                    "results": results,
                    "current_depth": current_depth,
                    "hostaged": hostaged,
                    "cum_logprob": cum_logps, 
                    "state_value": None,
                }

                try:
                    dump_with_rich(step_dict, os.path.join(self.output_dir, f"tmp{parent_i}-{k}.txt"))
                    if inject_mode == "cot":
                        dump_with_rich(step_dict, os.path.join(self.output_dir, "tmp_cot-hostaged.txt"))
                except Exception:
                    pass

                child_specs_flat.append(
                    (parent, step_dict, node_messages, new_context, should_terminate, parent_i, k)
                )

        # ---------------------------------------------------------------------
        # 4) Build one big batch for value_fn WITHOUT re-tokenizing full chat turns.
        #    full_ids = prompt_ids || completion_ids
        # ---------------------------------------------------------------------
        all_inputs: List[torch.Tensor] = []
        all_resp:   List[torch.Tensor] = []
        all_prompt: List[torch.Tensor] = []
        rev: List[Tuple[int, "Node", Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], bool, int, int]] = []

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0

        def _completion_pool_mask(comp_ids_1d: torch.Tensor) -> torch.Tensor:
            """1 for tokens up to and including first EOS, else 0 after EOS."""
            m = torch.ones_like(comp_ids_1d, dtype=torch.long)
            if eos_id is None:
                return m
            pos = (comp_ids_1d == int(eos_id)).nonzero(as_tuple=False)
            if pos.numel() > 0:
                first = int(pos[0].item())
                m[first + 1:] = 0   # keep eos, mask after eos
            return m

        max_len = getattr(self, "max_model_len", None)
        max_len = int(max_len) if (max_len is not None and int(max_len) > 0) else None

        for row_idx, (leaf, stp, msgs, ctx, should_terminate, leaf_i, k_i) in enumerate(child_specs_flat):
            try:
                p_ids = stp["prompt_ids"]
                c_ids = stp["completion_ids"]

                if not torch.is_tensor(p_ids):
                    p_ids = torch.as_tensor(p_ids, dtype=torch.long)
                if not torch.is_tensor(c_ids):
                    c_ids = torch.as_tensor(c_ids, dtype=torch.long)

                p_ids = p_ids.detach().to("cpu", dtype=torch.long).view(-1)
                c_ids = c_ids.detach().to("cpu", dtype=torch.long).view(-1)

                if c_ids.numel() == 0:
                    stp["disabled"] = True
                    stp["error"] = "empty completion_ids (cannot evaluate value)"
                    continue

                # response_mask: 0 for prompt tokens, 1 for completion tokens (up to EOS)
                c_mask = _completion_pool_mask(c_ids)
                r_mask = torch.cat([torch.zeros_like(p_ids, dtype=torch.long), c_mask], dim=0)

                # prompt_mask: 1 for prompt tokens, 0 for completion tokens
                p_mask = torch.cat(
                    [torch.ones_like(p_ids, dtype=torch.long),
                    torch.zeros_like(c_ids, dtype=torch.long)],
                    dim=0,
                )

                full_ids = torch.cat([p_ids, c_ids], dim=0)

                # Left-truncate to respect max context length (MUST truncate masks too!)
                if max_len is not None and int(full_ids.numel()) > max_len:
                    start = int(full_ids.numel()) - max_len
                    full_ids = full_ids[start:]
                    r_mask   = r_mask[start:]
                    p_mask   = p_mask[start:]

                # Sanity: masks must align with full_ids
                if int(r_mask.numel()) != int(full_ids.numel()) or int(p_mask.numel()) != int(full_ids.numel()):
                    raise RuntimeError(
                        f"mask length mismatch after truncation: "
                        f"full={int(full_ids.numel())}, r={int(r_mask.numel())}, p={int(p_mask.numel())}"
                    )

                # If response_mask became empty (should be rare), fall back to pooling on all tokens
                if int(r_mask.sum().item()) <= 0:
                    r_mask = torch.ones_like(full_ids, dtype=torch.long)

                all_inputs.append(full_ids)
                all_resp.append(r_mask)
                all_prompt.append(p_mask)
                rev.append((row_idx, leaf, stp, msgs, ctx, should_terminate, leaf_i, k_i))

            except Exception as e:
                stp["disabled"] = True
                stp["error"] = f"value batch build failed: {type(e).__name__}: {e}"
                continue

        if not all_inputs:
            return [(leaf, []) for leaf in leaves]

        # Pad to a rectangular batch
        B = len(all_inputs)
        Lmax = max(int(t.numel()) for t in all_inputs)

        ids2d  = torch.full((B, Lmax), fill_value=int(pad_id), dtype=torch.long)
        attn2d = torch.zeros((B, Lmax), dtype=torch.long)
        resp2d = torch.zeros((B, Lmax), dtype=torch.long)
        pm2d   = torch.zeros((B, Lmax), dtype=torch.long)

        for i, (ids1d, r1d, p1d) in enumerate(zip(all_inputs, all_resp, all_prompt)):
            L = int(ids1d.numel())
            ids2d[i, :L]  = ids1d
            attn2d[i, :L] = 1
            resp2d[i, :L] = r1d
            pm2d[i, :L]   = p1d

        # ---------------------------------------------------------------------
        # 5) One (distributed) value forward for ALL children
        # ---------------------------------------------------------------------
        h_batch, v_batch = self.value_fn(
            input_ids=ids2d,
            attention_mask=attn2d,
            response_mask=resp2d,
            prompt_mask=pm2d,
            root_h0=self.root_h0, 
            return_h0=False,
        )

        assert h_batch.size(0) == v_batch.size(0) == len(rev)

        # ---------------------------------------------------------------------
        # 6) Materialize children & attach
        # ---------------------------------------------------------------------
        created_per_parent: Dict[int, List["Node"]] = {i: [] for i in range(len(frontier_nodes))}

        for row, (_, parent, stp, msgs, ctx, should_terminate, parent_i, k_i) in enumerate(rev):
            # v_pred is the model's value_fn scalar (e.g., 1/(1+d))
            v_pred = float(v_batch[row].item())
            p_prior = float(priors_groups[parent_i][k_i]) if priors_groups[parent_i] else (1.0 / max(1, breadth))
            stp["p_prior"] = p_prior

            true_r = max(f(stp["completion"], ground_truth) for f in self.reward_fns)
            if (self.num_pos_sim < self.num_sim) and (true_r >= self.passk_threshold):
                self.pos_counter += 1
                
            if should_terminate:
                state_value = float(true_r)
            else:
                state_value = float(self.value_trust * v_pred + (1.0 - self.value_trust) * self.v_prior)

            stp["v_pred"] = v_pred
            stp["state_value"] = state_value

            # Store the hyperbolic embedding into hid_bank if available
            if getattr(self, "hid_bank", None) is not None:
                hid_idx = self.hid_bank.add(h_batch[row:row + 1])
            else:
                hid_idx = None

            stp["hid_idx"] = hid_idx
            stp["hid"] = h_batch[row].float().cpu().numpy().astype(np.float16).tolist()
            stp["disabled"] = False

            child = Node(
                parent    = parent,
                depth     = stp["current_depth"],
                p_prior   = p_prior,
                step_dict = stp,
                messages  = msgs,
                context   = ctx,
            )
            child.is_terminal = bool(should_terminate)
            child.v_pred = v_pred
            child.state_value = state_value

            parent.children.append(child)
            self._all_nodes.append(child)
            created_per_parent[parent_i].append(child)

        return [(frontier_nodes[i], created_per_parent.get(i, [])) for i in range(len(frontier_nodes))]


def _extract_answer_text(completion: str) -> str:
    ms = re.findall(r"<answer>(.*?)</answer>", completion or "", flags=re.S)
    return (ms[-1].strip() if ms else "")

def _to_float(x, default=0.0):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = float(arr.mean())
    sd = float(arr.std()) + 1e-6
    return (arr - mu) / sd

def _poincare_dist(u: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> float:
    uu = float(u @ u)
    vv = float(v @ v)
    uv_sq = float(max(0.0, uu + vv - 2.0 * float(u @ v)))
    denom = max(eps, (1.0 - uu) * (1.0 - vv))
    arg = 1.0 + 2.0 * uv_sq / denom
    arg = max(arg, 1.0 + 1e-7)
    return float(np.arccosh(arg))

def pick_best_leaf(
    chains,
    *,
    prefer_answer: bool = True,
    tau_group: float = 0.8,          # group soft-vote temperature
    k_nn: int = 5,                   # density kNN
    weights=None,                    # feature weights
):
    """
    Best leaf selection:
      1) build leaf candidates (prefer answered leaves)
      2) compute per-leaf features: Q/N/logp/v_pred/monotonicity/len/density
      3) group by extracted answer (canonicalized)
      4) choose answer group by soft evidence (logsumexp)
      5) choose best leaf inside group
    """
    if weights is None:
        weights = dict(
            zQ=1.0,         # backed-up mean value
            zlogN=0.8,      # search posterior mass proxy
            zlogp=0.4,      # generation likelihood (path evidence)
            zv=0.3,         # leaf v_pred (lightweight)
            zmono=0.3,      # path monotonicity (less oscillation)
            zdens=0.4,      # hyperbolic density/centrality
            zlen=0.2,       # length penalty (prevent rambles)
        )

    # -------- 1) collect (chain, leaf) candidates --------
    items = []
    for ch in chains or []:
        if not ch:
            continue
        leaf = ch[-1]
        comp = leaf.get("completion", "") or ""
        has_ans = bool(re.compile(r"<answer>.*?</answer>", re.S).search(comp))
        ans = _extract_answer_text(comp) if has_ans else ""
        if leaf.get("disabled") or leaf.get("_disabled"):
            continue

        # per-chain v_pred trace
        vs = []
        for st in ch:
            vv = st.get("v_pred", None)
            if vv is None:
                continue
            vs.append(_to_float(vv, default=np.nan))
        vs = [v for v in vs if np.isfinite(v)]
        v_leaf = vs[-1] if vs else _to_float(leaf.get("v_pred", 0.0))

        # monotonicity penalty (how much it *decreases*)
        mono_pen = 0.0
        eps_dec = 1e-4
        for a, b in zip(vs[:-1], vs[1:]):
            mono_pen += max(0.0, (a - b) - eps_dec)

        # length penalty (prefer shorter answers when quality similar)
        cids = leaf.get("completion_ids", None)
        if hasattr(cids, "numel"):
            clen = int(cids.numel())
        elif isinstance(cids, (list, tuple)):
            clen = len(cids)
        else:
            clen = len(comp)
        len_pen = math.log(1.0 + max(0, clen))

        # search stats (if you patched them in dfs)
        Q = _to_float(leaf.get("_Q", leaf.get("state_value", v_leaf)))
        N = float(max(0, int(leaf.get("_N", 0))))
        logN = math.log1p(N)

        # path likelihood evidence
        logp = 0.0
        has_lp = False
        for st in ch:
            if "cum_logprob" in st:
                logp += _to_float(st.get("cum_logprob", 0.0))
                has_lp = True
            elif "p_prior" in st:
                p = max(1e-12, _to_float(st.get("p_prior", 0.0)))
                logp += math.log(p)
                has_lp = True
        if not has_lp:
            logp = 0.0

        # hid for density (optional)
        hid = leaf.get("hid", None)
        hid_vec = None
        if isinstance(hid, (list, tuple)) and len(hid) >= 2:
            try:
                hid_vec = np.asarray(hid, dtype=np.float32)
            except Exception:
                hid_vec = None

        items.append(dict(
            chain=ch,
            leaf=leaf,
            has_ans=has_ans,
            ans=ans,
            Q=Q,
            logN=logN,
            logp=logp,
            v=v_leaf,
            mono=-mono_pen,   # higher is better
            neg_len=-len_pen, # higher is better
            hid=hid_vec,
        ))

    if not items:
        return None

    if prefer_answer:
        ans_items = [it for it in items if it["has_ans"] and it["ans"]]
        if ans_items:
            items = ans_items

    # -------- 2) density / centrality in hyperbolic space (kNN) --------
    # default density=0 if no hid
    dens = np.zeros((len(items),), dtype=np.float32)
    valid = [i for i,it in enumerate(items) if it["hid"] is not None]
    if len(valid) >= 3:
        # compute kNN mean distance (smaller => denser => higher score)
        for i in valid:
            di = []
            ui = items[i]["hid"]
            for j in valid:
                if i == j:
                    continue
                dj = _poincare_dist(ui, items[j]["hid"])
                di.append(dj)
            di.sort()
            k = min(k_nn, len(di))
            if k > 0:
                dens[i] = -float(sum(di[:k]) / k)
    for i,it in enumerate(items):
        it["dens"] = float(dens[i])

    # -------- 3) feature normalization (z-score) --------
    Qz     = _zscore(np.asarray([it["Q"]     for it in items], dtype=np.float32))
    logNz  = _zscore(np.asarray([it["logN"]  for it in items], dtype=np.float32))
    logpz  = _zscore(np.asarray([it["logp"]  for it in items], dtype=np.float32))
    vz     = _zscore(np.asarray([it["v"]     for it in items], dtype=np.float32))
    monoz  = _zscore(np.asarray([it["mono"]  for it in items], dtype=np.float32))
    densz  = _zscore(np.asarray([it["dens"]  for it in items], dtype=np.float32))
    lenz   = _zscore(np.asarray([it["neg_len"] for it in items], dtype=np.float32))

    logits = (
        weights["zQ"]    * Qz +
        weights["zlogN"] * logNz +
        weights["zlogp"] * logpz +
        weights["zv"]    * vz +
        weights["zmono"] * monoz +
        weights["zdens"] * densz +
        weights["zlen"]  * lenz
    )

    for it, lg in zip(items, logits.tolist()):
        it["logit"] = float(lg)

    # -------- 4) group by answer: soft evidence (logsumexp) --------
    from collections import defaultdict
    groups = defaultdict(list)
    for it in items:
        key = it["ans"] if it["ans"] else "__NOANS__"
        groups[key].append(it)

    def _logsumexp(xs):
        xs = np.asarray(xs, dtype=np.float32)
        m = float(xs.max())
        return float(m + np.log(np.exp(xs - m).sum() + 1e-12))

    best_ans, best_ev = None, -1e18
    for ans, lst in groups.items():
        # evidence = logsumexp(logit / tau)
        ev = _logsumexp([it["logit"] / max(1e-6, tau_group) for it in lst])
        # tiny tie-break: prefer bigger support
        ev += 0.05 * math.log1p(len(lst))
        if ev > best_ev:
            best_ev, best_ans = ev, ans

    winner = groups[best_ans]
    # -------- 5) pick representative leaf inside winning answer group --------
    # prefer max logit; tie-break by higher Q, higher logN
    winner.sort(key=lambda it: (it["logit"], it["Q"], it["logN"]), reverse=True)
    return winner[0]["leaf"]
