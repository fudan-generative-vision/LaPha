import os
import copy
import traceback
from collections import defaultdict

from tqdm.auto import tqdm
from typing import Any, Callable, Optional, Union, Mapping, List, Dict, Set
from types import SimpleNamespace
from unittest.mock import patch

import time
import json
import random 
import socket
import inspect

import re
import math
import numpy as np
from statistics import mean

import transformers
from deepspeed import zero

from accelerate.utils import broadcast_object_list, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trainer.agent import MCTSAgent, dump_with_rich, pick_best_leaf
from trainer.mtpo_config import MTPOConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams
from trainer.vllm_client import VLLMClient, _VLLMServerAdapter
from trainer.latent_bank import LatentBank

# What we call a reward function is a callable that takes a list of prompts and completion and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import html as _html
import plotly.graph_objects as go


def _mobius_add_c(x, y, c: float = 1.0, eps: float = 1e-9):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(eps)


def has_answer(st: dict) -> bool:
    """Check whether the completion contains an <answer>...</answer> block."""
    return bool(re.compile(r"<answer>.*?</answer>", re.S).search(st.get("completion", "") or ""))


class LinearValueHead(PreTrainedModel):
    """
    Linear value head with root-centered euclidean translation BEFORE exp0.

    base_lm last_hidden
      -> context mean pooling (pool_mask) -> h0_raw
      -> y_state = Exp0((h0_raw - root_h0)/scale)   # for bank/prune/V_map/L_mono
      -> v_pred  = sigmoid(W h0_raw)                # for MSE / MCTS

    pool_mask rule (same as your current):
        pool_mask = ( (response_mask if provided else attention_mask) OR prompt_mask ) AND attention_mask

    - root_h0: (H,) or (1,H) or (B,H). If provided, apply euclidean centering before exp0.
    - return_h0: if True, also return h0_raw (float32), so caller can cache root_h0.
    """
    _no_split_modules = ["LinearValueHead"]

    def __init__(
        self,
        base_lm: PreTrainedModel,
        curvature: float = 1.0,
        eps: float = 1e-6,
        eps_ball: float = 1e-4,
        *,
        no_head_scale: float = 0.0,          # 0 -> sqrt(H)
        value_activation: str = "sigmoid",   # "sigmoid" or "none"
    ):
        super().__init__(base_lm.config)
        self.base_lm = base_lm

        self.no_head_scale = float(no_head_scale)
        self.c = float(curvature)
        self.eps = float(eps)
        self.eps_ball = float(eps_ball)

        H = int(self.base_lm.config.hidden_size)
        self.value_head = nn.Linear(H, 1, bias=True)

        self.value_activation = str(value_activation).lower()
        if self.value_activation not in ("sigmoid", "none"):
            raise ValueError("value_activation must be 'sigmoid' or 'none'")

        self.post_init()
        p = next(self.base_lm.parameters())
        self.to(device=p.device, dtype=p.dtype)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        if mask_2d.dim() != 2:
            mask_2d = mask_2d.view(mask_2d.size(0), -1)
        m = mask_2d.to(dtype=x.dtype, device=x.device)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _assert_mask_nonempty_for_valid_rows(
        mask_2d: torch.Tensor,
        attention_mask: torch.Tensor,
        name: str,
    ):
        mask_sum = mask_2d.sum(dim=1)
        attn_sum = attention_mask.sum(dim=1)
        bad = (attn_sum > 0) & (mask_sum == 0)
        if bad.any():
            idx = bad.nonzero(as_tuple=False).view(-1)[:8]
            raise RuntimeError(
                f"{name} all-zero on non-empty sequences. "
                f"idx={idx.tolist()}, attn_sum={attn_sum[idx].tolist()}, mask_sum={mask_sum[idx].tolist()}"
            )

    def _exp0_poincare(self, v: torch.Tensor) -> torch.Tensor:
        c = float(max(self.c, 1e-8))
        sqrt_c = math.sqrt(c)
        vnorm = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.tanh(sqrt_c * vnorm) / (sqrt_c * vnorm)
        y = scale * v
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(self.eps)
        max_norm = 1.0 - self.eps_ball
        factor = torch.clamp(max_norm / y_norm, max=1.0)
        return y * factor

    def generate(self, *args, **kwargs):
        return self.base_lm.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        return self.base_lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        return self.base_lm.gradient_checkpointing_disable(**kwargs)

    def forward(
        self,
        input_ids: Optional[torch.IntTensor] = None,
        attention_mask: Optional[torch.IntTensor] = None,
        *,
        value_output: bool = False,
        response_mask: Optional[torch.IntTensor] = None,
        prompt_mask: Optional[torch.IntTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,

        root_h0: Optional[torch.Tensor] = None,   # (H,) or (1,H) or (B,H)  (euclidean pooled root)
        return_h0: bool = False,                  # if True: return (y_state, v_pred, h0_raw)

        **kwargs,
    ):
        if not value_output:
            return self.base_lm(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # ---- last_hidden ----
        if hidden_states is None:
            out = self.base_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]
        else:
            last_hidden = hidden_states

        B, L, H = last_hidden.size()
        dev = last_hidden.device

        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=dev, dtype=torch.long)
        if attention_mask.dim() != 2:
            attention_mask = attention_mask.view(B, L)
        attn = attention_mask.to(device=dev, dtype=torch.long)

        # ---- pool_mask = (response_mask or attn) OR prompt_mask, then AND attn ----
        if response_mask is None:
            pool = attn
        else:
            pool = response_mask
            if pool.dim() != 2:
                pool = pool.view(B, L)
            pool = pool.to(device=dev, dtype=torch.long)

        if prompt_mask is not None:
            pm = prompt_mask
            if pm.dim() != 2:
                pm = pm.view(B, L)
            pm = pm.to(device=dev, dtype=torch.long)
            pool = ((pool > 0) | (pm > 0)).long()

        pool = ((pool > 0) & (attn > 0)).long()
        self._assert_mask_nonempty_for_valid_rows(pool, attn, "pool_mask(context)")

        # =========================================================================
        # h0_raw: context mean pooling (float32 for stability)
        # =========================================================================
        h0_raw = self._masked_mean(last_hidden.to(torch.float32), pool)  # (B,H) float32

        # =========================================================================
        # Euclidean root-centering before exp0
        # =========================================================================
        if root_h0 is not None:
            rh = root_h0
            if not torch.is_tensor(rh):
                rh = torch.as_tensor(rh)
            rh = rh.to(device=dev, dtype=torch.float32)

            if rh.dim() == 1:
                rh = rh.view(1, -1)

            if rh.size(0) == 1:
                rh = rh.expand(B, -1)
            elif rh.size(0) != B:
                raise RuntimeError(
                    f"root_h0 batch mismatch: root_h0={tuple(rh.shape)} vs h0_raw={tuple(h0_raw.shape)}"
                )

            if rh.size(1) != H:
                raise RuntimeError(
                    f"root_h0 hidden mismatch: root_h0={tuple(rh.shape)} vs H={H}"
                )

            h0_centered = h0_raw - rh
        else:
            h0_centered = h0_raw

        # =========================================================================
        # y_state: exp0 in Poincaré ball (root-centered)
        # =========================================================================
        scale = self.no_head_scale
        if scale <= 0.0:
            scale = float(math.sqrt(H))
        y_state = self._exp0_poincare(h0_centered / scale)  # (B,H) float32

        # =========================================================================
        # v_pred: linear head on ORIGINAL h0_raw (NOT centered)
        # =========================================================================
        h0_for_v = h0_raw.to(dtype=self.value_head.weight.dtype)  # match weight dtype (bf16/fp16)
        v_logit = self.value_head(h0_for_v).squeeze(-1)          # (B,)

        if self.value_activation == "sigmoid":
            v_pred = torch.sigmoid(v_logit).to(torch.float32)
        else:
            v_pred = v_logit.to(torch.float32)

        if return_h0:
            return y_state, v_pred, h0_raw  # h0_raw float32 (good for caching as root_h0)
        return y_state, v_pred
    

def _artanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable inverse tanh."""
    x = x.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin: R^D -> Poincaré ball (||x|| < 1).
    We also project to keep points strictly inside the unit ball.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.tanh((c**0.5) * v_norm) / ((c**0.5) * v_norm)
    x = factor * v
    # Project inside the unit ball with a small margin
    x_norm = x.norm(dim=-1, keepdim=True)
    max_norm = 1.0 - 1e-5
    scale = torch.clamp(max_norm / x_norm, max=1.0)
    return x * scale

def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin (inverse of expmap0 for points inside the ball).
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    factor = _artanh((c**0.5) * x_norm) / ((c**0.5) * x_norm)
    return factor * x

def proj_ball(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Project points to inside the Poincaré ball with a margin.
    Enforces ||x|| <= (1-eps)/sqrt(c).
    """
    c = float(max(c, 1e-8))
    max_norm = (1.0 - float(eps)) / math.sqrt(c)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return x * factor

def poincare_dist_stable(x: torch.Tensor, y: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Numerically-stable Poincaré distance (curvature c>0).
    Returns shape (B,).
    """
    c = float(max(c, 1e-8))
    # Expect x,y already float32 for stability
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    d2 = ((x - y) * (x - y)).sum(dim=-1, keepdim=True).clamp_min(0.0)

    denom = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    z = 1.0 + 2.0 * c * d2 / denom
    z = z.clamp_min(1.0 + 1e-7)

    if hasattr(torch, "acosh"):
        d = torch.acosh(z)
    else:
        d = torch.log(z + torch.sqrt(z * z - 1.0))

    # Correct curvature scaling
    return (d / math.sqrt(c)).squeeze(-1)

def poincare_dist_matrix_stable(
    X: torch.Tensor,   # (M,H)
    Z: torch.Tensor,   # (C,H)
    *,
    c: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pairwise Poincaré distances between X and Z: (M,C), curvature-scaled ( /sqrt(c) ).
    """
    X = X.to(torch.float32)
    Z = Z.to(torch.float32)
    c = float(max(c, 1e-8))

    x2 = (X * X).sum(dim=-1, keepdim=True)      # (M,1)
    z2 = (Z * Z).sum(dim=-1, keepdim=True)      # (C,1)
    sq = (x2 + z2.t() - 2.0 * (X @ Z.t())).clamp_min(0.0)  # (M,C)

    one_minus_cx2 = (1.0 - c * x2).clamp_min(eps)  # (M,1)
    one_minus_cz2 = (1.0 - c * z2).clamp_min(eps)  # (C,1)
    denom = (one_minus_cx2 @ one_minus_cz2.t()).clamp_min(eps)  # (M,C)

    arg = 1.0 + 2.0 * c * sq / denom
    arg = arg.clamp_min(1.0 + 1e-7)

    if hasattr(torch, "acosh"):
        dist = torch.acosh(arg)
    else:
        dist = torch.log(arg + torch.sqrt(arg * arg - 1.0))

    return dist / math.sqrt(c)

class RiemannianGradScale(torch.autograd.Function):
    """
    Apply g^{-1} scaling to gradients on the Poincaré ball:
        grad_R = ((1 - c||x||^2)^2 / 4) * grad_E
    This mimics the key stabilizing effect of Riemannian SGD near the boundary.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, c: float, eps: float, gamma: float):
        ctx.save_for_backward(x)
        ctx.c = float(c)
        ctx.eps = float(eps)
        ctx.gamma = float(gamma)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        c = ctx.c
        eps = ctx.eps
        gamma = ctx.gamma

        x2 = (x * x).sum(dim=-1, keepdim=True)
        factor = ((1.0 - c * x2).clamp_min(eps) ** 2) * (gamma / 4.0)
        return grad_out * factor, None, None, None

def _validate_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools_registry: Mapping[str, Callable],
) -> None:
    """
    Validate each tool_call against a tool registry:
    - name must exist in registry
    - arguments must be a dict (after JSON-decoding)
    - arguments keys must match function signature (unless **kwargs present)
    - all required params (without defaults) except 'context' must be present
    """
    import inspect

    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict) or tc.get("type") != "function":
            raise ValueError(f"tool_calls[{i}] is not a function call object")

        func = tc.get("function", {})
        name = func.get("name", None)
        if not name or not isinstance(name, str):
            raise ValueError(f"tool_calls[{i}] missing valid function.name")

        if name not in tools_registry:
            raise ValueError(f"Unknown tool '{name}'")

        args = func.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError(f"tool_calls[{i}].function.arguments must be a JSON object")

        # Signature checks
        fn = tools_registry[name]
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkw:
            unexpected = [k for k in args.keys() if k not in sig.parameters]
            if unexpected:
                raise ValueError(f"Unexpected args for '{name}': {unexpected}")

        required = [
            p.name for p in sig.parameters.values()
            if p.default is inspect._empty
               and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
               and p.name != "context"
        ]
        missing = [k for k in required if k not in args]
        if missing:
            raise ValueError(f"Missing required args for '{name}': {missing}")


def parse_tool_calls(
    content: str,
    *,
    tools_registry: Optional[Mapping[str, Callable]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Parse <tool_call>...</tool_call> blocks. If strict=True and tools_registry is provided,
    additionally validate tool name and arguments against the registry/signature.
    """
    # Tolerate a single missing closing tag (common truncation case)
    if "<tool_call>" in content and "</tool_call>" not in content:
        content = content + "</tool_call>"

    tool_calls: List[Dict[str, Any]] = []
    offset = 0
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    decoder = json.JSONDecoder(strict=False)

    for i, m in enumerate(pattern.finditer(content)):
        if i == 0:
            offset = m.start()

        raw = m.group(1).strip()
        func = decoder.decode(raw)  # may raise JSONDecodeError

        # Normalize "arguments" to a dict
        args = func.get("arguments", {})
        if isinstance(args, str):
            args = decoder.decode(args)  # may raise JSONDecodeError
        func["arguments"] = args

        tool_calls.append({"type": "function", "function": func})

    if tool_calls:
        # Strict validation (optional)
        if strict and tools_registry is not None:
            _validate_tool_calls(tool_calls, tools_registry)

        # The content before the first <tool_call>
        c = content[:offset].strip() if offset > 0 and content[:offset].strip() else ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}

    # No tool calls → return plain assistant content (strip trailing <|im_end|>)
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


class MTPOTrainer(Trainer):
    """
    MTPOTrainer that uses BFS-like expansions for ReAct steps.
    """
    _tag_names = ["trl", "mtpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        agent_cls_list: Optional[List[MCTSAgent]], 
        reward_fns: Union[Callable, PreTrainedModel, str, List[Union[Callable, PreTrainedModel, str]]],
        args: MTPOConfig,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
    ):

        self.agent_cls_list = agent_cls_list
        # Prepare the MTPOConfig if needed
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = MTPOConfig(f"{model_name}-MTPO")
        # tree
        self.depth      = args.depth
        self.breadth    = args.breadth
        # mcts
        self.c_puct      = args.c_puct
        self.num_sim     = args.num_sim
        self.num_pos_sim = args.num_pos_sim
        self.prune_per   = args.prune_per
        # loss
        self.beta       = args.beta
        # pass@k
        self.passk_threshold  = args.passk_threshold
        self.passk_k          = args.passk_k
        # reward
        self.distance_metric = args.distance_metric
        self.distance_alpha = args.distance_alpha
        # record
        self.writer   = SummaryWriter(log_dir=args.output_dir)
        self._metrics = defaultdict(list)
        # latent state bank
        self._hid_bank = None

        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `MTPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `MTPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Reference model
        if is_deepspeed_zero3_enabled() and self.beta > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.name_or_path,
                padding_side="left"
            )
        # Reward functions
        if not isinstance(reward_fns, list):
            reward_fns = [reward_fns]
        for i, reward_func in enumerate(reward_fns):
            if isinstance(reward_func, str):
                reward_fns[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_fns = reward_fns

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_fns)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_fns):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_fns)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations

        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode

        model.warnings_issued["estimate_tokens"] = True
        # Initialize the metrics
        self._metrics = defaultdict(list)

        # Value head
        vh_type = str(getattr(args, "value_head_type", "qwen2")).lower()
        if vh_type in ("linear", "lin", "simple"):
            self.model = LinearValueHead(
                base_lm=model,
                curvature=float(getattr(args, "curvature", 1.0)),
                eps=float(getattr(args, "hyp_eps", 1e-6)),
                eps_ball=float(getattr(args, "hyp_eps_ball", 1e-4)),
                no_head_scale=float(getattr(args, "no_head_scale", 0.0)),
                value_activation=str(getattr(args, "value_activation", "sigmoid")),
            )
        elif vh_type in ("qwen2", "block", "decoder"):
            self.model = Qwen2ValueHead(
                base_lm=model,
                curvature=float(getattr(args, "curvature", 1.0)),
                eps=float(getattr(args, "hyp_eps", 1e-6)),
                eps_ball=float(getattr(args, "hyp_eps_ball", 1e-4)),
                no_head_scale=float(getattr(args, "no_head_scale", 0.0)),  # 0 -> sqrt(H)
                value_activation=str(getattr(args, "value_activation", "sigmoid")),
            )
        else:
            raise ValueError(f"Unknown value_head_type={vh_type!r}. Use 'qwen2' or 'linear'.")

        self.value_acc = 1.0
        super().__init__(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # vLLM can be used in two modes:
            #   • "server": trainer connects to an external vLLM server (which can occupy multiple GPUs)
            #   • "colocate": trainer spawns a colocated vLLM engine (can use TP across a subset/all trainer GPUs)
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            # Always create a SamplingParams object so downstream agent code that checks for attributes (e.g., `.n`)
            # keeps working, regardless of the selected vLLM mode. These fields are also used as defaults by the
            # server adapter when calling the HTTP API.
            if "SamplingParams" in globals():
                self.sampling_params = SamplingParams(
                    temperature=args.temperature, 
                    top_p=args.top_p, 
                    top_k=args.top_k, 
                    logprobs=1, 
                    repetition_penalty=args.repetition_penalty, 
                    max_tokens=self.max_completion_length,
                    # Requesting token-level logprobs from the HTTP server is not supported in the TRL client stub;
                    # keep the field for API compatibility (ignored by server mode).
                )
            else:
                self.sampling_params = None  # Should not happen if vLLM is available

            # Determine an upper bound on model context length to help vLLM planner.
            if self.max_prompt_length is not None and self.max_completion_length is not None:
                self.max_model_len = self.max_prompt_length + self.max_completion_length
            else:
                self.max_model_len = None
            # Keep one flag to avoid redundant weight pushes during grad accumulation.
            self._last_loaded_step = -1
            # Optional knobs used by adapters (fall back to sensible defaults if missing in args)
            self._gen_top_p = getattr(args, "top_p", 1.0)
            self._gen_top_k = getattr(args, "top_k", -1)
            self._gen_min_p = getattr(args, "min_p", 0.0)
            self._gen_rep_penalty = getattr(args, "repetition_penalty", 1.0)
            self._vllm_guided_regex = getattr(args, "vllm_guided_decoding_regex", None)
            self._vllm_generation_kwargs = getattr(args, "generation_kwargs", None)

            if self.vllm_mode == "server":
                # Main process initializes the client and the NCCL broadcast group used for weight sync.
                if self.accelerator.is_main_process:
                    base_url = getattr(args, "vllm_server_base_url", None)
                    if base_url is None:
                        host = getattr(args, "vllm_server_host", "0.0.0.0")
                        port = getattr(args, "vllm_server_port", 8000)
                        base_url = f"http://{host}:{port}"
                    timeout = getattr(args, "vllm_server_timeout", 0.0)
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=timeout)
                    # Bind the client's weight-comm group to this trainer device to broadcast tensors to the server group.
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())
                else:
                    self.vllm_client = None  # Only rank-0 talks to the server directly

                # Build adapter on rank-0 and a no-op placeholder on other ranks. Non-main ranks never call generate().
                self.llm = _VLLMServerAdapter(self.vllm_client, defaults=dict())

                # All processes should stay in sync before training proceeds.
                self.accelerator.wait_for_everyone()

            elif self.vllm_mode == "colocate":
                # Prepare distributed environment variables for the colocated driver if they are not set.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "xxxxx")

                # Determine an upper bound on model context length to help vLLM planner.
                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    self.max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    self.max_model_len = None

                # Spawn a colocated vLLM engine. The external launcher backend allows us to reuse the existing process group.
                tp_size = getattr(self, "vllm_tensor_parallel_size", 1)
                self.llm = LLM(
                    model=model_id,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * max(1, tp_size)
                    * max(1, getattr(self.args, "steps_per_generation", 1)),
                    max_model_len=self.max_model_len,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // max(1, tp_size),
                    # Keep a conservative cap to avoid FP memory checks failing on some vLLM versions.
                    model_impl=getattr(self.args, "vllm_model_impl", "auto"),
                )

                # All processes should stay in sync before training proceeds.
                self.accelerator.wait_for_everyone()

            else:
                raise ValueError(f"Unknown `vllm_mode`: {self.vllm_mode!r}. Use 'server' or 'colocate'.")

        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_fns):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_fns[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.reward_fns.append(self.self_evolving)

    @staticmethod
    def _pack_span_hidden(
        last_hidden: torch.Tensor,          # (B,L,H)
        attention_mask: torch.Tensor,       # (B,L)
        response_mask: Optional[torch.Tensor] = None,  # (B,L) or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract the span tokens (response_mask==1 if provided else attention_mask==1),
        and return padded:
          hidden_span: (B, Ls_max, H)
          am_span:     (B, Ls_max)  (1 for real span tokens)
          pos_span:    (B, Ls_max)  absolute RoPE positions = offset + arange(Ls)
        offset is computed as "#non-pad tokens before the first span token",
        which equals prompt_len when span is completion.
        """
        if attention_mask.dim() != 2:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        attn = attention_mask.to(dtype=torch.long)
        B, L, H = last_hidden.shape
        dev = last_hidden.device

        if response_mask is None:
            span_mask = attn > 0
        else:
            if response_mask.dim() != 2:
                response_mask = response_mask.view(B, L)
            span_mask = (response_mask.to(dev) > 0) & (attn > 0)

        span_int = span_mask.to(dtype=torch.long)
        span_len = span_int.sum(dim=1)  # (B,)

        max_len = int(span_len.max().item()) if B > 0 else 0
        if max_len <= 0:
            # degenerate: return a 1-token dummy to avoid shape errors upstream
            hidden_span = last_hidden.new_zeros((B, 1, H))
            am_span     = attn.new_zeros((B, 1))
            pos_span    = attn.new_zeros((B, 1))
            return hidden_span, am_span, pos_span

        # offset = number of non-pad tokens before the first span token
        has_span = span_len > 0
        first_idx = torch.argmax(span_int, dim=1)  # if no span -> 0, handled by has_span
        attn_cum = attn.cumsum(dim=1)
        idx_minus1 = (first_idx - 1).clamp_min(0)
        prefix_cnt = torch.gather(attn_cum, 1, idx_minus1.unsqueeze(1)).squeeze(1)
        offset = torch.where(has_span & (first_idx > 0), prefix_cnt, torch.zeros_like(prefix_cnt))

        hidden_span = last_hidden.new_zeros((B, max_len, H))
        am_span     = attn.new_zeros((B, max_len))
        pos_span    = attn.new_zeros((B, max_len))

        for i in range(B):
            idx = span_mask[i].nonzero(as_tuple=False).view(-1)
            li = int(idx.numel())
            if li <= 0:
                continue
            hidden_span[i, :li] = last_hidden[i, idx]
            am_span[i, :li] = 1
            pos_span[i, :li] = offset[i] + torch.arange(li, device=dev, dtype=torch.long)

        return hidden_span, am_span, pos_span

    def _set_signature_columns_if_needed(self):
        """
        Override so we only keep the columns we use in compute_loss (which is basically 'prompt', 'answer', etc.).
        """
        if self._signature_columns is None:
            self._signature_columns = ["question", "answer", "support_material_path"]

    def _prepare_inputs(self, inputs: dict) -> dict:
        # No default tokenization/cuda move is needed from Trainer, 
        # because we do custom logic in compute_loss
        return inputs

    def _sync_vllm_weights_if_needed(self, model):
        """
        Synchronize policy weights to vLLM when needed.

        Key design changes to avoid NCCL watchdog timeouts with ZeRO-3:
        1) Use per-parameter GatheredParameters instead of gathering the whole model at once.
        2) Perform the remote vLLM update *inside* each per-parameter gather context so that
        all ranks remain lock-step on the same parameter and no collective gets out of order.
        """
        if not getattr(self, "use_vllm", False):
            return

        step = int(self.state.global_step)
        if getattr(self, "_last_loaded_step", None) == step:
            return

        # Pre-stage barrier to avoid de-synchronization across ranks.
        self.accelerator.wait_for_everyone()

        # Unwrap and get the base language model (strip the value head wrapper).
        base_mod = self.accelerator.unwrap_model(model)
        if hasattr(base_mod, "module"):  # e.g., DeepSpeedEngine
            base_mod = base_mod.module
        base_lm = getattr(base_mod, "base_lm", base_mod)

        is_zero3 = False
        try:
            is_zero3 = is_deepspeed_zero3_enabled()
        except Exception:
            is_zero3 = False

        # SERVER MODE: push each param to remote vLLM (GPU-to-GPU via PyNcclCommunicator)
        if self.vllm_mode == "server":
            for name, param in base_lm.named_parameters():
                if is_zero3:
                    # Gather only this parameter on all ranks; rank-0 gets full weights.
                    with zero.GatheredParameters([param], modifier_rank=0):
                        if self.accelerator.is_main_process:
                            # Keep the tensor on GPU; the communicator expects device tensors.
                            weights = param.data
                            # Remote update happens *inside* the gather ctx to keep ranks in lock-step on this param.
                            self.vllm_client.update_named_param(name, weights)
                        # All other ranks do nothing but still participate in the same gather/scope.
                else:
                    # Non-ZeRO case: no param sharding; rank-0 can directly push.
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

            # Reset the prefix cache after all layers have been updated.
            if self.accelerator.is_main_process:
                self.vllm_client.reset_prefix_cache()

        # COLOCATE MODE: load weights into colocated vLLM engine per parameter
        elif self.vllm_mode == "colocate":
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            for name, param in base_lm.named_parameters():
                if is_zero3:
                    with zero.GatheredParameters([param], modifier_rank=0):
                        if self.accelerator.is_main_process:
                            # Keep on GPU for fast handover. vLLM loader accepts device tensors.
                            llm_model.load_weights([(name, param.data)])
                else:
                    if self.accelerator.is_main_process:
                        llm_model.load_weights([(name, param.data)])

            if self.accelerator.is_main_process:
                self.llm.reset_prefix_cache()

        else:
            raise ValueError(f"Unknown `vllm_mode`: {self.vllm_mode!r}.")

        # Post-stage synchronization and bookkeeping.
        self.accelerator.wait_for_everyone()
        self._last_loaded_step = step

    def _value_forward_server(self):
        """
        Mirror ranks serve rank0's distributed value_fn.
        Must match rank0's scatter/broadcast/all_gather order EXACTLY.
        """
        try:
            need_mirror = dist.is_available() and dist.is_initialized()
        except Exception:
            need_mirror = False
        if (not need_mirror) or self.accelerator.is_main_process:
            return

        dev = self.accelerator.device
        world_size = dist.get_world_size()

        while True:
            pkt = [None]
            broadcast_object_list(pkt, from_process=0)
            msg = pkt[0]
            tag = (msg or {}).get("tag", None)

            if tag == "STOP":
                break

            if tag == "VALUE_SCATTER":
                B_pad = int(msg["B_pad"]); L = int(msg["L"]); chunk = int(msg["chunk"])
                has_rm = bool(msg.get("has_response_mask", False))
                has_pm = bool(msg.get("has_prompt_mask", False))

                # root_h0 + need_h0
                has_root_h0 = bool(msg.get("has_root_h0", False))
                root_dim    = int(msg.get("root_h0_dim", 0))
                need_h0     = bool(msg.get("need_h0", False))

                # 0) MUST match rank0 order: broadcast root_h0 first (if any)
                root_h0_dev = None
                if has_root_h0:
                    if root_dim <= 0:
                        raise RuntimeError(f"[rank {self.accelerator.process_index}] invalid root_h0_dim={root_dim}")
                    root_h0_dev = torch.empty((root_dim,), device=dev, dtype=torch.float32)
                    dist.broadcast(root_h0_dev, src=0)

                # 1) scatter ids/masks
                recv_ids  = torch.empty((chunk, L), dtype=torch.long, device=dev)
                recv_am   = torch.empty((chunk, L), dtype=torch.long, device=dev)
                dist.scatter(recv_ids, scatter_list=None, src=0)
                dist.scatter(recv_am,  scatter_list=None, src=0)

                if has_rm:
                    recv_rm = torch.empty((chunk, L), dtype=torch.long, device=dev)
                    dist.scatter(recv_rm, scatter_list=None, src=0)
                else:
                    recv_rm = None

                if has_pm:
                    recv_pm = torch.empty((chunk, L), dtype=torch.long, device=dev)
                    dist.scatter(recv_pm, scatter_list=None, src=0)
                else:
                    recv_pm = None

                # 2) forward
                with torch.no_grad():
                    out = self.model.base_lm(
                        input_ids=recv_ids,
                        attention_mask=recv_am,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_hidden = out.hidden_states[-1]

                    out2 = self.model(
                        input_ids=recv_ids,
                        attention_mask=recv_am,
                        hidden_states=last_hidden,
                        response_mask=recv_rm,
                        prompt_mask=recv_pm,
                        root_h0=root_h0_dev,
                        return_h0=bool(need_h0),
                        value_output=True,
                    )

                    if need_h0:
                        y_state, v_pred, h0_local = out2
                    else:
                        y_state, v_pred = out2
                        h0_local = None

                    # 3) MUST match rank0 order: all_gather y, v, (optional) h0
                    h_list = [torch.empty_like(y_state) for _ in range(world_size)]
                    v_list = [torch.empty_like(v_pred)  for _ in range(world_size)]
                    dist.all_gather(h_list, y_state)
                    dist.all_gather(v_list, v_pred)

                    if need_h0:
                        h0_list = [torch.empty_like(h0_local) for _ in range(world_size)]
                        dist.all_gather(h0_list, h0_local)

                # cleanup
                del recv_ids, recv_am, recv_rm, recv_pm
                del out, last_hidden, out2, y_state, v_pred, h0_local
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                continue

            raise RuntimeError(f"[rank {self.accelerator.process_index}] Unexpected header tag={tag!r}")

        self.accelerator.wait_for_everyone()

    def value_fn(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        
        root_h0: Optional[torch.Tensor] = None,
        return_h0: bool = False,
    ):
        """
        Value function for MCTS:
        - y_state: Poincaré point of current state
        - v_pred : scalar value.
        Returns (y_state_cpu[B,H], v_cpu[B])
        """
        assert self.accelerator.is_main_process, "value_fn must be called on main process only."

        # ---- stage to CPU long (keep your convention) ----
        if not torch.is_tensor(input_ids):
            ids_t = torch.tensor(input_ids, device="cpu", dtype=torch.long)
        else:
            ids_t = input_ids.to("cpu", dtype=torch.long, non_blocking=True)

        if not torch.is_tensor(attention_mask):
            am_t = torch.tensor(attention_mask, device="cpu", dtype=torch.long)
        else:
            am_t = attention_mask.to("cpu", dtype=torch.long, non_blocking=True)

        rm_t = None
        if response_mask is not None:
            rm_t = response_mask.to("cpu", dtype=torch.long, non_blocking=True) if torch.is_tensor(response_mask) \
                else torch.tensor(response_mask, device="cpu", dtype=torch.long)

        pm_t = None
        if prompt_mask is not None:
            pm_t = prompt_mask.to("cpu", dtype=torch.long, non_blocking=True) if torch.is_tensor(prompt_mask) \
                else torch.tensor(prompt_mask, device="cpu", dtype=torch.long)

        B, L = int(ids_t.size(0)), int(ids_t.size(1))
        pad_id = int(self.processing_class.pad_token_id or 0)

        # shape checks
        if am_t.dim() != 2 or am_t.size(0) != B or am_t.size(1) != L:
            raise ValueError(f"attention_mask must be (B,L). Got {tuple(am_t.shape)} vs ({B},{L})")
        if rm_t is not None and (rm_t.dim() != 2 or rm_t.size(0) != B or rm_t.size(1) != L):
            raise ValueError(f"response_mask must be (B,L). Got {tuple(rm_t.shape)} vs ({B},{L})")
        if pm_t is not None and (pm_t.dim() != 2 or pm_t.size(0) != B or pm_t.size(1) != L):
            raise ValueError(f"prompt_mask must be (B,L). Got {tuple(pm_t.shape)} vs ({B},{L})")

        # dist?
        use_dist = False
        try:
            use_dist = (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        except Exception:
            use_dist = False

        c_hyp = float(getattr(self.model, "c", 1.0))
        c_hyp = max(c_hyp, 1e-8)

        # -------------------------
        # Single-process path
        # -------------------------
        if not use_dist:
            dev = self.accelerator.device
            ids = ids_t.to(dev, non_blocking=True)
            am  = am_t.to(dev,  non_blocking=True)
            rm  = rm_t.to(dev,  non_blocking=True) if rm_t is not None else None
            pm  = pm_t.to(dev,  non_blocking=True) if pm_t is not None else None

            with torch.no_grad():
                out = self.model.base_lm(
                    input_ids=ids,
                    attention_mask=am,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = out.hidden_states[-1]

                root_h0_dev = None
                if root_h0 is not None:
                    if torch.is_tensor(root_h0):
                        root_h0_dev = root_h0.detach()
                    else:
                        root_h0_dev = torch.as_tensor(root_h0)
                    root_h0_dev = root_h0_dev.to(dev, dtype=torch.float32).view(-1)  # (H,)

                out2 = self.model(
                    input_ids=ids,
                    attention_mask=am,
                    hidden_states=last_hidden,
                    response_mask=rm,
                    prompt_mask=pm,
                    root_h0=root_h0_dev,
                    return_h0=bool(return_h0),
                    value_output=True,
                )

                if return_h0:
                    y_state, v, h0 = out2
                    return y_state.detach().to("cpu"), v.detach().to("cpu"), h0.detach().to("cpu")
                else:
                    y_state, v = out2
                    return y_state.detach().to("cpu"), v.detach().to("cpu")

        # -------------------------
        # Distributed scatter path
        # -------------------------
        dev = self.accelerator.device
        world_size = dist.get_world_size()
        chunk = int(math.ceil(B / world_size))
        B_pad = chunk * world_size
        
        # pad rows
        if B_pad != B:
            pad_rows = B_pad - B
            pad_ids = torch.full((pad_rows, L), fill_value=pad_id, dtype=torch.long)
            pad_am  = torch.zeros((pad_rows, L), dtype=torch.long)
            ids_t = torch.cat([ids_t, pad_ids], dim=0)
            am_t  = torch.cat([am_t,  pad_am],  dim=0)
            if rm_t is not None:
                pad_rm = torch.zeros((pad_rows, L), dtype=torch.long)
                rm_t = torch.cat([rm_t, pad_rm], dim=0)
            if pm_t is not None:
                pad_pm = torch.zeros((pad_rows, L), dtype=torch.long)
                pm_t = torch.cat([pm_t, pad_pm], dim=0)

        # chunk lists on device for scatter_list (rank0 only)
        ids_chunks, am_chunks = [], []
        rm_chunks = [] if rm_t is not None else None
        pm_chunks = [] if pm_t is not None else None

        for r in range(world_size):
            s = r * chunk
            e = s + chunk
            ids_chunks.append(ids_t[s:e].to(dev, non_blocking=True))
            am_chunks.append(am_t[s:e].to(dev,  non_blocking=True))
            if rm_chunks is not None:
                rm_chunks.append(rm_t[s:e].to(dev, non_blocking=True))
            if pm_chunks is not None:
                pm_chunks.append(pm_t[s:e].to(dev, non_blocking=True))
        
        root_h0_t = None
        if root_h0 is not None:
            if torch.is_tensor(root_h0):
                root_h0_t = root_h0.detach().to("cpu", dtype=torch.float32).view(-1)
            else:
                root_h0_t = torch.as_tensor(root_h0, dtype=torch.float32).view(-1)

        # broadcast
        header = [{
            "tag": "VALUE_SCATTER",
            "B_pad": B_pad,
            "L": L,
            "chunk": chunk,
            "has_response_mask": bool(rm_t is not None),
            "has_prompt_mask":   bool(pm_t is not None),

            "has_root_h0": bool(root_h0_t is not None),
            "root_h0_dim": int(root_h0_t.numel()) if root_h0_t is not None else 0,
            "need_h0": bool(return_h0),
        }]
        broadcast_object_list(header, from_process=0)

        root_h0_dev = None
        if root_h0_t is not None:
            root_h0_dev = root_h0_t.to(dev, non_blocking=True)  # (H,) float32
            dist.broadcast(root_h0_dev, src=0)

        # scatter
        recv_ids  = torch.empty_like(ids_chunks[0])
        recv_am   = torch.empty_like(am_chunks[0])
        dist.scatter(recv_ids, scatter_list=ids_chunks, src=0)
        dist.scatter(recv_am,  scatter_list=am_chunks,  src=0)

        if rm_chunks is not None:
            recv_rm = torch.empty_like(rm_chunks[0])
            dist.scatter(recv_rm, scatter_list=rm_chunks, src=0)
        else:
            recv_rm = None

        if pm_chunks is not None:
            recv_pm = torch.empty_like(pm_chunks[0])
            dist.scatter(recv_pm, scatter_list=pm_chunks, src=0)
        else:
            recv_pm = None

        with torch.no_grad():
            out = self.model.base_lm(
                input_ids=recv_ids,
                attention_mask=recv_am,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]
            
            out2 = self.model(
                input_ids=recv_ids,
                attention_mask=recv_am,
                hidden_states=last_hidden,
                response_mask=recv_rm,
                prompt_mask=recv_pm,
                root_h0=root_h0_dev,
                return_h0=bool(return_h0),
                value_output=True,
            )
            if return_h0:
                y_state_local, v_local, h0_local = out2
            else:
                y_state_local, v_local = out2
                h0_local = None
            
        h_list = [torch.empty_like(y_state_local) for _ in range(world_size)]
        v_list = [torch.empty_like(v_local)       for _ in range(world_size)]
        dist.all_gather(h_list, y_state_local)
        dist.all_gather(v_list, v_local)

        if return_h0:
            h0_list = [torch.empty_like(h0_local) for _ in range(world_size)]
            dist.all_gather(h0_list, h0_local)
            H0_cat = torch.cat(h0_list, dim=0)[:B].detach().to("cpu")

        H_cat = torch.cat(h_list, dim=0)[:B].detach().to("cpu")
        V_cat = torch.cat(v_list, dim=0)[:B].detach().to("cpu")

        if return_h0:
            return H_cat, V_cat, H0_cat
        return H_cat, V_cat


    def _bank_add_vec(self, bank, y_cpu: torch.Tensor) -> int:
        """
        Add one hyperbolic vector to LatentBank and return its index.
        y_cpu: (Dp,) on CPU.
        """
        dev = getattr(bank, "device", self.accelerator.device)
        dt  = getattr(bank, "dtype", torch.bfloat16)

        y = y_cpu
        if y.dim() == 2 and y.size(0) == 1:
            y = y[0]
        y = y.to(device=dev, dtype=dt)

        # Try common bank APIs
        if hasattr(bank, "add"):
            try:
                return int(bank.add(y))
            except Exception:
                return int(bank.add(y.unsqueeze(0)))
        if hasattr(bank, "append"):
            try:
                return int(bank.append(y))
            except Exception:
                return int(bank.append(y.unsqueeze(0)))
        if hasattr(bank, "push"):
            try:
                return int(bank.push(y))
            except Exception:
                return int(bank.push(y.unsqueeze(0)))

        raise AttributeError("LatentBank needs an add/append/push method that returns an index.")

    def _ensure_hid_idx_coverage(
        self,
        chains: list[list[dict]],
        bank,
        *,
        root_step: dict | None = None,
        batch_size: int = 32,
    ):
        """
        Ensure EVERY node in `chains` (and optionally `root_step`) has st["hid_idx"] in `bank`.
        Embedding is computed via value_fn with response_mask pooling on completion span.
        """
        tok = self.processing_class
        pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
        eos_id = int(getattr(tok, "eos_token_id", pad_id) or pad_id)
        maxP   = int(getattr(self, "max_prompt_length", 0) or 0)

        def _tolist(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().to("cpu").view(-1).tolist()
            return list(x)

        # items: (step_dict, full_ids[list[int]], response_mask[list[int]], prompt_mask[list[int]])
        items = []

        # root_step
        if root_step is not None and root_step.get("hid_idx", None) is None:
            p_list = _tolist(root_step.get("prompt_ids", None))
            if p_list:
                if maxP > 0:
                    p_list = p_list[-maxP:]
                ids = p_list
                rm  = [1] * len(ids)  # pool prompt as state
                pm  = [1] * len(ids)  # prompt span
                items.append((root_step, ids, rm, pm))

        # 2) every step node (pool over completion tokens)
        seen = set()
        for ch in chains:
            for st in ch:
                sid = id(st)
                if sid in seen:
                    continue
                seen.add(sid)
                if st.get("hid_idx", None) is not None:
                    continue

                p_list = _tolist(st.get("prompt_ids", None))
                c_list = _tolist(st.get("completion_ids", None))
                if not p_list or not c_list:
                    continue

                if maxP > 0:
                    p_list = p_list[-maxP:]

                cm = [1] * len(c_list)
                if eos_id in c_list:
                    j = c_list.index(eos_id)
                    for k in range(j + 1, len(cm)):
                        cm[k] = 0

                full_ids = p_list + c_list
                rm = ([0] * len(p_list)) + cm
                pm = ([1] * len(p_list)) + ([0] * len(c_list))
                
                max_len = self.max_model_len
                if max_len > 0 and len(full_ids) > max_len:
                    start = len(full_ids) - max_len
                    full_ids = full_ids[start:]
                    rm = rm[start:]
                    pm = pm[start:]                

                items.append((st, full_ids, rm, pm))
                
        if not items:
            return
        
        # Batch value_fn calls
        for s in range(0, len(items), batch_size):
            batch = items[s : s + batch_size]
            B = len(batch)
            Lmax = max(len(ids) for _, ids, _, _ in batch)

            ids_t = torch.full((B, Lmax), pad_id, dtype=torch.long)   # CPU
            rm_t  = torch.zeros((B, Lmax), dtype=torch.long)          # CPU
            pm_t  = torch.zeros((B, Lmax), dtype=torch.long)          # CPU

            for i, (_, ids, rm, pm) in enumerate(batch):
                L = len(ids)
                ids_t[i, :L] = torch.tensor(ids, dtype=torch.long)
                rm_t[i, :L]  = torch.tensor(rm,  dtype=torch.long)
                pm_t[i, :L]  = torch.tensor(pm,  dtype=torch.long)

            am_t = (ids_t != pad_id).long()

            root_h0 = None
            if root_step is not None:
                rh = root_step.get("root_h0", None)
                if rh is not None:
                    root_h0 = rh.detach().to("cpu", dtype=torch.float32).view(-1) if torch.is_tensor(rh) \
                            else torch.as_tensor(rh, dtype=torch.float32).view(-1)

            y_cpu, _ = self.value_fn(
                input_ids=ids_t,
                attention_mask=am_t,
                response_mask=rm_t,
                prompt_mask=pm_t,
                root_h0=root_h0,
                return_h0=False,
            )

            for i, (st, _, _, _) in enumerate(batch):
                idx = self._bank_add_vec(bank, y_cpu[i])
                st["hid_idx"] = idx

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("MTPOTrainer does not support `return_outputs=True`.")

        hostname = socket.gethostname().split(".")[0]
        dbg_enabled = bool(getattr(self.args, "debug_print", True))

        def _now() -> str:
            return time.strftime("%H:%M:%S")

        def _cuda_mem() -> str:
            try:
                dev = self.accelerator.device
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    alloc = torch.cuda.memory_allocated() / 1e9
                    resv = torch.cuda.memory_reserved() / 1e9
                    return f"cuda{torch.cuda.current_device()}: alloc={alloc:.2f}G reserved={resv:.2f}G"
                return "cpu"
            except Exception:
                return "n/a"

        def _p(msg: str):
            if not dbg_enabled:
                return
            rid = getattr(self.accelerator, "process_index", -1)
            step = getattr(self.state, "global_step", -1)
            print(f"[{_now()}][{hostname}][rank{rid}][step{step}] {msg} | {_cuda_mem()}",
                flush=True)

        device = self.accelerator.device
        pad_token = int(self.processing_class.pad_token_id)
        eos_id = int(self.processing_class.eos_token_id)

        # =========================================================================
        # 0) Synchronize weights between HF model and vLLM (if using vLLM)
        # =========================================================================
        t0 = time.perf_counter()
        _p("sync_vllm_weights_if_needed: start")
        self._sync_vllm_weights_if_needed(model)
        _p(f"sync_vllm_weights_if_needed: done in {time.perf_counter() - t0:.3f}s")

        # =========================================================================
        # 1) Rank0: MCTS rollout + reward computation + chain sampling
        # =========================================================================
        if self.accelerator.is_main_process:
            t_rollout_all = time.perf_counter()
            
            avgAcc_list: list[float] = []
            passAt_1_list: list[float] = []
            roots_meta: list[dict] = []
            step_samples: list[dict] = []
            tree_ground_truth: list[object] = []

            self.num_groups = getattr(self, "num_groups", 8)
            # Value_loss training set control:
            #   num_trees = -1 -> Train only on step_samples (preserves original behavior)
            #   num_trees != -1 -> Train on all nodes of the first num_trees "non-zero trees"
            num_trees_cfg = int(getattr(self.args, "num_trees", -1))
            mse_nodes: list[dict] = []
            mse_tree_cnt = 0

            eps_reward = 1e-12
            eps_vt = 1e-8

            global_group_count = 0
            early_stop = False

            def _best_var_window_constrained(vals: np.ndarray, ok_mask: np.ndarray, k: int, eps_pos: float = 1e-12):
                n = int(vals.shape[0])
                if k <= 1 or k > n:
                    return None, float("-inf")
                ps = np.cumsum(vals, dtype=np.float64)
                ps2 = np.cumsum(vals * vals, dtype=np.float64)
                psc = np.cumsum(ok_mask.astype(np.int32))
                psp = np.cumsum((vals > eps_pos).astype(np.int32))

                def rsum(prefix, s, e):
                    return float(prefix[e - 1] - (prefix[s - 1] if s > 0 else 0.0))

                best_var, best_s = float("-inf"), None
                for s in range(0, n - k + 1):
                    e = s + k
                    cnt_ok = int(psc[e - 1] - (psc[s - 1] if s > 0 else 0))
                    cnt_pos = int(psp[e - 1] - (psp[s - 1] if s > 0 else 0))
                    if cnt_ok <= 0 or cnt_pos <= 0:
                        continue
                    S = rsum(ps, s, e)
                    SS = rsum(ps2, s, e)
                    var_unbiased = (SS - (S * S) / k) / (k - 1)
                    if var_unbiased > best_var + 1e-12:
                        best_var, best_s = var_unbiased, s
                return (best_s, best_var) if best_s is not None else (None, float("-inf"))

            try:
                _p("MCTS loop: start")
                for idx, inp in enumerate(inputs):
                    if global_group_count >= self.num_groups:
                        _p(f"MCTS: reached num_groups={self.num_groups}, stop at idx={idx}.")
                        break

                    question = inp["question"]
                    ground_truth = inp["ground_truth"]
                    support_material_path = inp["support_material_path"]
                    cot = inp.get("cot", None)

                    self.question = question
                    self.cot = cot

                    hid_bank = LatentBank(
                        device=self.accelerator.device,
                        dtype=torch.bfloat16,
                        store_cpu_copy=True,
                        normalize=False,
                    )

                    agent = random.choice(self.agent_cls_list)(
                        tokenizer=self.processing_class,
                        depth=self.depth,
                        breadth=self.breadth,
                        output_dir=self.args.output_dir,
                        llm=self.llm,
                        sampling_params=self.sampling_params,
                        max_model_len=self.max_model_len,
                        value_fn=self.value_fn,
                        reward_fns=self.reward_fns,
                        c_puct=self.c_puct,
                        value_trust=self.value_acc,
                        num_sim=self.num_sim,
                        prune_per=self.prune_per,
                        num_pos_sim=self.num_pos_sim,
                        passk_threshold=self.passk_threshold,
                    )
                    agent.hid_bank = hid_bank

                    expansions = agent.search(
                        question=question,
                        support_material_path=support_material_path,
                        ground_truth=ground_truth,
                        cot=None,
                    )

                    self._ensure_hid_idx_coverage(
                        expansions,
                        hid_bank,
                        root_step=agent._root_step,
                        batch_size=2,
                    )

                    self._hid_bank = hid_bank
                    tree_id = len(tree_ground_truth)
                    avgAcc, passAt_1, _ = self.compute_action_rewards(
                        chains=expansions,
                        reward_fns=self.reward_fns,
                        ground_truth=ground_truth,
                        root_step=agent._root_step,
                        tree_id=tree_id,
                        cot=cot,
                    )
                    self._hid_bank = None

                    avgAcc_list.append(avgAcc)
                    passAt_1_list.append(passAt_1)
                    tree_ground_truth.append(ground_truth)

                    # First, determine if the tree is an "all-zero signal tree" (if all 0s, neither the policy nor the MSE will be trained).
                    has_sig = any(abs(float(st.get("v_target", 0.0))) > eps_vt for ch in expansions for st in ch)
                    if not has_sig:
                        _p(f"MCTS[{idx}]: no v_target signal (all-zero tree), skip tree.")
                        roots_meta.append({"tree_id": tree_id, "prompt_ids": []})
                        continue

                    # num_chains deprecated: all chains are included; chain_id uses chain_idx from expansions
                    # To handle "shared/duplicate nodes": prompt_key -> the chain_idx of its first occurrence
                    prompt_key_to_chain_id: dict[tuple, int] = {}
                    for chain_idx, chain in enumerate(expansions):
                        for st in chain:
                            p_ids = st.get("prompt_ids", None)
                            if p_ids is None:
                                continue
                            if torch.is_tensor(p_ids):
                                p_list = p_ids.detach().to("cpu").view(-1).tolist()
                            else:
                                p_list = list(p_ids)
                            if not p_list:
                                continue
                            key = tuple(p_list[-self.max_prompt_length:])
                            if key not in prompt_key_to_chain_id:
                                prompt_key_to_chain_id[key] = int(chain_idx)

                    # dedup by step object id
                    local_samples: list[dict] = []
                    seen_step_ids: set[int] = set()

                    for chain_idx, chain in enumerate(expansions):
                        for st in chain:
                            sid = id(st)
                            if sid in seen_step_ids:
                                continue

                            p_ids = st.get("prompt_ids", None)
                            c_ids = st.get("completion_ids", None)
                            if p_ids is None or c_ids is None:
                                continue

                            if torch.is_tensor(p_ids):
                                p_list = p_ids.detach().to("cpu").view(-1).tolist()
                            else:
                                p_list = list(p_ids)

                            if torch.is_tensor(c_ids):
                                c_list = c_ids.detach().to("cpu").view(-1).tolist()
                            else:
                                c_list = list(c_ids)

                            if not p_list or not c_list:
                                continue

                            key = tuple(p_list[-self.max_prompt_length:])
                            cid = int(prompt_key_to_chain_id.get(key, chain_idx))

                            sample = dict(
                                prompt_ids     = p_list[-self.max_prompt_length:],
                                chain_id       = int(prompt_key_to_chain_id[key]),
                                tree_id        =                          tree_id,
                                completion_ids =                           c_list,
                                state_value    =         float(st["state_value"]),
                                reward         =              float(st["reward"]),
                                is_leaf        =              bool(st["is_leaf"]),
                                depth          =         int(st["current_depth"]),
                                is_correct     =           bool(st["is_correct"]),
                                on_path        =              bool(st["on_path"]),
                                v_target       =            float(st["v_target"]),
                                v_pred         =              float(st["v_pred"]),
                                has_answer     =             bool(has_answer(st)),
                            )
                            local_samples.append(sample)
                            seen_step_ids.add(sid)

                    if not local_samples:
                        roots_meta.append({"tree_id": tree_id, "prompt_ids": []})
                        continue

                    if (int(num_trees_cfg) != -1) and (mse_tree_cnt < int(num_trees_cfg)):
                        for s0 in local_samples:
                            mse_nodes.append(
                                dict(
                                    prompt_ids=list(map(int, s0["prompt_ids"])),
                                    completion_ids=list(map(int, s0["completion_ids"])),
                                    v_target=float(s0.get("v_target", 0.0)),
                                )
                            )
                        mse_tree_cnt += 1

                    # skip: Trees with a height of avgAcc will not be included in step_samples.
                    if avgAcc >= 0.8:
                        _p(f"MCTS[{idx}]: avgAcc >= 0.8, skip tree for training stability.")
                        roots_meta.append({"tree_id": tree_id, "prompt_ids": []})
                        continue

                    # Root prompt IDs: Only required for trees participating in step_samples (for the y_root of L_mono)
                    root_step = agent._root_step
                    r_ids = root_step["prompt_ids"]
                    if torch.is_tensor(r_ids):
                        r_list = r_ids.detach().to("cpu").view(-1).tolist()
                    else:
                        r_list = list(r_ids)
                    r_list = r_list[-self.max_prompt_length:]
                    roots_meta.append({"tree_id": tree_id, "prompt_ids": r_list})

                    _p(f"MCTS[{idx}]: Collecting...")

                    buckets: dict[tuple, list[dict]] = defaultdict(list)
                    for s in local_samples:
                        key = tuple(int(t) for t in s["prompt_ids"])
                        buckets[key].append(s)

                    tree_group_count = 0
                    for key, samples in buckets.items():
                        _p(f"        {global_group_count}/{self.num_groups}...")
                        if global_group_count >= self.num_groups:
                            early_stop = True
                            break
                        
                        if tree_group_count >= 2:
                            break

                        if self.breadth > 0 and len(samples) < self.breadth:
                            _p("        len(samples) < self.breadth")
                            continue

                        r_vals = [float(s["reward"]) for s in samples]
                        if (max(r_vals) - min(r_vals)) <= eps_reward:
                            _p("        reward range too small")
                            continue

                        vt_vals = [float(s["v_target"]) for s in samples]
                        if max(vt_vals) <= eps_vt:
                            _p("        v_target max too small")
                            continue

                        ss = sorted(samples, key=lambda s: float(s["reward"]), reverse=True)
                        vals_np = np.asarray([float(s["reward"]) for s in ss], dtype=np.float32)
                        ok_mask = np.ones_like(vals_np, dtype=bool)

                        start_idx, _ = _best_var_window_constrained(vals_np, ok_mask, self.breadth)
                        chosen = ss[:self.breadth] if start_idx is None else ss[start_idx:start_idx + self.breadth]

                        step_samples.extend(chosen)
                        global_group_count += 1
                        tree_group_count += 1

                    if early_stop:
                        _p(
                            f"MCTS loop: early stop after tree_id={tree_id}, "
                            f"global_group_count={global_group_count}, steps={len(step_samples)}"
                        )
                        break

                _p(
                    f"MCTS loop: done in {time.perf_counter() - t_rollout_all:.3f}s | "
                    f"trees={len(tree_ground_truth)} | groups={global_group_count} | steps={len(step_samples)}"
                )

            finally:
                _p("broadcast STOP to workers")
                try:
                    broadcast_object_list([{"tag": "STOP"}], from_process=0)
                except Exception as e:
                    _p(f"STOP broadcast failed: {type(e).__name__}: {e}")
                self.accelerator.wait_for_everyone()

            batch_avgAcc = float(mean(avgAcc_list)) if avgAcc_list else 0.0
            batch_passAt1 = float(mean(passAt_1_list)) if passAt_1_list else 0.0
            self._metrics["avgAcc"].append(batch_avgAcc)
            self._metrics["passAt_1"].append(batch_passAt1)
            if hasattr(self, "writer"):
                self.writer.add_scalar("avgAcc", batch_avgAcc, self.state.global_step)
                self.writer.add_scalar("pass@1", batch_passAt1, self.state.global_step)

            wrapper = [{
                "tag": "STEPS",
                "payload": step_samples,
                "num_trees": int(num_trees_cfg),
                "mse_payload": (mse_nodes if int(num_trees_cfg) != -1 else None),
            }]
            proj_wrapper = [{"tag": "PROJ", "roots": roots_meta}]

            
            # save training samples
            try:
                dump_root = os.path.join(self.args.output_dir, "train", f"step-{self.state.global_step}")
                os.makedirs(dump_root, exist_ok=True)

                prompt2gid = {}
                gid_counts = defaultdict(int)

                for s in step_samples:
                    key = tuple(int(t) for t in s["prompt_ids"])
                    if key not in prompt2gid:
                        prompt2gid[key] = len(prompt2gid)
                    gid = int(prompt2gid[key])

                    local_i = int(gid_counts[gid])
                    gid_counts[gid] = local_i + 1

                    group_dir = os.path.join(dump_root, f"group-{gid}")
                    os.makedirs(group_dir, exist_ok=True)

                    p_ids_list = list(map(int, s["prompt_ids"]))
                    c_ids_list = list(map(int, s["completion_ids"]))
                    p_ids_t = torch.tensor(p_ids_list, dtype=torch.long)
                    c_ids_t = torch.tensor(c_ids_list, dtype=torch.long)

                    prompt_txt = self.processing_class.decode(p_ids_list, skip_special_tokens=False)
                    comp_txt = self.processing_class.decode(c_ids_list, skip_special_tokens=False)

                    t_id = int(s["tree_id"])
                    gt = tree_ground_truth[t_id] if (0 <= t_id < len(tree_ground_truth)) else None

                    dump_step = {
                        "reward": float(s["reward"]),
                        "state_value": float(s["state_value"]),
                        "prompt_ids": p_ids_t,
                        "completion_ids": c_ids_t,
                        "prompt": prompt_txt,
                        "completion": comp_txt,
                        "ground_truth": gt,
                    }
                    dump_with_rich(dump_step, os.path.join(group_dir, f"tmp{local_i}.txt"))
            except Exception as e:
                _p(f"dump samples failed: {type(e).__name__}: {e}")

        else:
            _p("non-main: value_forward_server (mirror loop)")
            self._value_forward_server()
            wrapper = [None]
            proj_wrapper = [None]

        # =========================================================================
        # 1.3) Broadcast STEPS
        # =========================================================================
        num_trees_cfg = int(getattr(self.args, "num_trees", -1))
        mse_samples = None

        self.accelerator.wait_for_everyone()
        broadcast_object_list(wrapper, from_process=0)

        msg = wrapper[0]
        if not (isinstance(msg, dict) and msg.get("tag") == "STEPS"):
            raise RuntimeError(f"[rank {self.accelerator.process_index}] Expected STEPS, got {type(msg)}")

        step_samples = msg.get("payload", []) or []

        try:
            num_trees_cfg = int(msg.get("num_trees", num_trees_cfg))
        except Exception:
            num_trees_cfg = int(getattr(self.args, "num_trees", -1))

        mse_samples = msg.get("mse_payload", None)
        self.accelerator.wait_for_everyone()
        broadcast_object_list(proj_wrapper, from_process=0)
        self.accelerator.wait_for_everyone()

        if not step_samples:
            self._metrics["loss"].append(0.0)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # =========================================================================
        # 2) Pack per-step samples into padded tensors
        # =========================================================================
        def _as_1d_long(x):
            if torch.is_tensor(x):
                return x.detach().to(device=device, dtype=torch.long).view(-1)
            return torch.as_tensor(x, device=device, dtype=torch.long).view(-1)

        cleaned = []
        cleaned_meta = []
        for stp in step_samples:
            try:
                p = _as_1d_long(stp["prompt_ids"])[-self.max_prompt_length:]
                c = _as_1d_long(stp["completion_ids"])
                if c.numel() == 0:
                    continue
                cleaned.append((p, c))
                cleaned_meta.append(stp)
            except Exception:
                continue

        if not cleaned:
            self._metrics["loss"].append(0.0)
            return torch.tensor(0.0, device=device, requires_grad=True)

        step_samples = cleaned_meta

        from trl.trainer.utils import pad
        prompt_ids = pad([p for (p, _) in cleaned], padding_value=pad_token)
        completion_ids = pad([c for (_, c) in cleaned], padding_value=pad_token)

        B_total = int(prompt_ids.size(0))
        T = int(completion_ids.size(1))

        # =========================================================================
        # 3) Policy forward: logps + (y_state_new, v_pred_new)
        # =========================================================================
        def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # logits: (B, T, V), labels: (B, T) in [0, V)
            logp = torch.log_softmax(logits, dim=-1)
            return logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        def _completion_eos_mask_1d(ids_1d: torch.Tensor, eos_id_int: int) -> torch.Tensor:
            """
            ids_1d: (Lc,) completion ids (no pad)
            Returns: (Lc,) long mask, with 1 before EOS (inclusive) and 0 after EOS; if there is no EOS, all values ​​are 1.
            """
            m = torch.ones_like(ids_1d, dtype=torch.long)
            if eos_id_int is None:
                return m
            pos = (ids_1d == int(eos_id_int)).nonzero(as_tuple=False)
            if pos.numel() > 0:
                first = int(pos[0].item())
                if first + 1 < m.numel():
                    m[first + 1:] = 0
            return m

        def policy_forward_logps_yv(
            model_wrapped,
            prompt_ids: torch.Tensor,
            completion_ids: torch.Tensor,
            pad_id: int,
            temperature: float,
            eos_id_int: int,
        ):
            """
            Returns (all differentiable w.r.t model params):
              per_token_logps: (B, T) float32
              y_state_ctx    : (B, H) same dtype as y_state
              v_pred_ctx     : (B,) float32
            """
            base_mod = model_wrapped
            if hasattr(base_mod, "module"):  # deepspeed / ddp wrapper
                base_mod = base_mod.module
            base = getattr(base_mod, "base_lm", base_mod)

            device_ = completion_ids.device
            B, T = completion_ids.shape
            temperature = float(temperature) if float(temperature) > 0 else 1.0

            am_prompt = (prompt_ids != pad_id)
            am_comp = (completion_ids != pad_id)

            per_token_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            v_list: List[torch.Tensor] = []

            for i in range(B):
                p_trim = prompt_ids[i][am_prompt[i]].unsqueeze(0)   # (1,Lp)
                c_trim = completion_ids[i][am_comp[i]].unsqueeze(0) # (1,Lc)
                if c_trim.numel() == 0 or p_trim.numel() == 0:
                    # Keep the shape consistent (zero padding), but maintain stackability
                    per_token_list.append(torch.zeros((T,), device=device_, dtype=torch.float32))
                    y_list.append(torch.zeros((1,), device=device_, dtype=torch.float32))  # Placeholder, dimensions will be corrected later.
                    v_list.append(torch.zeros((), device=device_, dtype=torch.float32))
                    continue

                Lp = int(p_trim.size(1))
                Lc = int(c_trim.size(1))

                input_full = torch.cat([p_trim, c_trim], dim=1)  # (1, L)
                attn_full = torch.ones_like(input_full, dtype=torch.long)

                out = base(
                    input_ids=input_full,
                    attention_mask=attn_full,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                logits = out.logits
                last_hidden = out.hidden_states[-1]
                try:
                    out.hidden_states = None
                except Exception:
                    pass
                del out

                logits_next = logits[:, :-1, :] / temperature
                targets = input_full[:, 1:]
                logps_all = selective_log_softmax(logits_next, targets)  # (1, L-1)

                # The prediction position corresponding to completion: starting from (Lp-1), with a length of Lc
                logps_trim = logps_all[:, (Lp - 1):(Lp - 1 + Lc)].squeeze(0)  # (Lc,)
                logps_trim_f = logps_trim.to(torch.float32)                 # (Lc,)
                if Lc < T:
                    pad = torch.zeros((T - Lc,), device=device_, dtype=torch.float32)
                    row = torch.cat([logps_trim_f, pad], dim=0)             # (T,)
                else:
                    row = logps_trim_f[:T]
                per_token_list.append(row)

                # masks: completion up to EOS + prompt span
                c_mask_1d = _completion_eos_mask_1d(c_trim.view(-1), eos_id_int).to(device_)
                resp_full = torch.cat(
                    [torch.zeros((Lp,), device=device_, dtype=torch.long), c_mask_1d],
                    dim=0,
                ).view(1, -1)
                pm_full = torch.cat(
                    [torch.ones((Lp,), device=device_, dtype=torch.long), torch.zeros((Lc,), device=device_, dtype=torch.long)],
                    dim=0,
                ).view(1, -1)

                y_i, v_i = model_wrapped(
                    input_ids=input_full,
                    attention_mask=attn_full,
                    hidden_states=last_hidden,
                    response_mask=resp_full,
                    prompt_mask=pm_full,
                    value_output=True,
                )

                y_list.append(y_i.squeeze(0))              # (H,)
                v_list.append(v_i.squeeze(0).to(torch.float32))  # ()

            per_token_logps = torch.stack(per_token_list, dim=0)  # (B,T)

            # Correct the placeholder dimension of y_state: use the first non-placeholder y to determine H
            # (If all samples are empty, it degenerates to (B,1))
            y_valid = [y for y in y_list if y.dim() == 1 and y.numel() > 1]
            if y_valid:
                H = int(y_valid[0].numel())
                y_state_ctx = []
                for y in y_list:
                    if y.dim() == 1 and y.numel() == H:
                        y_state_ctx.append(y)
                    else:
                        y_state_ctx.append(torch.zeros((H,), device=device_, dtype=y_valid[0].dtype))
                y_state_ctx = torch.stack(y_state_ctx, dim=0)  # (B,H)
            else:
                y_state_ctx = torch.zeros((B, 1), device=device_, dtype=torch.float32)

            v_pred_ctx = torch.stack(v_list, dim=0).view(B)  # (B,)

            return per_token_logps, y_state_ctx, v_pred_ctx

        micro_bs = 1
        logps_chunks = []
        y_chunks = []
        vpred_chunks = []
        t_lm = time.perf_counter()
        for s in range(0, B_total, micro_bs):
            e = min(s + micro_bs, B_total)
            p_chunk = prompt_ids[s:e]
            c_chunk = completion_ids[s:e]
            logps_chunk, y_chunk, vpred_chunk = policy_forward_logps_yv(
                model,
                p_chunk,
                c_chunk,
                pad_token,
                self.args.temperature,
                eos_id,
            )
            logps_chunks.append(logps_chunk)
            y_chunks.append(y_chunk)
            vpred_chunks.append(vpred_chunk)

        per_token_logps = torch.cat(logps_chunks, dim=0)
        y_state_new = torch.cat(y_chunks, dim=0)
        v_pred_new = torch.cat(vpred_chunks, dim=0).to(torch.float32)  # (B_total,) —— h1 head

        t_lm2 = time.perf_counter()
        if self.accelerator.is_main_process:
            _p(f"⏱ policy forward logps+y_state+v_pred: {(t_lm2 - t_lm):.2f}s")

        # completion mask
        B, T = completion_ids.size()
        seq_index = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        is_eos = completion_ids.eq(eos_id)
        has_eos = is_eos.any(dim=1)
        first_eos = is_eos.int().argmax(dim=1)
        eos_end = torch.full((B,), T, dtype=torch.long, device=device)
        eos_end[has_eos] = (first_eos[has_eos] + 1).clamp(max=T)
        completion_mask = ((completion_ids != pad_token) & (seq_index < eos_end.unsqueeze(1))).int()

        mask_f = completion_mask.to(torch.float32)

        # =========================================================================
        # 4) Optional reference KL penalty
        # =========================================================================
        beta = float(getattr(self, "beta", 0.0))
        if beta > 0.0:
            def ref_forward_logps(
                ref_model_like,
                prompt_ids: torch.Tensor,
                completion_ids: torch.Tensor,
                pad_id: int,
                temperature: float,
            ):
                base_ref = getattr(ref_model_like, "base_lm", ref_model_like)
                device_ = completion_ids.device
                B, T = completion_ids.shape
                temperature = float(temperature) if float(temperature) > 0 else 1.0
                am_prompt = (prompt_ids != pad_id)
                am_comp = (completion_ids != pad_id)

                out_logps = torch.zeros((B, T), device=device_, dtype=torch.float32)
                with torch.inference_mode():
                    for i in range(B):
                        p_trim = prompt_ids[i][am_prompt[i]].unsqueeze(0)
                        c_trim = completion_ids[i][am_comp[i]].unsqueeze(0)
                        if c_trim.numel() == 0:
                            continue
                        input_full = torch.cat([p_trim, c_trim], dim=1)
                        out = base_ref(
                            input_ids=input_full,
                            attention_mask=torch.ones_like(input_full),
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                        logits_next = out.logits[:, :-1, :] / temperature
                        targets = input_full[:, 1:]
                        logps_all = selective_log_softmax(logits_next, targets)
                        Lp = int(p_trim.size(1))
                        Lc = int(c_trim.size(1))
                        logps_trim = logps_all[:, (Lp - 1):(Lp - 1 + Lc)]
                        out_logps[i, :Lc] = logps_trim[0].to(torch.float32)
                return out_logps

            try:
                if self.ref_model is not None:
                    base_ref = self.ref_model
                else:
                    uw = self.accelerator.unwrap_model(model)
                    base_ref = getattr(uw, "base_lm", uw)

                ref_micro_bs = int(getattr(self.args, "ref_micro_bs", 1))
                ref_micro_bs = max(1, ref_micro_bs)

                ref_chunks = []
                for s in range(0, B_total, ref_micro_bs):
                    e = min(s + ref_micro_bs, B_total)
                    ref_chunks.append(
                        ref_forward_logps(
                            base_ref,
                            prompt_ids[s:e],
                            completion_ids[s:e],
                            pad_token,
                            getattr(self.args, "temperature", 1.0),
                        )
                    )
                ref_per_token_logps = torch.cat(ref_chunks, dim=0)
            except Exception as e:
                _p(f"ref forward failed: {type(e).__name__}: {e}")
                ref_per_token_logps = torch.zeros_like(per_token_logps)

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                ref_per_token_logps - per_token_logps
            ) - 1.0
        else:
            per_token_kl = None

        # =========================================================================
        # 5) Value loss = MSE(v_pred)        (rank loss removed)
        # =========================================================================
        use_all_nodes_mse = (int(num_trees_cfg) != -1) and (mse_samples is not None) and (len(mse_samples) > 0)
        value_w = float(getattr(self.args, "value_w", 1.0))
        
        if use_all_nodes_mse:
            mse_micro_bs = int(getattr(self.args, "mse_micro_bs", micro_bs))
            mse_micro_bs = max(1, mse_micro_bs)

            # unwrap base lm (for hidden extraction)
            base_mod = model
            if hasattr(base_mod, "module"):  # deepspeed / ddp
                base_mod = base_mod.module
            base = getattr(base_mod, "base_lm", base_mod)
            base_tfm = getattr(base, "model", None)

            # group by prompt_ids (keep your existing grouping behavior)
            groups = defaultdict(list)
            for it in mse_samples:
                pids = it.get("prompt_ids", None)
                if not pids:
                    continue
                try:
                    key = tuple(int(x) for x in pids)
                except Exception:
                    continue
                groups[key].append(it)

            mse_sum = torch.zeros((), device=device, dtype=torch.float32)
            mse_cnt = 0

            for _, items in groups.items():
                # build valid seqs for this group
                seqs: list[torch.Tensor] = []
                rm_list: list[torch.Tensor] = []
                pm_list: list[torch.Tensor] = []
                tgt_list: list[float] = []

                for it in items:
                    try:
                        p = torch.as_tensor(it["prompt_ids"], device=device, dtype=torch.long).view(-1)
                        c = torch.as_tensor(it["completion_ids"], device=device, dtype=torch.long).view(-1)
                    except Exception:
                        continue

                    # Defense: Eliminate any pads that may have been mixed in
                    p = p[p != pad_token]
                    c = c[c != pad_token]
                    if p.numel() == 0 or c.numel() == 0:
                        continue

                    full = torch.cat([p, c], dim=0)  # (L,)
                    c_mask = _completion_eos_mask_1d(c, eos_id).to(device=device, dtype=torch.long)  # (Lc,)
                    resp = torch.cat([torch.zeros_like(p), c_mask], dim=0)  # (L,)
                    pm   = torch.cat([torch.ones_like(p), torch.zeros_like(c)], dim=0)  # (L,)

                    seqs.append(full)
                    rm_list.append(resp)
                    pm_list.append(pm)
                    tgt_list.append(float(it.get("v_target", 0.0)))

                if not seqs:
                    continue

                tgts = torch.as_tensor(tgt_list, device=device, dtype=torch.float32).clamp(0.0, 1.0)

                # forward in micro-batches
                for s in range(0, len(seqs), mse_micro_bs):
                    chunk_seqs = seqs[s : s + mse_micro_bs]
                    chunk_rm   = rm_list[s : s + mse_micro_bs]
                    chunk_pm   = pm_list[s : s + mse_micro_bs]
                    tgt_chunk  = tgts[s : s + mse_micro_bs]

                    Lmax = max(int(x.numel()) for x in chunk_seqs)
                    Bm = len(chunk_seqs)

                    ids_full = torch.full((Bm, Lmax), pad_token, device=device, dtype=torch.long)
                    rm_full  = torch.zeros((Bm, Lmax), device=device, dtype=torch.long)
                    pm_full  = torch.zeros((Bm, Lmax), device=device, dtype=torch.long)

                    for i in range(Bm):
                        li = int(chunk_seqs[i].numel())
                        ids_full[i, :li] = chunk_seqs[i]
                        rm_full[i, :li]  = chunk_rm[i]
                        pm_full[i, :li]  = chunk_pm[i]

                    am_full = (ids_full != pad_token).long()

                    # Base forward (To truncate the gradient, use last_hidden.detach())
                    if base_tfm is not None:
                        out0 = base_tfm(
                            input_ids=ids_full,
                            attention_mask=am_full,
                            use_cache=False,
                            return_dict=True,
                        )
                        last_hidden = out0.last_hidden_state
                    else:
                        out0 = base(
                            input_ids=ids_full,
                            attention_mask=am_full,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                        last_hidden = out0.hidden_states[-1]

                    # value head forward
                    _y, v_pred_chunk = model(
                        input_ids=ids_full,
                        attention_mask=am_full,
                        hidden_states=last_hidden,  # last_hidden.detach()
                        response_mask=rm_full,
                        prompt_mask=pm_full,
                        value_output=True,
                    )
                    v_pred_chunk = v_pred_chunk.to(torch.float32)

                    mse_sum = mse_sum + F.mse_loss(v_pred_chunk, tgt_chunk, reduction="sum")
                    mse_cnt += int(v_pred_chunk.numel())

            value_mse = (mse_sum / float(max(mse_cnt, 1))).to(torch.float32)
        else:
            v_target = torch.as_tensor(
                [float(st.get("v_target", 0.0)) for st in step_samples],
                device=device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)

            value_mse = F.mse_loss(v_pred_new, v_target)

        # value_loss:
        value_loss = value_mse
        # metrics / logging
        self._metrics.setdefault("value_loss", []).append(float(value_loss.item()))
        if hasattr(self, "writer"):
            step_id = self.state.global_step
            self.writer.add_scalar("Loss/ValueLoss", float(value_loss.item()), step_id)

        # =========================================================================
        # 6) Policy loss
        # =========================================================================
        loss_type = str(getattr(self.args, "loss_type")).lower()

        level = str(getattr(self.args, "importance_sampling_level", "token")).lower()
        eps_low = float(getattr(self.args, "epsilon", 0.2))
        eps_high = float(getattr(self.args, "epsilon_high", eps_low))

        rewards_t = torch.as_tensor(
            [float(st["reward"]) for st in step_samples],
            device=device,
            dtype=torch.float32,
        )

        # group id by prompt_ids
        prompt_key_to_gid: dict[tuple, int] = {}
        gid_for_sample: list[int] = []
        for st in step_samples:
            key = tuple(int(t) for t in st["prompt_ids"])
            if key not in prompt_key_to_gid:
                prompt_key_to_gid[key] = len(prompt_key_to_gid)
            gid_for_sample.append(prompt_key_to_gid[key])
        group_ids = torch.tensor(gid_for_sample, device=device, dtype=torch.long)
        K = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0

        # Advantage computation
        scale_rewards = getattr(self.args, "scale_rewards", "group")
        if isinstance(scale_rewards, bool):
            scale_rewards = "group" if scale_rewards else "none"
        scale_rewards = str(scale_rewards).lower()

        if K > 0:
            one = torch.ones_like(rewards_t)
            g_cnt = torch.zeros(K, device=device, dtype=torch.float32)
            g_sum = torch.zeros(K, device=device, dtype=torch.float32)
            g_cnt.scatter_add_(0, group_ids, one)
            g_sum.scatter_add_(0, group_ids, rewards_t)
            g_mean = g_sum / (g_cnt + 1e-8)

            centered = rewards_t - g_mean[group_ids]

            if scale_rewards in ("none", "false", "0"):
                advantages = centered
            elif scale_rewards in ("batch", "global"):
                denom = centered.std(unbiased=False) + 1e-4
                advantages = centered / denom
            else:
                # default "group"
                g_sumsq = torch.zeros(K, device=device, dtype=torch.float32)
                g_sumsq.scatter_add_(0, group_ids, centered * centered)
                g_var = g_sumsq / (g_cnt + 1e-8)  # unbiased=False (population) for stability
                g_std = torch.sqrt(g_var.clamp_min(0.0))
                advantages = centered / (g_std[group_ids] + 1e-4)
        else:
            centered = rewards_t - rewards_t.mean()
            denom = centered.std(unbiased=False) + 1e-4
            advantages = centered / denom

        A = advantages.unsqueeze(1).to(torch.float32)  # (B,1)

        # on-policy "old logps" (single update)
        old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps  # (B,T)

        if level == "token":
            log_w = log_ratio
        elif level == "sequence":
            denom_len = mask_f.sum(-1).clamp(min=1.0)
            seq_log_ratio = (log_ratio * mask_f).sum(-1) / denom_len
            log_w = seq_log_ratio.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance_sampling_level: {level}")

        ratio = torch.exp(log_w)  # (B,T) or (B,1)
        ratio_clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

        obj1 = ratio * A
        obj2 = ratio_clipped * A
        per_token_loss = -torch.minimum(obj1, obj2)

        # KL penalty
        if per_token_kl is not None:
            per_token_loss = per_token_loss + beta * per_token_kl

        max_comp_len = int(getattr(self, "max_completion_length", T))
        if loss_type in ("grpo",):
            policy_loss = ((per_token_loss * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1.0)).mean()
        elif loss_type in ("bnpo",):
            policy_loss = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        elif loss_type in ("dr_grpo",):
            policy_loss = (per_token_loss * mask_f).sum() / (per_token_loss.size(0) * max_comp_len)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self._metrics["policy_loss"].append(float(policy_loss.item()))
        if hasattr(self, "writer"):
            self.writer.add_scalar("Loss/PolicyLoss", float(policy_loss.item()), step_id)
            with torch.no_grad():
                denom_tok = mask_f.sum().clamp(min=1.0)
                if per_token_kl is not None:
                    mean_kl = (per_token_kl * mask_f).sum() / denom_tok
                    self._metrics["kl"].append(float(mean_kl.item()))
                if self._metrics.get("kl"):
                    self.writer.add_scalar("Metrics/KL", self._metrics["kl"][-1], step_id)

        # =========================================================================
        # 7) Final Loss
        # =========================================================================
        loss = policy_loss + value_w * value_loss  # + struc_w * struc_loss
        self._metrics["loss"].append(float(loss.item()))

        prompt_len = (prompt_ids != pad_token).int().sum(dim=1)            # (B,)
        comp_len   = completion_mask.sum(dim=1).int()                      # (B,)
        ctx_len    = (prompt_len + comp_len).to(torch.float32)             # (B,)

        # step_samples and align with prompt_ids/completion_ids
        ans_mask = torch.as_tensor(
            [1.0 if bool(st.get("has_answer", False)) else 0.0 for st in step_samples],
            device=device,
            dtype=torch.float32,
        )  # (B,)
        sum_ctx = (ctx_len * ans_mask).sum()   # scalar
        cnt_ctx = ans_mask.sum()              # scalar
        stats = torch.stack([sum_ctx, cnt_ctx], dim=0).unsqueeze(0)   # (1, 2)
        stats_all = self.accelerator.gather_for_metrics(stats)        # (world, 2)
        if stats_all.dim() == 1:
            stats_all = stats_all.view(-1, 2)
        sum_ctx_all = stats_all[:, 0].sum()
        cnt_ctx_all = stats_all[:, 1].sum()
        avg_ctx_len = (sum_ctx_all / cnt_ctx_all.clamp_min(1.0)).item() if cnt_ctx_all.item() > 0 else 0.0

        self._metrics["context_length"].append(float(avg_ctx_len))

        if hasattr(self, "writer"):
            self.writer.add_scalar("Loss/Loss", float(loss.item()), step_id)
            self.writer.add_scalar("Metrics/ContextLength", float(avg_ctx_len), step_id)

        return loss

    def compute_action_rewards(
        self,
        chains: List[List[dict]],
        reward_fns: List[Callable[[str, Any], float]],
        ground_truth: Any,
        tree_id: int,
        *,
        cot = None, 
        agg_leaf: Callable[[List[float]], float] | None = None,
        agg_internal: Callable[[List[float]], float] | None = None,
        root_step: Optional[dict] = None,
    ) -> tuple[float, float, List[List[dict]]]:
        """
        Compute per-node rewards and potentials on an MCTS tree.

        Core invariants / conventions:
        - We treat the search structure as a DAG built from `chains`
            (each chain is a path of step dicts).
        - "terminal leaf": a node that has no children AND either
                (contains `<answer>...</answer>` in completion) OR
                (depth >= self.depth).
        - avgAcc:     (#correct terminal leaves) / (#all terminal leaves),
                        if there are no terminal leaves, defined as 0.0.
        - pass@1:     Start from the root, greedily follow the child
                        with the largest state_value until the first
                        answer node, and check if it is correct.
        - This function writes back the following fields into each step:
                st["is_leaf"]          : bool (terminal vs non-terminal)
                st["is_correct"]  : bool (terminal and above threshold)
                st["on_path"]  : bool (on any correct leaf path)
                st["state_value"]      : float ∈ [0, 1]
                st["reward"]           : float ∈ [0, 1]
        """
        from functools import lru_cache

        if agg_leaf is None:
            agg_leaf = max
        if agg_internal is None:
            agg_internal = lambda xs: sum(xs) / len(xs)

        passk_threshold  = float(getattr(self, "passk_threshold", 1.0))
        out_dir          = self.args.output_dir

        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("No tokenizer found: expected self.processing_class or self.tokenizer.")

        pad_tok = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        eos_tok = getattr(tokenizer, "eos_token_id", None)

        def _as_1d_long(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().to("cpu", dtype=torch.long).view(-1)
            # list/tuple/np array etc
            return torch.as_tensor(list(x), dtype=torch.long).view(-1)

        def _cot_to_completion_ids(cot_obj):
            """
            Returns 1D LongTensor of completion token ids.
            Accepts:
            - str: tokenize with add_special_tokens=False
            - list[int] / tuple[int] / tensor: treated as already-tokenized ids
            """
            if cot_obj is None:
                return None
            if isinstance(cot_obj, str):
                ids = tokenizer(cot_obj, return_tensors="pt", add_special_tokens=False)["input_ids"].view(-1).long()
            else:
                ids = _as_1d_long(cot_obj)

            if ids is None:
                return None

            # make it "finish-like" (optional but recommended)
            if eos_tok is not None:
                eos_id = int(eos_tok)
                if ids.numel() == 0 or int(ids[-1].item()) != eos_id:
                    ids = torch.cat([ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)
            return ids

        def _embed_prompt_plus_completion(prompt_ids_1d, completion_ids_1d):
            """
            Run value_fn once to get the hyperbolic embedding y(s) for (prompt||completion),
            pooling on completion tokens (response_mask=1 on completion).
            Returns: y (1, D)
            """
            p_ids = _as_1d_long(prompt_ids_1d)
            c_ids = _as_1d_long(completion_ids_1d)
            if p_ids is None or c_ids is None or c_ids.numel() == 0:
                return None

            full_ids = torch.cat([p_ids, c_ids], dim=0)

            # masks (same truncation rule as your batch builder)
            r_mask = torch.cat([torch.zeros_like(p_ids), torch.ones_like(c_ids)], dim=0)
            p_mask = torch.cat([torch.ones_like(p_ids), torch.zeros_like(c_ids)], dim=0)

            max_len = int(getattr(self, "max_model_len", 0) or 0)
            if max_len > 0 and int(full_ids.numel()) > max_len:
                start = int(full_ids.numel()) - max_len
                full_ids = full_ids[start:]
                r_mask   = r_mask[start:]
                p_mask   = p_mask[start:]

            ids2d  = full_ids.view(1, -1)
            attn2d = torch.ones_like(ids2d, dtype=torch.long)
            resp2d = r_mask.view(1, -1)
            pm2d   = p_mask.view(1, -1)

            # device align
            try:
                dev = next(self.model.parameters()).device
            except Exception:
                # fallback: some wrappers
                dev = ids2d.device

            ids2d  = ids2d.to(dev)
            attn2d = attn2d.to(dev)
            resp2d = resp2d.to(dev)
            pm2d   = pm2d.to(dev)

            root_h0 = None
            if root_step is not None and ("root_h0" in root_step):
                rh = root_step["root_h0"]
                root_h0 = rh.detach().to("cpu", dtype=torch.float32).view(-1) if torch.is_tensor(rh) \
                        else torch.as_tensor(rh, dtype=torch.float32).view(-1)

            with torch.no_grad():
                y, _v = self.value_fn(
                    input_ids=ids2d,
                    attention_mask=attn2d,
                    response_mask=resp2d,
                    prompt_mask=pm2d,
                    root_h0=root_h0,
                    return_h0=False,
                )
            return y.detach()

        def _is_terminal_leaf(st: dict, children: set) -> bool:
            """
            A terminal leaf MUST:
            - have no children; and
            - either contain <answer>...</answer> OR have reached max depth.
            """
            if children:
                return False
            if has_answer(st):
                return True
            curd = int(st["current_depth"] or 0)
            return curd >= self.depth

        def _fmt_bonus(c: str) -> float:
            """
            Structural / formatting bonus for ReAct-style steps.

            We reward completions that look like:

                STEP-i:\n
                <think> ... </think>
                [<answer>...</answer> or <tool_call>...</tool_call> or empty]

            Returns:
                1.0 if formatting is considered "good", else 0.0.
            """
            if not re.match(r'^STEP-\d+:\r?\n', c):
                return 0.0
            rest = re.sub(r'^STEP-\d+:\r?\n', '', c, count=1)
            if re.search(r'STEP-\d+:', rest):
                return 0.0
            think = re.match(r'<think>.*?</think>', rest, re.S)
            if not think:
                return 0.0
            remain = rest[think.end():].strip()
            if not remain:
                return 1.0
            return 1.0 if re.fullmatch(r'<answer>.*?</answer>', remain, re.S) or \
                        re.fullmatch(r'<tool_call>.*?</tool_call>', remain, re.S) else 0.0
                        
        # =========================================================================
        # 1) Build graph (DAG) and optionally attach root_step
        # =========================================================================
        ch: dict[int, set[int]] = defaultdict(set)   # adjacency: sid -> children set
        par: dict[int, int] = defaultdict(int)       # in-degree count
        parent_of: dict[int, int] = {}               # parent pointer
        id2: dict[int, dict] = {}                    # sid -> step dict

        for chain in chains:
            for i, st in enumerate(chain):
                sid = id(st); id2[sid] = st
                if i + 1 < len(chain):
                    cid = id(chain[i + 1]); id2[cid] = chain[i + 1]
                    if cid not in ch[sid]:
                        ch[sid].add(cid)
                        par[cid] += 1
                        parent_of.setdefault(cid, sid)
        roots = [sid for sid in id2 if par[sid] == 0]

        root_sid: Optional[int] = None
        if root_step is not None:
            # We explicitly add a "super root" that connects to all existing roots.
            root_sid = id(root_step); id2[root_sid] = root_step
            ch.setdefault(root_sid, set())
            for r in roots:
                ch[root_sid].add(r)
                parent_of[r] = root_sid
            roots = [root_sid]
        for sid in list(id2.keys()):
            ch.setdefault(sid, set())

        # =========================================================================
        # 2) Bottom-up win_rate / value propagation
        # =========================================================================
        @lru_cache(None)
        def dfs_wr(sid: int) -> Optional[float]:
            """
            Recursively compute "win_rate" (scalar reward signal) for each node:

            - For leaf nodes:
                * If terminal, win_rate is the aggregated reward from reward_fns.
                * If non-terminal, win_rate is None (ignored in internal aggregation).
            - For internal nodes:
                * Aggregate all non-None children's win_rate via `agg_internal`.
            """
            st = id2[sid]
            children = ch[sid]
            if not children:
                # Leaf: distinguish between terminal vs non-terminal
                is_leaf = _is_terminal_leaf(st, children)
                st["is_leaf"] = bool(is_leaf)
                if is_leaf:
                    comp = st.get("completion", "")
                    r = agg_leaf([f(comp, ground_truth) for f in reward_fns])
                    st["win_rate"] = float(r)
                    return st["win_rate"]
                else:
                    st["win_rate"] = None
                    return None
            # Internal node: aggregate all defined child win_rates
            vals = []
            for c in children:
                vc = dfs_wr(c)
                if vc is not None:
                    vals.append(float(vc))
            if vals:
                wr = float(agg_internal(vals))
                st["win_rate"] = wr
                st["is_leaf"] = False
                return wr
            else:
                st["win_rate"] = None
                st["is_leaf"] = False
                return None

        for r in roots:
            dfs_wr(r)

        # =========================================================================
        # 3) Count terminal & correct leaves; avgAcc
        # =========================================================================
        terminal_leaf_sids = []
        answered_leaf_sids = []
        correct_leaf_sids  = []
        for sid, st in id2.items():
            if not ch[sid] and bool(st.get("is_leaf", False)):
                terminal_leaf_sids.append(sid)
                if has_answer(st):
                    answered_leaf_sids.append(sid)
                wr = st["win_rate"]
                is_correct = (wr is not None) and (float(wr) >= passk_threshold)
                st["is_correct"] = bool(is_correct)
                if is_correct:
                    correct_leaf_sids.append(sid)
            else:
                st["is_correct"] = False

        if len(terminal_leaf_sids) > 0:
            avgAcc = float(len(correct_leaf_sids)) / float(len(terminal_leaf_sids))
        else:
            avgAcc = 0.0

        # =========================================================================
        # 4) Mark success path nodes (all ancestors of correct leaves)
        # =========================================================================
        success_path_nodes: set[int] = set()
        for leaf_sid in correct_leaf_sids:
            cur = leaf_sid
            while True:
                if cur in success_path_nodes:
                    parent = parent_of.get(cur, None)
                    if parent is None:
                        break
                    cur = parent
                    continue
                success_path_nodes.add(cur)
                parent = parent_of.get(cur, None)
                if parent is None:
                    break
                cur = parent
        for sid in id2.keys():
            id2[sid]["on_path"] = (sid in success_path_nodes)
            
        # ===========================================================================
        # 5) Build V_map(s) for shaping / v_target
        #   - remove CoT anchor entirely
        #   - potential from:
        #       d_goal(s) = min_{g in correct_leaves} d(y(s), y(g))
        #       d_root(s) = d(y(s), y_root)
        #       V(s)      = d_root / (d_root + d_goal + eps)
        # ===========================================================================
        V_map: dict[int, float] = {}
        rho_by_sid: dict[int, float] = {}

        bank = getattr(self, "_hid_bank", None)
        if bank is None or (not chains):
            V_map = {sid: 0.0 for sid in id2.keys()}
        else:
            node_sids, node_idx = [], []
            for sid, st in id2.items():
                idx = st.get("hid_idx", None)
                if idx is not None:
                    node_sids.append(sid)
                    node_idx.append(int(idx))

            if not node_idx:
                V_map = {sid: 0.0 for sid in id2.keys()}
            else:
                with torch.no_grad():
                    Y = bank.index_select(node_idx).to(torch.float32)  # (N, Dp)

                sid2row = {sid: i for i, sid in enumerate(node_sids)}
                c_hyp = float(getattr(self.model, "c", 1.0))
                c_hyp = max(c_hyp, 1e-8)

                # radius diagnostics for pass@1 selection
                rho_all = torch.linalg.norm(Y, dim=-1)  # (N,)
                for sid, row in sid2row.items():
                    rho_by_sid[sid] = float(rho_all[row].item())

                # ---- anchor set for d_goal: (real correct leaves) + (cot anchor if provided) ----
                anchors = []
                # (1) real correct leaves from search
                cr_rows = [sid2row[s] for s in correct_leaf_sids if s in sid2row]
                if len(cr_rows) > 0:
                    corr_rows_t = torch.as_tensor(cr_rows, device=Y.device, dtype=torch.long)
                    anchors.append(Y.index_select(0, corr_rows_t))  # (C, Dp)
                # (2) always include cot as an extra "successful correct leaf" anchor (if provided)
                y_cot = None
                if cot is not None:
                    # choose a prompt_ids to pair with cot
                    p_ids = None
                    if root_step is not None:
                        p_ids = root_step.get("prompt_ids", None)
                    if p_ids is None and roots:
                        # fallback: pick any root node's prompt_ids
                        p_ids = id2[roots[0]].get("prompt_ids", None)

                    c_ids = _cot_to_completion_ids(cot)
                    if p_ids is not None and c_ids is not None:
                        y_cot = _embed_prompt_plus_completion(p_ids, c_ids)
                        if y_cot is not None:
                            y_cot = y_cot.to(device=Y.device, dtype=Y.dtype)  # (1, Dp)
                            anchors.append(y_cot)

                # If still no anchors, it's a dead tree
                if not anchors:
                    V_map = {sid: 0.0 for sid in id2.keys()}
                else:
                    y_root = Y[sid2row[root_sid]]
                    y_corr = torch.cat(anchors, dim=0)  # (C + 1, Dp) if cot exists
                    # ---- distances ----
                    d_goal = poincare_dist_matrix_stable(Y, y_corr, c=c_hyp).min(dim=1).values.float()  # (N,)
                    d_root = poincare_dist_stable(Y, y_root.view(1, -1).expand_as(Y), c=c_hyp).float()  # (N,)

                    eps = 1e-8
                    V_nodes = (d_root / (d_root + d_goal + eps)).clamp(0.0, 1.0)

                    V_map = {}
                    for sid, row in sid2row.items():
                        V_map[sid] = float(V_nodes[row].item())
                    for sid in id2.keys():
                        V_map[sid]

                    # optional logging
                    self._metrics.setdefault("vmap_mean", []).append(float(V_nodes.mean().item()))
                    self._metrics.setdefault("vmap_std",  []).append(float(V_nodes.std(unbiased=False).item()))
                    if hasattr(self, "writer"):
                        step_id = self.state.global_step
                        self.writer.add_scalar("VMap/mean", float(V_nodes.mean().item()), step_id)
                        self.writer.add_scalar("VMap/std",  float(V_nodes.std(unbiased=False).item()), step_id)

            # ---- group-wise masking (keep as before, since no cot anchor now) ----
            group2sids: dict[tuple[int, ...], list[int]] = defaultdict(list)
            group_has_onpath: dict[tuple[int, ...], bool] = defaultdict(bool)

            max_prompt_len = int(getattr(self, "max_prompt_length", 0) or 0)
            pad_key_prefix = ("__no_prompt__",)

            for sid, st in id2.items():
                p_ids = st.get("prompt_ids", None)
                if p_ids is None:
                    key = (pad_key_prefix, sid)
                else:
                    if torch.is_tensor(p_ids):
                        p_list = p_ids.detach().to("cpu").view(-1).tolist()
                    else:
                        p_list = list(p_ids)
                    if max_prompt_len > 0:
                        p_list = p_list[-max_prompt_len:]
                    key = tuple(int(t) for t in p_list)

                group2sids[key].append(sid)
                if bool(st.get("on_path", False)):
                    group_has_onpath[key] = True
            
            # Disable not "on_path" nodes only
            # for sid in id2.keys():
            #     if not bool(id2[sid].get("on_path", False)):
            #         V_map[sid] = 0.0
            # Disable groups without any "on_path" node
            # for key, sids_in_group in group2sids.items():
            #     if not group_has_onpath.get(key, False):
            #         for sid in sids_in_group:
            #             V_map[sid] = 0.0

        # Write back v_target for every node
        for sid, st in id2.items():
            st["v_target"] = float(V_map[sid])

        # =========================================================================
        # 6) pass@1
        # =========================================================================
        # best_leaf = pick_best_leaf(chains, prefer_answer=True)
        # passAt_1 = 1.0 if bool(best_leaf["is_correct"]) else 0.0
        passAt_1 = 0.0
        if answered_leaf_sids:
            best_sid = max(answered_leaf_sids, key=lambda sid: float(id2[sid].get("v_pred", -1e9)))
            passAt_1 = 1.0 if bool(id2[best_sid].get("is_correct", False)) else 0.0

        # =========================================================================
        # 7) Step-level rewards: ΔV + structural bonus / adaptive mixing
        # =========================================================================
        adaptive = bool(getattr(self.args, "adaptive_fmt_bonus", True))
        if not adaptive:
            # Strict formatting
            for sid, st in id2.items():
                if sid == root_sid:
                    st["reward"] = 0.0
                    continue
                p = parent_of.get(sid, None)
                if p is None:
                    st["reward"] = 0.0
                    continue
                dv = V_map[sid] - V_map[p]  # float(max(0.0, V_map[sid] - V_map[p]))
                fb = 1.0 if (_fmt_bonus(st.get("completion", "")) > 0.0) else 0.0
                st["reward"] = dv
        else:
            # Tree-level statistics for adaptive mixing
            fmt_flags = []   # formatting is good (1.0) vs bad (0.0) per edge
            dv_list   = []   # ΔV per edge (child-parent)
            for sid, st in id2.items():
                if sid == root_sid:
                    continue
                p = parent_of.get(sid, None)
                if p is None:
                    continue
                dv = V_map[sid] - V_map[p]
                dv_list.append(dv)
                fmt_flags.append(1.0 if (_fmt_bonus(st.get("completion", "")) > 0.0) else 0.0)

            p_fmt_good = float(np.mean(fmt_flags)) if fmt_flags else 0.0
            terminal_cnt = max(1, len(terminal_leaf_sids))
            leaf_correct_rate = float(len(correct_leaf_sids)) / float(terminal_cnt)

            # "Deficit" signals: how bad we are at formatting vs correctness
            def_fmt  = max(0.0, 1.0 - p_fmt_good)
            def_cont = max(0.0, 1.0 - leaf_correct_rate)

            dv_arr = np.asarray(dv_list, dtype=np.float32)
            dv_var_eps  = float(getattr(self.args, "adapt_dv_var_eps", 1e-12))
            dv_sum_eps  = float(getattr(self.args, "adapt_dv_sum_eps", 1e-9))
            has_dv_sig  = bool((dv_arr.size > 0) and (float(dv_arr.var()) > dv_var_eps)
                                and (float(dv_arr.sum()) > dv_sum_eps))

            alpha_fmt = float(getattr(self.args, "adapt_alpha_fmt", 1.0))
            alpha_dv  = float(getattr(self.args, "adapt_alpha_dv",  1.0))
            raw_fmt   = (def_fmt  ** alpha_fmt)
            raw_dv    = (def_cont ** alpha_dv) if has_dv_sig else 0.0

            eps_w   = float(getattr(self.args, "adapt_eps", 1e-8))
            min_w   = float(getattr(self.args, "adapt_min_weight", 0.0))
            denom   = raw_fmt + raw_dv + eps_w
            w_fmt   = raw_fmt / denom
            w_dv    = raw_dv  / denom
            if (raw_fmt > 0.0) and (raw_dv > 0.0) and (min_w > 0.0):
                w_fmt = float(np.clip(w_fmt, min_w, 1.0 - min_w))
                w_dv  = 1.0 - w_fmt
                
            for sid, st in id2.items():
                if sid == root_sid:
                    st["reward"] = 0.0
                    continue
                p = parent_of.get(sid, None)
                if p is None:
                    st["reward"] = 0.0
                    continue
                # ΔV
                dv = float(max(0.0, V_map[sid] -V_map[p]))
                fb = 1.0 if (_fmt_bonus(st.get("completion", "")) > 0.0) else 0.0

                r = w_dv * dv + w_fmt * fb
                st["reward"] = float(np.clip(r, 0.0, 1.0))
        
        # =========================================================================
        # 8) Visualization (unchanged from original)
        # =========================================================================
        enable_viz = bool(getattr(self.args, "viz", True))
        if enable_viz:
            os.makedirs(out_dir, exist_ok=True)

            # Collect all nodes that have a hyperbolic embedding
            node_sids, node_idx = [], []
            for sid, st in id2.items():
                idx = st.get("hid_idx", None)
                if idx is not None:
                    node_sids.append(sid)
                    node_idx.append(idx)
            if not node_sids:
                raise RuntimeError("No Poincaré embeddings available for visualization.")

            with torch.no_grad():
                Y = self._hid_bank.index_select(node_idx).to(torch.float32)
            N, Dp = int(Y.size(0)), int(Y.size(1))
            sid2row = {sid: i for i, sid in enumerate(node_sids)}

            # Choose root embedding y0 for centering in the disk visualization
            if (root_sid is not None) and (root_sid in sid2row):
                ri = sid2row[root_sid]
                y0 = Y[ri].unsqueeze(0)
            else:
                ri = None
                y0 = torch.zeros((1, Dp), dtype=Y.dtype, device=Y.device)

            c_hyp  = float(getattr(self.model, "c", 1.0))
            rho_cap = float(getattr(self.args, "viz_rho_cap", 0.98))
            Yc = _mobius_add_c(-y0, Y, c=c_hyp)

            if Dp == 2:
                V_tan = logmap0(Yc, c=c_hyp)
                if ri is not None: V_tan[ri].zero_()
                norms = torch.linalg.norm(V_tan, dim=-1)
                if norms.numel() > 0 and float(norms.max()) > 1e-12:
                    target = math.atanh(min(0.999, rho_cap * math.sqrt(c_hyp))) / math.sqrt(c_hyp)
                    V_tan = V_tan * (target / float(norms.max()))
                Y2 = expmap0(V_tan, c=c_hyp).cpu().numpy()
                if ri is not None: Y2[ri, :] = 0.0
            else:
                try:
                    from sklearn.decomposition import PCA as _PCA
                    V_tan = logmap0(Yc, c=c_hyp)
                    if ri is not None: V_tan[ri].zero_()
                    V_np = V_tan.cpu().numpy()
                    m = int(min(50, V_np.shape[1], max(2, N - 1)))
                    V_red = _PCA(n_components=m, random_state=0).fit_transform(V_np) if m < V_np.shape[1] else V_np
                except Exception:
                    V_tan = logmap0(Yc, c=c_hyp)
                    if ri is not None: V_tan[ri].zero_()
                    V_red = V_tan[:, :min(50, V_tan.size(1))].cpu().numpy()
                try:
                    from sklearn.manifold import TSNE as _TSNE
                    perpl = int(getattr(self.args, "viz_tsne_perplexity", 30))
                    perpl = max(5, min(perpl, max(2, N - 1)))
                    tsne = _TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perpl, verbose=False)
                    Z2 = tsne.fit_transform(V_red)
                except Exception as e_tsne:
                    print(f"[viz] TSNE unavailable, fallback to SVD: {type(e_tsne).__name__}: {e_tsne}")
                    try:
                        U, S, Vt = torch.linalg.svd(V_tan, full_matrices=False)
                        Z2 = (V_tan @ Vt[:2, :].T).cpu().numpy()
                    except Exception:
                        Z2 = V_tan[:, :2].cpu().numpy()
                if ri is not None: Z2 = Z2 - Z2[ri][None, :]
                else: Z2 = Z2 - Z2.mean(axis=0, keepdims=True)
                norms = np.linalg.norm(Z2, axis=1)
                if norms.size > 0 and float(norms.max()) > 1e-12:
                    s = math.atanh(min(0.999, rho_cap * math.sqrt(c_hyp))) / (math.sqrt(c_hyp) * float(norms.max()))
                else:
                    s = 1.0
                Z2s = torch.tensor(Z2 * s, dtype=torch.float32)
                Y2 = expmap0(Z2s, c=c_hyp).cpu().numpy()
                if ri is not None: Y2[ri, :] = 0.0

            vals = np.array([float(V_map[sid]) for sid in node_sids], dtype=np.float32)
            v_pred_vals = []
            for sid in node_sids:
                vp = id2[sid].get("v_pred", None)
                if vp is None:
                    v_pred_vals.append(np.nan)
                else:
                    try:
                        v_pred_vals.append(float(vp))
                    except Exception:
                        v_pred_vals.append(np.nan)
                        
            v_pred_vals = np.asarray(v_pred_vals, dtype=np.float32)
            va = getattr(self.model, "value_activation", "sigmoid")
            if str(va).lower() == "sigmoid":
                v_pred_plot = np.clip(np.nan_to_num(v_pred_vals, nan=0.0), 0.0, 1.0)
            else:
                v_pred_plot = np.nan_to_num(v_pred_vals, nan=0.0)

            viz_max_edges = int(getattr(self.args, "viz_max_edges", 3000))
            edges = []
            for p_sid, children in ch.items():
                if (p_sid in sid2row):
                    for c_sid in children:
                        if c_sid in sid2row:
                            edges.append((p_sid, c_sid))
            if len(edges) > viz_max_edges:
                rng = np.random.RandomState(0)
                edges = [edges[i] for i in rng.choice(len(edges), size=viz_max_edges, replace=False).tolist()]

            try:
                def geodesic_arc_2d(x, y, n=24, eps=1e-9):
                    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
                    M = np.array([[2*x[0], 2*x[1]], [2*y[0], 2*y[1]]], dtype=np.float64)
                    b = np.array([x.dot(x) + 1.0, y.dot(y) + 1.0], dtype=np.float64)
                    det = np.linalg.det(M)
                    if abs(det) < eps:
                        t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
                        return (1 - t) * x[None, :] + t * y[None, :]
                    c = np.linalg.solve(M, b)
                    r2 = float(c.dot(c) - 1.0)
                    if r2 <= eps:
                        t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
                        return (1 - t) * x[None, :] + t * y[None, :]
                    r = float(np.sqrt(max(r2, 0.0)))
                    a0 = math.atan2(x[1] - c[1], x[0] - c[0]); a1 = math.atan2(y[1] - c[1], y[0] - c[0])
                    d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                    ts = np.linspace(0.0, 1.0, n, dtype=np.float64)
                    ang = a0 + d * ts
                    return np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)

                def plot_one(fig, ax, cvals, cmap, cbar_label: str, title: str):
                    circ = Circle((0.0, 0.0), radius=1.0, fill=False, linewidth=1.0, linestyle="--", alpha=0.6)
                    ax.add_patch(circ)

                    for (p_sid, c_sid) in edges:
                        pi = sid2row[p_sid]; ci = sid2row[c_sid]
                        pts = geodesic_arc_2d(Y2[pi], Y2[ci], n=24)
                        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.35, alpha=0.25, zorder=1)

                    sc = ax.scatter(
                        Y2[:, 0], Y2[:, 1],
                        c=cvals,
                        cmap=cmap,
                        s=18.0,
                        edgecolors="none",
                        linewidths=0.0,
                        zorder=2,
                    )

                    if ri is not None:
                        ax.scatter([0.0], [0.0], marker="s", s=110, facecolors="none",
                                    edgecolors="red", linewidths=1.2, zorder=3)

                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label(cbar_label, rotation=270, labelpad=12)

                    ax.set_aspect("equal")
                    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(title)

                viz_dpi = int(getattr(self.args, "viz_dpi", 220))

                fig3, axes = plt.subplots(2, 1, figsize=(6.6, 13.2), dpi=viz_dpi)
                plot_one(
                    fig3, axes[0],
                    vals, "viridis", 
                    cbar_label="state value",
                    title="",
                )
                plot_one(
                    fig3, axes[1],
                    v_pred_plot, "coolwarm", 
                    cbar_label="value (estimated)",
                    title="",
                )
                fig3.suptitle(f"Poincaré Disk", y=0.98)
                fig3.tight_layout()
                plt.savefig(os.path.join(out_dir, f"tree{tree_id}_disk.png"), bbox_inches="tight")
                plt.close(fig3)

            except Exception as e:
                print(f"[viz] failed: {type(e).__name__}: {e}")

        return avgAcc, passAt_1, chains

    def self_evolving(self, model_output, ground_truth):
        matches = re.findall(r'<answer>(.*?)</answer>', model_output)
        if matches:
            model_output = matches[-1]
            if str(ground_truth) not in model_output:
                return 0.0
            elif str(ground_truth) == model_output:
                return 0.8
        else:
            return 0.0

        prompt = f"""\
Evaluate the model's answer against the human-annotated ground truth.

## Instructions
1. Return a correctness score **either 0 or 1** (1 represents model_output == ground_truth).  
3. Wrap **only** the final score in `<answer>…</answer>`.  

## Query
{self.question.split('👆')[0]}

## Model Output
{model_output}

## Ground Truth
{ground_truth}"""
        msg = [{"role":"user", "content": prompt}]
        prompt = self.processing_class.apply_chat_template(
            conversation=msg, 
            tokenize=False, 
            add_generation_prompt=True
        )
        judge_sampling_params = copy.copy(self.sampling_params)
        judge_sampling_params.n = 1
        generation_result = self.llm.generate(
            prompts=[prompt], 
            sampling_params=judge_sampling_params, 
            use_tqdm=False
        )
        output_obj = generation_result[0].outputs[0]
        token_ids = output_obj.token_ids
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        result = self.processing_class.decode(token_ids, skip_special_tokens=True)
        
        matches = re.findall(r'<answer>(.*?)</answer>', result)
        if matches:
            try:
                score = float(matches[-1])
                if score == 1.0:
                    return 1.0
                else:
                    return 0.0
            except Exception as e:
                return 0.0
        else:
            return 0.0

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Overridden to incorporate the metrics we collected. 
        """
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()} if self._metrics else {}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()