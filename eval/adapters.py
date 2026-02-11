# -*- coding: utf-8 -*-
"""
Engine adapters that expose a vLLM-like generate() interface to the rest of the codebase.

We support exactly TWO backends:
  1) vllm : remote HTTP server via trainer.vllm_client.VLLMClient
  2) jet  : local JetEngine via `from jetengine import LLM, SamplingParams`

Why an adapter?
  - The agents & eval code want a single interface:
        engine.generate(prompts, sampling_params) -> List[LLMOutput]
  - Each LLMOutput must mimic vLLM's schema:
        out.outputs[0].token_ids : List[int]  (COMPLETION-ONLY token ids)
        out.outputs[0].text      : Optional[str]
  - rollouts must write `completion` as a decoded STRING, not a dict/object dump.

Key pitfall (your "completion starts with '{'" bug):
  - JetEngine may sometimes return python dict-like objects (or objects with dict fields),
    and naive fallback `str(req_out)` tokenization produces something like:
        "{'text': '...', 'token_ids': [...]}"

  - If you then decode those "token ids" (tokenized from the dict string),
    you literally get outputs beginning with "{" in completion.
  - Fix: robustly extract *actual* completion token ids / text from request outputs.
    If extraction fails, return empty completion (or raise), but NEVER tokenize `str(req_out)`.

Tensor Parallel (TP) for JetEngine:
  - JetEngine supports multi-GPU *within a single process* via tensor_parallel_size.
  - Do NOT use torchrun data-parallel to "simulate" TP.
  - You control TP with CLI/env:
        --tensor-parallel-size or JET_TP_SIZE
  - JetEngine uses visible GPUs (CUDA_VISIBLE_DEVICES) and TP slices the model across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import inspect
import os

from trainer.vllm_client import VLLMClient, _VLLMServerAdapter
try:
    from transformers import PreTrainedTokenizerBase
except Exception:
    PreTrainedTokenizerBase = object  # type: ignore



def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe getattr with a default."""
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _as_list_int(x: Any) -> Optional[List[int]]:
    """
    Normalize arbitrary token id containers (list/tuple/tensor/numpy/iterable) to List[int].
    """
    if x is None:
        return None
    # Common cases: list/tuple
    if isinstance(x, (list, tuple)):
        try:
            return [int(v) for v in x]
        except Exception:
            return None
    # Torch tensor / numpy array
    if hasattr(x, "tolist"):
        try:
            y = x.tolist()
            if isinstance(y, list):
                return [int(v) for v in y]
        except Exception:
            return None
    # Generic iterable
    try:
        return [int(v) for v in list(x)]
    except Exception:
        return None


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    JetEngine versions may differ. This filters kwargs to only those accepted by `fn(...)`.
    """
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # If signature introspection fails, return kwargs untouched (may raise TypeError later).
        return kwargs

# -----------------------------
# Engine-agnostic sampling config
# -----------------------------
@dataclass
class GenParams:
    """
    Minimal *engine-agnostic* sampling config.

    For vLLM (AR sampling):
      - temperature, top_p, top_k, min_p, repetition_penalty, max_tokens

    For JetEngine / SDAR:
      - It still consumes temp/topk/topp/max_tokens
      - plus diffusion-specific parameters (see SDAR README):
            remasking_strategy, block_length, denoising_steps, dynamic_threshold
    """
    # Common sampling knobs
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 256

    # SDAR / diffusion knobs (JetEngine)
    remasking_strategy: str = "low_confidence_dynamic"
    block_length: int = 4
    denoising_steps: int = 4
    dynamic_threshold: float = 0.9


# -----------------------------
# Output structs (vLLM-like)
# -----------------------------
@dataclass
class LLMResponse:
    """
    vLLM CompletionOutput-like object.

    IMPORTANT:
      token_ids MUST be completion-only tokens (NOT including prompt).
    """
    token_ids: List[int]
    text: Optional[str] = None


@dataclass
class LLMOutput:
    """
    vLLM RequestOutput-like object.

    IMPORTANT:
      prompt_token_ids is best-effort (for debugging). Agents mainly need outputs[0].token_ids.
    """
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[LLMResponse]


# -----------------------------
# JetEngine engine
# -----------------------------
class JetEngine:
    """
    Adapter over JetEngine as used by SDAR.

    TP (tensor parallel) is controlled by `tensor_parallel_size`.
    This is SINGLE-PROCESS multi-GPU (no torchrun needed).

    Typical SDAR usage:
        from jetengine import LLM, SamplingParams
        llm = LLM(model_path, tensor_parallel_size=N, ...)
        outs = llm.generate(prompts, sampling_params) OR generate_streaming(...)
    """
    def __init__(
        self,
        model_path: str,
        tokenizer: "PreTrainedTokenizerBase",
        *,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
        mask_token_id: Optional[int] = None,
        block_length: Optional[int] = None,
        # Optional args (version-dependent)
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_active: Optional[int] = None,
        extra_llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.tp = int(tensor_parallel_size)

        try:
            from jetengine import LLM  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import `jetengine`. Install JetEngine/SDAR runtime in this environment."
            ) from e

        # Build kwargs defensively: JetEngine signatures vary by version.
        llm_kwargs: Dict[str, Any] = {
            "tensor_parallel_size": self.tp,
            "enforce_eager": bool(enforce_eager),
        }
        if mask_token_id is not None:
            llm_kwargs["mask_token_id"] = int(mask_token_id)
        if block_length is not None:
            llm_kwargs["block_length"] = int(block_length)

        # Optional performance knobs
        if max_model_len and int(max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs and int(max_num_seqs) > 0:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_active and int(max_active) > 0:
            llm_kwargs["max_active"] = int(max_active)

        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        # Filter kwargs to what the installed JetEngine actually accepts (prevents TypeError)
        llm_kwargs = _filter_kwargs_for_callable(LLM, llm_kwargs)
        self.llm = LLM(model_path, **llm_kwargs)

    def _to_jet_sampling_params(self, p: Any) -> Any:
        """
        Convert GenParams (or a SamplingParams-like object) into jetengine.SamplingParams.

        SDAR expects names like: topk, topp, dynamic_threshold, denoising_steps, block_length, remasking_strategy.
        """
        from jetengine import SamplingParams  # type: ignore

        # Support both our GenParams and arbitrary attribute objects.
        temperature = float(_get_attr(p, "temperature", 1.0))
        top_k = int(_get_attr(p, "top_k", 0))
        top_p = float(_get_attr(p, "top_p", 1.0))
        max_tokens = int(_get_attr(p, "max_tokens", 256))

        # Diffusion / SDAR params (read from params or env defaults)
        remasking_strategy = str(_get_attr(p, "remasking_strategy", os.getenv("SDAR_REMASKING_STRATEGY", "low_confidence_dynamic")))
        block_length = int(_get_attr(p, "block_length", int(os.getenv("SDAR_BLOCK_LENGTH", "4"))))
        denoising_steps = int(_get_attr(p, "denoising_steps", int(os.getenv("SDAR_DENOISING_STEPS", "4"))))
        dynamic_threshold = float(_get_attr(p, "dynamic_threshold", float(os.getenv("SDAR_DYNAMIC_THRESHOLD", "0.9"))))

        sp_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            topk=top_k,
            topp=top_p,
            max_tokens=max_tokens,
            remasking_strategy=remasking_strategy,
            block_length=block_length,
            denoising_steps=denoising_steps,
            dynamic_threshold=dynamic_threshold,
        )

        # Filter kwargs for SamplingParams signature
        sp_kwargs = _filter_kwargs_for_callable(SamplingParams, sp_kwargs)
        return SamplingParams(**sp_kwargs)

    def _collect_stream_final(self, stream: Iterable[Any]) -> List[Any]:
        """
        JetEngine may return a streaming iterator that yields partial / incremental outputs.
        We collect the final output per request_id in insertion order.
        """
        final_by_id: Dict[str, Any] = {}
        order: List[str] = []

        def _rid(o: Any, fallback: int) -> str:
            for key in ("request_id", "req_id", "id"):
                v = _get_attr(o, key, None)
                if v is not None:
                    return str(v)
            return f"idx{fallback}"

        i = 0
        for item in stream:
            outs = item if isinstance(item, list) else [item]
            for o in outs:
                rid = _rid(o, i)
                if rid not in final_by_id:
                    order.append(rid)
                final_by_id[rid] = o
                i += 1

        return [final_by_id[rid] for rid in order]

    def _extract_completion_from_req_out(self, req_out: Any) -> Tuple[Optional[List[int]], Optional[str], Optional[List[int]]]:
        """
        Extract (completion_token_ids, completion_text, maybe_full_token_ids) from JetEngine output.

        We handle both:
          - vLLM-like objects: req_out.outputs[0].token_ids / .text
          - dict-shaped outputs: {"outputs":[{"token_ids":[...], "text":"..."}], ...}

        Returns:
          completion_ids (preferred), completion_text, full_ids (if we detect full sequence ids)
        """
        # ---- Case 1: dict-shaped output ----
        if isinstance(req_out, dict):
            outs = req_out.get("outputs") or req_out.get("output") or None
            if isinstance(outs, list) and outs:
                o0 = outs[0]
                if isinstance(o0, dict):
                    cids = _as_list_int(o0.get("token_ids") or o0.get("token_ids_list"))
                    txt = o0.get("text")
                    full = _as_list_int(o0.get("all_token_ids") or o0.get("full_token_ids"))
                    return cids, (str(txt) if txt is not None else None), full
                # fallback: outs[0] is an object
                cids = _as_list_int(_get_attr(outs[0], "token_ids", None))
                txt = _get_attr(outs[0], "text", None)
                return cids, (str(txt) if txt is not None else None), None

            # Sometimes top-level dict has token_ids/text directly
            cids = _as_list_int(req_out.get("token_ids"))
            txt = req_out.get("text")
            return cids, (str(txt) if txt is not None else None), None

        # ---- Case 2: object-shaped output (vLLM-like) ----
        outs = _get_attr(req_out, "outputs", None)
        if isinstance(outs, (list, tuple)) and outs:
            o0 = outs[0]
            cids = _as_list_int(_get_attr(o0, "token_ids", None))
            txt = _get_attr(o0, "text", None)
            full = _as_list_int(_get_attr(o0, "all_token_ids", None) or _get_attr(o0, "full_token_ids", None))
            return cids, (str(txt) if txt is not None else None), full

        # Top-level fallbacks (still object)
        cids = _as_list_int(_get_attr(req_out, "token_ids", None))
        txt = _get_attr(req_out, "text", None)
        return cids, (str(txt) if txt is not None else None), None

    def generate(self, prompts: List[str], sampling_params: Any, use_tqdm: bool = False) -> List[LLMOutput]:
        jet_params = self._to_jet_sampling_params(sampling_params)

        # Prefer non-streaming generate() if available; else streaming.
        if hasattr(self.llm, "generate"):
            raw = self.llm.generate(prompts, jet_params)
            req_outs = raw if isinstance(raw, list) else [raw]
        else:
            stream = self.llm.generate_streaming(prompts, jet_params)
            req_outs = self._collect_stream_final(stream)

        results: List[LLMOutput] = []

        for prompt, req_out in zip(prompts, req_outs):
            # Prompt token ids for debugging & optional prefix-stripping
            try:
                prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            except Exception:
                prompt_ids = []

            comp_ids, comp_text, full_ids = self._extract_completion_from_req_out(req_out)

            # If we did not get completion token ids but got text, tokenize the TEXT (not the object!)
            if comp_ids is None and comp_text is not None:
                try:
                    comp_ids = self.tokenizer(comp_text, add_special_tokens=False)["input_ids"]
                except Exception:
                    comp_ids = []

            # If we only got full sequence ids, try to strip prompt prefix to get completion-only ids.
            # This is best-effort and does not crash if mismatch.
            if comp_ids is None and full_ids is not None:
                comp_ids = full_ids

            if comp_ids is None:
                # IMPORTANT: NEVER tokenize(str(req_out)) -> that reintroduces the "{'text':...}" bug.
                comp_ids = []

            # Best-effort: if returned ids *accidentally include the prompt*, strip it.
            if prompt_ids and len(comp_ids) > len(prompt_ids) and comp_ids[: len(prompt_ids)] == prompt_ids:
                comp_ids = comp_ids[len(prompt_ids):]

            results.append(
                LLMOutput(
                    prompt=prompt,
                    prompt_token_ids=list(prompt_ids),
                    outputs=[LLMResponse(token_ids=list(comp_ids), text=comp_text)],
                )
            )

        return results


# -----------------------------
# Factory
# -----------------------------
def build_engine_adapter(
    *,
    name: str,
    tokenizer: "PreTrainedTokenizerBase",
    vllm_client: Optional[Any] = None,
    jet_model_path: Optional[str] = None,
    jet_tp_size: int = 1,
    jet_enforce_eager: bool = True,
    jet_mask_token_id: Optional[int] = None,
    jet_block_length: Optional[int] = None,
    jet_max_model_len: Optional[int] = None,
    jet_max_num_seqs: Optional[int] = None,
    jet_max_active: Optional[int] = None,
) -> Union[VLLMClient, JetEngine]:
    """
    Return an engine adapter with a vLLM-like .generate() interface.

    Only two names are allowed: "vllm" and "jet".
    """
    name = (name or "").strip().lower()
    if name == "vllm":
        vllm_client = VLLMClient(base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"))
        return _VLLMServerAdapter(vllm_client, defaults={})

    if name == "jet":
        if not jet_model_path:
            raise ValueError("ENGINE=jet requires jet_model_path (model directory).")
        return JetEngine(
            model_path=jet_model_path,
            tokenizer=tokenizer,
            tensor_parallel_size=int(jet_tp_size),
            enforce_eager=bool(jet_enforce_eager),
            mask_token_id=jet_mask_token_id,
            block_length=jet_block_length,
            max_model_len=jet_max_model_len,
            max_num_seqs=jet_max_num_seqs,
            max_active=jet_max_active,
        )

    raise ValueError(f"Unknown engine name: {name!r}. Expected 'jet' or 'vllm'.")
