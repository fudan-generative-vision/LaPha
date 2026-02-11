#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a wrapper checkpoint (LM_Nd_ValueHead / Qwen2HyperbolicProjHead style) into:
  (1) a policy-only HF directory for vLLM,
  (2) a value-head weights file (.pt).

This version is hardened for:
- Qwen2HyperbolicProjHead(new): extra params live in value_block.* (Qwen2DecoderLayer) etc.
- DDP/DeepSpeed saves that prefix keys with "module."
- preserving original dtype (bf16/fp16) when re-saving policy model
- patching config.architectures away from wrapper names
- optionally stripping auto_map for built-in model_types to avoid trust_remote_code in serving

Usage:
  python split_valuehead.py \
    --src /path/to/checkpoint-50 \
    --out-policy /path/to/policy_model_ckpt50 \
    --out-vhead /path/to/value_head_ckpt50.pt \
    --copy-tokenizer \
    --max-shard-size 5GB \
    --trust-remote-code
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from safetensors.torch import load_file as safe_load_file
    HAVE_SAFE = True
except Exception:
    HAVE_SAFE = False


# Minimal map for common model_type -> CausalLM class name (used to patch architectures)
ARCH_MAP = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "mistral": "MistralForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gpt_neox": "GPTNeoXForCausalLM",
    "phi": "PhiForCausalLM",
    "baichuan": "BaichuanForCausalLM",
    "yi": "YiForCausalLM",
}

# Wrapper class names whose architectures we want to "downgrade" to pure CausalLM
WRAPPER_ARCH_NAMES = {
    "LM_Nd_ValueHead",
    "Qwen2HyperbolicProjHead",
}

TOKENIZER_SIDE_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "generation_config.json",
    "configuration.json",
    "README.md",
    "LICENSE",
]


def _strip_module_prefix(k: str) -> str:
    # DDP / DeepSpeed occasionally prefixes "module."
    return k[len("module."):] if k.startswith("module.") else k


def _load_state(fp: Path) -> Dict[str, torch.Tensor]:
    """Load state from a shard file into CPU."""
    if fp.suffix == ".safetensors":
        if not HAVE_SAFE:
            raise RuntimeError("safetensors not installed; can't read *.safetensors")
        return safe_load_file(str(fp), device="cpu")
    return torch.load(str(fp), map_location="cpu")


def _collect_weight_files(src: Path) -> Tuple[list[Path], dict]:
    """Collect all weight files and optional weight_map from index."""
    if not src.exists():
        raise FileNotFoundError(f"src dir does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"src is not a directory: {src}")

    print(f"[info] scanning weights under: {src}")

    idx_files = [
        src / "model.safetensors.index.json",
        src / "pytorch_model.bin.index.json",
    ]
    for idx in idx_files:
        if idx.exists():
            print(f"[info] found index file: {idx.name}")
            with idx.open("r", encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
            shard_names = sorted({name for name in weight_map.values()})
            files = [src / name for name in shard_names]
            return files, weight_map

    # Fallback: single-file candidates
    cands = [src / "model.safetensors", src / "pytorch_model.bin"]
    files = [p for p in cands if p.exists()]
    if not files:
        raise FileNotFoundError(
            f"No model weights found in {src}. "
            f"Expected one of: model.safetensors, pytorch_model.bin, "
            f"or an index json (model.safetensors.index.json / pytorch_model.bin.index.json)."
        )
    print(f"[info] using single weight file(s): {[p.name for p in files]}")
    return files, {}


def _guess_base_prefix(weight_keys: Iterable[str]) -> Optional[str]:
    """
    Guess wrapper base prefix.
    We intentionally do NOT include "model." here (that's real CausalLM prefix).
    """
    cands = [
        "base_lm.",
        "base_model.",
        "policy_model.",
        "backbone.",
        "actor.",
    ]
    keys = list(weight_keys)
    for cand in cands:
        if any(_strip_module_prefix(k).startswith(cand) for k in keys):
            return cand
    return None


def _ensure_architectures(cfg) -> Optional[str]:
    """
    Ensure cfg.architectures has a valid causal LM class name.
    If architectures only contain wrapper names, overwrite them with ARCH_MAP[cfg.model_type].
    """
    archs = list(getattr(cfg, "architectures", []) or [])
    if archs and all(a and a not in WRAPPER_ARCH_NAMES for a in archs):
        return archs[0]

    mt = getattr(cfg, "model_type", None)
    arch = ARCH_MAP.get(mt)
    if arch:
        cfg.architectures = [arch]
    else:
        cfg.architectures = []
    return arch


def _maybe_drop_auto_map(cfg, keep_auto_map: bool) -> None:
    """
    For built-in model_type in ARCH_MAP, auto_map is usually unnecessary and can force
    trust_remote_code in loaders. Drop it unless user asks to keep.
    """
    if keep_auto_map:
        return
    mt = getattr(cfg, "model_type", None)
    if mt in ARCH_MAP and hasattr(cfg, "auto_map"):
        try:
            delattr(cfg, "auto_map")
            print("[config] removed auto_map for built-in model_type to avoid trust_remote_code requirement")
        except Exception:
            # fallback: overwrite to None
            try:
                cfg.auto_map = None
            except Exception:
                pass


def _infer_dtype_from_policy(policy_sd: Dict[str, torch.Tensor]) -> torch.dtype:
    """
    Infer floating dtype from policy weights. Fallback to fp16 if none.
    """
    for _, v in policy_sd.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v.dtype
    return torch.float16


def _dtype_to_cfg_string(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "float16"
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float32:
        return "float32"
    return str(dt).replace("torch.", "")


def _instantiate_policy_model(cfg, trust_remote_code: bool, dtype: torch.dtype):
    """
    Create model from config while preserving dtype as much as possible.
    transformers versions differ: some accept torch_dtype in from_config, some don't.
    """
    # Ensure config records dtype for save_pretrained
    try:
        cfg.torch_dtype = _dtype_to_cfg_string(dtype)
    except Exception:
        pass

    # Try the clean path first
    try:
        return AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code, torch_dtype=dtype)
    except TypeError:
        # Older transformers: no torch_dtype arg
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
        # Cast params so load_state_dict doesn't upcast weights to fp32
        try:
            m.to(dtype=dtype)
        except Exception:
            pass
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Wrapper checkpoint dir (Trainer output)")
    ap.add_argument("--out-policy", type=str, required=True, help="Output dir for policy-only HF model")
    ap.add_argument("--out-vhead", type=str, required=True, help="Output .pt for value head state_dict")
    ap.add_argument("--copy-tokenizer", action="store_true", help="Copy tokenizer/config files into policy dir")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when reading config/building model")
    ap.add_argument("--keep-auto-map", action="store_true", help="Keep config.auto_map in saved policy config.json")
    ap.add_argument("--max-shard-size", type=str, default="5GB", help="HF shard size for saving policy weights")
    ap.add_argument(
        "--base-prefix",
        type=str,
        default="auto",
        help="Wrapper base prefix (e.g. 'base_lm.'). Use 'auto' to detect.",
    )
    ap.add_argument(
        "--save-meta",
        action="store_true",
        help="Write a meta json next to out-vhead describing split details.",
    )
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_policy = Path(args.out_policy).resolve()
    out_vhead = Path(args.out_vhead).resolve()

    files, weight_map = _collect_weight_files(src)

    # Guess base prefix early from weight_map (cheap); else fallback later
    guessed_prefix = None
    if args.base_prefix == "auto":
        if weight_map:
            guessed_prefix = _guess_base_prefix(weight_map.keys())
        # If no index/weight_map, we'll guess after loading first shard.
    else:
        guessed_prefix = args.base_prefix

    # Split:
    policy_sd: Dict[str, torch.Tensor] = {}
    vhead_sd: Dict[str, torch.Tensor] = {}
    non_base_prefixes = set()

    detected_prefix = guessed_prefix

    for i, fp in enumerate(files):
        print(f"[info] loading shard: {fp.name}")
        sd = _load_state(fp)

        # If still unknown, detect from first loaded shard keys
        if detected_prefix is None and args.base_prefix == "auto" and i == 0:
            detected_prefix = _guess_base_prefix(sd.keys())
            if detected_prefix:
                print(f"[split] detected base prefix: {detected_prefix}")

        for k, v in sd.items():
            k = _strip_module_prefix(k)
            if not isinstance(v, torch.Tensor):
                continue

            # If no wrapper prefix found, treat as pure policy model checkpoint
            if detected_prefix is None:
                policy_sd[k] = v.detach().cpu()
                continue

            # 1) base prefix -> policy (strip prefix)
            if k.startswith(detected_prefix):
                new_k = k[len(detected_prefix):]
                policy_sd[new_k] = v.detach().cpu()
                continue

            # 2) wrapper top-level lm_head.* -> policy (rare, but keep)
            if k.startswith("lm_head."):
                policy_sd[k] = v.detach().cpu()
                continue

            # 3) otherwise -> head / aux module weights
            if k.startswith("value_head."):
                vh_key = k[len("value_head."):]
            else:
                vh_key = k
            vhead_sd[vh_key] = v.detach().cpu()
            top = k.split(".", 1)[0]
            non_base_prefixes.add(top)

    if not policy_sd:
        raise RuntimeError(
            "No policy parameters collected. "
            "If this is a wrapper checkpoint, pass --base-prefix explicitly (e.g. --base-prefix base_lm.)."
        )

    if detected_prefix is None:
        print(
            "[warn] Could not detect wrapper base prefix. "
            "Treating checkpoint as pure CausalLM (no value-head extracted)."
        )
    else:
        print(f"[split] base prefix: {detected_prefix}")

    print(f"[split] value-head prefixes (non-base): {sorted(non_base_prefixes)}")
    if not vhead_sd:
        print(
            "[warn] No value-head parameters found outside base prefix. "
            "Expected if use_projection_head=False (no extra params) or checkpoint already pure CausalLM."
        )

    # Infer dtype to preserve
    policy_dtype = _infer_dtype_from_policy(policy_sd)
    print(f"[dtype] inferred policy dtype: {policy_dtype}")

    # Build a proper base config and model for the policy branch
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=args.trust_remote_code)
    resolved_arch = _ensure_architectures(cfg)
    _maybe_drop_auto_map(cfg, keep_auto_map=args.keep_auto_map)
    print(f"[config] model_type={getattr(cfg, 'model_type', None)}, architectures={getattr(cfg, 'architectures', None)}")

    # Instantiate a real base model and load the policy weights
    policy_model = _instantiate_policy_model(cfg, trust_remote_code=args.trust_remote_code, dtype=policy_dtype)

    incompatible = policy_model.load_state_dict(policy_sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    # If lm_head.weight missing, try tying from embed_tokens
    if "lm_head.weight" in missing:
        print("[split] lm_head.weight missing – trying to tie it to model.embed_tokens.weight")
        try:
            with torch.no_grad():
                if hasattr(policy_model, "lm_head") and hasattr(policy_model, "model") and hasattr(policy_model.model, "embed_tokens"):
                    policy_model.lm_head.weight.data.copy_(policy_model.model.embed_tokens.weight.data)
            missing = [k for k in missing if k != "lm_head.weight"]
        except Exception as e:
            print(f"[warn] failed tying lm_head.weight: {e}")

    print(f"[load policy] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first 20 missing:", missing[:20])
    if unexpected:
        print("  first 20 unexpected:", unexpected[:20])

    try:
        policy_model.tie_weights()
    except Exception:
        pass

    # Save policy-only
    out_policy.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(
        out_policy,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    # Copy tokenizer/aux files (optional)
    if args.copy_tokenizer:
        for name in TOKENIZER_SIDE_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, out_policy / name)

    # Save value head
    out_vhead.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vhead_sd, str(out_vhead))

    # Optional meta
    if args.save_meta:
        meta = {
            "src": str(src),
            "out_policy": str(out_policy),
            "out_vhead": str(out_vhead),
            "detected_base_prefix": detected_prefix,
            "num_policy_tensors": len(policy_sd),
            "num_vhead_tensors": len(vhead_sd),
            "vhead_top_prefixes": sorted(list(non_base_prefixes)),
            "inferred_policy_dtype": _dtype_to_cfg_string(policy_dtype),
            "resolved_architecture": resolved_arch,
            "model_type": getattr(cfg, "model_type", None),
        }
        meta_path = out_vhead.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[meta] wrote: {meta_path}")

    # Print saved config summary
    cfg_path = out_policy / "config.json"
    try:
        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(f"[config(saved)] model_type={cfg_json.get('model_type')} architectures={cfg_json.get('architectures')}")
        if "auto_map" in cfg_json:
            print("[config(saved)] auto_map is present (you used --keep-auto-map or model_type unknown).")
    except Exception:
        pass

    print(f"[OK] Policy-only saved to: {out_policy}")
    print(f"[OK] Value head saved to: {out_vhead}")
    if not resolved_arch:
        print(
            "[WARN] Could not resolve architectures from model_type. "
            "You may need --trust-remote-code when serving."
        )


if __name__ == "__main__":
    main()
