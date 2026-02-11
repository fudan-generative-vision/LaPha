from __future__ import annotations

from typing import Dict, List
from types import ModuleType
from tools.local_python_executor import LocalPythonExecutor
from tools.tool_base import Tool

import re
import json
import reprlib

import signal
import multiprocessing as mp
import os
import resource
import traceback

from logging import getLogger
from contextlib import contextmanager

from dataclasses import dataclass
from functools import singledispatch


pa = r'(\b[\w\d\(\)\+\-\*/\s]+\b)\s*\*\*\s*(\d{6,}|(\(*\s*\d+\s*\**\s*\**\s*\d+\s*\)*))'
DEFAULT_VARS = {}
logger = getLogger(__file__)


class TimeoutException(Exception):
    """Raised when code execution exceeds the allowed time."""
    pass




@dataclass(frozen=True)
class ReprOptions:
    list_items: int = 10      # 序列/映射最多展示的元素数（对称裁剪）
    str_chars:  int = 120     # 字符串最大长度
    max_depth:  int = 2       # 最大递归深度（避免深层嵌套轰炸）
    np_threshold: int = 20    # numpy.array2string 的 threshold
    df_head: int = 3          # DataFrame/Series 预览 head 行数
    df_tail: int = 3          # DataFrame/Series 预览 tail 行数
    df_max_cols: int = 10     # DataFrame 预览最多显示的列数


def _repr(
    obj,
    *,
    list_items: int = 10,
    str_chars: int = 128,
    max_depth: int = 2,
    np_threshold: int = 20,
    df_head: int = 3,
    df_tail: int = 3,
    df_max_cols: int = 10,
) -> str:

    opts = ReprOptions(
        list_items=list_items,
        str_chars=str_chars,
        max_depth=max_depth,
        np_threshold=np_threshold,
        df_head=df_head,
        df_tail=df_tail,
        df_max_cols=df_max_cols,
    )

    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import pandas as _pd
    except Exception:
        _pd = None

    ELL = object()
    seen = set()

    def shorten_str(s: str) -> str:
        return s if len(s) <= opts.str_chars else s[: opts.str_chars - 1] + "…"

    def clip_seq(seq, limit: int):
        seq = list(seq)
        if limit <= 0 or len(seq) <= limit:
            return seq
        half = limit // 2
        return seq[:half] + [ELL] + seq[-half:]

    def clip_items(items, limit: int):
        items = list(items)
        if limit <= 0 or len(items) <= limit:
            return items
        half = limit // 2
        return items[:half] + [(ELL, ELL)] + items[-half:]

    def bracketed(parts, left, right) -> str:
        return f"{left}{', '.join(parts)}{right}"

    @singledispatch
    def _repr(x, depth: int) -> str:
        r = reprlib.Repr()
        r.maxstring = opts.str_chars
        r.maxother = opts.str_chars
        return r.repr(x)

    @_repr.register(type(None))
    def _repr_none(x, depth):  # noqa
        return "None"

    @_repr.register(bool)
    @_repr.register(float)
    @_repr.register(complex)
    def _repr_atomic(x, depth):  # noqa
        return repr(x)
    
    @_repr.register(int)
    def _repr_int(x: int, depth: int) -> str:  # noqa
        s = repr(x)

        sign = ""
        if s[0] in "+-":
            sign, s = s[0], s[1:]

        def sci_notation_from_str(digits: str, sig: int = 6) -> str:
            if not digits or set(digits) == {"0"}:
                return "0e+0"
            exp = len(digits) - 1
            sig = max(1, sig)
            head = digits[:sig]
            mantissa = head[0] if len(head) == 1 else (head[0] + "." + head[1:])
            return f"{mantissa}e+{exp}"

        sci = sci_notation_from_str(s, sig=16)

        max_len = max(1, opts.str_chars - len(sign))
        if len(s) <= max_len:
            full_disp = sign + s
            return full_disp
        else:
            if max_len <= 2:
                full_disp = sign + s[:1] + "…"
            else:
                keep = (max_len - 1) // 2
                full_disp = f"{sign}{s[:keep]}…{s[-keep:]}"
                
            return f"{full_disp} (≈ {sign}{sci})"

    @_repr.register(str)
    def _repr_str(x: str, depth: int) -> str:
        return shorten_str(x)

    @_repr.register(bytes)
    def _repr_bytes(x: bytes, depth: int) -> str:
        r = reprlib.Repr()
        r.maxstring = opts.str_chars
        return r.repr(x)

    @_repr.register(list)
    def _repr_list(x: list, depth: int) -> str:
        if depth >= opts.max_depth:
            return f"[… {len(x)} items …]"
        parts = []
        for it in clip_seq(x, opts.list_items):
            parts.append("…" if it is ELL else render(it, depth + 1))
        return bracketed(parts, "[", "]")

    @_repr.register(tuple)
    def _repr_tuple(x: tuple, depth: int) -> str:
        fields = getattr(x, "_fields", None)
        if fields and depth < opts.max_depth:
            items = list(zip(fields, x))
            items = clip_items(items, opts.list_items)
            parts = []
            for k, v in items:
                parts.append("…" if k is ELL else f"{k}={render(v, depth + 1)}")
            return f"{type(x).__name__}(" + ", ".join(parts) + ")"
        if depth >= opts.max_depth:
            return f"(… {len(x)} items …)"
        parts = []
        for it in clip_seq(x, opts.list_items):
            parts.append("…" if it is ELL else render(it, depth + 1))
        s = bracketed(parts, "(", ")")
        return s

    @_repr.register(set)
    def _repr_set(x: set, depth: int) -> str:
        if depth >= opts.max_depth:
            return f"{{… {len(x)} items …}}"
        seq = sorted(x, key=lambda e: repr(e))
        parts = []
        for it in clip_seq(seq, opts.list_items):
            parts.append("…" if it is ELL else render(it, depth + 1))
        return bracketed(parts, "{", "}")

    try:
        from collections.abc import Mapping
    except Exception:
        Mapping = dict

    @_repr.register(dict)
    @_repr.register(Mapping)
    def _repr_mapping(x: Mapping, depth: int) -> str:  # noqa
        if depth >= opts.max_depth:
            return f"{{… {len(x)} pairs …}}"
        items = clip_items(x.items(), opts.list_items)
        parts = []
        for k, v in items:
            parts.append("…" if k is ELL else f"{render(k, depth + 1)}: {render(v, depth + 1)}")
        return bracketed(parts, "{", "}")

    if _np is not None:

        @_repr.register(_np.ndarray)
        def _repr_ndarray(x, depth: int) -> str:  # noqa
            header = f"<ndarray shape={x.shape} dtype={x.dtype}>"
            body = _np.array2string(x, threshold=opts.np_threshold)
            return f"{header}\n{body}"

    if _pd is not None:

        @_repr.register(_pd.Series)
        def _repr_series(x, depth: int) -> str:  # noqa
            info = f"<Series len={len(x)} dtype={x.dtype}>"
            head = x.head(opts.df_head).to_string(max_rows=opts.df_head)
            if len(x) > opts.df_head + opts.df_tail and opts.df_tail > 0:
                tail = x.tail(opts.df_tail).to_string(max_rows=opts.df_tail)
                return f"{info}\n{head}\n...\n{tail}"
            return f"{info}\n{head}"

        @_repr.register(_pd.DataFrame)
        def _repr_dataframe(x, depth: int) -> str:  # noqa
            info = f"<DataFrame shape={x.shape}>"
            head = x.head(opts.df_head).to_string(
                max_rows=opts.df_head, max_cols=opts.df_max_cols, show_dimensions=False
            )
            if len(x) > opts.df_head + opts.df_tail and opts.df_tail > 0:
                tail = x.tail(opts.df_tail).to_string(
                    max_rows=opts.df_tail, max_cols=opts.df_max_cols, show_dimensions=False
                )
                return f"{info}\n{head}\n...\n{tail}"
            return f"{info}\n{head}"


    def render(x, depth: int) -> str:
        obj_id = id(x)
        if obj_id in seen:
            return f"<recursion {type(x).__name__}>"
        seen.add(obj_id)
        try:
            return _repr(x, depth)
        finally:
            seen.remove(obj_id)

    return render(obj, depth=0)


def format_variables(variables: dict) -> str:
    """
    Build the nicely truncated dump string.
    """
    pieces = []
    for k, v in list(variables.items())[-10:]:
        if not isinstance(v, ModuleType):
            pieces.append(f"Var: {k}; Type: {type(v).__name__}\n{_repr(v)}")
    return "\n".join(pieces)


@contextmanager
def time_limit(seconds: int):
    """
    Context manager to limit execution time of a code block.
    Works on POSIX systems that support SIGALRM.
    """
    def _handle_timeout(signum, frame):
        raise TimeoutException(f"Execution exceeded {seconds}s time limit")

    original_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)


class IPythonInterpreter(Tool):
    """
    Tool for executing Python code similar to a Jupyter notebook
    """
    
    def __init__(
            self, 
            local_dict, 
            default_timeout: int = 10
        ):
        
        self.default_timeout = default_timeout
        
        name = "ipython_interpreter"
        description = "Execute Python code and return the results, use print() to get detailed infomathon of varibales."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
            },
            "required": ["code"]
        }
        
        super().__init__(name, description, parameters)
        
        # Initialize the execution environment
        self.globals_dict = {}
        self.locals_dict = local_dict if local_dict else DEFAULT_VARS

        self._is_finish = False
        
        # Import commonly used libraries
        # ...
        
        # Initialize python code executor
        self.executor = LocalPythonExecutor(
            additional_authorized_imports=["*"],
        )
        self.executor.send_tools({})
    
    def execute(self, code: str, timeout: int | None = None) -> dict:
        if self.locals_dict:
            self.executor.send_variables(self.locals_dict)

        timeout = self.default_timeout if timeout is None else timeout
        try:
            with time_limit(timeout):
                self.executor(code)
            return {"status": "success", "results": self.executor.state}

        except TimeoutException as e:
            self._is_finish = False
            return {"status": "failed", "results": str(e)}
        except Exception as e:
            self._is_finish = False
            return {"status": "failed", "results": str(e)}

    def batch_execute(self, args_list: List[Dict[str, str]]) -> List[str]:
        results = []
        for args in args_list:
            code:str = args.get('code', '')
            if isinstance(code,str):
                match = re.findall(pa,code)
                if match:
                    logger.error(f"Can not get code: {args}")
                    results.append({"status": 'failed', "result": f'Do not calculate the {match[0]} directily, much use the packages or cause OOM'})
                    continue
            if not code:
                logger.error(f"Can not get code: {args}")

            results.append(self.execute(code))
        return results

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for code execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            if result_obj.get("status") == "success":
                return 0.1  # Reward successful execution
            else:
                return -0.1  # Small penalty for execution errors
        except:    # noqa: E722
            return -0.1  # Penalty for invalid result format

    def reset_environment(self):
        """
        Reset the execution environment to its initial state
        """
        self.globals_dict = {}
        self.locals_dict = {}
        print("[DEBUG] IPython interpreter environment reset")
    
    def is_finish(self, output_str:str) -> bool:
        return self._is_finish


def execute_python_code(code, context: dict = dict(), timeout: int = 10, output_string_limit: int = 1024):
    interpreter = IPythonInterpreter(context)

    state = interpreter.execute(code, timeout=timeout)
    status = state.get("status")
    results = state.get("results")

    if status == "success":    
        results.pop("__name__")
        results.pop("_operations_count")
        print_outputs = results.get("_print_outputs")
        results.pop("_print_outputs")
        
        variables = results.copy()
        context.update(variables)

        output_str = format_variables(
            variables = variables, 
        )
    else:
        output_str = results
        print_outputs = None
    output_str = f" ----- terminal output log ----- \n{print_outputs}\n ------------------------------- \n{output_str}"    
    
    return output_str[: output_string_limit], context


description = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code in a sandboxed environment.\n"
                "Usage (two equivalent forms):\n"
                "  1) As a <tool_call> block with JSON arguments, e.g.:\n"
                "     {\"name\": \"execute_python_code\", \"arguments\": {\"code\": \"...\"}}\n"
                "  2) As a fenced code block after <think></think>:\n"
                "     ```python\n"
                "     ... code ...\n"
                "     ```\n"
                "The system automatically converts the fenced block into a tool call.\n\n"
                "Return behavior:\n"
                "  • Only variables that were explicitly assigned in the code (latest 10) are displayed.\n"
                "  • Module objects are ignored.\n"
                "  • print() output is captured and included.\n\n"
                "Display policy:\n"
                "  • Long strings and large arrays are truncated.\n"
                "  • DataFrame and Series outputs include only head/tail samples.\n"
                "  • Default timeout: 10 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source code to execute. Assign the variables you want to show, "
                            "for example:\n"
                            "  result = solve(...)\n"
                            "  summary = df.describe()\n"
                        ),
                    }
                },
                "required": ["code"]
            }
        }
    }
]



if __name__ == "__main__":
    test_str = """
<think>首先，我们需要分析方程 $x^3 = ax + a + 1$ 有偶数根的条件。因为题目要求根是偶数，我们可以将方程写为 $x^3 - ax - a - 1 = 0$，然后考虑$x$为偶数的情况。
对于偶数根$x=2k$（$k$为整数），代入方程 $x^3 - ax - a - 1 = 0$，可以得到：
[ (2k)^3 - a(2k) - a - 1 = 0 \]
                           
化简得：
[ 8k^3 - 2ak - a - 1 = 0 \]
[ a(2k - 1) = 8k^3 - 1 \]
[ a = \frac{8k^3 - 1}{2k - 1} \]            
                           
我们需要找到所有满足$|x| < 1000$的偶数根对应的$a$值，即$|2k| < 1000$。这意味着$k$的取值范围是$-499 \leq k \leq 
499$。

现在，我们需要验证这些$k$值是否都能使得$a$为实数。我们可以通过计算$a$的值来检查。</think>
<tool_call>                
{"name": "execute_python_code", "arguments": {"code": "results = []
for k in range(-499, 500): 
    a = (8 * k**3 - 1) / (2 * k - 1)
    results.append(a)
results = list(set(results)) # Remove duplicates
len(results)"}}
</tool_call>"""

    def parse_tool_calls(content: str):
        tool_calls = []
        offset = 0
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        decoder = json.JSONDecoder(strict=False)

        for i, m in enumerate(pattern.finditer(content)):
            if i == 0:
                offset = m.start()

            raw = m.group(1).strip()
            func = decoder.decode(raw)

            if isinstance(func.get("arguments"), str):
                func["arguments"] = decoder.decode(func["arguments"])

            tool_calls.append({"type": "function", "function": func})

        if tool_calls:
            c = content[:offset].strip() if offset > 0 and content[:offset].strip() else ""
            return {"role": "assistant", "content": c, "tool_calls": tool_calls}

        return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}
    
    code = parse_tool_calls(test_str)["tool_calls"][0]["function"]["arguments"]["code"]
    print(execute_python_code(code))

