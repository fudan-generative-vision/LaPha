from __future__ import annotations

import re
import json
import signal
import uuid
import reprlib
import logging
import sys
from typing import Dict, List, Optional, Any
from types import ModuleType
from dataclasses import dataclass
from functools import singledispatch
from contextlib import contextmanager
from logging import getLogger

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from tools.local_python_executor import LocalPythonExecutor
    from tools.tool_base import Tool
except ImportError:
    print("Warning: 'tools' module not found. Using MockExecutor.")
    class Tool:
        def __init__(self, name, description, parameters): pass
    
    class LocalPythonExecutor:
        def __init__(self, additional_authorized_imports): self.state = {}
        def send_tools(self, tools): pass
        def send_variables(self, vars): self.state.update(vars)
        def __call__(self, code):
            old_stdout = sys.stdout
            from io import StringIO
            redirected_output = StringIO()
            sys.stdout = redirected_output
            try:
                exec(code, {}, self.state)
                self.state["_print_outputs"] = redirected_output.getvalue()
                self.state["status"] = "success"
            except Exception as e:
                self.state["status"] = "failed"
                raise e
            finally:
                sys.stdout = old_stdout

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pa = r'(\b[\w\d\(\)\+\-\*/\s]+\b)\s*\*\*\s*(\d{6,}|(\(*\s*\d+\s*\**\s*\**\s*\d+\s*\)*))'
SESSIONS: Dict[str, Dict[str, Any]] = {}

class TimeoutException(Exception): pass

@dataclass(frozen=True)
class ReprOptions:
    list_items: int = 10
    str_chars:  int = 120
    max_depth:  int = 2
    np_threshold: int = 20
    df_head: int = 3
    df_tail: int = 3
    df_max_cols: int = 10

def _repr(obj, **kwargs):
    return repr(obj) 
    
def format_variables(variables: dict) -> str:
    pieces = []
    filter_keys = {"_print_outputs", "__builtins__", "status"}
    for k, v in list(variables.items())[-10:]:
        if k in filter_keys or isinstance(v, ModuleType) or k.startswith('_'):
            continue
        pieces.append(f"Var: {k}; Type: {type(v).__name__}\n{_repr(v)}")
    return "\n".join(pieces)

@contextmanager
def time_limit(seconds: int):
    if sys.platform == "win32":
        yield
        return
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
    def __init__(self, local_dict, default_timeout):
        self.default_timeout = default_timeout
        super().__init__("ipython_interpreter", "Execute Python code.", {"code": {"type": "string"}})
        self.locals_dict = local_dict if local_dict is not None else {}
        self.executor = LocalPythonExecutor(additional_authorized_imports=["*"])
        self.executor.send_tools({})

    def execute(self, code: str, timeout: int | None = None) -> dict:
        if isinstance(code, str):
            if re.findall(pa, code):
                return {"status": "failed", "results": "Power expression too large."}
        if self.locals_dict:
            self.executor.send_variables(self.locals_dict)
        timeout = self.default_timeout if timeout is None else timeout
        try:
            with time_limit(timeout):
                self.executor(code)
            return {"status": "success", "results": self.executor.state}
        except Exception as e:
            return {"status": "failed", "results": str(e)}

app = FastAPI()

class ExecuteRequest(BaseModel):
    code: str
    session_id: Optional[str] = None
    timeout: int = 10

class ExecuteResponse(BaseModel):
    session_id: str
    output: str
    status: str
    context: Dict[str, str]

@app.post("/execute", response_model=ExecuteResponse)
async def execute_endpoint(req: ExecuteRequest):
    # 1. Session Management
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {}
    elif session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    
    current_context = SESSIONS[session_id]

    # 2. Execute
    interpreter = IPythonInterpreter(local_dict=current_context, default_timeout=req.timeout)
    state = interpreter.execute(req.code, timeout=req.timeout)
    
    status = state.get("status")
    results = state.get("results")
    
    output_str = ""
    safe_context = {}

    if status == "success" and isinstance(results, dict):
        results.pop("__name__", None)
        results.pop("_operations_count", None)
        print_outputs = results.get("_print_outputs", "")
        results.pop("_print_outputs", None)
        
        current_context.update(results)
        # {format_variables(results)}
        output_str = f"------- terminal output -------\n{print_outputs}\n-------------------------------\n"

        filter_keys = {"__builtins__", "quit", "exit", "In", "Out"}
        for k, v in results.items():
            if k in filter_keys or k.startswith('_') or isinstance(v, ModuleType):
                continue
            safe_context[k] = str(v) 
    else:
        output_str = str(results)

    return ExecuteResponse(
        session_id=session_id,
        output=output_str,
        status=status or "failed",
        context=safe_context
    )

"""
1. pip install gunicorn
2. gunicorn code_server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --max-requests 1000
"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)