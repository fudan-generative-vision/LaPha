# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import base64
import logging
import socket
import time

from io import BytesIO
from typing import Optional, Union
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from torch import nn
from transformers.utils.import_utils import _is_package_available


if _is_package_available("requests"):
    import requests
    from requests import ConnectionError


if _is_package_available("vllm"):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if _is_package_available("vllm_ascend"):
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator(device="cuda")
        >>> client.update_model_params(model)
        ```

        There are several ways to initialize the client:

        ```python
        VLLMClient(base_url="http://localhost:8000")
        VLLMClient(base_url="http://192.168.1.100:8000")
        VLLMClient(host="localhost", server_port=8000)
        VLLMClient(host="192.168.1.100", server_port=8000)
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not _is_package_available("requests"):
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not _is_package_available("vllm"):
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.base_url}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        images: Optional[list] = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
        logprobs: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            images (`list[PIL.Image]` or `None`, *optional*, defaults to `None`):
                List of PIL Images to send along with the prompts.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict` or `None`, *optional*, defaults to `None`):
                Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"{self.base_url}/generate/"

        def pil_to_base64(image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")

        # Sanitize inputs (convert None to server-acceptable defaults)
        n = 1 if n is None else int(n)
        repetition_penalty = 1.0 if repetition_penalty is None else float(repetition_penalty)
        temperature = 1.0 if temperature is None else float(temperature)
        top_p = 1.0 if top_p is None else float(top_p)
        # vLLM uses -1 to disable top-k; keep that convention
        top_k = -1 if top_k is None else int(top_k)
        min_p = 0.0 if min_p is None else float(min_p)
        max_tokens = 16 if max_tokens is None else int(max_tokens)

        # Convert PIL images to base64 strings
        images = [pil_to_base64(img) for img in images] if images else None

        # Merge generation kwargs and ensure logprobs is included if provided
        payload_generation_kwargs = dict(generation_kwargs or {})
        if logprobs is not None and "logprobs" not in payload_generation_kwargs:
            payload_generation_kwargs["logprobs"] = int(logprobs)

        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "images": images,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "generation_kwargs": payload_generation_kwargs,
            },
        )
        if response.status_code == 200:
            # Keep backward compatibility: return only completion_ids
            data = response.json()
            return data
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self, device: Union[torch.device, str, int] = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization.
                Can be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(
            url,
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=device)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.base_url}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")



class _VLLMServerAdapter:
    """
    Thin adapter that:
    - Accepts vLLM-like `generate(prompts, sampling_params, use_tqdm=False)`
    - Normalizes SamplingParams to concrete scalars (no `None`)
    - Sends ALL prompts in a single HTTP call
    - Wraps the response into vLLM-like objects:
        request_outputs = [ SimpleNamespace(outputs=[ SimpleNamespace(token_ids=[...], cumulative_logprob=...) , ... ]), ... ]
    """

    def __init__(self, client, defaults: dict):
        """
        defaults: dict with keys like temperature, top_p, top_k, min_p, repetition_penalty, max_tokens, guided_decoding_regex
        """
        self.client = client
        self.defaults = defaults

    def _get_or(self, sp, name: str, default):
        val = getattr(sp, name, None)
        return default if val is None else val

    def generate(self, prompts, sampling_params, use_tqdm: bool = False):
        """
        vLLM-server adapter:
        - Calls TRL vLLM server in one HTTP request for all prompts.
        - If logprobs are requested (sampling_params.logprobs > 0) and the server returns them,
            we compute cumulative_logprob = sum(token_logprobs) per completion.

        The returned structure mimics vLLM:
        request_outputs = [
            SimpleNamespace(outputs=[
            SimpleNamespace(token_ids=[...], cumulative_logprob=..., token_logprobs=[...]),
            ...
            ]),
            ...
        ]
        """

        def _get_or(sp, name: str, default):
            val = getattr(sp, name, None)
            return default if val is None else val

        def _list_depth(x) -> int:
            """Return nesting depth for python lists: [..]=1, [[..]]=2, [[[..]]]=3. Non-list -> 0."""
            if not isinstance(x, list):
                return 0
            if len(x) == 0:
                return 1
            return 1 + _list_depth(x[0])

        def _normalize_prompt_major(raw, num_prompts: int, n: int):
            """
            Normalize server outputs into:
            per_prompt[p] -> list_of_completions
            where each completion is itself a list (token ids or token logprobs).
            """
            if raw is None:
                return [[None for _ in range(n)] for _ in range(num_prompts)]

            d = _list_depth(raw)

            # Depth 1: [t0, t1, ...]  (single prompt, single completion)
            if d == 1:
                if num_prompts != 1:
                    # Best-effort: assign to the first prompt, empty for others.
                    out = [[None for _ in range(n)] for _ in range(num_prompts)]
                    out[0] = [raw]
                    return out
                return [ [raw] ]

            # Depth 2: [[...], [...], ...]
            if d == 2:
                L = len(raw)

                # Case A: one completion per prompt (common when n==1)
                if L == num_prompts and not (num_prompts == 1 and n > 1):
                    return [[raw[i]] for i in range(num_prompts)]

                # Case B: single prompt, multiple completions (n>1)
                if num_prompts == 1:
                    return [raw]

                # Case C: flattened prompt-major list (length == num_prompts * n)
                if n > 0 and L == num_prompts * n:
                    return [raw[i * n : (i + 1) * n] for i in range(num_prompts)]

                # Case D: fallback chunk by equal split if divisible
                if num_prompts > 0 and (L % num_prompts == 0):
                    per = L // num_prompts
                    return [raw[i * per : (i + 1) * per] for i in range(num_prompts)]

                raise ValueError(f"Unexpected depth-2 server output shape: len={L}, prompts={num_prompts}, n={n}")

            # Depth 3: [[[...], ...], [[...], ...], ...]  (prompt-major already)
            if d >= 3:
                L = len(raw)
                if L == num_prompts:
                    return raw

                # Flattened prompt-major blocks
                if n > 0 and L == num_prompts * n:
                    # This would mean each element is itself a list-of-completions, which is unusual;
                    # still do a best-effort regrouping.
                    return [raw[i * n : (i + 1) * n] for i in range(num_prompts)]

                raise ValueError(f"Unexpected depth-3+ server output shape: len={L}, prompts={num_prompts}, n={n}")

            raise ValueError(f"Unsupported server output type: {type(raw)}")

        def _safe_sum_token_logprobs(token_logprobs):
            """Sum token logprobs safely (expects list[float]-like)."""
            if token_logprobs is None:
                return 0.0
            s = 0.0
            for x in token_logprobs:
                try:
                    s += float(x)
                except Exception:
                    # Ignore non-numeric entries (best-effort).
                    continue
            return float(s)

        prompts = list(prompts)
        num_prompts = len(prompts)

        # Read and sanitize sampling params with fallback to trainer defaults
        n = int(getattr(sampling_params, "n", 1))

        temperature = float(_get_or(sampling_params, "temperature", self.defaults.get("temperature", 1.0)))
        top_p = float(_get_or(sampling_params, "top_p", self.defaults.get("top_p", 1.0)))
        top_k_raw = _get_or(sampling_params, "top_k", self.defaults.get("top_k", -1))
        top_k = -1 if top_k_raw is None else int(top_k_raw)

        min_p_raw = _get_or(sampling_params, "min_p", self.defaults.get("min_p", 0.0))
        min_p = 0.0 if min_p_raw is None else float(min_p_raw)

        repetition_penalty = float(_get_or(sampling_params, "repetition_penalty", self.defaults.get("repetition_penalty", 1.0)))
        max_tokens = int(_get_or(sampling_params, "max_tokens", self.defaults.get("max_tokens", 16)))
        guided_decoding_regex = getattr(sampling_params, "guided_decoding_regex", None) or self.defaults.get("guided_decoding_regex")

        # Request token logprobs if needed (e.g. used for priors / importance sampling / seq scores).
        logprobs_k = int(getattr(sampling_params, "logprobs", 0) or 0)
        want_logprobs = (logprobs_k > 0)

        # One HTTP call for all prompts.
        # IMPORTANT: return_full_response=True so we can read "logprobs" when the server provides them. :contentReference[oaicite:2]{index=2}
        resp = self.client.generate(
            prompts=prompts,
            images=None,
            n=n,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            guided_decoding_regex=guided_decoding_regex,
            generation_kwargs=None,
            logprobs=logprobs_k if want_logprobs else None,
        )

        # Parse server response.
        if isinstance(resp, dict):
            completion_raw = resp.get("completion_ids", None)
            logprobs_raw = resp.get("logprobs", None) if want_logprobs else None
        else:
            # Backward-compat: older client might return only completion_ids.
            completion_raw = resp
            logprobs_raw = None

        # Normalize to prompt-major: per_prompt[p] -> list of completions
        completion_by_prompt = _normalize_prompt_major(completion_raw, num_prompts=num_prompts, n=n)

        # Normalize logprobs structure if available; else fill with None
        if logprobs_raw is not None:
            logprobs_by_prompt = _normalize_prompt_major(logprobs_raw, num_prompts=num_prompts, n=n)
        else:
            logprobs_by_prompt = [[None for _ in range(len(completion_by_prompt[p]))] for p in range(num_prompts)]

        # Build vLLM-like objects
        request_outputs = []
        for p in range(num_prompts):
            outs = []
            comps = completion_by_prompt[p]
            lpss = logprobs_by_prompt[p] if p < len(logprobs_by_prompt) else [None] * len(comps)

            for j, tok_ids in enumerate(comps):
                tok_ids_list = list(tok_ids) if tok_ids is not None else []
                tok_lps = lpss[j] if (j < len(lpss)) else None

                cumulative = _safe_sum_token_logprobs(tok_lps)
                outs.append(
                    SimpleNamespace(
                        token_ids=tok_ids_list,
                        cumulative_logprob=float(cumulative),
                        token_logprobs=tok_lps,  # optional, but useful
                    )
                )

            request_outputs.append(SimpleNamespace(outputs=outs))

        return request_outputs