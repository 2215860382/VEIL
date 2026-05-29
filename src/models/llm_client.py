"""Text-only LLM wrapper (causal LM). Which weights you load is entirely ``model_path`` / YAML / CLI.

Used by planner / verifier / text-only answerer paths. Backends:

  - **transformers** (default): ``AutoModelForCausalLM`` from ``model_path``
  - **vllm**: ``use_vllm=True`` — still uses ``model_path`` as the served model id/path
  - **api**: ``api_url`` — OpenAI-compatible ``/v1/chat/completions``; ``api_model`` overrides the id
"""
from __future__ import annotations

import threading
from typing import List

import torch


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}


class LLMClient:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        enable_thinking: bool = False,
        max_new_tokens: int = 512,
        use_vllm: bool = False,
        gpu_memory_utilization: float = 0.85,
        api_url: str | None = None,
        api_model: str | None = None,
    ):
        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self.default_max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.model = None
        self._llm  = None
        self._api_endpoints: list[str] | None = None
        self._api_rr = 0
        self._api_lock = threading.Lock()
        self._api_model = api_model or model_path

        if api_url:
            import requests as _req
            self._requests = _req
            bases = [b.strip() for b in api_url.split(",") if b.strip()]
            self._api_endpoints = [b.rstrip("/") + "/v1/chat/completions" for b in bases]
            if not api_model:
                try:
                    r = _req.get(bases[0].rstrip("/") + "/v1/models", timeout=10)
                    self._api_model = r.json()["data"][0]["id"]
                except Exception:
                    self._api_model = model_path
        elif use_vllm:
            self._init_vllm(model_path, dtype, device, gpu_memory_utilization)
        else:
            self._init_transformers(model_path, dtype, device, attn_impl)

    def _init_transformers(self, model_path: str, dtype: str, device: str, attn_impl: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=_DTYPE_MAP.get(dtype, torch.bfloat16),
            attn_implementation=attn_impl,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._llm = None

    def _init_vllm(self, model_path: str, dtype: str, device: str, gpu_memory_utilization: float):
        import os
        from transformers import AutoTokenizer
        from vllm import LLM

        # Map device string (e.g. "cuda:2") to CUDA_VISIBLE_DEVICES
        if ":" in device:
            gpu_idx = device.split(":")[1]
            existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not existing:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        vllm_dtype = "bfloat16" if dtype in ("bfloat16", "auto") else dtype
        self._llm = LLM(
            model=model_path,
            dtype=vllm_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            trust_remote_code=True,
        )
        self.device = device
        self.model = None

    @torch.inference_mode()
    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        n_tokens = max_new_tokens or self.default_max_new_tokens

        if self._api_endpoints:
            return self._chat_api(messages, n_tokens, temperature, et)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=et,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if self.use_vllm:
            return self._chat_vllm(text, n_tokens, temperature)
        else:
            return self._chat_transformers(text, n_tokens, temperature)

    def _chat_api(self, messages: List[dict], max_new_tokens: int, temperature: float,
                  enable_thinking: bool = False) -> str:
        payload = {
            "model":       self._api_model,
            "messages":    messages,
            "max_tokens":  max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
            # Optional; tokenizer chat templates that support it (e.g. thinking modes) consume this.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        with self._api_lock:
            ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
            self._api_rr += 1
        resp = self._requests.post(ep, json=payload, timeout=120)
        if resp.status_code in (400, 500):
            # Strip image_url items from the last message and retry once.
            msgs = payload["messages"]
            has_images = any(
                isinstance(m.get("content"), list) and
                any(c.get("type") == "image_url" for c in m["content"])
                for m in msgs
            )
            if has_images:
                import logging
                logging.getLogger(__name__).warning(
                    "LLM API %s error with images — retrying without images", resp.status_code
                )
                payload["messages"] = [
                    {**m, "content": (
                        [c for c in m["content"] if c.get("type") != "image_url"]
                        if isinstance(m.get("content"), list) else m["content"]
                    )}
                    for m in msgs
                ]
                with self._api_lock:
                    ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
                    self._api_rr += 1
                resp = self._requests.post(ep, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        if "</think>" in result:
            result = result.split("</think>", 1)[1].strip()
        return result.strip()

    def _chat_transformers(self, text: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()
        return decoded.strip()

    def _chat_vllm(self, text: str, max_new_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
        )
        outputs = self._llm.generate([text], params)
        result = outputs[0].outputs[0].text
        if "</think>" in result:
            result = result.split("</think>", 1)[1].strip()
        return result.strip()
