"""Qwen3 text-only LLM client (used by planner / verifier / text-only answerer).

Single transformers backend. Pulled in lazily so importing this file doesn't load weights.
"""
from __future__ import annotations

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
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self.default_max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=_DTYPE_MAP.get(dtype, torch.bfloat16),
            attn_implementation=attn_impl,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=et,
            )
        except TypeError:
            # Tokenizers without enable_thinking kw — fall back to plain template.
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        # Strip a stray <think>...</think> if a thinking-trained model returned one anyway.
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()
        return decoded.strip()
